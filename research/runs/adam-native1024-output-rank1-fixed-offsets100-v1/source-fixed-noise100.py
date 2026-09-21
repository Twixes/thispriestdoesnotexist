"""Guarded 100-iteration native1024 output-rank1 FIXED-NOISE stability experiment.

Starts from verified original weights, never resumes or changes the objective. It does not
estimate Fisher importance, perform main adaptation or approve a serving model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = ROOT / 'research'
GDIR = RESEARCH / 'runs/inference-cpu/ffhq1024/baseline-bundle'
DDIR = RESEARCH / 'models/ffhq1024-discriminator'
DATA = RESEARCH / 'data/flux-priest-domain-v1'
G_HASH = 'f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d'
D_HASH = 'd91ebf17ce8ef94ba50db60d5452583085a4c4723eb7b294daea92510558f1a5'
DATA_HASH = '6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b'
GIB = 2**30
SMOKE_SHA256 = 'a0f58197d99dea71e19f6b1ce0ade2491173bba1576c8e4ba44ed844faf48e63'
MODULATION_SHA256 = 'b032f7965f397a5ebdc93fb1081adaeace651d31a614347123ab83fc46ce456a'
SOURCE_PROBE_SHA256 = 'f77541904b211a60ca6be24f89e22153c63a0d74afd8fb37fbb638191e5d28a1'
OUTPUT_RANK1_PROBE_SHA256 = 'e72acd143ce7d269549e8ed91b75e897a7ca9130b517ff630070aedd6e5cb73e'
EVAL_Z_SOURCE = RESEARCH/'runs/adam-native1024-probing500-v1/eval-z.npz'
EVAL_Z_SHA256 = 'c4eb3bee33c6f9a0502c62b620a1a984aa431d1b43eadcaab1bbc53c8ebe804e'
ITERATIONS = 100
CHECKPOINT_STEPS = (100,)
PREVIEW_STEPS = (0, 10, 25, 50, 100)



def _noise_offset(name):
    return name.endswith('.parametrizations.noise_strength.0.b_vector')


def stability_named_parameters(model):
    from modulation import probing_named_parameters
    return [(n,p) for n,p in probing_named_parameters(model) if not _noise_offset(n)]


def set_stability_grad(model, enabled):
    """Use for every G toggle; never allow additive noise offsets into gradients."""
    model.requires_grad_(False)
    for _,p in stability_named_parameters(model):
        p.requires_grad_(enabled)
    for name,p in model.named_parameters():
        if _noise_offset(name):
            assert not p.requires_grad and p.grad is None, 'Noise offset unexpectedly enabled or accumulated a gradient'


def capture_noise_reference(model, expected_count=None):
    import torch
    offsets={n:p for n,p in model.named_parameters() if _noise_offset(n)}
    if expected_count is not None:
        assert len(offsets)==expected_count
    assert offsets, 'No noise offsets discovered'
    references={}
    for name,offset in offsets.items():
        assert not offset.requires_grad and offset.grad is None and not torch.count_nonzero(offset)
        references[name]=model.get_parameter(name.replace('.0.b_vector','.original')).detach().clone()
    return references


def assert_fixed_noise(model, references, optimizer=None):
    import torch
    offsets={n:p for n,p in model.named_parameters() if _noise_offset(n)}
    assert offsets.keys()==references.keys(), 'Noise offset inventory changed'
    optimizer_ids=set() if optimizer is None else {id(p) for group in optimizer.param_groups for p in group['params']}
    for name,offset in offsets.items():
        original=model.get_parameter(name.replace('.0.b_vector','.original'))
        assert not offset.requires_grad and offset.grad is None, 'Noise offset gradients enabled'
        assert not torch.count_nonzero(offset), 'Noise offset is no longer zero'
        assert not original.requires_grad and original.grad is None, 'Original noise strength unfrozen'
        assert torch.equal(original,references[name]), 'Original noise strength changed'
        assert id(offset) not in optimizer_ids and id(original) not in optimizer_ids, 'Noise parameter entered optimizer'


def expected_optimizer_steps(iterations):
    return {'g': iterations + (iterations + 3)//4, 'd': iterations + (iterations + 15)//16}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(out):
    import psutil
    assert not out.exists(), 'Use a fresh output directory'
    assert sha(GDIR/'generator.safetensors') == G_HASH
    assert sha(DDIR/'D.safetensors') == D_HASH
    assert sha(DATA/'manifest.json') == DATA_HASH
    data = json.loads((DATA/'manifest.json').read_text())
    training = []
    for e in data['entries']:
        if e['split'] != 'train':
            continue
        p = DATA/'train'/f"{e['id']}.png"
        assert sha(p) == e['native_sha256']
        training.append({'id':e['id'], 'path':str(p.relative_to(ROOT)), 'sha256':sha(p)})
    assert len(training) == 20
    assert sha(HERE/'smoke.py') == SMOKE_SHA256, 'Frozen smoke source changed'
    assert sha(HERE/'modulation.py') == MODULATION_SHA256, 'Validated modulation source changed'
    assert sha(HERE/'probe.py') == SOURCE_PROBE_SHA256, 'Frozen source probe changed'
    assert sha(HERE/'probe_output_rank1.py') == OUTPUT_RANK1_PROBE_SHA256
    assert sha(EVAL_Z_SOURCE) == EVAL_Z_SHA256
    sources = [HERE/'probe_output_rank1_fixed_noise100.py', EVAL_Z_SOURCE, HERE/'probe_output_rank1.py', HERE/'probe.py', HERE/'smoke.py', HERE/'modulation.py', HERE/'upstream/provenance.json', GDIR/'model.json', DDIR/'metadata.json']
    sources += sorted((RESEARCH/'vendor/stylegan2-ada-pytorch').rglob('*.py'))
    pins = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    memory = psutil.virtual_memory()
    assert memory.available/memory.total >= .35, 'Need 35% available system memory before launch'
    protocol = {'name':'adam-native1024-output-rank1-fixed-noise100-stability', 'conv_layout':'output_rank1', 'derived_from_probe_sha256':SOURCE_PROBE_SHA256, 'iterations':ITERATIONS, 'checkpoint_steps':list(CHECKPOINT_STEPS), 'derived_from_smoke_sha256':SMOKE_SHA256, 'resolution':1024,
        'device':'cpu', 'threads':1, 'batch':1, 'seed':2026092209,
        'g_weights_sha256':G_HASH, 'd_weights_sha256':D_HASH, 'dataset_manifest_sha256':DATA_HASH,
        'training':training, 'pins':pins, 'g_lr':.002*4/5, 'd_lr':.002*16/17,
        'g_beta2':.99**(4/5), 'd_beta2':.99**(16/17), 'r1_gamma':10, 'r1_every':16,
        'path_weight':2, 'path_every':4, 'path_decay':.01, 'style_mixing_probability':.9,
        'ema_decay':.5**(32/10000), 'augmentation':'random horizontal flip of real images only',
        'preview_seed':2026092210, 'preview_count':4, 'preview_steps':list(PREVIEW_STEPS), 'eval_z_source_sha256':EVAL_Z_SHA256,
        'sampling':'independent Gaussian z; no prompt; psi1; constant synthesis noise for review only; random synthesis noise during updates',
        'deviations':['Output-rank1 convolution layout differs from released AdAM source_flattened control',
                      'All17 original G noise strengths retained;17 additive noise offsets fixed zero and excluded from optimizer, unlike AdAM',
                      'NVIDIA native1024 networks rather than Rosinality256; preserve own equalized scaling',
                      'CPU FP32 batch1 instead of CUDA batch4; minibatch variance is zero, feature includes sqrt(epsilon)',
                      'Start G from original EMA export, D from original checkpoint',
                      'Exactly100 stability iterations only; no Fisher/main adaptation and no quality-candidate/convergence claim'],
        'guards':{'start_available_fraction':.35,'runtime_available_fraction':.20,'rss_gib':12,'swap_growth_mib':512,'seconds':2400},
        'initialization':'fresh original G_ema and D, not the smoke checkpoint',
        'execution_policy':'Forty-minute wall deadline from supervisor start; no restart, resume, changed objective or reduced-resolution fallback',
        'expected_optimizer_steps':expected_optimizer_steps(ITERATIONS),
        'noise_policy':{'offset_count':17,'offset_value':0,'optimizer_excluded':True,'raw_and_ema_fixed':True,'original_strengths_unchanged':True},
        'derived_from_output_rank1_probe_sha256':OUTPUT_RANK1_PROBE_SHA256,
        'production_approved':False,'server_latency_proven':False}
    out.mkdir(parents=True)
    (out/'.gitignore').write_text('*.partial\n')
    write(out/'protocol.json', protocol)
    shutil.copy2(HERE/'probe_output_rank1_fixed_noise100.py',out/'probe_output_rank1_fixed_noise100.py')
    shutil.copy2(HERE/'probe_output_rank1.py',out/'source-output-rank1-probe.py')
    shutil.copy2(HERE/'probe.py',out/'source-probe.py')
    shutil.copy2(HERE/'smoke.py',out/'source-smoke.py')
    shutil.copy2(HERE/'modulation.py',out/'modulation.py')
    return protocol


def supervise(out):
    import psutil
    protocol = prepare(out)
    start = time.monotonic(); swap = psutil.swap_memory().used
    proc = None; failure = None; peak = 0; min_available = 1.0
    try:
        with (out/'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(HERE/'probe_output_rank1_fixed_noise100.py'),'--worker','--output',str(out)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                mem = psutil.virtual_memory()
                try:
                    p = psutil.Process(proc.pid)
                    rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                except psutil.NoSuchProcess:
                    continue
                peak = max(peak,rss); min_available = min(min_available,mem.available/mem.total)
                if rss > 12*GIB: failure='RSS guard'
                elif mem.available/mem.total < .20: failure='available memory guard'
                elif psutil.swap_memory().used-swap > 512*2**20: failure='swap growth guard'
                elif time.monotonic()-start > protocol['guards']['seconds']: failure='wall time guard'
                if failure:
                    os.killpg(proc.pid,signal.SIGKILL); break
                time.sleep(.1)
            code = proc.wait()
        if code != 0 and failure is None: failure=f'worker exit {code}'
        if failure is None:
            result=json.loads((out/'result.json').read_text())
            assert result['complete'] and result['iterations']==ITERATIONS and result['protocol_sha256']==sha(out/'protocol.json')
            assert {'g':result['g_optimizer_steps'],'d':result['d_optimizer_steps']}==expected_optimizer_steps(ITERATIONS)
            assert [c['step'] for c in result['checkpoints']]==list(CHECKPOINT_STEPS)
            for step in CHECKPOINT_STEPS:
                checkpoint_record=json.loads((out/f'checkpoint-{step:03}.json').read_text())
                assert checkpoint_record['complete'] and checkpoint_record['step']==step
                assert checkpoint_record['checkpoint_sha256']==sha(out/checkpoint_record['checkpoint'])
            for step in PREVIEW_STEPS:
                manifest=json.loads((out/f'step-{step:03}'/'manifest.json').read_text())
                assert manifest['complete'] and manifest['raw_and_ema_noise_fixed'] and len(manifest['images'])==8
                for image in manifest['images']:
                    assert sha(out/f'step-{step:03}'/image['path'])==image['sha256']
    except BaseException:
        failure=traceback.format_exc()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        raise
    finally:
        write(out/'supervisor.json',{'complete':failure is None,'failure':failure,
            'seconds':time.monotonic()-start,'peak_rss_gib':peak/GIB,'min_available_fraction':min_available,
            'protocol_sha256':sha(out/'protocol.json')})
    if failure: raise RuntimeError(failure)


def worker(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['pins'].items(): assert sha(ROOT/p)==h, p
    import copy
    import platform
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from safetensors.torch import load_file, save_file
    sys.path.insert(0,str(RESEARCH/'vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator, Discriminator
    from modulation import install_modulation, probing_parameters, probing_named_parameters, set_probing_grad, frozen_parameters, fold_modulation
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(protocol['seed'])
    write(out/'runtime.json',{'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
                             'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads()})
    def digest(state):
        h=hashlib.sha256()
        for n,t in sorted(state.items()):
            t=t.detach().cpu().contiguous(); h.update(n.encode());h.update(str(t.shape).encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
        return h.hexdigest()
    def frozen(model):
        return digest({**dict(frozen_parameters(model)),**dict(model.named_buffers())})
    def state_digest(value):
        h=hashlib.sha256()
        def visit(item):
            if isinstance(item,torch.Tensor):
                h.update(b'tensor');h.update(digest({'value':item}).encode())
            elif isinstance(item,dict):
                h.update(b'dict')
                for key in sorted(item,key=lambda x:(type(x).__name__,repr(x))):
                    visit(key);visit(item[key])
            elif isinstance(item,(list,tuple)):
                h.update(type(item).__name__.encode())
                for child in item:visit(child)
            else:h.update((type(item).__name__+':'+repr(item)).encode())
        visit(value)
        return h.hexdigest()
    def save_tensor(t,path):
        a=((t.detach().cpu()[0].permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
        tmp=Path(str(path)+'.partial')
        Image.fromarray(a).save(tmp,format='PNG');tmp.replace(path)
    gmeta=json.loads((GDIR/'model.json').read_text()); dmeta=json.loads((DDIR/'metadata.json').read_text())
    G=Generator(**gmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    G.load_state_dict(load_file(str(GDIR/'generator.safetensors')),strict=True)
    D=Discriminator(**dmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    D.load_state_dict(load_file(str(DDIR/'D.safetensors')),strict=True)
    eval_rng=torch.Generator().manual_seed(protocol['preview_seed'])
    eval_z=torch.randn(4,512,generator=eval_rng)
    with np.load(EVAL_Z_SOURCE,allow_pickle=False) as fixture:
        assert torch.equal(eval_z,torch.from_numpy(fixture['z'])), 'Existing four review latents changed'
    np.savez(out/'eval-z.npz',z=eval_z.numpy())
    baseline=[]
    with torch.no_grad():
        for i,z in enumerate(eval_z):
            x=G(z[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False)
            baseline.append(x.cpu());save_tensor(x,out/f'baseline-{i:03}.png')
        d_input=baseline[0];d_reference=D(d_input,None,force_fp32=True)
    write(out/'baseline-manifest.json',{'complete':True,'count':4,'protocol_sha256':sha(out/'protocol.json'),
        'eval_z_sha256':sha(out/'eval-z.npz'),'images':[{'path':f'baseline-{i:03}.png','sha256':sha(out/f'baseline-{i:03}.png')} for i in range(4)]})
    inventories={'G':install_modulation(G,component='G',conv_layout='output_rank1'),'D':install_modulation(D,component='D',conv_layout='output_rank1')}
    set_stability_grad(G,True)
    noise_reference=capture_noise_reference(G,expected_count=17)
    inventories['G']['stability_noise_policy']={'frozen_zero_offsets':list(noise_reference),'count':17}
    inventories['G']['currently_enabled_names']=[n for n,p in G.named_parameters() if p.requires_grad]
    inventories['G']['stability_optimizer_parameter_names']=[n for n,_ in stability_named_parameters(G)]
    write(out/'modulation-inventory.json',inventories)
    frozen_g=frozen(G); frozen_d=frozen(D)
    with torch.no_grad():
        for z,x in zip(eval_z,baseline):
            assert torch.equal(G(z[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False),x), 'G initial output changed'
        assert torch.equal(D(d_input,None,force_fp32=True),d_reference), 'D initial output changed'
    write(out/'initial-equality.json',{'generator_exact':True,'discriminator_exact':True,'g_cases':4,'d_cases':1})
    del baseline,d_reference,d_input
    Gema=copy.deepcopy(G).eval().requires_grad_(False)
    Dema=copy.deepcopy(D).eval().requires_grad_(False)
    gp=[p for _,p in stability_named_parameters(G)]; dp=list(probing_parameters(D))
    go=torch.optim.Adam(gp,lr=protocol['g_lr'],betas=(0.0,protocol['g_beta2']))
    do=torch.optim.Adam(dp,lr=protocol['d_lr'],betas=(0.0,protocol['d_beta2']))
    assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
    write(out/'fixed-noise-policy.json',{'offset_names':list(noise_reference),'original_values':{n:float(t) for n,t in noise_reference.items()},
        'optimizer_parameter_names':[n for n,_ in stability_named_parameters(G)],'originals_retained':True})
    train_rng=torch.Generator().manual_seed(protocol['seed']+2)
    optimizer_steps={'g':0,'d':0}
    def apply_step(opt):
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        opt.step()
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        optimizer_steps['g' if opt is go else 'd']+=1
    def gradient_metrics(model):
        stats={}
        for family in ('u_vector','v_vector','b_vector'):
            parameters=[p for n,p in stability_named_parameters(model) if n.endswith(family)]
            nonzero=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in parameters)
            finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
            assert finite and nonzero>0, f'{family} has missing/nonfinite/all-zero gradients'
            stats[family]={'parameters':len(parameters),'nonzero_gradients':nonzero,'finite':finite}
        return stats
    real_images=[]
    for e in protocol['training']:
        p=ROOT/e['path'];assert sha(p)==e['sha256']
        with Image.open(p) as im:
            assert im.size==(1024,1024)
            real_images.append(torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()/127.5-1)
    np.savez(out/'initial-training-rng.npz',torch_rng=torch.get_rng_state().numpy(),sampling_rng=train_rng.get_state().numpy())
    def generate(model, const=False, mix=True):
        z=torch.randn(1,512,generator=train_rng)
        ws=model.mapping(z,None,skip_w_avg_update=True)
        if mix and torch.rand((),generator=train_rng)<.9:
            cut=int(torch.randint(1,model.num_ws,(),generator=train_rng))
            ws2=model.mapping(torch.randn(1,512,generator=train_rng),None,skip_w_avg_update=True)
            ws=torch.cat([ws[:,:cut],ws2[:,cut:]],dim=1)
        image=model.synthesis(ws,noise_mode='const' if const else 'random',force_fp32=True,fused_modconv=False)
        return image,ws
    def update(opt,params,loss):
        assert torch.isfinite(loss), 'nonfinite loss'
        opt.zero_grad(set_to_none=True);loss.backward()
        grads=[p.grad for p in params if p.grad is not None]
        assert grads and all(torch.isfinite(t).all() for t in grads), 'nonfinite/missing gradients'
        assert any(torch.count_nonzero(t)>0 for t in grads), 'all gradients zero'
        apply_step(opt)
        assert all(torch.isfinite(p).all() for p in params), 'nonfinite updated parameters'
        return float(loss.detach())
    def previews(step):
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        folder=out/f'step-{step:03}';folder.mkdir()
        old_rng=torch.get_rng_state();old_train=train_rng.get_state()
        with torch.no_grad():
            for label,model in [('raw',G),('ema',Gema)]:
                for i,z in enumerate(eval_z):
                    save_tensor(model(z[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False),folder/f'{label}-{i:03}.png')
        assert torch.equal(old_rng,torch.get_rng_state()) and torch.equal(old_train,train_rng.get_state())
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        assert len(list(folder.glob('*.png')))==8
        write(folder/'manifest.json',{'complete':True,'step':step,'protocol_sha256':sha(out/'protocol.json'),
            'eval_z_sha256':sha(out/'eval-z.npz'),'rng_unchanged':True,'raw_and_ema_noise_fixed':True,
            'images':[{'path':p.name,'sha256':sha(p)} for p in sorted(folder.glob('*.png'))]})
    def checkpoint(step, current_path_mean):
        assert optimizer_steps==expected_optimizer_steps(step)
        payload={'G':G.state_dict(),'D':D.state_dict(),'Gema':Gema.state_dict(),'Dema':Dema.state_dict(),
            'g_optimizer':go.state_dict(),'d_optimizer':do.state_dict(),
            'torch_rng':torch.get_rng_state(),'sampling_rng':train_rng.get_state(),
            'path_mean':current_path_mean,'iterations':step,'optimizer_steps':dict(optimizer_steps),
            'protocol':protocol,'protocol_sha256':sha(out/'protocol.json'),
            'runtime_sha256':sha(out/'runtime.json'),'modulation_inventory_sha256':sha(out/'modulation-inventory.json')}
        expected=state_digest(payload)
        path=out/f'checkpoint-{step:03}.pt';tmp=Path(str(path)+'.partial')
        torch.save(payload,tmp);tmp.replace(path)
        restored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert state_digest(restored)==expected,'Serialized checkpoint differs'
        # Controlled perturbation verifies restoration without another optimizer update.
        with torch.no_grad():gp[0].add_(1)
        torch.rand(());torch.rand((),generator=train_rng)
        for name,model in [('G',G),('D',D),('Gema',Gema),('Dema',Dema)]:
            model.load_state_dict(restored[name],strict=True)
        go.load_state_dict(restored['g_optimizer']);do.load_state_dict(restored['d_optimizer'])
        torch.set_rng_state(restored['torch_rng']);train_rng.set_state(restored['sampling_rng'])
        restored_path_mean=restored['path_mean'].clone()
        actual={**restored,'G':G.state_dict(),'D':D.state_dict(),'Gema':Gema.state_dict(),'Dema':Dema.state_dict(),
            'g_optimizer':go.state_dict(),'d_optimizer':do.state_dict(),
            'torch_rng':torch.get_rng_state(),'sampling_rng':train_rng.get_state(),
            'path_mean':restored_path_mean,'optimizer_steps':dict(optimizer_steps)}
        assert state_digest(actual)==expected,'Complete model/optimizer/RNG state restoration differs'
        assert frozen(G)==frozen_g and frozen(D)==frozen_d
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        record={'complete':True,'step':step,'checkpoint':path.name,'checkpoint_sha256':sha(path),
            'state_sha256':expected,'protocol_sha256':sha(out/'protocol.json'),
            'snapshot_manifest_sha256':sha(out/f'step-{step:03}'/'manifest.json'),
            'optimizer_steps':dict(optimizer_steps),'model_optimizer_rng_path_state_restored_exactly':True}
        write(out/f'checkpoint-{step:03}.json',record)
        return restored_path_mean,record
    previews(0)
    path_mean=torch.zeros(());metrics=[];checkpoints=[];start=time.monotonic()
    for iteration in range(ITERATIONS):
        tick=time.monotonic();record={'iteration':iteration,'step':iteration+1}
        set_stability_grad(G,False);set_probing_grad(D,True)
        idx=int(torch.randint(20,(),generator=train_rng));real=real_images[idx][None]
        flip=bool(torch.rand((),generator=train_rng)<.5)
        if flip: real=real.flip(-1)
        record.update(real_id=protocol['training'][idx]['id'],horizontal_flip=flip)
        with torch.no_grad(): fake,_=generate(G)
        # Separate backwards lower peak memory without changing the summed loss.
        do.zero_grad(set_to_none=True)
        loss_fake=F.softplus(D(fake,None,force_fp32=True)).mean();loss_fake.backward()
        loss_real=F.softplus(-D(real,None,force_fp32=True)).mean();loss_real.backward()
        assert torch.isfinite(loss_fake) and torch.isfinite(loss_real)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in dp)
        record['d_gradient_families']=gradient_metrics(D)
        apply_step(do);record['d_loss']=float(loss_fake.detach()+loss_real.detach())
        del fake,loss_fake,loss_real
        if iteration%16==0:
            real_r1=real.detach().requires_grad_(True)
            logits=D(real_r1,None,force_fp32=True)
            gradient=torch.autograd.grad(logits.sum(),real_r1,create_graph=True)[0]
            penalty=gradient.square().flatten(1).sum(1).mean()
            record['r1_loss']=update(do,dp,penalty*(10/2)*16+logits.sum()*0)
            del real_r1,logits,gradient,penalty
        set_probing_grad(D,False);set_stability_grad(G,True)
        fake,_=generate(G)
        record['g_loss']=update(go,gp,F.softplus(-D(fake,None,force_fp32=True)).mean())
        record['g_gradient_families']=gradient_metrics(G)
        del fake
        if iteration%4==0:
            fake,ws=generate(G)
            image_noise=torch.randn(fake.shape,generator=train_rng)/1024
            gradient=torch.autograd.grad((fake*image_noise).sum(),ws,create_graph=True)[0]
            lengths=gradient.square().sum(2).mean(1).sqrt()
            target=path_mean+.01*(lengths.mean()-path_mean)
            penalty=(lengths-target).square().mean()
            record['path_loss']=update(go,gp,2*4*penalty+fake[0,0,0,0]*0)
            path_mean=target.detach()
            del fake,ws,image_noise,gradient,lengths,target,penalty
        assert frozen(G)==frozen_g and frozen(D)==frozen_d, 'frozen source changed'
        assert all(torch.isfinite(p).all() for p in gp+dp)
        with torch.no_grad():
            for ema,model in [(Gema,G),(Dema,D)]:
                for pe,p in zip(ema.parameters(),model.parameters()):pe.copy_(p.lerp(pe,protocol['ema_decay']))
        assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
        record.update(raw_and_ema_noise_fixed=True,seconds=time.monotonic()-tick,path_mean=float(path_mean),optimizer_steps=dict(optimizer_steps))
        metrics.append(record)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
        if iteration+1 in PREVIEW_STEPS:
            previews(iteration+1)
        if iteration+1 in CHECKPOINT_STEPS:
            path_mean,checkpoint_record=checkpoint(iteration+1,path_mean)
            checkpoints.append(checkpoint_record)
    assert_fixed_noise(G,noise_reference,go);assert_fixed_noise(Gema,noise_reference)
    folded=fold_modulation(G,inplace=False).eval().requires_grad_(False)
    errors=[]
    with torch.no_grad():
        for z in eval_z:
            a=G(z[None],None,noise_mode='const',force_fp32=True,fused_modconv=False)
            b=folded(z[None],None,noise_mode='const',force_fp32=True,fused_modconv=False)
            errors.append(float((a-b).abs().max()))
            assert torch.equal(a,b),'folded native output differs'
    folded_path=out/'folded-generator.safetensors'
    folded_tmp=Path(str(folded_path)+'.partial')
    save_file({k:v.detach().contiguous() for k,v in folded.state_dict().items()},str(folded_tmp))
    folded_tmp.replace(folded_path)
    assert optimizer_steps==expected_optimizer_steps(ITERATIONS)
    write(out/'result.json',{'complete':True,'iterations':ITERATIONS,'g_optimizer_steps':optimizer_steps['g'],'d_optimizer_steps':optimizer_steps['d'],
        'frozen_g_unchanged':frozen(G)==frozen_g,'frozen_d_unchanged':frozen(D)==frozen_d,
        'checkpoint_model_optimizer_rng_path_state_restored_exactly':True,'checkpoints':checkpoints,'folded_native_max_errors':errors,
        'seconds_training_and_previews':time.monotonic()-start,'metrics':metrics,
        'checkpoint_sha256':sha(out/'checkpoint-100.pt'),'folded_generator_sha256':sha(out/'folded-generator.safetensors'),
        'protocol_sha256':sha(out/'protocol.json'),'unconditional':True,
        'raw_and_ema_noise_fixed':True,'stability_test_only':True,'fisher_estimated':False,'main_adaptation_run':False,'production_approved':False,'server_latency_proven':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:
        try:worker(out)
        except BaseException:
            write(out/'failure.json',{'traceback':traceback.format_exc(),'complete':False});raise
    else:supervise(out)
