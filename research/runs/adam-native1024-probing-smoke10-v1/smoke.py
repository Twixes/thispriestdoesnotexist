"""Guarded ten-iteration native1024 AdAM-style importance-probing experiment.

This tests the full probing loss and portable modulation mechanics. It does not
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
    sources = [HERE/'smoke.py', HERE/'modulation.py', HERE/'upstream/provenance.json', GDIR/'model.json', DDIR/'metadata.json']
    sources += sorted((RESEARCH/'vendor/stylegan2-ada-pytorch').rglob('*.py'))
    pins = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    memory = psutil.virtual_memory()
    assert memory.available/memory.total >= .35, 'Need 35% available system memory before launch'
    protocol = {'name':'adam-native1024-probing-smoke10', 'iterations':10, 'resolution':1024,
        'device':'cpu', 'threads':1, 'batch':1, 'seed':2026092209,
        'g_weights_sha256':G_HASH, 'd_weights_sha256':D_HASH, 'dataset_manifest_sha256':DATA_HASH,
        'training':training, 'pins':pins, 'g_lr':.002*4/5, 'd_lr':.002*16/17,
        'g_beta2':.99**(4/5), 'd_beta2':.99**(16/17), 'r1_gamma':10, 'r1_every':16,
        'path_weight':2, 'path_every':4, 'path_decay':.01, 'style_mixing_probability':.9,
        'ema_decay':.5**(32/10000), 'augmentation':'random horizontal flip of real images only',
        'preview_seed':2026092210, 'preview_count':4, 'preview_steps':[0,5,10],
        'sampling':'independent Gaussian z; no prompt; psi1; constant synthesis noise for review only; random synthesis noise during updates',
        'deviations':['NVIDIA native1024 networks rather than Rosinality256; preserve own equalized scaling',
                      'CPU FP32 batch1 instead of CUDA batch4; minibatch variance is zero, feature includes sqrt(epsilon)',
                      'Start G from original EMA export, D from original checkpoint',
                      'Ten probing iterations only; no Fisher/main adaptation and no convergence claim'],
        'guards':{'start_available_fraction':.35,'runtime_available_fraction':.20,'rss_gib':12,'swap_growth_mib':512,'seconds':1200},
        'production_approved':False,'server_latency_proven':False}
    out.mkdir(parents=True)
    write(out/'protocol.json', protocol)
    shutil.copy2(HERE/'smoke.py',out/'smoke.py')
    shutil.copy2(HERE/'modulation.py',out/'modulation.py')
    return protocol


def supervise(out):
    import psutil
    protocol = prepare(out)
    start = time.monotonic(); swap = psutil.swap_memory().used
    proc = None; failure = None; peak = 0; min_available = 1.0
    try:
        with (out/'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(HERE/'smoke.py'),'--worker','--output',str(out)],
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
                elif time.monotonic()-start > 1200: failure='wall time guard'
                if failure:
                    os.killpg(proc.pid,signal.SIGKILL); break
                time.sleep(.1)
            code = proc.wait()
        if code != 0 and failure is None: failure=f'worker exit {code}'
        if failure is None:
            result=json.loads((out/'result.json').read_text())
            assert result['complete'] and result['iterations']==10 and result['protocol_sha256']==sha(out/'protocol.json')
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
    def tensor_digest(t): return digest({'tensor':t})
    def save_tensor(t,path):
        a=((t.detach().cpu()[0].permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
        Image.fromarray(a).save(path)
    gmeta=json.loads((GDIR/'model.json').read_text()); dmeta=json.loads((DDIR/'metadata.json').read_text())
    G=Generator(**gmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    G.load_state_dict(load_file(str(GDIR/'generator.safetensors')),strict=True)
    D=Discriminator(**dmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    D.load_state_dict(load_file(str(DDIR/'D.safetensors')),strict=True)
    eval_rng=torch.Generator().manual_seed(protocol['preview_seed'])
    eval_z=torch.randn(4,512,generator=eval_rng)
    np.savez(out/'eval-z.npz',z=eval_z.numpy())
    baseline=[]
    with torch.no_grad():
        for i,z in enumerate(eval_z):
            x=G(z[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False)
            baseline.append(x.cpu());save_tensor(x,out/f'baseline-{i:03}.png')
        d_input=baseline[0];d_reference=D(d_input,None,force_fp32=True)
    inventories={'G':install_modulation(G,component='G'),'D':install_modulation(D,component='D')}
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
    gp=list(probing_parameters(G)); dp=list(probing_parameters(D))
    go=torch.optim.Adam(gp,lr=protocol['g_lr'],betas=(0,protocol['g_beta2']))
    do=torch.optim.Adam(dp,lr=protocol['d_lr'],betas=(0,protocol['d_beta2']))
    train_rng=torch.Generator().manual_seed(protocol['seed']+2)
    optimizer_steps={'g':0,'d':0}
    def apply_step(opt):
        opt.step()
        optimizer_steps['g' if opt is go else 'd']+=1
    def gradient_metrics(model):
        stats={}
        for family in ('u_vector','v_vector','b_vector'):
            parameters=[p for n,p in probing_named_parameters(model) if n.endswith(family)]
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
        folder=out/f'step-{step:03}';folder.mkdir()
        old_rng=torch.get_rng_state();old_train=train_rng.get_state()
        with torch.no_grad():
            for label,model in [('raw',G),('ema',Gema)]:
                for i,z in enumerate(eval_z):
                    save_tensor(model(z[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False),folder/f'{label}-{i:03}.png')
        assert torch.equal(old_rng,torch.get_rng_state()) and torch.equal(old_train,train_rng.get_state())
    previews(0)
    path_mean=torch.zeros(());metrics=[];start=time.monotonic()
    for iteration in range(10):
        tick=time.monotonic();record={'iteration':iteration,'step':iteration+1}
        set_probing_grad(G,False);set_probing_grad(D,True)
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
        set_probing_grad(D,False);set_probing_grad(G,True)
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
        record.update(seconds=time.monotonic()-tick,path_mean=float(path_mean))
        metrics.append(record)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
        if iteration+1 in (5,10):previews(iteration+1)
    checkpoint={'G':G.state_dict(),'D':D.state_dict(),'Gema':Gema.state_dict(),'Dema':Dema.state_dict(),
                'g_optimizer':go.state_dict(),'d_optimizer':do.state_dict(),
                'torch_rng':torch.get_rng_state(),'sampling_rng':train_rng.get_state(),
                'path_mean':path_mean,'iterations':10,'protocol_sha256':sha(out/'protocol.json')}
    torch.save(checkpoint,out/'checkpoint.pt')
    restored=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    expected={k:digest(checkpoint[k]) for k in ('G','D','Gema','Dema')}
    with torch.no_grad():gp[0].add_(1)
    G.load_state_dict(restored['G'],strict=True);D.load_state_dict(restored['D'],strict=True);Gema.load_state_dict(restored['Gema'],strict=True);Dema.load_state_dict(restored['Dema'],strict=True)
    go.load_state_dict(restored['g_optimizer']);do.load_state_dict(restored['d_optimizer'])
    torch.set_rng_state(restored['torch_rng']);train_rng.set_state(restored['sampling_rng'])
    assert all(digest(m.state_dict())==expected[k] for k,m in [('G',G),('D',D),('Gema',Gema),('Dema',Dema)])
    folded=fold_modulation(G,inplace=False).eval().requires_grad_(False)
    errors=[]
    with torch.no_grad():
        for z in eval_z:
            a=G(z[None],None,noise_mode='const',force_fp32=True,fused_modconv=False)
            b=folded(z[None],None,noise_mode='const',force_fp32=True,fused_modconv=False)
            errors.append(float((a-b).abs().max()))
            assert torch.equal(a,b),'folded native output differs'
    save_file({k:v.detach().contiguous() for k,v in folded.state_dict().items()},str(out/'folded-generator.safetensors'))
    assert optimizer_steps=={'g':13,'d':11}
    write(out/'result.json',{'complete':True,'iterations':10,'g_optimizer_steps':optimizer_steps['g'],'d_optimizer_steps':optimizer_steps['d'],
        'frozen_g_unchanged':frozen(G)==frozen_g,'frozen_d_unchanged':frozen(D)==frozen_d,
        'checkpoint_model_state_restored_exactly':True,'folded_native_max_errors':errors,
        'seconds_training_and_previews':time.monotonic()-start,'metrics':metrics,
        'checkpoint_sha256':sha(out/'checkpoint.pt'),'folded_generator_sha256':sha(out/'folded-generator.safetensors'),
        'protocol_sha256':sha(out/'protocol.json'),'unconditional':True,
        'fisher_estimated':False,'main_adaptation_run':False,'production_approved':False,'server_latency_proven':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:
        try:worker(out)
        except BaseException:
            write(out/'failure.json',{'traceback':traceback.format_exc(),'complete':False});raise
    else:supervise(out)
