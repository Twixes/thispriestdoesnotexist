"""Prepared supervised continuation of a successful retained250 native100.

No automatic launch/restart. The exact frozen AdaptationEngine is imported;
all source G/D/GEMA/Adam/RNG/path/mask state is restored before global step101.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import main_adaptation_fixed_offsets250_10 as base
import continue_main_adaptation_fixed_offsets250_100 as previous

HERE, ROOT, RESEARCH = base.HERE, base.ROOT, base.RESEARCH
sha, read, write, under_root = base.sha, base.read, base.write, base.under_root
state_digest, expected_steps = base.state_digest, base.expected_steps
AdaptationEngine = base.AdaptationEngine
PARENT_RUNNER_SHA = '7a23948814df313e106057c903a73675cfec542988017f1f7ffbc116b48a2b23'
SOURCE_RUN = 'research/runs/adam-native1024-retained250-main10-to100-v1'
SOURCE_PROTOCOL_SHA = 'a21187fa9476db357b144067e084847f394d82730827d82f94f23c4251d915b9'
START = 100
END = 250
CHECKPOINT_STEPS = (150,250)


def validate_parent(run):
    """Require completed100 from the exact named run; no future checkpoint hash assumed."""
    run=run.resolve()
    assert str(run.relative_to(ROOT))==SOURCE_RUN, 'Unexpected source main100 run'
    assert sha(HERE/'continue_main_adaptation_fixed_offsets250_100.py')==PARENT_RUNNER_SHA
    assert sha(run/'protocol.json')==SOURCE_PROTOCOL_SHA
    protocol=read(run/'protocol.json');result=read(run/'result.json');supervisor=read(run/'supervisor.json')
    assert protocol['name']=='adam-native1024-retained250-main-adaptation10-to100'
    assert protocol['iterations']==100 and protocol['expected_optimizer_steps']==expected_steps(100)
    assert protocol['threads']==1 and protocol['device']=='cpu' and protocol['resolution']==1024 and protocol['batch']==1
    assert protocol['source_main_step']==10 and protocol['importance_source_step']==250
    own_relative=str((HERE/'continue_main_adaptation_fixed_offsets250_100.py').relative_to(ROOT))
    assert protocol['pins'][own_relative]==PARENT_RUNNER_SHA
    for relative,expected in protocol['pins'].items():assert sha(under_root(relative))==expected,relative
    assert result['complete'] is True and result['iterations']==100 and result['first_completed_step']==11
    assert result['additional_iterations']==90 and result['optimizer_steps']==expected_steps(100)
    assert result['fixed_offsets_and_mask_invariants'] is True
    assert result['folded_native_max_errors']=={'raw':[0.0]*4,'ema':[0.0]*4}
    assert result['main_adaptation_run'] is True and result['development_continuation_only'] is True
    assert supervisor['complete'] is True and supervisor['failure'] is None
    for record in (result,supervisor):assert record['protocol_sha256']==SOURCE_PROTOCOL_SHA
    assert [r['step'] for r in result['checkpoints']]==[50,100]
    for step in (50,100):
        record=read(run/f'checkpoint-{step:03}.json')
        assert record==next(r for r in result['checkpoints'] if r['step']==step)
        assert record['complete'] is True and record['step']==step
        assert record['checkpoint']==f'checkpoint-{step:03}.pt'
        assert record['protocol_sha256']==SOURCE_PROTOCOL_SHA
        assert record['model_optimizer_rng_path_state_restored_exactly'] is True
        assert record['optimizer_steps']==expected_steps(step)
        assert sha(run/record['checkpoint'])==record['checkpoint_sha256']
        assert sha(run/f'step-{step:03}/manifest.json')==record['snapshot_manifest_sha256']
    for step in (10,50,100):
        manifest=read(run/f'step-{step:03}/manifest.json')
        assert manifest['complete'] is True and manifest['step']==step and manifest['rng_unchanged'] is True
        assert manifest['protocol_sha256']==SOURCE_PROTOCOL_SHA
        assert manifest['eval_z_sha256']==sha(run/'eval-z.npz')
        names={f'{label}-{i:03}.png' for label in ('raw','ema') for i in range(4)}
        assert len(manifest['images'])==8 and {entry['path'] for entry in manifest['images']}==names
        for entry in manifest['images']:assert sha(run/f'step-{step:03}'/entry['path'])==entry['sha256']
        if step==10:assert manifest['parent_pngs_exact'] is True
    parent10,parent_marker=previous.validate_parent(under_root(protocol['source_main_run']))
    assert protocol['selections']==parent10['selections'] and protocol['training']==parent10['training']
    assert protocol['source_main_checkpoint_sha256']==parent_marker['checkpoint_sha256']==result['parent_checkpoint_sha256']
    assert protocol['source_main_state_sha256']==parent_marker['state_sha256']
    restoration=read(run/'restoration.json')
    assert restoration['complete'] is True and restoration['step']==10
    assert restoration['full_engine_state_restored_exactly'] is True
    assert restoration['parent_checkpoint_sha256']==parent_marker['checkpoint_sha256']
    assert restoration['parent_state_sha256']==parent_marker['state_sha256']
    assert restoration['optimizer_steps']==expected_steps(10)
    return protocol,read(run/'checkpoint-100.json')


def restore_checkpoint(engine,payload,*,expected_digest,protocol_sha,runtime_sha,inventory_sha,expected_iterations):
    """Authenticate serialized envelope before the frozen engine mutates state."""
    assert state_digest(payload)==expected_digest,'Checkpoint state digest changed'
    assert payload['protocol_sha256']==protocol_sha,'Checkpoint protocol changed'
    assert payload['runtime_sha256']==runtime_sha,'Checkpoint runtime changed'
    assert payload['modulation_inventory_sha256']==inventory_sha,'Checkpoint inventory changed'
    assert payload['engine']['iterations']==expected_iterations
    assert payload['engine']['optimizer_steps']==expected_steps(expected_iterations)
    engine.restore(payload['engine'])
    assert state_digest(engine.state())==state_digest(payload['engine']),'Restored engine state differs'


def prepare(run,out):
    import psutil
    assert not out.exists(),'Use a fresh output directory'
    source,marker=validate_parent(run)
    assert psutil.virtual_memory().available/psutil.virtual_memory().total>=.35
    pins=dict(source['pins'])
    files=[HERE/'continue_main_adaptation_fixed_offsets250_250.py',HERE/'continue_main_adaptation_fixed_offsets250_100.py']
    files += [run/name for name in ('protocol.json','result.json','supervisor.json','checkpoint-100.json','checkpoint-100.pt','restoration.json','runtime.json','modulation-inventory.json','eval-z.npz')]
    for step in (10,50,100):files += [run/f'step-{step:03}/manifest.json',*sorted((run/f'step-{step:03}').glob('*.png'))]
    for path in files:pins[str(path.resolve().relative_to(ROOT))]=sha(path)
    protocol=copy.deepcopy(source)
    protocol.update(name='adam-native1024-retained250-main-adaptation100-to250',iterations=END,
        source_main_run=str(run.relative_to(ROOT)),source_main_step=START,
        source_main_protocol_sha256=sha(run/'protocol.json'),source_main_checkpoint_sha256=marker['checkpoint_sha256'],
        source_main_state_sha256=marker['state_sha256'],source_main_runner_sha256=PARENT_RUNNER_SHA,
        initialization='Authenticate original G/D references and identical masks, then restore all native100 model/Adam/RNG/path state; no reset enters updates',
        preview_steps=[100,150,250],checkpoint_steps=list(CHECKPOINT_STEPS),expected_optimizer_steps=expected_steps(END),
        metrics_range={'first_completed_step':101,'last_completed_step':250,'additional_iterations':150},
        additional_optimizer_steps={key:expected_steps(END)[key]-expected_steps(START)[key] for key in ('g','d')},
        pins=pins,automatic_restart=False,automatic_launch=False,native100_visual_review_required_before_launch=True,
        numerical_disjoint_stability_review_required_before_launch=True)
    protocol['guards']=dict(source['guards'],seconds=3600)
    protocol['deviations']=[item for item in source['deviations'] if not item.startswith('Exact native10 main-adaptation continuation')]
    protocol['deviations'].append('Exact native100 main-adaptation continuation to250; checkpoint/previews150/250; no convergence or quality approval')
    out.mkdir(parents=True);(out/'.gitignore').write_text('*.partial\n');write(out/'protocol.json',protocol)
    for path in files:
        if path.parent==HERE:shutil.copy2(path,out/path.name)
    return protocol


def worker(out):
    import numpy as np
    import platform
    import torch
    from PIL import Image
    from safetensors.torch import load_file,save_file
    from modulation import install_modulation,fold_modulation
    sys.path.insert(0,str(RESEARCH/'vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator,Discriminator
    protocol=read(out/'protocol.json')
    for relative,expected in protocol['pins'].items():assert sha(under_root(relative))==expected,relative
    run=under_root(protocol['source_main_run']);source,marker=validate_parent(run)
    assert sha(run/'protocol.json')==protocol['source_main_protocol_sha256']
    assert marker['checkpoint_sha256']==protocol['source_main_checkpoint_sha256']
    assert marker['state_sha256']==protocol['source_main_state_sha256']
    assert sha(base.GDIR/'generator.safetensors')==base.G_HASH and sha(base.DDIR/'D.safetensors')==base.D_HASH
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(source['seed'])
    runtime={'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'threads':1}
    assert runtime==read(run/'runtime.json'),'Parent runtime differs; exact continuation not established'
    write(out/'runtime.json',runtime)
    G=Generator(**read(base.GDIR/'model.json')['init_kwargs']).cpu().eval().requires_grad_(False)
    D=Discriminator(**read(base.DDIR/'metadata.json')['init_kwargs']).cpu().eval().requires_grad_(False)
    G.load_state_dict(load_file(str(base.GDIR/'generator.safetensors')),strict=True)
    D.load_state_dict(load_file(str(base.DDIR/'D.safetensors')),strict=True)
    inventory={'G':install_modulation(G,component='G',conv_layout='output_rank1'),'D':install_modulation(D,component='D',conv_layout='output_rank1')}
    assert inventory==read(run/'modulation-inventory.json'),'Parent inventory differs'
    write(out/'modulation-inventory.json',inventory)
    engine=AdaptationEngine(G,D,source['selections'],seed=source['sampling_seed'])
    assert len(engine.fixed)==34
    payload=torch.load(run/'checkpoint-100.pt',weights_only=True,map_location='cpu',mmap=True)
    restore_checkpoint(engine,payload,expected_digest=marker['state_sha256'],protocol_sha=sha(run/'protocol.json'),
        runtime_sha=sha(run/'runtime.json'),inventory_sha=sha(run/'modulation-inventory.json'),expected_iterations=START)
    assert state_digest(engine.state())==state_digest(payload['engine'])
    write(out/'restoration.json',{'complete':True,'step':100,'parent_checkpoint_sha256':marker['checkpoint_sha256'],
        'parent_state_sha256':marker['state_sha256'],'full_engine_state_restored_exactly':True,'optimizer_steps':dict(engine.counts)})
    del payload
    with np.load(run/'eval-z.npz',allow_pickle=False) as fixture:z=torch.from_numpy(fixture['z'].copy())
    with np.load(base.EVAL_Z,allow_pickle=False) as fixture:assert torch.equal(z,torch.from_numpy(fixture['z']))
    assert z.shape==(4,512);shutil.copy2(run/'eval-z.npz',out/'eval-z.npz')
    def render(model,latent):
        return model(latent[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False)
    def save_image(value,path):
        data=((value[0].detach().permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
        temporary=Path(str(path)+'.partial');Image.fromarray(data).save(temporary,format='PNG');temporary.replace(path)
    def preview(step):
        engine.check();old_rng=torch.get_rng_state();old_sampling=engine.rng.get_state()
        directory=out/f'step-{step:03}';directory.mkdir()
        with torch.no_grad():
            for label,model in [('raw',G),('ema',engine.Gema)]:
                for index,latent in enumerate(z):save_image(render(model,latent),directory/f'{label}-{index:03}.png')
        assert torch.equal(old_rng,torch.get_rng_state()) and torch.equal(old_sampling,engine.rng.get_state())
        engine.check()
        images=[{'path':p.name,'sha256':sha(p)} for p in sorted(directory.glob('*.png'))]
        assert len(images)==8
        if step==START:
            assert images==read(run/'step-100/manifest.json')['images'],'Parent100 PNG bytes differ; no update permitted'
        write(directory/'manifest.json',{'complete':True,'step':step,'rng_unchanged':True,'protocol_sha256':sha(out/'protocol.json'),
            'eval_z_sha256':sha(out/'eval-z.npz'),'parent_pngs_exact':step==START,'images':images})
    preview(START)
    real_images=[]
    for entry in source['training']:
        path=under_root(entry['path']);assert '/train/' in str(path) and sha(path)==entry['sha256']
        with Image.open(path) as image:
            assert image.size==(1024,1024)
            real_images.append(torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1)
    assert len(real_images)==20
    np.savez(out/'restored-training-rng.npz',torch_rng=torch.get_rng_state().numpy(),sampling_rng=engine.rng.get_state().numpy())
    def checkpoint(step):
        engine.check();assert engine.iterations==step
        payload={'engine':engine.state(),'protocol_sha256':sha(out/'protocol.json'),'runtime_sha256':sha(out/'runtime.json'),
            'modulation_inventory_sha256':sha(out/'modulation-inventory.json'),'source_main_checkpoint_sha256':marker['checkpoint_sha256']}
        expected=state_digest(payload);path=out/f'checkpoint-{step:03}.pt';temporary=Path(str(path)+'.partial')
        torch.save(payload,temporary);temporary.replace(path)
        restored=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        assert state_digest(restored)==expected
        with torch.no_grad():engine.cg.parameters()[0].add_(1)
        torch.rand(());torch.rand((),generator=engine.rng)
        restore_checkpoint(engine,restored,expected_digest=expected,protocol_sha=sha(out/'protocol.json'),
            runtime_sha=sha(out/'runtime.json'),inventory_sha=sha(out/'modulation-inventory.json'),expected_iterations=step)
        assert state_digest({**restored,'engine':engine.state()})==expected
        record={'complete':True,'step':step,'checkpoint':path.name,'checkpoint_sha256':sha(path),'state_sha256':expected,
            'protocol_sha256':sha(out/'protocol.json'),'snapshot_manifest_sha256':sha(out/f'step-{step:03}/manifest.json'),
            'optimizer_steps':dict(engine.counts),'model_optimizer_rng_path_state_restored_exactly':True}
        write(out/f'checkpoint-{step:03}.json',record)
        return record
    started=time.monotonic();checkpoints=[]
    for iteration in range(START,END):
        assert engine.iterations==iteration
        tick=time.monotonic();record=engine.iteration(real_images);record['seconds']=time.monotonic()-tick
        record['real_id']=source['training'][record['image_index']]['id']
        with (out/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
        if engine.iterations in CHECKPOINT_STEPS:
            preview(engine.iterations);checkpoints.append(checkpoint(engine.iterations))
    errors={}
    for label,model in [('raw',G),('ema',engine.Gema)]:
        folded=fold_modulation(model,inplace=False).eval().requires_grad_(False)
        with torch.no_grad():errors[label]=[float((render(model,latent)-render(folded,latent)).abs().max()) for latent in z]
        assert errors[label]==[0.0]*4
        temporary=out/f'folded-{label}.safetensors.partial'
        save_file({key:value.detach().contiguous() for key,value in folded.state_dict().items()},str(temporary));temporary.replace(out/f'folded-{label}.safetensors')
        del folded
    engine.check();assert engine.iterations==END
    write(out/'result.json',{'complete':True,'iterations':END,'first_completed_step':101,'additional_iterations':150,
        'optimizer_steps':dict(engine.counts),'additional_optimizer_steps':protocol['additional_optimizer_steps'],
        'protocol_sha256':sha(out/'protocol.json'),'parent_checkpoint_sha256':marker['checkpoint_sha256'],
        'checkpoints':checkpoints,'folded_native_max_errors':errors,'fixed_offsets_and_mask_invariants':True,
        'seconds_training_previews_export':time.monotonic()-started,'main_adaptation_run':True,
        'development_continuation_only':True,'production_approved':False,'server_latency_proven':False})


def supervise(run,out):
    import psutil
    protocol=prepare(run,out)
    started=time.monotonic();swap=psutil.swap_memory().used;proc=None;failure=None;peak=0;minimum=1
    try:
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'continue_main_adaptation_fixed_offsets250_250.py'),'--worker','--output',str(out)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                memory=psutil.virtual_memory()
                try:
                    process=psutil.Process(proc.pid);rss=process.memory_info().rss+sum(p.memory_info().rss for p in process.children(recursive=True))
                except psutil.NoSuchProcess:continue
                peak=max(peak,rss);minimum=min(minimum,memory.available/memory.total)
                if rss>12*2**30:failure='RSS guard'
                elif memory.available/memory.total<.20:failure='available memory guard'
                elif psutil.swap_memory().used-swap>512*2**20:failure='swap growth guard'
                elif time.monotonic()-started>protocol['guards']['seconds']:failure='wall time guard'
                if failure:os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=proc.wait()
        if code and failure is None:failure=f'worker exit {code}'
        if failure is None:
            result=read(out/'result.json')
            assert result['complete'] is True and result['iterations']==END and result['optimizer_steps']==expected_steps(END)
            assert result['protocol_sha256']==sha(out/'protocol.json')
            assert [record['step'] for record in result['checkpoints']]==list(CHECKPOINT_STEPS)
            for step in CHECKPOINT_STEPS:
                record=read(out/f'checkpoint-{step:03}.json')
                assert record['complete'] is True and sha(out/record['checkpoint'])==record['checkpoint_sha256']
                assert sha(out/f'step-{step:03}/manifest.json')==record['snapshot_manifest_sha256']
    except BaseException:
        failure=traceback.format_exc()
        if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        write(out/'supervisor.json',{'complete':failure is None,'failure':failure,'seconds':time.monotonic()-started,
            'peak_rss_gib':peak/2**30,'min_available_fraction':minimum,'protocol_sha256':sha(out/'protocol.json')})
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.output.resolve())
    else:
        if args.run is None:parser.error('--run is required')
        supervise(args.run.resolve(),args.output.resolve())
