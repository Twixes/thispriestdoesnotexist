"""Exact-state continuation from completed main250, then this runner's endpoints.

Only the horizon, snapshot steps and deadline are configurable. No native run
starts automatically; source endpoint review is a root scheduling decision.
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
import continue_main_adaptation_fixed_offsets250_250 as legacy

HERE,ROOT,RESEARCH=base.HERE,base.ROOT,base.RESEARCH
sha,read,write,under_root=base.sha,base.read,base.write,base.under_root
state_digest,expected_steps=base.state_digest,base.expected_steps
AdaptationEngine=base.AdaptationEngine
restore_checkpoint=legacy.restore_checkpoint
OWN_FILE=HERE/'continue_main_adaptation.py'
NAME='adam-native1024-retained250-exact-continuation'
BOOTSTRAP_RUN='research/runs/adam-native1024-retained250-main100-to250-v1'
BOOTSTRAP_PROTOCOL_SHA='2c10cfc0ebec3ab0568d2778f3fd5363d231e52931d81ce4522cc4660b038c33'
LEGACY_SHA='bc5d9d5682a9584338dbf72c28b5e7cad6daa9dc97d24d932414b95af1f08c06'
TRAINING_FIELDS=('conv_layout','fixed_offset_policy','seed','sampling_seed','importance_run','importance_pairs',
    'importance_source_run','importance_source_step','quantile','selections','g_weights_sha256','d_weights_sha256',
    'dataset_manifest_sha256','training','resolution','device','threads','batch','g_lr','d_lr','g_beta2','d_beta2',
    'r1_gamma','r1_every','path_weight','path_every','path_decay','style_mixing_probability','ema_decay','ema_components','augmentation')


def schedule(start,target,snapshots,seconds):
    assert type(start) is int and type(target) is int and 0<=start<target,'Target must exceed source endpoint'
    assert type(seconds) is int and seconds>0,'Explicit positive integer deadline required'
    assert isinstance(snapshots,(list,tuple)) and snapshots,'At least terminal snapshot required'
    assert all(type(step) is int and start<step<=target for step in snapshots),'Snapshot outside continuation'
    assert list(snapshots)==sorted(set(snapshots)) and snapshots[-1]==target,'Snapshots must be sorted, unique and include target'
    return tuple(snapshots)


def validate_source(run,seen=None):
    """Authenticate complete endpoints only, back to the pinned bootstrap protocol."""
    run=run.resolve();run.relative_to(RESEARCH/'runs')
    seen=set() if seen is None else set(seen)
    assert str(run) not in seen,'Source lineage cycle';seen.add(str(run))
    bootstrap=under_root(BOOTSTRAP_RUN)
    assert sha(bootstrap/'protocol.json')==BOOTSTRAP_PROTOCOL_SHA
    anchor=read(bootstrap/'protocol.json')
    protocol=read(run/'protocol.json');protocol_sha=sha(run/'protocol.json')
    result=read(run/'result.json');supervisor=read(run/'supervisor.json')
    is_bootstrap=run==bootstrap
    assert {key:protocol[key] for key in TRAINING_FIELDS}=={key:anchor[key] for key in TRAINING_FIELDS},'Training configuration changed'
    for key in ('start_available_fraction','runtime_available_fraction','rss_gib','swap_growth_mib'):
        assert protocol['guards'][key]==anchor['guards'][key],'Resource guard changed'
    if is_bootstrap:
        assert protocol['name']=='adam-native1024-retained250-main-adaptation100-to250'
        assert protocol['iterations']==250 and protocol['source_main_step']==100
        assert protocol['checkpoint_steps']==[150,250] and protocol['preview_steps']==[100,150,250]
        assert sha(HERE/'continue_main_adaptation_fixed_offsets250_250.py')==LEGACY_SHA
        assert protocol['pins'][str((HERE/'continue_main_adaptation_fixed_offsets250_250.py').relative_to(ROOT))]==LEGACY_SHA
    else:
        assert protocol['name']==NAME and protocol['continuation_schema']==1,'Unrecognized continuation format'
        assert protocol['bootstrap_run']==BOOTSTRAP_RUN and protocol['bootstrap_protocol_sha256']==BOOTSTRAP_PROTOCOL_SHA
        assert protocol['pins'][str(OWN_FILE.relative_to(ROOT))]==sha(OWN_FILE),'Continuation runner changed'
    start,end=protocol['source_main_step'],protocol['iterations']
    snapshots=schedule(start,end,protocol['checkpoint_steps'],protocol['guards']['seconds'])
    assert protocol['preview_steps']==[start,*snapshots]
    for relative,expected in protocol['pins'].items():assert sha(under_root(relative))==expected,relative
    assert result['complete'] is True and result['iterations']==end
    assert result['first_completed_step']==start+1 and result['additional_iterations']==end-start
    assert result['optimizer_steps']==expected_steps(end)==protocol['expected_optimizer_steps']
    assert result['additional_optimizer_steps']=={key:expected_steps(end)[key]-expected_steps(start)[key] for key in ('g','d')}
    assert result['fixed_offsets_and_mask_invariants'] is True and result['main_adaptation_run'] is True
    assert result['development_continuation_only'] is True and result['folded_native_max_errors']=={'raw':[0.0]*4,'ema':[0.0]*4}
    assert supervisor['complete'] is True and supervisor['failure'] is None
    for record in (result,supervisor):assert record['protocol_sha256']==protocol_sha
    assert [record['step'] for record in result['checkpoints']]==list(snapshots)
    for step in snapshots:
        record=read(run/f'checkpoint-{step:03}.json')
        assert record==next(item for item in result['checkpoints'] if item['step']==step)
        assert record['complete'] is True and record['step']==step and record['checkpoint']==f'checkpoint-{step:03}.pt'
        assert record['protocol_sha256']==protocol_sha and record['optimizer_steps']==expected_steps(step)
        assert record['model_optimizer_rng_path_state_restored_exactly'] is True
        assert sha(run/record['checkpoint'])==record['checkpoint_sha256']
        assert sha(run/f'step-{step:03}/manifest.json')==record['snapshot_manifest_sha256']
    for step in protocol['preview_steps']:
        manifest=read(run/f'step-{step:03}/manifest.json')
        assert manifest['complete'] is True and manifest['step']==step and manifest['rng_unchanged'] is True
        assert manifest['protocol_sha256']==protocol_sha and manifest['eval_z_sha256']==sha(run/'eval-z.npz')
        names={f'{label}-{i:03}.png' for label in ('raw','ema') for i in range(4)}
        assert len(manifest['images'])==8 and {entry['path'] for entry in manifest['images']}==names
        for entry in manifest['images']:assert sha(run/f'step-{step:03}'/entry['path'])==entry['sha256']
        if step==start:assert manifest['parent_pngs_exact'] is True
    parent_run=under_root(protocol['source_main_run'])
    parent,parent_marker=legacy.validate_parent(parent_run) if is_bootstrap else validate_source(parent_run,seen)
    assert parent['iterations']==start and {key:protocol[key] for key in TRAINING_FIELDS}=={key:parent[key] for key in TRAINING_FIELDS}
    assert protocol['source_main_protocol_sha256']==sha(parent_run/'protocol.json')
    assert protocol['source_main_checkpoint_sha256']==parent_marker['checkpoint_sha256']==result['parent_checkpoint_sha256']
    assert protocol['source_main_state_sha256']==parent_marker['state_sha256']
    restoration=read(run/'restoration.json')
    assert restoration['complete'] is True and restoration['step']==start and restoration['full_engine_state_restored_exactly'] is True
    assert restoration['parent_checkpoint_sha256']==parent_marker['checkpoint_sha256'] and restoration['parent_state_sha256']==parent_marker['state_sha256']
    assert restoration['optimizer_steps']==expected_steps(start)
    return protocol,read(run/f'checkpoint-{end:03}.json')


def prepare(run,out,target,snapshots,seconds):
    import psutil
    out.resolve().relative_to(RESEARCH/'runs')
    assert not out.exists(),'Use a fresh output directory'
    source,marker=validate_source(run);start=source['iterations']
    snapshots=schedule(start,target,snapshots,seconds)
    assert psutil.virtual_memory().available/psutil.virtual_memory().total>=.35
    pins=dict(source['pins'])
    files=[OWN_FILE,HERE/'continue_main_adaptation_fixed_offsets250_250.py']
    files += [run/name for name in ('protocol.json','result.json','supervisor.json',f'checkpoint-{start:03}.json',
        f'checkpoint-{start:03}.pt','restoration.json','runtime.json','modulation-inventory.json','eval-z.npz')]
    for step in source['preview_steps']:files += [run/f'step-{step:03}/manifest.json',*sorted((run/f'step-{step:03}').glob('*.png'))]
    for path in files:pins[str(path.resolve().relative_to(ROOT))]=sha(path)
    protocol=copy.deepcopy(source)
    protocol.update(name=NAME,continuation_schema=1,iterations=target,bootstrap_run=BOOTSTRAP_RUN,bootstrap_protocol_sha256=BOOTSTRAP_PROTOCOL_SHA,
        source_main_run=str(run.relative_to(ROOT)),source_main_step=start,source_main_protocol_sha256=sha(run/'protocol.json'),
        source_main_checkpoint_sha256=marker['checkpoint_sha256'],source_main_state_sha256=marker['state_sha256'],
        source_main_runner_sha256=LEGACY_SHA if str(run.relative_to(ROOT))==BOOTSTRAP_RUN else sha(OWN_FILE),
        initialization='Reconstruct original protected references, then restore the complete authenticated source endpoint; no reset enters updates',
        preview_steps=[start,*snapshots],checkpoint_steps=list(snapshots),expected_optimizer_steps=expected_steps(target),
        metrics_range={'first_completed_step':start+1,'last_completed_step':target,'additional_iterations':target-start},
        additional_optimizer_steps={key:expected_steps(target)[key]-expected_steps(start)[key] for key in ('g','d')},
        pins=pins,automatic_restart=False,automatic_launch=False,source_endpoint_visual_review_required_before_launch=True)
    protocol['guards']=dict(source['guards'],seconds=seconds)
    protocol['deviations']=[item for item in source['deviations'] if not item.startswith(('Exact native','Exact endpoint'))]
    protocol['deviations'].append('Exact endpoint continuation; no convergence or quality approval')
    out.mkdir(parents=True);(out/'.gitignore').write_text('*.partial\n');write(out/'protocol.json',protocol)
    for path in files:
        if path.parent==HERE:shutil.copy2(path,out/path.name)
    return protocol


def run_updates(engine,real_images,target,snapshots,on_step,on_checkpoint):
    """Only orchestration: the frozen engine owns every training calculation."""
    start=engine.iterations;snapshots=schedule(start,target,snapshots,1)
    for iteration in range(start,target):
        assert engine.iterations==iteration
        tick=time.monotonic();record=engine.iteration(real_images);record['seconds']=time.monotonic()-tick
        on_step(record)
        if engine.iterations in snapshots:on_checkpoint(engine.iterations)


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
    start,target=protocol['source_main_step'],protocol['iterations']
    snapshots=schedule(start,target,protocol['checkpoint_steps'],protocol['guards']['seconds'])
    for relative,expected in protocol['pins'].items():assert sha(under_root(relative))==expected,relative
    run=under_root(protocol['source_main_run']);source,marker=validate_source(run)
    assert source['iterations']==start
    assert {key:protocol[key] for key in TRAINING_FIELDS}=={key:source[key] for key in TRAINING_FIELDS}
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
    payload=torch.load(run/marker['checkpoint'],weights_only=True,map_location='cpu',mmap=True)
    restore_checkpoint(engine,payload,expected_digest=marker['state_sha256'],protocol_sha=sha(run/'protocol.json'),
        runtime_sha=sha(run/'runtime.json'),inventory_sha=sha(run/'modulation-inventory.json'),expected_iterations=start)
    assert state_digest(engine.state())==state_digest(payload['engine'])
    write(out/'restoration.json',{'complete':True,'step':start,'parent_checkpoint_sha256':marker['checkpoint_sha256'],
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
        if step==start:
            assert images==read(run/f'step-{start:03}/manifest.json')['images'],'Parent PNG bytes differ; no update permitted'
        write(directory/'manifest.json',{'complete':True,'step':step,'rng_unchanged':True,'protocol_sha256':sha(out/'protocol.json'),
            'eval_z_sha256':sha(out/'eval-z.npz'),'parent_pngs_exact':step==start,'images':images})
    preview(start)
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
    def on_step(record):
        record['real_id']=source['training'][record['image_index']]['id']
        with (out/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
    def on_checkpoint(step):
        preview(step);checkpoints.append(checkpoint(step))
    run_updates(engine,real_images,target,snapshots,on_step,on_checkpoint)
    errors={}
    for label,model in [('raw',G),('ema',engine.Gema)]:
        folded=fold_modulation(model,inplace=False).eval().requires_grad_(False)
        with torch.no_grad():errors[label]=[float((render(model,latent)-render(folded,latent)).abs().max()) for latent in z]
        assert errors[label]==[0.0]*4
        temporary=out/f'folded-{label}.safetensors.partial'
        save_file({key:value.detach().contiguous() for key,value in folded.state_dict().items()},str(temporary));temporary.replace(out/f'folded-{label}.safetensors')
        del folded
    engine.check();assert engine.iterations==target
    write(out/'result.json',{'complete':True,'iterations':target,'first_completed_step':start+1,'additional_iterations':target-start,
        'optimizer_steps':dict(engine.counts),'additional_optimizer_steps':protocol['additional_optimizer_steps'],
        'protocol_sha256':sha(out/'protocol.json'),'parent_checkpoint_sha256':marker['checkpoint_sha256'],
        'checkpoints':checkpoints,'folded_native_max_errors':errors,'fixed_offsets_and_mask_invariants':True,
        'seconds_training_previews_export':time.monotonic()-started,'main_adaptation_run':True,
        'development_continuation_only':True,'production_approved':False,'server_latency_proven':False})


def supervise(run,out,target,snapshots,seconds):
    import psutil
    protocol=prepare(run,out,target,snapshots,seconds)
    started=time.monotonic();swap=psutil.swap_memory().used;proc=None;failure=None;peak=0;minimum=1
    try:
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(OWN_FILE),'--worker','--output',str(out)],
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
            assert result['complete'] is True and result['iterations']==target and result['optimizer_steps']==expected_steps(target)
            assert result['protocol_sha256']==sha(out/'protocol.json')
            assert [record['step'] for record in result['checkpoints']]==list(snapshots)
            for step in snapshots:
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path);parser.add_argument('--until',type=int)
    parser.add_argument('--snapshots',type=int,nargs='+');parser.add_argument('--seconds',type=int)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.output.resolve())
    else:
        if args.run is None or args.until is None or args.snapshots is None or args.seconds is None:
            parser.error('--run, --until, --snapshots and --seconds are required')
        supervise(args.run.resolve(),args.output.resolve(),args.until,args.snapshots,args.seconds)
