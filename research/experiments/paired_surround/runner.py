"""Research-only 50-update collar-surround objective forks; imports models only in guarded workers."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PINS=HERE/'parent-pins.json'
UPDATES=50
MILESTONES=(0,25,50)
BRANCHES=('baseline','collar-surround')
MAX_RSS=8*1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def memory_guard():
    if sys.platform!='darwin':raise RuntimeError('Real probe requires reviewed macOS resource guard')
    text=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True,timeout=15).stdout
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',text)
    if not match or int(match.group(1))<25:raise RuntimeError('Defer: at least25% free memory required')
    return text


def rss_bytes():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform=='darwin' else peak*1024)


def watchdog(output,stop):
    def watch():
        while not stop.wait(.25):
            if rss_bytes()>MAX_RSS:
                write_json(output/'failure.json',{'complete':False,'reason':'8GiB observed peak-RSS ceiling exceeded','peak_rss_bytes':rss_bytes()})
                os._exit(70)
    thread=threading.Thread(target=watch,daemon=True);thread.start();return thread


def verify_pins():
    pins=json.loads(PINS.read_text())
    for pathkey,hashkey in [('parent_checkpoint','parent_checkpoint_sha256'),('parent_config','parent_config_sha256'),('manifest','manifest_sha256')]:
        if sha(ROOT/pins[pathkey])!=pins[hashkey]:raise ValueError(f'Pinned {pathkey} changed')
    cfg=json.loads((ROOT/pins['parent_config']).read_text())
    if cfg['provenance']!=pins['provenance'] or cfg['options']!=pins['options']:raise ValueError('Parent config changed')
    vendor=ROOT/'research/vendor/stylegan2-ada-pytorch'
    actual={'regions_trainer':sha(ROOT/'research/experiments/paired_regions/trainer.py'),'paired_edit_helpers':sha(ROOT/'research/experiments/paired_edit/trainer.py')}
    for directory in ['training','torch_utils','dnnlib']:
        for path in sorted((vendor/directory).rglob('*')):
            if path.is_file() and path.suffix in ['.py','.cpp','.cu','.h','.hpp']:actual[str(path.relative_to(vendor))]=sha(path)
    if actual!=pins['provenance']['source_files']:raise ValueError('Parent helper/vendor source hashes changed')
    for pair in pins['provenance']['pairs']:
        for info in pair['files'].values():
            path=Path(info['path'])
            if not path.resolve().is_relative_to(ROOT) or sha(path)!=info['sha256']:raise ValueError('Pinned paired data changed')
    manifest=json.loads((ROOT/pins['manifest']).read_text())
    if len(manifest['pairs'])!=7 or [p['id'] for p in manifest['pairs'] if p['split']=='validation']!=['028']:raise ValueError('Expected six train plus028 only')
    bundle=((ROOT/pins['manifest']).parent/manifest['source_bundle']).resolve()
    for name,key in [('generator.safetensors','source_weights_sha256'),('model.json','source_metadata_sha256')]:
        if sha(bundle/name)!=pins['provenance'][key]:raise ValueError('Original source bundle changed')
    return pins,bundle


def equal_tree(a,b,torch):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k],torch) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y,torch) for x,y in zip(a,b))
    return a==b


def restore_optimizer(student,parent_optimizer,trainer,torch,expected_old_names=None):
    """Restore exactly the parent freeze policy and complete existing Adam state."""
    names=trainer.freeze_student(student)
    if expected_old_names is not None and names!=expected_old_names:raise ValueError('Original parameter order differs')
    optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999))
    optimizer.load_state_dict(copy.deepcopy(parent_optimizer))
    if not equal_tree(optimizer.state_dict(),parent_optimizer,torch):raise ValueError('Existing Adam state transfer changed')
    if len(optimizer.param_groups)!=1 or optimizer.param_groups[0]['lr']!=1e-4 or tuple(optimizer.param_groups[0]['betas'])!=(.9,.999):raise ValueError('Unexpected parent optimizer hyperparameters')
    return optimizer,{'parameter_names':names,'optimizer_state_exact':True,'new_parameters':False,'freeze_policy_changed':False}


def predict_draws(sampling,preservation,pairs,options,z_dim,torch):
    s=torch.Generator(device='cpu');s.set_state(sampling.get_state())
    p=torch.Generator(device='cpu');p.set_state(preservation.get_state())
    indices=torch.randint(len(pairs),(options['batch'],),generator=s).tolist()
    z=torch.randn(1,z_dim,generator=p) if options['fresh_weight']>0 else None
    return {'pair_indices':indices,'pair_ids':[pairs[i]['id'] for i in indices],
            'fresh_z_sha256':hashlib.sha256(z.numpy().tobytes()).hexdigest() if z is not None else None},s.get_state(),p.get_state()


def invariant_state(student,source,trainer):
    return {'student_frozen':trainer.state_digest(trainer.frozen_state(student)),
            'source':trainer.state_digest(source.state_dict()),
            'student_buffers':trainer.state_digest(dict(student.named_buffers()))}


def snapshot(student,pairs,fixed_z,output,probe_step,trainer,torch,sampling,preservation):
    """Full native student renders only; no source/target pixel compositing."""
    from PIL import Image,ImageDraw
    rng=trainer.capture_rng('cpu',sampling,preservation)
    before=trainer.state_digest(student.state_dict());started=time.monotonic();rows=[];thumbs=[]
    directory=output/f'preview-{probe_step:03}';directory.mkdir()
    with torch.no_grad():
        requests=[(p['id'],p['split'],p['w'],p) for p in pairs]
        requests += [(f'unseen-{i:03}','unseen',student.mapping(z[None],None,truncation_psi=1,skip_w_avg_update=True),None) for i,z in enumerate(fixed_z)]
        for ident,split,w,pair in requests:
            rgb=student.synthesis(w,noise_mode='const',force_fp32=True)
            if not bool(torch.isfinite(rgb).all()):raise ValueError('Nonfinite preview')
            mono=trainer.grayscale(rgb);names={}
            for label,tensor in [('rgb',rgb),('gray',mono.repeat(1,3,1,1))]:
                path=directory/f'{ident}-{label}.png';image=Image.fromarray(trainer.quantized(tensor));image.save(path)
                names[label]={'path':path.name,'sha256':sha(path)}
                if label=='gray':thumbs.append((ident,split,image.resize((256,256),Image.Resampling.LANCZOS)))
            metrics={}
            if pair is not None:
                metrics['protected_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['original']),1-pair['mask']))
                metrics['clothing_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['target']),pair['mask']))
                if pair['collar'] is not None:
                    metrics['tab_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['target']),pair['collar']))
                    metrics['rest_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['target']),pair['rest']))
                    metrics['surround_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['target']),pair['surround']))
                    metrics['remainder_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['target']),pair['remainder']))
            rows.append({'id':ident,'split':split,'outputs':names,'metrics_minus1_to1':metrics})
    trainer.restore_rng(rng,'cpu',sampling,preservation)
    if trainer.state_digest(student.state_dict())!=before:raise ValueError('Snapshot mutated student state')
    contact=Image.new('RGB',(4*256,3*280),(24,24,24));draw=ImageDraw.Draw(contact)
    for i,(ident,split,img) in enumerate(thumbs):
        x=(i%4)*256;y=(i//4)*280;contact.paste(img,(x,y));draw.text((x+3,y+258),f'{ident} {split}',fill='white')
    contact.save(directory/'contact.png')
    report={'probe_step':probe_step,'parent_step':600,'count':len(rows),'source_compositing':False,'rows':rows,'state_sha256':before,'seconds':time.monotonic()-started,'rng_restored':True}
    write_json(directory/'evaluation.json',report)
    return report


def worker(branch,output):
    started=time.monotonic();pressure=memory_guard();pins,bundle=verify_pins();pressure2=memory_guard()
    output.mkdir(parents=True);(output/'memory-before.txt').write_text(pressure+'\nBEFORE IMPORT:\n'+pressure2)
    stop=threading.Event();watch=watchdog(output,stop)
    try:
        os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        import numpy as np
        import PIL
        from safetensors.torch import load_file
        sys.path.insert(0,str(ROOT))
        from research.experiments.paired_regions import trainer
        checkpoint=torch.load(ROOT/pins['parent_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        if checkpoint['format']!='paired-regions-research-v1' or checkpoint['step']!=600 or checkpoint['production_approved'] is not False:raise ValueError('Expected completed unapproved regions600')
        if checkpoint['provenance']!=pins['provenance'] or checkpoint['options']!=pins['options']:raise ValueError('Checkpoint/config provenance mismatch')
        versions={'torch':str(torch.__version__),'numpy':np.__version__,'pillow':PIL.__version__}
        if versions!=pins['provenance']['versions']:raise ValueError('Runtime versions changed')
        metadata=json.loads((bundle/'model.json').read_text())
        if checkpoint['init_kwargs']!=metadata['init_kwargs'] or metadata['truncation_psi']!=1:raise ValueError('Source architecture mismatch')
        source=trainer.Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
        source.load_state_dict(load_file(str(bundle/'generator.safetensors'),device='cpu'),strict=True)
        if trainer.state_digest(source.state_dict())!=checkpoint['source_digest']:raise ValueError('Original source differs')
        student=trainer.Generator(**metadata['init_kwargs']).cpu().eval();student.load_state_dict(checkpoint['student'],strict=True)
        trainer.freeze_student(student)
        if trainer.state_digest(trainer.frozen_state(student))!=checkpoint['frozen_digest']:raise ValueError('Parent frozen state differs')
        parent_student_digest=trainer.state_digest(student.state_dict())
        for model in [source,student]:
            if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):raise ValueError('Nonfinite initial state')
        # Reuse exact source-regeneration/PNG/W checks; target/mask semantics unchanged.
        pairs,provenance=trainer.load_pairs(ROOT/pins['manifest'],source,source_max_error=0,w_atol=0)
        if provenance!=pins['provenance']['pairs']:raise ValueError('Regenerated source/pair provenance differs')
        from research.experiments.paired_surround import objective
        region_counts=objective.attach_regions(pairs,json.loads((ROOT/pins['manifest']).read_text()),trainer)
        train_pairs=[p for p in pairs if p['split']=='train']
        if len(train_pairs)!=6 or len(pairs)!=7:raise ValueError('Unexpected split')
        optimizer,transfer=restore_optimizer(student,checkpoint['optimizer'],trainer,torch,pins['trainable_parameters'])
        options=copy.deepcopy(checkpoint['options'])
        if options['threads']!=2 or options['device']!='cpu' or options['lr']!=1e-4 or options['batch']!=1:raise ValueError('Unexpected parent settings')
        fixed_z=checkpoint['fixed_z'].clone()
        if tuple(fixed_z.shape)!=(4,source.z_dim):raise ValueError('Expected original four fixed unseen latents')
        train_z={hashlib.sha256(p['z'].numpy().tobytes()).hexdigest() for p in pairs}
        if any(hashlib.sha256(z[None].numpy().tobytes()).hexdigest() in train_z for z in fixed_z):raise ValueError('Fixed unseen latent overlaps paired identities')
        sampling=torch.Generator(device='cpu');preservation=torch.Generator(device='cpu')
        trainer.restore_rng(checkpoint['rng'],'cpu',sampling,preservation)
        if trainer.state_digest(student.state_dict())!=parent_student_digest:raise ValueError('Fork construction altered initial weights')
        initial_rng=trainer.capture_rng('cpu',sampling,preservation)
        invariants=invariant_state(student,source,trainer)
        fork={'format':'paired-surround-fork-v1','production_approved':False,'exact_resume':False,'branch':branch,
              'parent_checkpoint_sha256':pins['parent_checkpoint_sha256'],'parent_step':600,'updates':UPDATES,
              'objective': 'parent0.5*tab+0.5*rest' if branch=='baseline' else '(tab+surround+remainder)/3; protected/fresh unchanged',
              'region_counts':region_counts,'surround_radius_pixels':30,'objective_sha256':sha(HERE/'objective.py'),
              'freeze_policy':'unchanged parent mapping/noise/buffers/blocks<=32 frozen in both branches',
              'optimizer_transfer':transfer,'parent_student_state_sha256':parent_student_digest,
              'runner_sha256':sha(__file__),'pins_sha256':sha(PINS),'options':options,'parent_provenance':pins['provenance'],
              'interop_threads':1,'fixed_z_sha256':hashlib.sha256(fixed_z.numpy().tobytes()).hexdigest(),
              'rss_abort_bytes':MAX_RSS,'rss_limit_is_sampled_not_hard_allocation_cap':True}
        write_json(output/'config.json',fork)
        del checkpoint
        last_metrics={};schedule=[];milestones=[]
        def save_milestone(probe_step):
            if invariant_state(student,source,trainer)!=invariants:raise AssertionError('Source/frozen parameters/buffers changed')
            if any(p.grad is not None for p in source.parameters()):raise AssertionError('Source acquired gradients')
            evaluation=snapshot(student,pairs,fixed_z,output,probe_step,trainer,torch,sampling,preservation)
            if invariant_state(student,source,trainer)!=invariants:raise AssertionError('Snapshot changed invariants')
            state={'format':'paired-surround-fork-v1','production_approved':False,'probe_step':probe_step,'step':600+probe_step,
                   'parent_step':600,'branch':branch,'student':student.state_dict(),'optimizer':optimizer.state_dict(),
                   'rng':trainer.capture_rng('cpu',sampling,preservation),'fixed_z':fixed_z,'init_kwargs':metadata['init_kwargs'],
                   'options':options,'fork':fork,'invariants':invariants,'source_digest':invariants['source'],
                   'frozen_digest':invariants['student_frozen'],'last_metrics':last_metrics,'schedule':schedule}
            path=output/f'checkpoint-{probe_step:03}.pt';trainer.atomic_save(state,path)
            milestones.append({'probe_step':probe_step,'checkpoint':path.name,'sha256':sha(path),'student_sha256':evaluation['state_sha256'],'preview_seconds':evaluation['seconds']})
        save_milestone(0)
        if not equal_tree(initial_rng,trainer.capture_rng('cpu',sampling,preservation),torch):raise ValueError('Step0 snapshot changed RNG')
        with (output/'metrics.jsonl').open('w') as log:
            for probe_step in range(1,UPDATES+1):
                draw,expected_sampling,expected_preservation=predict_draws(sampling,preservation,train_pairs,options,source.z_dim,torch)
                tick=time.monotonic()
                if branch=='baseline':
                    metrics=trainer.update(student,source,optimizer,train_pairs,options,sampling,preservation,'cpu')
                else:
                    metrics=objective.update(student,source,optimizer,train_pairs,options,sampling,preservation,trainer)
                if metrics['pair_ids']!=draw['pair_ids'] or not torch.equal(sampling.get_state(),expected_sampling) or not torch.equal(preservation.get_state(),expected_preservation):raise AssertionError('Actual sampling differed from predicted parent RNG schedule')
                metrics.update(probe_step=probe_step,parent_step=600,step=600+probe_step,update_seconds=time.monotonic()-tick,**{k:v for k,v in draw.items() if k!='pair_ids'})
                last_metrics=metrics;schedule.append(draw)
                log.write(json.dumps(metrics)+'\n');log.flush();print(json.dumps(metrics),flush=True)
                if probe_step in MILESTONES:save_milestone(probe_step)
        # Final immutable checkpoint deserialization proof, with no model allocation.
        if sha(__file__)!=fork['runner_sha256'] or sha(HERE/'objective.py')!=fork['objective_sha256'] or sha(PINS)!=fork['pins_sha256']:raise ValueError('Probe source changed during execution')
        final=torch.load(output/'checkpoint-050.pt',weights_only=True,map_location='cpu',mmap=True)
        if final['probe_step']!=50 or not equal_tree(final['optimizer'],optimizer.state_dict(),torch) or trainer.state_digest(final['student'])!=trainer.state_digest(student.state_dict()):raise ValueError('Final full checkpoint roundtrip mismatch')
        write_json(output/'complete.json',{'complete':True,'branch':branch,'updates':50,'milestones':milestones,'schedule':schedule,
                   'initial_student_state_sha256':parent_student_digest,'final_student_state_sha256':trainer.state_digest(student.state_dict()),
                   'invariants':invariants,'source_and_frozen_checks_passed':True,'full_checkpoint_roundtrip':True,
                   'peak_rss_bytes':rss_bytes(),'wall_seconds':time.monotonic()-started,'runner_sha256':sha(__file__)})
    except Exception as exc:
        write_json(output/'failure.json',{'complete':False,'error':str(exc),'peak_rss_bytes':rss_bytes()});raise
    finally:
        stop.set();watch.join(timeout=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,required=True,help='NEW research directory, creates two sequential branch subdirectories')
    parser.add_argument('--worker',choices=BRANCHES,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required; real weights remain unloaded otherwise')
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/'research').resolve()) or output.exists():parser.error('Use a new research output directory; resume/overwrite unsupported')
    if args.worker:
        worker(args.worker,output);return
    pressure=memory_guard();pins,_=verify_pins();output.mkdir(parents=True)
    (output/'memory-before.txt').write_text(pressure)
    write_json(output/'plan.json',{'branches':list(BRANCHES),'updates_each':50,'milestones':list(MILESTONES),'parent_checkpoint_sha256':pins['parent_checkpoint_sha256'],'runner_sha256':sha(__file__),'objective_sha256':sha(HERE/'objective.py'),'pins_sha256':sha(PINS),'exact_resume':False,'only_experimental_change':'clothing partition0.5/0.5 to equal tab/surround/remainder thirds' })
    code_pins={str(path):sha(path) for path in [Path(__file__),HERE/'objective.py',PINS]}
    completed=[]
    for branch in BRANCHES:
        # Separate child processes guarantee all tensors/allocator state are gone
        # before the next branch loads; no concurrent paired-surround workers.
        if any(sha(Path(path))!=expected for path,expected in code_pins.items()):raise ValueError('Probe code/pins changed between branches')
        pressure=memory_guard();(output/f'{branch}-memory-before.txt').write_text(pressure)
        with (output/f'{branch}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--execute','--worker',branch,'--output',str(output/branch)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode!=0:raise RuntimeError(f'{branch} failed with exit{result.returncode}; second branch will not start after failure')
        completed.append(json.loads((output/branch/'complete.json').read_text()))
    baseline,variant=completed
    if baseline['schedule']!=variant['schedule'] or baseline['initial_student_state_sha256']!=variant['initial_student_state_sha256']:raise AssertionError('Branches did not share identical initial weights and draws')
    write_json(output/'comparison-mechanics.json',{'complete':True,'initial_student_states_exact':True,'all50_pair_and_fresh_latent_draws_exact':True,'baseline':baseline,'variant':variant,'quality_approved':False})


if __name__=='__main__':main()
