"""Isolated, research-only 2x2 pixel/feature and b32-capacity forks. No implicit model load."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from research.experiments.paired_surround import runner as shared
sha=shared.sha;write_json=shared.write_json;equal_tree=shared.equal_tree
PINS=HERE/'parent-pins.json'
ARMS=('A','B','C','D')
UPDATES=300
MILESTONES=(0,100,200,300)
MAX_RSS=8*1024**3
MAX_SECONDS=4*3600


def own_hashes():
    return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}|{'parent-pins.json':sha(PINS)}


def verify_pins():
    old,bundle=shared.verify_pins();pins=json.loads(PINS.read_text())
    if any(pins[k]!=v for k,v in old.items()):raise ValueError('Parent pin set mismatch')
    for key in ['helper_files','d_files']:
        for path,digest in pins[key].items():
            if sha(ROOT/path)!=digest:raise ValueError(f'Pinned {path} changed')
    for key in ['extra_manifest','source30_manifest']:
        if sha(ROOT/pins[key]['path'])!=pins[key]['sha256']:raise ValueError(f'Pinned {key} changed')
    for path,digest in pins['extra_manifest']['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Held-out data changed')
    original=json.loads((ROOT/pins['manifest']).read_text());extra=json.loads((ROOT/pins['extra_manifest']['path']).read_text())
    if extra['pairs'][:7]!=original['pairs'] or [p['id'] for p in extra['pairs'][7:]]!=pins['extra_manifest']['heldout_ids']:raise ValueError('Paired10 no longer extends identical original seven records')
    review=json.loads((ROOT/pins['d_bundle']/'root-extraction-review.json').read_text())
    if review['approved_for_bounded_research_smoke'] is not True or review['production_approved'] is not False:raise ValueError('D extraction not approved for research')
    if review['weights_sha256']!=pins['d_files'][pins['d_bundle']+'/D.safetensors']:raise ValueError('D review hash mismatch')
    return pins,bundle


def guard_thread(output,stop,deadline):
    def watch():
        while not stop.wait(.25):
            reason=None
            if shared.rss_bytes()>MAX_RSS:reason='8 GiB observed peak RSS exceeded'
            if time.time()>deadline:reason='Four-hour total deadline exceeded'
            if reason:
                write_json(output/'failure.json',{'complete':False,'reason':reason,'peak_rss_bytes':shared.rss_bytes()});os._exit(70)
    worker=threading.Thread(target=watch,daemon=True);worker.start();return worker


def balanced_plan(state,count,torch):
    if count!=6:raise ValueError('Exactly six original training identities required')
    generator=torch.Generator(device='cpu');generator.set_state(state)
    return [i for _ in range(50) for i in torch.randperm(count,generator=generator).tolist()]


def planned_draw(step,plan,sampling,preservation,pairs,z_dim,torch):
    """Consume one permutation per cycle; forecast fresh draw without advancing its live RNG."""
    if step%6==0:
        order=torch.randperm(6,generator=sampling).tolist()
        if order!=plan[step:step+6]:raise AssertionError('Balanced schedule RNG changed')
    g=torch.Generator(device='cpu');g.set_state(preservation.get_state());z=torch.randn(1,z_dim,generator=g)
    index=plan[step]
    return {'pair_index':index,'pair_id':pairs[index]['id'],'fresh_z_sha256':hashlib.sha256(z.numpy().tobytes()).hexdigest()},g.get_state()


def load_prefix(pins,torch):
    from safetensors.torch import load_file
    from training.networks import Discriminator
    from research.experiments.clothing_structure.losses import FrozenDPrefix
    bundle=ROOT/pins['d_bundle'];meta=json.loads((bundle/'metadata.json').read_text())
    if meta['constructor']!='training.networks.Discriminator' or meta['init_args']!=[] or meta['production_approved'] is not False:raise ValueError('Unexpected D constructor/provenance')
    if meta['source_pickle_sha256']!='a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c':raise ValueError('Unexpected D origin')
    model=Discriminator(**meta['init_kwargs']).cpu().eval().requires_grad_(False)
    state=load_file(str(bundle/'D.safetensors'),device='cpu');model.load_state_dict(state,strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):raise ValueError('Nonfinite D state')
    prefix=FrozenDPrefix(model,1024,(512,256));del model,state
    return prefix,meta


def worker(arm,output,deadline,expected_hashes,smoke=False):
    started=time.monotonic()
    if time.time()>=deadline:raise RuntimeError('Total deadline expired before worker')
    pressure=shared.memory_guard();pins,bundle=verify_pins()
    if own_hashes()!=expected_hashes:raise ValueError('Experiment code changed')
    pressure2=shared.memory_guard();output.mkdir(parents=True)
    (output/'memory-before.txt').write_text(pressure+'\nBEFORE IMPORT:\n'+pressure2)
    stop=threading.Event();watch=guard_thread(output,stop,deadline)
    try:
        os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        import numpy as np
        import PIL
        from safetensors.torch import load_file
        from research.experiments.paired_regions import trainer
        from research.experiments.paired_factorial.optimizer import fork_optimizer
        from research.experiments.paired_factorial.update import update
        from research.experiments.paired_factorial import render
        from research.experiments.paired_surround import objective as regions
        checkpoint=torch.load(ROOT/pins['parent_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        if checkpoint['format']!='paired-regions-research-v1' or checkpoint['step']!=600 or checkpoint['production_approved'] is not False:raise ValueError('Expected unapproved original regions600 checkpoint')
        if checkpoint['provenance']!=pins['provenance'] or checkpoint['options']!=pins['options']:raise ValueError('Parent config/provenance mismatch')
        versions={'torch':str(torch.__version__),'numpy':np.__version__,'pillow':PIL.__version__}
        if versions!=pins['provenance']['versions']:raise ValueError('Runtime versions changed')
        meta=json.loads((bundle/'model.json').read_text())
        if checkpoint['init_kwargs']!=meta['init_kwargs'] or meta['truncation_psi']!=1:raise ValueError('Source architecture mismatch')
        source=trainer.Generator(**meta['init_kwargs']).cpu().eval().requires_grad_(False)
        source.load_state_dict(load_file(str(bundle/'generator.safetensors'),device='cpu'),strict=True)
        if trainer.state_digest(source.state_dict())!=checkpoint['source_digest']:raise ValueError('Source state changed')
        student=trainer.Generator(**meta['init_kwargs']).cpu().eval();student.load_state_dict(checkpoint['student'],strict=True)
        trainer.freeze_student(student)
        if trainer.state_digest(trainer.frozen_state(student))!=checkpoint['frozen_digest']:raise ValueError('Parent frozen state changed')
        initial_digest=trainer.state_digest(student.state_dict())
        if any(not bool(torch.isfinite(t).all()) for t in student.state_dict().values()):raise ValueError('Nonfinite initial student')
        pairs,pair_provenance=trainer.load_pairs(ROOT/pins['extra_manifest']['path'],source,source_max_error=0,w_atol=0)
        if pair_provenance[:7]!=pins['provenance']['pairs']:raise ValueError('Original pair provenance differs')
        # Evaluation-only surround metrics use the already reviewed 30px partition.
        regions.attach_regions(pairs,json.loads((ROOT/pins['extra_manifest']['path']).read_text()),trainer)
        for pair in pairs[7:]:pair['split']='diagnostic-heldout'
        train_pairs=[p for p in pairs if p['split']=='train']
        if len(pairs)!=10 or [p['id'] for p in train_pairs]!=[p['id'] for p in pins['provenance']['pairs'] if p['split']=='train']:raise ValueError('Training/held-out identity split mismatch')
        optimizer,transfer=fork_optimizer(student,checkpoint['optimizer'],arm in ('B','D'),trainer,torch,pins['trainable_parameters'])
        options=copy.deepcopy(checkpoint['options'])
        if options['threads']!=2 or options['device']!='cpu' or options['batch']!=1 or options['fresh_weight']!=1:raise ValueError('Unexpected parent options')
        fixed_z=checkpoint['fixed_z'].clone()
        if tuple(fixed_z.shape)!=(4,source.z_dim):raise ValueError('Expected four original fixed unseen latents')
        if any(torch.equal(z[None],p['z']) for z in fixed_z for p in pairs):raise ValueError('Fixed unseen overlaps a paired identity')
        prefix=d_meta=None
        if arm in ('C','D'):prefix,d_meta=load_prefix(pins,torch)
        d_digest=trainer.state_digest(prefix.state_dict()) if prefix is not None else None
        sampling=torch.Generator(device='cpu');preservation=torch.Generator(device='cpu')
        trainer.restore_rng(checkpoint['rng'],'cpu',sampling,preservation)
        initial_rng=trainer.capture_rng('cpu',sampling,preservation);plan=balanced_plan(sampling.get_state(),6,torch)
        invariants=shared.invariant_state(student,source,trainer)
        if trainer.state_digest(student.state_dict())!=initial_digest:raise ValueError('Fork altered initial weights')
        fork={'format':'paired-factorial-smoke-v1' if smoke else 'paired-factorial-fork-v1','production_approved':False,'exact_resume':False,
              'arm':arm,'smoke':smoke,'parent_step':600,'parent_checkpoint_sha256':pins['parent_checkpoint_sha256'],
              'updates':1 if smoke else UPDATES,'actual_optimizer_calls':2 if smoke else UPDATES,
              'b32_open':arm in ('B','D'),'feature_weight':.05 if prefix is not None else 0.,
              'pixel_objective':'original 0.5 tab + 0.5 remaining clothing; protected/fresh unchanged',
              'schedule':'50 shuffled cycles of all six identities using restored parent sampling RNG; fresh RNG unchanged',
              'train_ids':[p['id'] for p in train_pairs],'diagnostic_heldout_ids':pins['extra_manifest']['heldout_ids'],
              'optimizer_transfer':transfer,'source_files':expected_hashes,'parent_provenance':pins['provenance'],
              'pair_provenance':pair_provenance,'options':options,'d_metadata':d_meta,'d_prefix_digest':d_digest,
              'parent_student_state_sha256':initial_digest,'fixed_z_sha256':hashlib.sha256(fixed_z.numpy().tobytes()).hexdigest(),
              'rss_abort_bytes':MAX_RSS,'rss_limit_is_sampled_not_hard_allocation_cap':True,'deadline_unix':deadline,
              'torch_threads':2,'interop_threads':1}
        write_json(output/'config.json',fork)
        schedule=[];milestones=[];last_metrics={};diagnostic_report=None
        def check_invariants():
            if shared.invariant_state(student,source,trainer)!=invariants:raise AssertionError('Source/frozen parameters/buffers changed')
            if any(p.grad is not None for p in source.parameters()):raise AssertionError('Source gradient acquired')
            if prefix is not None and (trainer.state_digest(prefix.state_dict())!=d_digest or any(p.grad is not None for p in prefix.parameters())):raise AssertionError('Frozen D changed/acquired gradients')
        def save(step):
            check_invariants();evaluation=render.snapshot(student,pairs,fixed_z,output,step,trainer,torch,sampling,preservation);check_invariants()
            state={'format':fork['format'],'production_approved':False,'fork_step':step,'step':600+step,'parent_step':600,
                   'student':student.state_dict(),'optimizer':optimizer.state_dict(),'rng':trainer.capture_rng('cpu',sampling,preservation),
                   'fixed_z':fixed_z,'init_kwargs':meta['init_kwargs'],'options':options,'fork':fork,'invariants':invariants,
                   'source_digest':invariants['source'],'frozen_digest':invariants['student_frozen'],'last_metrics':last_metrics,
                   'schedule':schedule,'balanced_plan':plan,'next_plan_offset':len(schedule)}
            path=output/f'checkpoint-{step:03}.pt';trainer.atomic_save(state,path)
            milestones.append({'fork_step':step,'checkpoint':path.name,'sha256':sha(path),'student_sha256':evaluation['state_sha256'],'preview_seconds':evaluation['seconds']})
        save(0)
        if not equal_tree(initial_rng,trainer.capture_rng('cpu',sampling,preservation),torch):raise AssertionError('Initial preview altered RNG')
        if smoke:
            # Two independent one-update replicas, both starting at the same immutable parent.
            calibration=next(p for p in train_pairs if p['id']=='calibration-original')
            tick=time.monotonic();diagnostic_report=update(student,source,optimizer,calibration,options,preservation,trainer,prefix,True)
            diagnostic_report['seconds_with_two_extra_autograd_passes']=time.monotonic()-tick
            check_invariants();diag_student=trainer.state_digest(student.state_dict());diag_optimizer=trainer.state_digest({f'{k}/{n}':v for k,s in optimizer.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v)})
            diag_rng=trainer.capture_rng('cpu',sampling,preservation)
            student.load_state_dict(checkpoint['student'],strict=True)
            optimizer,transfer_again=fork_optimizer(student,checkpoint['optimizer'],True,trainer,torch,pins['trainable_parameters'])
            if transfer_again!=transfer:raise AssertionError('Smoke reinitialization optimizer transfer differs')
            trainer.restore_rng(initial_rng,'cpu',sampling,preservation)
            tick=time.monotonic();last_metrics=update(student,source,optimizer,calibration,options,preservation,trainer,prefix,False)
            last_metrics['ordinary_update_seconds']=time.monotonic()-tick
            normal_optimizer=trainer.state_digest({f'{k}/{n}':v for k,s in optimizer.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v)})
            if trainer.state_digest(student.state_dict())!=diag_student or normal_optimizer!=diag_optimizer or not equal_tree(diag_rng,trainer.capture_rng('cpu',sampling,preservation),torch):raise AssertionError('Diagnostic and ordinary smoke updates differ')
            diagnostic_report['ordinary_update_seconds']=last_metrics['ordinary_update_seconds'];diagnostic_report['independent_updates_equal']=True
            write_json(output/'gradient-calibration.json',diagnostic_report);write_json(output/'ordinary-update.json',last_metrics)
            save(1)
        else:
            del checkpoint
            with (output/'metrics.jsonl').open('w') as log:
                for i in range(UPDATES):
                    draw,expected_preservation=planned_draw(i,plan,sampling,preservation,train_pairs,source.z_dim,torch)
                    tick=time.monotonic();metrics=update(student,source,optimizer,train_pairs[draw['pair_index']],options,preservation,trainer,prefix)
                    if metrics['pair_ids']!=[draw['pair_id']] or not torch.equal(preservation.get_state(),expected_preservation):raise AssertionError('Actual draw differed from matching plan')
                    metrics.update(fork_step=i+1,parent_step=600,step=601+i,update_seconds=time.monotonic()-tick,**draw)
                    last_metrics=metrics;schedule.append(draw);log.write(json.dumps(metrics)+'\n');log.flush();print(json.dumps(metrics),flush=True)
                    if i+1 in MILESTONES:save(i+1)
            final_evaluation=render.source30(student,source,ROOT/pins['source30_manifest']['path'],train_pairs,output,trainer,torch,sampling,preservation)
            check_invariants()
        if own_hashes()!=expected_hashes:raise ValueError('Experiment sources changed during execution')
        final=torch.load(output/f'checkpoint-{1 if smoke else UPDATES:03}.pt',weights_only=True,map_location='cpu',mmap=True)
        if not equal_tree(final['optimizer'],optimizer.state_dict(),torch) or trainer.state_digest(final['student'])!=trainer.state_digest(student.state_dict()) or not equal_tree(final['rng'],trainer.capture_rng('cpu',sampling,preservation),torch):raise AssertionError('Final full checkpoint roundtrip mismatch')
        write_json(output/'complete.json',{'complete':True,'arm':arm,'smoke':smoke,'updates_per_trajectory':1 if smoke else UPDATES,
                   'actual_optimizer_calls':2 if smoke else UPDATES,'milestones':milestones,'schedule':schedule,
                   'initial_student_state_sha256':initial_digest,'initial_rng_sha256':trainer.state_digest({k:v for k,v in initial_rng.items() if torch.is_tensor(v)}),
                   'final_student_state_sha256':trainer.state_digest(student.state_dict()),'source_and_frozen_checks_passed':True,
                   'full_checkpoint_roundtrip':True,'final_source30_count':0 if smoke else final_evaluation['count'],
                   'peak_rss_bytes':shared.rss_bytes(),'wall_seconds':time.monotonic()-started,'source_files':expected_hashes,
                   'ordinary_update_seconds':last_metrics.get('ordinary_update_seconds'),'gradient_calibration':diagnostic_report})
    except Exception as exc:
        write_json(output/'failure.json',{'complete':False,'error':str(exc),'peak_rss_bytes':shared.rss_bytes()});raise
    finally:stop.set();watch.join(timeout=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',choices=ARMS,help=argparse.SUPPRESS)
    parser.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--source-hashes',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve()
    if not args.execute:parser.error('Explicit --execute required; models otherwise remain unloaded')
    if not output.is_relative_to(ROOT/'research') or output.exists():parser.error('Use a NEW research directory; no overwrite/resume supported')
    if args.smoke and args.worker not in (None,'D'):parser.error('Smoke uses feature plus b32 arm D only')
    hashes=own_hashes();deadline=time.time()+MAX_SECONDS if args.deadline is None else args.deadline
    if deadline>time.time()+MAX_SECONDS:parser.error('Deadline cannot exceed four hours')
    if args.worker:
        if args.source_hashes is None:parser.error('Worker requires parent source hash file')
        worker(args.worker,output,deadline,json.loads(Path(args.source_hashes).read_text()),args.smoke);return
    shared.memory_guard();verify_pins();output.mkdir(parents=True);write_json(output/'source-hashes.json',hashes)
    write_json(output/'launch.json',{'arms':['D'] if args.smoke else list(ARMS),'smoke':args.smoke,'deadline_unix':deadline,'sources':hashes,'production_approved':False})
    results=[]
    for arm in (('D',) if args.smoke else ARMS):
        if time.time()>=deadline:raise RuntimeError('Total deadline expired between arms')
        shared.memory_guard()
        command=[sys.executable,str(HERE/'runner.py'),'--execute','--output',str(output/arm),'--worker',arm,'--deadline',str(deadline),'--source-hashes',str(output/'source-hashes.json')]
        if args.smoke:command.append('--smoke')
        with (output/f'{arm}.log').open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
        if completed.returncode!=0:raise RuntimeError(f'Arm {arm} failed with exit {completed.returncode}; see {arm}.log; do not rerun completed arms')
        result=json.loads((output/arm/'complete.json').read_text());results.append(result)
        print(json.dumps({'arm':arm,'complete':True,'seconds':result['wall_seconds'],'peak_rss_bytes':result['peak_rss_bytes']}),flush=True)
    if not args.smoke and (len({r['initial_student_state_sha256'] for r in results})!=1 or len({r['initial_rng_sha256'] for r in results})!=1 or any(r['schedule']!=results[0]['schedule'] for r in results)):raise AssertionError('Matched-arm initialization/sampling differed')
    write_json(output/'comparison.json',{'complete':True,'smoke':args.smoke,'arms':[r['arm'] for r in results],
               'initial_weights_and_sampling_matched':not args.smoke,'source_files':hashes,'quality_approved':False,
               'production_approved':False,'results':[{k:r[k] for k in ['arm','peak_rss_bytes','wall_seconds','final_student_state_sha256']} for r in results]})


if __name__=='__main__':main()
