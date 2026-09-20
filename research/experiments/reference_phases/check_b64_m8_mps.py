"""One paper256-sized MPS update: batch64/micro8, mirrored 110-image dataset."""
import hashlib
import json
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'runs/reference256-b64-m8-smoke'
pressure = subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True).stdout
(HERE/'b64-m8-preflight.txt').write_text(pressure)
free = int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1))
if free < 25:
    raise SystemExit(f'Deferred: memory free {free}% below required 25%')
if (OUT/'resume.pt').exists():
    raise SystemExit('Completed evidence already exists; refusing to overwrite')
print(f'Memory preflight: {free}% free; exactly one batch64/micro8 update; 110 originals / 220 virtual entries',flush=True)

import torch
from PIL import Image
import trainer

measurements = []
phase_calls = []
step_calls = []
started = time.monotonic()


def observe(stage):
    torch.mps.synchronize()
    record = {'stage':stage,'seconds':time.monotonic()-started,
              'current_bytes':torch.mps.current_allocated_memory(),
              'driver_bytes':torch.mps.driver_allocated_memory(),
              'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    measurements.append(record)
    return record


original_accumulate = trainer.StyleGAN2Loss.accumulate_gradients
original_step = torch.optim.Adam.step
original_snapshot = trainer.snapshot
original_save = trainer.atomic_save


def accumulate(self, phase, *args, **kwargs):
    result = original_accumulate(self,phase,*args,**kwargs)
    phase_calls.append(phase)
    observe('backward:'+phase+':'+str(phase_calls.count(phase)))
    return result


def step(self,*args,**kwargs):
    result = original_step(self,*args,**kwargs)
    step_calls.append(len(step_calls)+1)
    observe('optimizer:'+str(len(step_calls)))
    return result


def snapshot(*args,**kwargs):
    observe('before_snapshot')
    result = original_snapshot(*args,**kwargs)
    observe('after_snapshot')
    return result


def save(*args,**kwargs):
    result = original_save(*args,**kwargs)
    observe('checkpoint_saved')
    return result


trainer.StyleGAN2Loss.accumulate_gradients = accumulate
torch.optim.Adam.step = step
trainer.snapshot = snapshot
trainer.atomic_save = save
sys.argv = ['trainer.py','--device','mps','--threads','2','--run',str(OUT),'--steps','1',
            '--batch','64','--microbatch','8','--pl-batch-shrink','2','--mirror','--verify-updates',
            '--checkpoint-every','1','--snapshot-every','1']
trainer.main()
observe('completed_training')
assert phase_calls == ['Gmain']*8+['Greg']*8+['Dmain']*8+['Dreg']*8
assert len(step_calls)==4
metrics = [json.loads(line) for line in (OUT/'metrics.jsonl').read_text().splitlines()]
assert len(metrics)==1 and metrics[0]['images_seen']==64
assert (OUT/'resume.pt').stat().st_size>0 and not (OUT/'resume.pt.tmp').exists()
for step_index in [0,1]:
    for suffix in ['raw','untruncated','ema']:
        with Image.open(OUT/f'samples-{step_index:06}-{suffix}.png') as image:
            assert image.size==(1024,1024) and image.mode=='RGB'
# Verify actual serialization can be loaded and carries all expected state.
checkpoint = torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False)
for key in ['G','D','G_ema','G_opt','D_opt','pl_mean','augment','rng','training_stats','sampler','fixed_z']:
    assert key in checkpoint,key
assert checkpoint['batch_idx']==1 and checkpoint['images_seen']==64
assert float(checkpoint['pl_mean'])>0
assert checkpoint['rng']['mps'] is not None
assert checkpoint['recipe']['dataset_count']==220 and checkpoint['recipe']['mirror'] is True
assert len(list((ROOT/'alignment/collar-only/eyes42').glob('*.png')))==110
assert checkpoint['config']['architecture']['D'].find("'mbstd_group_size': 8") >= 0
safe_generator = torch.load(OUT/'generator-000001.pt',map_location='cpu',weights_only=True)
assert safe_generator['step']==1 and safe_generator['images_seen']==64
assert all(torch.equal(value,safe_generator['G_ema'][name]) for name,value in checkpoint['G_ema'].items())
del safe_generator
assert all(torch.isfinite(value).all() for name in ['G','D','G_ema'] for value in checkpoint[name].values())
result = {'passed':True,'preflight_free_percent':free,'device':'mps','batch':64,'microbatch':8,'original_image_count':110,'virtual_dataset_count':220,'mirror':True,
          'pl_batch_shrink':2,'updates':1,'optimizer_steps':4,'phase_calls':phase_calls,
          'heavy_regularized_update_seconds':metrics[0]['seconds'],
          'total_wall_seconds_including_setup_snapshots_checkpoint':time.monotonic()-started,
          'max_observed_current_bytes':max(item['current_bytes'] for item in measurements),
          'max_observed_driver_bytes':max(item['driver_bytes'] for item in measurements),
          'process_peak_rss_before_checkpoint_reload_bytes':max(item['process_peak_rss_bytes'] for item in measurements),
          'process_peak_rss_including_checkpoint_reload_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'memory_caveat':'allocator samples after each backward/step/snapshot, not true instantaneous peaks',
          'checkpoint_reload_validated':True,'weights_only_generator_validated':True,'source_mbstd_group_size':8,'checkpoint_bytes':(OUT/'resume.pt').stat().st_size,
          'snapshot_count':6,'snapshots_validated':True,'metrics':metrics[0],'memory_observations':measurements,
          'trainer_sha256':hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
          'source_hashes':checkpoint['recipe']['source_hashes']}
(HERE/'b64-m8-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key not in ['memory_observations','phase_calls','metrics','source_hashes']},indent=2),flush=True)
