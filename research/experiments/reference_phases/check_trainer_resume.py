"""Actual CPU accumulation + exact uninterrupted-versus-resumed trainer check."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE/'prototype-tests'


def equal(left, right, path, counts):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and torch.equal(left, right), path
        counts['tensors'] += 1
        counts['elements'] += left.numel()
    elif isinstance(left, np.ndarray):
        assert isinstance(right,np.ndarray) and np.array_equal(left,right), path
    elif isinstance(left, dict):
        assert left.keys()==right.keys(), path+'.keys'
        for key in left:
            equal(left[key],right[key],path+'.'+str(key),counts)
    elif isinstance(left,(tuple,list)):
        assert type(left) is type(right) and len(left)==len(right), path
        for index,(a,b) in enumerate(zip(left,right)):
            equal(a,b,path+'.'+str(index),counts)
    else:
        assert left==right,(path,left,right)


def run(name, steps, resume=None):
    directory = ARTIFACTS/name
    command = [sys.executable,'-u',str(HERE/'trainer.py'),'--device','cpu','--threads','2',
               '--run',str(directory),'--steps',str(steps),'--batch','2','--microbatch','1',
               '--pl-batch-shrink','1','--freeze-d-layers','4','--augment-p','.25',
               '--checkpoint-every','1','--no-snapshots','--verify-updates']
    if resume:
        command += ['--resume',str(resume)]
    started = time.monotonic()
    with (ARTIFACTS/(name+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    return {'name':name,'seconds':time.monotonic()-started,'command':command}


def main():
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if any((ARTIFACTS/name/'resume.pt').exists() for name in ['continuous','first','resumed']):
        raise SystemExit('Evidence already exists; select new artifact paths before repeating')
    phases = [run('continuous',5), run('first',1)]
    phases.append(run('resumed',5,ARTIFACTS/'first/resume.pt'))
    continuous = torch.load(ARTIFACTS/'continuous/resume.pt',map_location='cpu',weights_only=False)
    resumed = torch.load(ARTIFACTS/'resumed/resume.pt',map_location='cpu',weights_only=False)
    counts = {'tensors':0,'elements':0}
    checked = []
    for key in ['format_version','recipe','G','D','G_ema','G_opt','D_opt','pl_mean','augment',
                'batch_idx','images_seen','rng','sampler','training_stats','fixed_z']:
        equal(continuous[key],resumed[key],key,counts)
        checked.append(key)
    assert continuous['batch_idx']==5 and continuous['images_seen']==10
    assert float(continuous['pl_mean'])>0
    records = [json.loads(line) for line in (ARTIFACTS/'resumed/metrics.jsonl').read_text().splitlines()]
    assert records[-1]['step']==5 and any(item['phase']=='Greg' for item in records[-1]['phases'])
    assert all(item['rounds']==2 for record in records for item in record['phases'])
    assert float(continuous['augment']['p'])!=.25, 'ADA controller did not update in test'
    for name in ['continuous','first','resumed']:
        assert not (ARTIFACTS/name/'resume.pt.tmp').exists()
        completion = json.loads((ARTIFACTS/name/'completed.json').read_text())
        assert completion['finite_update_checks'] and completion['frozen_d_unchanged']
    result = {'passed':True,'device':'cpu','threads':2,'resolution':256,'batch':2,'microbatch':1,
              'pl_batch_shrink':1,'freeze_d_layers':4,'steps':5,'images_seen':10,
              'exact_uninterrupted_resume_match':True,'checked_fields':checked,'comparison_counts':counts,
              'pl_mean':float(continuous['pl_mean']),'augment_p':float(continuous['augment']['p']),
              'microbatch_rounds':2,'resumed_pl_phase_executed':True,'ada_update_executed':True,
              'all_phase_update_assertions_passed':True,'durable_checkpoint_no_temp_file':True,
              'subprocesses':phases,'trainer_sha256':hashlib.sha256((HERE/'trainer.py').read_bytes()).hexdigest()}
    (ARTIFACTS/'resume-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
