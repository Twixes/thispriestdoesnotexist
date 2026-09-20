"""Prepared-only bounded MPS A/B: one selected phase, no optimizer updates.

Run only after coordinating other MPS jobs. CPU equivalence must already pass.
This command is intentionally separate from active training and was not executed
as part of preparing the adapter.
"""
import argparse
import hashlib
import json
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/reference_phases'))
from smoke import FP32Call, BGC, legacy, StyleGAN2Loss, AugmentPipe, training_stats, conv2d_gradfix, grid_sample_gradfix
from adapter import DeferredStats
from check_cpu import reset_stats, tensor_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['Gmain','Greg','Dmain','Dreg'],required=True)
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--memory-cap-gib',type=float,required=True,
                        help='Hard MPS allocator limit in GiB; OOM aborts without changing phase math')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.trials<=3:parser.error('Use one to three bounded trials')
    if not 0<args.memory_cap_gib<=12:parser.error('Explicit memory cap must be greater than zero and at most 12 GiB')
    if not json.loads((HERE/'cpu-equivalence.json').read_text())['passed']:
        raise SystemExit('CPU equivalence prerequisite failed')
    if args.output.exists():raise SystemExit('Refusing to overwrite existing timing evidence')
    pressure=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True).stdout
    free=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1))
    if free<25:raise SystemExit(f'Deferred: {free}% free, below 25% timing gate')
    recommended_bytes=torch.mps.recommended_max_memory()
    cap_bytes=int(args.memory_cap_gib*2**30)
    fraction=cap_bytes/recommended_bytes
    if fraction>1:raise SystemExit('Requested cap exceeds the recommended MPS working set')
    torch.mps.set_per_process_memory_fraction(fraction)
    torch.set_num_threads(1);torch.manual_seed(9297);np.random.seed(9297)
    conv2d_gradfix.enabled=True;grid_sample_gradfix.enabled=True
    with (ROOT/'models/ffhq256.pkl').open('rb') as handle:nets=legacy.load_network_pkl(handle)
    g=nets['G'].to('mps').train().requires_grad_(args.phase.startswith('G'))
    d=nets['D'].to('mps').train().requires_grad_(args.phase.startswith('D'))
    del nets
    augment=AugmentPipe(**BGC).to('mps').train().requires_grad_(False)
    augment.p.copy_(torch.tensor(.25))
    loss=StyleGAN2Loss(device=torch.device('mps'),G_mapping=g.mapping,G_synthesis=FP32Call(g.synthesis),
                      D=FP32Call(d),augment_pipe=augment,r1_gamma=1,pl_batch_shrink=2)
    image=sorted((ROOT/'alignment/collar-only/eyes42').glob('*.png'))[0]
    real=torch.from_numpy(np.array(Image.open(image).convert('RGB'))).permute(2,0,1)[None].repeat(8,1,1,1).to('mps').float()/127.5-1
    c=torch.zeros(8,0,device='mps');z=torch.randn(8,g.z_dim,device='mps')
    saved_buffers=[{name:value.detach().clone() for name,value in model.named_buffers()} for model in [g,d]]
    cpu_rng=torch.get_rng_state();mps_rng=torch.mps.get_rng_state()
    upstream=training_stats.report
    gain={'Gmain':1,'Greg':4,'Dmain':1,'Dreg':16}[args.phase]
    results=[]
    for trial in range(-1,args.trials):
        pair={}
        order=['baseline','deferred'] if trial%2==0 else ['deferred','baseline']
        for variant in order:
            reset_stats()
            for model,buffers in zip([g,d],saved_buffers):
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name,value in model.named_buffers():value.copy_(buffers[name])
            loss.pl_mean.zero_()
            torch.set_rng_state(cpu_rng);torch.mps.set_rng_state(mps_rng)
            calls=[0]
            def immediate(name,value):
                calls[0]+=1
                return upstream(name,value.detach().cpu() if torch.is_tensor(value) else value)
            adapter=DeferredStats(immediate) if variant=='deferred' else None
            training_stats.report=adapter.report if adapter else immediate
            collector=training_stats.Collector(regex='Loss/.*')
            torch.mps.synchronize();started=time.monotonic()
            loss.accumulate_gradients(args.phase,real,c,z,c,sync=True,gain=gain)
            if adapter:adapter.flush();adapter.assert_empty()
            collector.update()
            torch.mps.synchronize();elapsed=time.monotonic()-started
            module=g if args.phase.startswith('G') else d
            grads={name:value.grad.detach().cpu().clone() for name,value in module.named_parameters() if value.grad is not None}
            assert grads and all(torch.isfinite(value).all() for value in grads.values())
            pair[variant]={'gradients':grads,'statistics':dict(collector.as_dict()),'pl_mean':float(loss.pl_mean),
                           'cpu_rng':tensor_record(torch.get_rng_state()),'mps_rng':tensor_record(torch.mps.get_rng_state())}
            if trial>=0:
                results.append({'trial':trial,'variant':variant,'seconds':elapsed,'reports':calls[0],
                                'packs':adapter.last_flush['packs'] if adapter else calls[0],
                                'current_bytes':torch.mps.current_allocated_memory(),
                                'driver_bytes':torch.mps.driver_allocated_memory()})
        assert pair['baseline']['gradients'].keys()==pair['deferred']['gradients'].keys()
        for name in pair['baseline']['gradients']:
            torch.testing.assert_close(pair['baseline']['gradients'][name],pair['deferred']['gradients'][name],rtol=1e-5,atol=1e-7)
        for name in ['statistics','pl_mean','cpu_rng','mps_rng']:
            assert pair['baseline'][name]==pair['deferred'][name],name
    result={'phase':args.phase,'batch':8,'pl_batch_shrink':2,'trials':args.trials,'one_warmup_pair_excluded':True,
            'mps_allocator_cap_bytes':cap_bytes,'mps_memory_fraction':fraction,
            'mps_recommended_max_memory_bytes':recommended_bytes,
            'optimizer_updates':0,'preflight_free_percent':free,'results':results,'gradient_rtol':1e-5,'gradient_atol':1e-7,
            'statistics_pl_rng_exact':True,'real_inputs':'one repeated real image; diagnostic only',
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'adapter_sha256':hashlib.sha256((HERE/'adapter.py').read_bytes()).hexdigest(),
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
