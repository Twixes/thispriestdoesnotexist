"""CPU-only actual FFHQ256 reporting equivalence; one thread, no active edits."""
import argparse
import copy
import hashlib
import json
import math
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/reference_phases'))
from smoke import FP32Call, BGC, legacy, StyleGAN2Loss, AugmentPipe, training_stats, conv2d_gradfix, grid_sample_gradfix
from adapter import DeferredStats


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return {'dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def module_record(module, gradients=False):
    return {name:tensor_record(value.grad if gradients else value)
            for name,value in (module.named_parameters() if gradients else module.state_dict().items())
            if not gradients or value.grad is not None}


def reset_stats():
    training_stats._counters = {}
    training_stats._cumulative = {}
    training_stats._sync_called = False


def fixtures():
    def execute(deferred):
        reset_stats()
        original = training_stats.report
        records = []
        def sink(name,value):
            records.append({'name':name,**tensor_record(torch.as_tensor(value).flatten())})
            return original(name,value)
        adapter = DeferredStats(sink) if deferred else None
        report = adapter.report if adapter else sink
        tensor = torch.tensor([1.25,-2.5],requires_grad=True)
        array = np.array([1+2**-30,1e-30],dtype=np.float64)
        sequence = [2.,-3.]
        values = [('repeated',tensor),('double',array),('list',sequence),('repeated',torch.tensor([65504.,.125],dtype=torch.float16)),
                  ('integer',torch.tensor([2**40+1,-3],dtype=torch.int64)),('boolean',torch.tensor([True,False])),
                  ('scalar',7.25),('noncontiguous',torch.arange(12,dtype=torch.float32).reshape(3,4).T),
                  ('empty',torch.empty(0,dtype=torch.float64))]
        for name,value in values:
            returned = report(name,value)
            if adapter:
                assert returned is value
        with torch.no_grad(): tensor.add_(100)
        array[:] = -100
        sequence[:] = [999.,999.]
        if adapter:
            try: adapter.assert_empty()
            except RuntimeError: pass
            else: raise AssertionError('Unflushed checkpoint guard did not reject pending values')
            assert all(not value.requires_grad and value.grad_fn is None for _,value in adapter.pending)
            adapter.flush();adapter.assert_empty()
        collector = training_stats.Collector(regex='.*')
        # Constructor consumes initial counters and clears moments; compare the
        # cumulative counters themselves as well as replay order/dtype/content.
        cumulative = {name:tensor_record(value) for name,value in training_stats._cumulative.items()}
        return {'records':records,'cumulative':cumulative,'names':collector.names()},adapter.last_flush if adapter else None
    baseline,_ = execute(False)
    candidate,packing = execute(True)
    assert baseline==candidate
    return {'passed':True,'mutation_snapshot_preserved':True,'dtype_order_values_exact':True,
            'mixed_types':['float16','float32','float64','int64','bool','Python scalar/list','NumPy','noncontiguous','empty'],
            'checkpoint_guard_passed':True,'no_autograd_graph_retained':True,'packing':packing}


def run_variant(variant):
    torch.set_num_threads(1)
    torch.manual_seed(9297)
    np.random.seed(9297)
    reset_stats()
    conv2d_gradfix.enabled = True;grid_sample_gradfix.enabled = True
    upstream = training_stats.report
    report_count = [0]
    def immediate(name,value):
        report_count[0] += 1
        return upstream(name,value.detach().cpu() if torch.is_tensor(value) else value)
    deferred = DeferredStats(immediate) if variant=='deferred' else None
    training_stats.report = deferred.report if deferred else immediate
    with (ROOT/'models/ffhq256.pkl').open('rb') as handle: nets=legacy.load_network_pkl(handle)
    g=nets['G'].cpu().train().requires_grad_(False)
    d=nets['D'].cpu().train().requires_grad_(False)
    del nets
    pipe=AugmentPipe(**BGC).train().requires_grad_(False)
    pipe.p.copy_(torch.tensor(.25))
    loss=StyleGAN2Loss(device=torch.device('cpu'),G_mapping=g.mapping,G_synthesis=FP32Call(g.synthesis),
                      D=FP32Call(d),augment_pipe=pipe,r1_gamma=1,pl_batch_shrink=1)
    phases=[]
    for name,module,interval in [('G',g,4),('D',d,16)]:
        ratio=interval/(interval+1)
        opt=torch.optim.Adam(module.parameters(),lr=.0025*ratio,betas=(0.,.99**ratio),eps=1e-8)
        phases.extend([(name+'main',module,opt,1),(name+'reg',module,opt,interval)])
    file=sorted((ROOT/'alignment/collar-only/eyes42').glob('*.png'))[0]
    real=torch.from_numpy(np.array(Image.open(file).convert('RGB'))).permute(2,0,1)[None].float()/127.5-1
    c=torch.zeros(1,0)
    latents=torch.randn(4,1,g.z_dim)
    metrics=training_stats.Collector(regex='Loss/.*')
    ada=training_stats.Collector(regex='Loss/signs/real')
    results=[];timings=[];packs=[]
    initial_rng=tensor_record(torch.get_rng_state())
    for (phase,module,opt,gain),z in zip(phases,latents):
        tick=time.monotonic()
        opt.zero_grad(set_to_none=True);module.requires_grad_(True)
        loss.accumulate_gradients(phase,real,c,z,c,sync=True,gain=gain)
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        gradients=module_record(module,gradients=True)
        module.requires_grad_(False);opt.step()
        assert all(torch.isfinite(p).all() for p in module.parameters())
        if deferred:
            packs.append(deferred.flush());deferred.assert_empty()
        metrics.update()
        results.append({'phase':phase,'gain':gain,'gradients':gradients,'module_state':module_record(module),
                        'pl_mean':tensor_record(loss.pl_mean),'statistics':dict(metrics.as_dict()),
                        'rng':tensor_record(torch.get_rng_state())})
        timings.append({'phase':phase,'seconds':time.monotonic()-tick})
    if deferred: deferred.assert_empty()
    ada.update()
    sign=ada['Loss/signs/real']
    adjustment=np.sign(sign-.6)*4/(100*1000)
    pipe.p.copy_((pipe.p+adjustment).clamp_min(0))
    if deferred: deferred.assert_empty()
    checkpoint_counters={name:tensor_record(value) for name,value in training_stats._cumulative.items()}
    result={'variant':variant,'device':'cpu','threads':1,'resolution':256,'batch':1,'pl_batch_shrink':1,
            'initial_rng':initial_rng,'latents':tensor_record(latents),'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'phases':results,'ada_sign_mean':sign,'ada_probability':tensor_record(pipe.p),'checkpoint_counters':checkpoint_counters,
            'final_g':module_record(g),'final_d':module_record(d),'report_count':report_count[0],
            'all_gradients_and_weights_finite':True,'packing':packs,'timings':timings,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'adapter_sha256':hashlib.sha256((HERE/'adapter.py').read_bytes()).hexdigest(),
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'loss_sha256':hashlib.sha256((ROOT/'vendor/stylegan2-ada-pytorch/training/loss.py').read_bytes()).hexdigest()}
    (HERE/f'cpu-{variant}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'variant':variant,'passed':True,'timings':timings,'report_count':report_count[0],
                      'peak_rss_bytes':result['process_peak_rss_bytes']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline','deferred']);args=parser.parse_args()
    if args.variant:
        run_variant(args.variant);return
    torch.set_num_threads(1)
    fixture=fixtures()
    (HERE/'fixtures.json').write_text(json.dumps(fixture,indent=2)+'\n')
    gates=[]
    for variant in ['baseline','deferred']:
        pressure=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True).stdout
        (HERE/f'memory-{variant}.txt').write_text(pressure)
        free=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1))
        if free<15: raise SystemExit(f'Deferred CPU test postponed: {free}% memory free')
        gates.append({'variant':variant,'free_percent':free})
        with (HERE/f'cpu-{variant}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--variant',variant],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    a=json.loads((HERE/'cpu-baseline.json').read_text());b=json.loads((HERE/'cpu-deferred.json').read_text())
    checked=[key for key in a if key not in ['variant','packing','timings','process_peak_rss_bytes']]
    for key in checked: assert a[key]==b[key],key
    summary={'passed':True,'exact_gradients_parameters_pl_rng_stats_ada_match':True,'checked_fields':checked,
             'fixtures':fixture,'memory_preflights':gates,'baseline_reports':a['report_count'],
             'candidate_packs':sum(x['packs'] for x in b['packing']),
             'baseline_seconds':sum(x['seconds'] for x in a['timings']),
             'deferred_seconds':sum(x['seconds'] for x in b['timings']),
             'timing_caveat':'CPU equivalence test with state hashing, not an MPS speed benchmark'}
    (HERE/'cpu-equivalence.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
