"""Bounded reuse of NVIDIA's actual loss phases, not a new loss implementation.

NVIDIA code/weights retain upstream licensing. This independent harness only
adapts device/FP32/logging and the outer four-phase schedule; see README.md.
"""
import argparse
import gc
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'vendor/stylegan2-ada-pytorch'
sys.path.insert(0, str(VENDOR))
import legacy
from training.loss import StyleGAN2Loss
from training.augment import AugmentPipe
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix

OUT = ROOT/'experiments/reference_phases'
BGC = dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
           brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
COLOR = dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)


class FP32Call(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs, force_fp32=True)


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def memory(device):
    result = {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == 'mps':
        result.update(current_bytes=torch.mps.current_allocated_memory(), driver_bytes=torch.mps.driver_allocated_memory())
    return result


def probe_bgc(device):
    pipe = AugmentPipe(**BGC).to(device).requires_grad_(False)
    pipe.p.copy_(torch.tensor(.5))
    pixels = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
    try:
        values = pipe(pixels)
        first = torch.autograd.grad(values.square().mean(), pixels, create_graph=True)[0]
        first.square().sum().backward()
        return {'passed': True, 'finite': bool(torch.isfinite(pixels.grad).all()), 'resolution': 32}
    except Exception as error:
        return {'passed': False, 'exception': type(error).__name__, 'message': str(error), 'resolution': 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--augmentation', choices=['none', 'color', 'bgc'], default='color')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7371)
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True
    result = {'device': args.device, 'threads': 2, 'batch': 1, 'resolution': 256,
              'torch_version': torch.__version__, 'augmentation': args.augmentation,
              'augmentation_probability': .5, 'pl_batch_shrink': 1,
              'r1_gamma': 1, 'pl_weight': 2, 'style_mixing_probability': .9,
              'source_hashes': {name: hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()
                                for name in ['training/loss.py', 'training/augment.py', 'training/training_loop.py']},
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'phases': []}
    result['bgc_second_derivative_probe'] = probe_bgc(args.device)
    print(json.dumps({'bgc_probe': result['bgc_second_derivative_probe']}), flush=True)
    # Logging counters alone use float64 upstream, unsupported on MPS. Preserve
    # actual loss tensors and gradients; move only detached reporting copies.
    original_report = training_stats.report
    if args.device == 'mps':
        def report_cpu(name, value):
            original_report(name, value.detach().cpu() if torch.is_tensor(value) else value)
            return value
        training_stats.report = report_cpu
        result['reporting_adapter'] = 'detached statistics copied to CPU; loss tensors unchanged'
    with (ROOT/'models/ffhq256.pkl').open('rb') as handle:
        nets = legacy.load_network_pkl(handle)
    g = nets['G'].to(args.device).train().requires_grad_(False)
    d = nets['D'].to(args.device).train().requires_grad_(False)
    del nets
    gc.collect()
    augment = None
    if args.augmentation != 'none':
        augment = AugmentPipe(**(COLOR if args.augmentation == 'color' else BGC)).to(args.device).train().requires_grad_(False)
        augment.p.copy_(torch.tensor(.5))
    loss = StyleGAN2Loss(device=torch.device(args.device), G_mapping=g.mapping,
                        G_synthesis=FP32Call(g.synthesis), D=FP32Call(d), augment_pipe=augment,
                        r1_gamma=1, pl_batch_shrink=1, pl_weight=2)
    phases = []
    result['optimizers'] = {}
    for name, module, interval in [('G', g, 4), ('D', d, 16)]:
        ratio = interval/(interval+1)
        lr = .0025*ratio
        betas = (0.0, .99**ratio)
        optimizer = torch.optim.Adam(module.parameters(), lr=lr, betas=betas, eps=1e-8)
        phases += [(name+'main', module, optimizer, 1), (name+'reg', module, optimizer, interval)]
        result['optimizers'][name] = {'base_lr': .0025, 'lazy_ratio': ratio, 'lr': lr,
                                      'betas': list(betas), 'same_optimizer_for_main_and_reg': True}
    path = sorted((ROOT/'alignment/collar-only/eyes42').glob('*.png'))[0]
    real = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1).unsqueeze(0).to(args.device).float()/127.5-1
    c = torch.zeros(1, 0, device=args.device)
    result['memory_before_phases'] = memory(args.device)
    collector = training_stats.Collector(regex='Loss/.*')
    started = time.monotonic()
    for name, module, optimizer, interval in phases:
        sync(args.device)
        tick = time.monotonic()
        before = next(module.parameters()).detach().cpu().clone()
        optimizer.zero_grad(set_to_none=True)
        module.requires_grad_(True)
        z = torch.randn(1, g.z_dim, device=args.device)
        try:
            loss.accumulate_gradients(name, real, c, z, c, sync=True, gain=interval)
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(grad).all() for grad in grads), name+' non-finite gradients'
            norm = sum(float(grad.detach().cpu().double().square().sum()) for grad in grads)**.5
            assert norm > 0, name+' zero gradient norm'
            module.requires_grad_(False)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in module.parameters()), name+' non-finite parameters'
            assert not torch.equal(before, next(module.parameters()).detach().cpu()), name+' first parameter unchanged'
            sync(args.device)
            collector.update()
            record = {'phase': name, 'gain': interval, 'seconds': time.monotonic()-tick,
                      'finite_gradients': True, 'gradient_l2': norm, 'gradient_tensors': len(grads),
                      'parameter_updated': True, 'pl_mean': float(loss.pl_mean), 'memory': memory(args.device),
                      'reported_statistics': {key: value.mean for key,value in collector.as_dict().items()}}
            result['phases'].append(record)
            print(json.dumps(record), flush=True)
        except Exception as error:
            result['phases'].append({'phase': name, 'error': type(error).__name__, 'message': str(error),
                                     'traceback': traceback.format_exc(), 'memory': memory(args.device)})
            raise
        finally:
            result['phase_seconds_total'] = time.monotonic()-started
            (OUT/f'smoke-{args.device}-{args.augmentation}.json').write_text(json.dumps(result,indent=2)+'\n')
    assert float(loss.pl_mean)>0
    result['all_four_phases_passed'] = True
    result['pl_second_derivative_passed'] = True
    result['r1_second_derivative_passed'] = True
    (OUT/f'smoke-{args.device}-{args.augmentation}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All four upstream phases passed', flush=True)


if __name__ == '__main__':
    main()
