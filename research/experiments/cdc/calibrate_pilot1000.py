"""No-update CDC gradient calibration and a separate bounded MPS diagnostic."""
import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'vendor/stylegan2-ada-pytorch'), str(ROOT / 'vendor/diffaugment')]
import legacy
from DiffAugment_pytorch import DiffAugment
from cdc_loss import cdc_loss

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
parser.add_argument('--mode', choices=['calibration', 'cdc-benchmark'], default='calibration')
args = parser.parse_args()
if args.device != 'cpu' and args.mode == 'calibration':
    parser.error('Full adversarial calibration is CPU-only; MPS uses cdc-benchmark')
torch.set_num_threads(2)
torch.manual_seed(99127)
if args.device == 'mps':
    torch.mps.manual_seed(99127)
base = ROOT / 'models/ffhq256.pkl'
resume = ROOT / 'runs/pilot256/resume.pt'
checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
assert checkpoint['step'] == 1000, f"Expected archived pilot1000, got {checkpoint['step']}"
with base.open('rb') as stream:
    networks = legacy.load_network_pkl(stream)
source = networks['G'].eval().requires_grad_(False)
target = copy.deepcopy(source).train().requires_grad_(True)
target.load_state_dict(checkpoint['G'])
discriminator = networks['D'].train().requires_grad_(False)
discriminator.load_state_dict(checkpoint['D'])
augment_p = float(checkpoint['aug_p'])
del networks, checkpoint
source = source.to(args.device)
target = target.to(args.device)
if args.mode == 'calibration':
    discriminator = discriminator.to(args.device)
else:
    del discriminator
latents = torch.randn(4, source.z_dim, device=args.device)
parameters = list(target.parameters())
names = [name for name, _ in target.named_parameters()]

def synchronize():
    if args.device == 'mps':
        torch.mps.synchronize()

def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def gradient_norm(gradients):
    return math.sqrt(sum(float(value.detach().cpu().double().square().sum())
                         for value in gradients if value is not None))

source_hash = state_hash(source)
target_hash = state_hash(target)
target_buffers = {name: value.detach().cpu().clone() for name, value in target.named_buffers()}
synchronize()
before_memory = {}
if args.device == 'mps':
    before_memory = {'current_allocated_bytes': torch.mps.current_allocated_memory(),
                     'driver_allocated_bytes': torch.mps.driver_allocated_memory()}
started = time.perf_counter()
loss, details = cdc_loss(source, target, latents)
synchronize()
forward_seconds = time.perf_counter() - started
started = time.perf_counter()
cdc_gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
synchronize()
backward_seconds = time.perf_counter() - started
assert all(torch.isfinite(value).all() for value in cdc_gradients if value is not None)
cdc_norm = gradient_norm(cdc_gradients)
assert math.isfinite(float(loss.detach())) and cdc_norm > 0
report = {'device': args.device, 'mode': args.mode, 'threads': 2, 'batch': 4,
          'resolution': 256, 'checkpoint_step': 1000, 'torch': torch.__version__,
          'platform': platform.platform(), 'source_state_sha256': source_hash,
          'target_state_sha256': target_hash, 'source_model': str(base.relative_to(ROOT.parent)),
          'pilot_checkpoint': str(resume.relative_to(ROOT.parent)),
          'helper_sha256': hashlib.sha256((HERE / 'cdc_loss.py').read_bytes()).hexdigest(),
          'calibration_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cdc_loss_unweighted': float(loss.detach().cpu()), 'cdc_gradient_l2': cdc_norm,
          'cdc_tensors_with_gradient': sum(value is not None for value in cdc_gradients),
          'cdc_details': details, 'seconds': {'cdc_forward': forward_seconds,
          'cdc_backward': backward_seconds, 'cdc_total': forward_seconds + backward_seconds},
          'optimizer_updates': 0}

if args.mode == 'calibration':
    started = time.perf_counter()
    styles = target.mapping(latents, None, skip_w_avg_update=True)
    fake = target.synthesis(styles, force_fp32=True, noise_mode='const', fused_modconv=False)
    flip = torch.rand(4, 1, 1, 1) < .5
    fake = torch.where(flip, fake.flip(3), fake)
    augmented = DiffAugment(fake, policy='color,translation,cutout')
    fake = torch.where(torch.rand(4, 1, 1, 1) < augment_p, augmented, fake)
    adversarial = F.softplus(-discriminator(fake, None, force_fp32=True)).mean()
    adversarial_gradients = torch.autograd.grad(adversarial, parameters, allow_unused=True)
    adversarial_seconds = time.perf_counter() - started
    assert all(torch.isfinite(value).all() for value in adversarial_gradients if value is not None)
    adversarial_norm = gradient_norm(adversarial_gradients)
    dot = sum(float((a.detach().double() * c.detach().double()).sum())
              for a, c in zip(adversarial_gradients, cdc_gradients) if a is not None and c is not None)
    per_group = {}
    for name, adv, cdc in zip(names, adversarial_gradients, cdc_gradients):
        group = name.split('.')[0] if name.startswith('mapping.') else '.'.join(name.split('.')[:2])
        values = per_group.setdefault(group, {'adversarial_l2_squared': 0., 'cdc_l2_squared': 0.})
        if adv is not None: values['adversarial_l2_squared'] += float(adv.detach().double().square().sum())
        if cdc is not None: values['cdc_l2_squared'] += float(cdc.detach().double().square().sum())
    report.update({'adversarial_loss': float(adversarial.detach()), 'adversarial_gradient_l2': adversarial_norm,
                   'adversarial_seconds': adversarial_seconds, 'augment_p': augment_p,
                   'gradient_cosine': dot / (adversarial_norm * cdc_norm),
                   'weight_for_equal_global_gradient_norm': adversarial_norm / cdc_norm,
                   'weighted_comparisons': [{'weight': weight, 'weighted_cdc_loss': weight * float(loss.detach()),
                                            'weighted_cdc_gradient_l2': weight * cdc_norm,
                                            'cdc_to_adversarial_norm_ratio': weight * cdc_norm / adversarial_norm,
                                            'combined_gradient_l2': math.sqrt(adversarial_norm ** 2 +
                                                 (weight * cdc_norm) ** 2 + 2 * weight * dot)}
                                           for weight in [100, 1000, 10000]],
                   'per_group': {name: {'adversarial_gradient_l2': math.sqrt(values['adversarial_l2_squared']),
                                        'cdc_gradient_l2': math.sqrt(values['cdc_l2_squared'])}
                                 for name, values in per_group.items()},
                   'adversarial_definition': 'Actual pilot D, softplus(-D), same DiffAugment/flip policy and saved p; matched unmixed z and constant noise for this calibration',
                   'scaling': 'Weighted gradients computed analytically by linear scaling of the one actual CDC gradient; not extra backward passes'})

assert all(parameter.grad is None for parameter in source.parameters()), 'Source received gradients'
assert source_hash == state_hash(source), 'Source changed'
assert target_hash == state_hash(target), 'Target changed without optimizer'
assert all(torch.equal(value, dict(target.named_buffers())[name].detach().cpu())
           for name, value in target_buffers.items()), 'Target buffer changed'
report['source_unchanged_no_gradients'] = True
report['target_weights_and_buffers_unchanged'] = True
report['finite_gradients'] = True
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['process_peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
if args.device == 'mps':
    report['mps_memory_before'] = before_memory
    report['mps_memory_after'] = {'current_allocated_bytes': torch.mps.current_allocated_memory(),
                                'driver_allocated_bytes': torch.mps.driver_allocated_memory()}
    report['memory_scope'] = 'Per-process MPS allocator snapshots, not device-wide peak; main training continues concurrently'
filename = 'calibration-pilot1000.json' if args.mode == 'calibration' else 'benchmark-mps-pilot1000.json'
(HERE / filename).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
