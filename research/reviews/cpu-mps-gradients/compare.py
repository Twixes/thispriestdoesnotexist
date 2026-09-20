"""Deterministic, no-update CPU/MPS gradient comparison of upstream StyleGAN2."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/stylegan2-ada-pytorch'))
import legacy

torch.set_num_threads(2)
torch.manual_seed(20260921)
base = ROOT / 'models/ffhq256.pkl'
with base.open('rb') as stream:
    networks = legacy.load_network_pkl(stream)
g_cpu = networks['G'].cpu().train()
d_cpu = networks['D'].cpu().train()
del networks
z = torch.randn(1, g_cpu.z_dim)
real_path = sorted((ROOT / 'alignment/collar-only/eyes42').glob('*.png'))[0]
real_cpu = torch.from_numpy(np.asarray(Image.open(real_path).convert('RGB')).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1

def generate(g, latent):
    ws = g.mapping(latent, None, skip_w_avg_update=True)
    return g.synthesis(ws, noise_mode='const', force_fp32=True, fused_modconv=False)

with torch.no_grad():
    fake_cpu = generate(g_cpu, z).detach()

def collect(module):
    return {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
            for name, parameter in module.named_parameters()}

def vector_stats(reference, candidate):
    reference = reference.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    error = candidate - reference
    norm = float(reference.norm())
    other_norm = float(candidate.norm())
    return {'count': reference.numel(), 'finite': bool(torch.isfinite(candidate).all()),
            'reference_l2': norm, 'candidate_l2': other_norm,
            'error_l2': float(error.norm()), 'relative_l2': float(error.norm()) / max(norm, 1e-30),
            'max_absolute': float(error.abs().max()),
            'cosine': float(reference.dot(candidate)) / max(norm * other_norm, 1e-30)}

def compare(reference, candidate):
    layers = {}
    sums = [0., 0., 0., 0.]
    missing = []
    for name, ref in reference.items():
        other = candidate[name]
        if ref is None or other is None:
            missing.append({'name': name, 'cpu_missing': ref is None, 'mps_missing': other is None})
            continue
        stats = vector_stats(ref, other)
        layers[name] = stats
        sums[0] += stats['reference_l2'] ** 2
        sums[1] += stats['candidate_l2'] ** 2
        sums[2] += stats['error_l2'] ** 2
        sums[3] += stats['cosine'] * stats['reference_l2'] * stats['candidate_l2']
    return {'all_finite': all(v['finite'] for v in layers.values()),
            'relative_l2': (sums[2] / max(sums[0], 1e-60)) ** .5,
            'cosine': sums[3] / max((sums[0] * sums[1]) ** .5, 1e-30),
            'compared_tensor_count': len(layers), 'missing_gradients': missing,
            'layers': layers}

results = {'torch': torch.__version__, 'seed': 20260921, 'batch_size': 1, 'threads': 2,
           'resolution': 256, 'noise_mode': 'const', 'fused_modconv': False,
           'augmentation': False, 'optimizer_updates': 0,
           'base_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
           'real_image': str(real_path.relative_to(ROOT.parent)),
           'real_image_sha256': hashlib.sha256(real_path.read_bytes()).hexdigest(),
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'timings': {}, 'losses': {}, 'comparisons': {}}
reference = {}
start = time.monotonic()
for device in ['cpu', 'mps']:
    device_started = time.monotonic()
    if device == 'cpu':
        g, d = g_cpu, d_cpu
    else:
        g, d = copy.deepcopy(g_cpu).to(device), copy.deepcopy(d_cpu).to(device)
    latent = z.to(device)
    real = real_cpu.to(device)
    fake = fake_cpu.to(device)
    outputs = {}
    for phase in ['generator', 'discriminator', 'r1']:
        phase_started = time.monotonic()
        g.zero_grad(set_to_none=True)
        d.zero_grad(set_to_none=True)
        if phase == 'generator':
            g.requires_grad_(True)
            d.requires_grad_(False)
            image = generate(g, latent)
            loss = F.softplus(-d(image, None, force_fp32=True)).mean()
            loss.backward()
            gradients = collect(g)
            outputs['generated_image'] = image.detach().cpu()
        elif phase == 'discriminator':
            g.requires_grad_(False)
            d.requires_grad_(True)
            loss = F.softplus(d(fake, None, force_fp32=True)).mean() + F.softplus(-d(real, None, force_fp32=True)).mean()
            loss.backward()
            gradients = collect(d)
        else:
            real = real.detach().requires_grad_(True)
            score = d(real, None, force_fp32=True)
            input_gradient = torch.autograd.grad(score.sum(), real, create_graph=True)[0]
            loss = input_gradient.square().sum((1, 2, 3)).mean()
            loss.backward()
            gradients = collect(d)
            outputs['r1_input_gradient'] = input_gradient.detach().cpu()
        if device == 'mps':
            torch.mps.synchronize()
        results['timings'][f'{device}_{phase}_seconds'] = time.monotonic() - phase_started
        results['losses'][f'{device}_{phase}'] = float(loss.detach().cpu())
        if device == 'cpu':
            reference[phase] = gradients
        else:
            results['comparisons'][phase] = compare(reference.pop(phase), gradients)
        print(device, phase, results['timings'][f'{device}_{phase}_seconds'], results['losses'][f'{device}_{phase}'], flush=True)
        del gradients, loss
    if device == 'cpu':
        reference.update(outputs)
    else:
        for name, value in outputs.items():
            results['comparisons'][name] = vector_stats(reference.pop(name), value)
        results['mps_memory'] = {'current_allocated_bytes': torch.mps.current_allocated_memory(),
                                 'driver_allocated_bytes': torch.mps.driver_allocated_memory()}
    results['timings'][f'{device}_total_seconds'] = time.monotonic() - device_started
results['timings']['total_seconds'] = time.monotonic() - start
results['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps({key: {k: v for k, v in value.items() if k != 'layers'} for key, value in results['comparisons'].items()}, indent=2))
