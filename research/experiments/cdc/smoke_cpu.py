"""Bounded real-256px CDC correctness/gradient/CPU benchmark (no optimizer)."""
import copy
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'vendor/stylegan2-ada-pytorch'))
import legacy
from cdc_loss import cdc_loss

torch.set_num_threads(2)
torch.manual_seed(8128)
model_path = ROOT / 'models/ffhq256.pkl'
with model_path.open('rb') as model_file:
    networks = legacy.load_network_pkl(model_file)
source = networks['G'].cpu().eval().requires_grad_(False)
target = copy.deepcopy(source).train().requires_grad_(True)
del networks
latents = torch.randn(4, source.z_dim)

def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()

def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024

initial_source_hash = state_hash(source)
initial_target_hash = state_hash(target)
rng_before = torch.get_rng_state().clone()
target_modes = [module.training for module in target.modules()]
target_buffers = {name: tensor.clone() for name, tensor in target.named_buffers()}
rss_before = peak_rss_bytes()
started = time.perf_counter()
identical, identical_details = cdc_loss(source, target, latents)
identical_seconds = time.perf_counter() - started
identical_value = float(identical.detach())
assert abs(identical_value) < 1e-7, f'Identical models produced {identical_value}'
assert initial_target_hash == state_hash(target), 'CDC forward changed target state'
assert torch.equal(rng_before, torch.get_rng_state()), 'CDC consumed RNG'
del identical

# Perturb one shared upstream style affine enough to change relationships among
# the four latents. No optimization is performed or checkpoint produced.
perturb_name = 'synthesis.b32.conv0.affine.weight'
parameter = dict(target.named_parameters())[perturb_name]
with torch.no_grad():
    perturbation = .15 * parameter.std() * torch.randn_like(parameter)
    parameter.add_(perturbation)
perturbed_hash = state_hash(target)
rng_before = torch.get_rng_state().clone()
started = time.perf_counter()
loss, details = cdc_loss(source, target, latents)
forward_seconds = time.perf_counter() - started
value = float(loss.detach())
assert value > 1e-8, f'Perturbation did not produce meaningful positive KL: {value}'
started = time.perf_counter()
loss.backward()
backward_seconds = time.perf_counter() - started
grads = {name: param.grad for name, param in target.named_parameters() if param.grad is not None}
assert grads and all(torch.isfinite(grad).all() for grad in grads.values()), 'Invalid target gradient'
gradient_norm = sum(float(grad.square().sum()) for grad in grads.values()) ** .5
assert gradient_norm > 0, 'All target gradients are zero'
assert target_modes == [module.training for module in target.modules()], 'Target modes changed'
assert all(torch.equal(value, dict(target.named_buffers())[name]) for name, value in target_buffers.items()), 'Target buffer changed'
assert all(param.grad is None for param in source.parameters()), 'Source received gradients'
assert initial_source_hash == state_hash(source), 'Source weights or buffers changed'
assert perturbed_hash == state_hash(target), 'Loss backward unexpectedly changed target weights'
assert torch.equal(rng_before, torch.get_rng_state()), 'CDC forward/backward consumed RNG'
assert all(not module._forward_hooks for module in source.modules()), 'Source hooks leaked'
assert all(not module._forward_hooks for module in target.modules()), 'Target hooks leaked'
rejected_small_batch = False
try:
    cdc_loss(source, target, latents[:2])
except ValueError:
    rejected_small_batch = True
assert rejected_small_batch

report = {'device': 'cpu', 'threads': 2, 'batch': 4, 'resolution': 256,
          'torch': torch.__version__, 'platform': platform.platform(),
          'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
          'source_state_sha256': initial_source_hash,
          'helper_sha256': hashlib.sha256((HERE / 'cdc_loss.py').read_bytes()).hexdigest(),
          'identical_loss': identical_value, 'identical_details': identical_details,
          'perturbed_loss': value, 'perturbation': {'tensor': perturb_name, 'std_fraction': .15},
          'target_gradient_l2': gradient_norm, 'target_tensors_with_gradient': len(grads),
          'target_gradients_finite': True, 'source_has_no_gradients': True,
          'source_unchanged': True, 'target_weights_unchanged_by_forward_backward': True,
          'target_buffers_unchanged': True, 'target_modes_unchanged': True,
          'rng_unchanged': True, 'hooks_removed': True, 'batch2_rejected': True,
          'timing_seconds': {'identical_forward': identical_seconds,
                             'perturbed_forward': forward_seconds,
                             'perturbed_backward': backward_seconds,
                             'perturbed_total': forward_seconds + backward_seconds},
          'memory': {'metric': 'process cumulative peak RSS; includes models and prior phases',
                     'before_calls_bytes': rss_before, 'after_backward_bytes': peak_rss_bytes(),
                     'incremental_peak_bytes': max(0, peak_rss_bytes() - rss_before)},
          'details': details, 'optimizer_updates': 0,
          'scope': 'CDC-only diagnostic, not a full paper reproduction or image-quality result'}
(HERE / 'smoke-cpu.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
