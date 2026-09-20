"""CPU-only FFHQ1024 feasibility measurement; these are not priest images."""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'vendor/stylegan2-ada-pytorch'))
import legacy

parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=int, choices=[1, 2], required=True)
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
checkpoint = ROOT / 'models/ffhq1024.pkl'
if not checkpoint.exists():
    checkpoint = ROOT / 'models/ffhq1024.download.pkl'
started = time.monotonic()
with checkpoint.open('rb') as stream:
    networks = legacy.load_network_pkl(stream)
generator = networks.pop('G_ema').cpu().eval().requires_grad_(False)
assert generator.img_resolution == 1024 and generator.img_channels == 3
assert next(generator.parameters()).device.type == 'cpu'
del networks
gc.collect()
load_seconds = time.monotonic() - started
with checkpoint.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
if checkpoint.name.endswith('.download.pkl'):
    canonical = ROOT / 'models/ffhq1024.pkl'
    if canonical.exists():
        raise RuntimeError('Refusing to overwrite a canonical checkpoint')
    checkpoint.rename(canonical)
    checkpoint = canonical
provenance = {
    'model': 'NVIDIA original StyleGAN2 FFHQ at 1024px',
    'source_url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
    'path': 'research/models/ffhq1024.pkl', 'sha256': digest,
    'bytes': checkpoint.stat().st_size, 'verified_resolution': 1024,
    'verified_channels': 3, 'loaded_network': 'G_ema',
    'license': 'NVIDIA Source Code License: research/evaluation use only; not MIT',
    'license_file': 'research/vendor/stylegan2-ada-pytorch/LICENSE.txt',
    'license_url': 'https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt',
    'verified_on': '2026-09-21', 'production_approved': False,
    'purpose': 'CPU feasibility benchmark of original FFHQ source; not a trained priest generator',
}
provenance_path = ROOT / 'models/ffhq1024.provenance.json'
if provenance_path.exists():
    assert json.loads(provenance_path.read_text())['sha256'] == digest
else:
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')

folder = OUT / f'threads-{args.threads}'
folder.mkdir(exist_ok=True)
records = []
for index in range(4):
    seed = 2026092100 + index
    z = torch.from_numpy(np.random.default_rng(seed).standard_normal((1, generator.z_dim)).astype('float32'))
    started = time.monotonic()
    with torch.inference_mode():
        output = generator(z, None, noise_mode='const', truncation_psi=1.0, force_fp32=True)
    forward_seconds = time.monotonic() - started
    assert list(output.shape) == [1, 3, 1024, 1024]
    assert torch.isfinite(output).all()
    pixels = ((output[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    image = Image.fromarray(pixels)
    path = folder / f'seed-{seed}.png'
    image.save(path)
    encoded = io.BytesIO()
    encoding_started = time.monotonic()
    image.save(encoded, format='WEBP', quality=90, method=4)
    encode_seconds = time.monotonic() - encoding_started
    records.append({'seed': seed, 'latent_rng': 'numpy PCG64 standard_normal float64 cast float32',
                    'kind': 'cold' if index == 0 else 'warm', 'forward_seconds': forward_seconds,
                    'webp_encode_seconds': encode_seconds, 'webp_bytes': len(encoded.getvalue()),
                    'shape': list(output.shape), 'image_path': str(path.relative_to(ROOT.parent)),
                    'png_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'rgb_pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
    del output, pixels, image
    print(json.dumps(records[-1]), flush=True)
result = {'model_sha256': digest, 'network': 'G_ema', 'resolution': 1024,
          'device': 'cpu', 'cpu_threads': args.threads, 'interop_threads': 1,
          'platform': platform.platform(), 'machine': platform.machine(), 'torch_version': torch.__version__,
          'noise_mode': 'const', 'truncation_psi': 1.0, 'force_fp32': True,
          'load_and_release_unused_networks_seconds': load_seconds,
          'warm_median_forward_seconds': statistics.median(row['forward_seconds'] for row in records[1:]),
          'warm_mean_forward_seconds': statistics.mean(row['forward_seconds'] for row in records[1:]),
          'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
          'peak_rss_includes_full_pickle_initial_load': True,
          'generator_parameter_bytes': sum(p.numel() * p.element_size() for p in generator.parameters()),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'production_approved': False, 'records': records}
(folder / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
