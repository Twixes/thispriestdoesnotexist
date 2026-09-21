"""New unfiltered 64-source batch; adapted from sources32, no training or MPS."""
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HERE.mkdir(exist_ok=True)
if (HERE / "manifest.json").exists() or any((HERE / "images").glob("*.png")):
    raise SystemExit("Refusing to overwrite existing source evidence")
preflight = subprocess.run(['memory_pressure'], capture_output=True, text=True, check=True).stdout
(HERE / 'host-memory-before.txt').write_text(preflight)
free_percent = int(re.search(r'System-wide memory free percentage: (\d+)%', preflight).group(1))
if free_percent < 25:
    raise SystemExit(f'Deferred: memory free {free_percent}% is below required 25%')

import numpy as np
from PIL import Image, ImageDraw, features, __version__ as pillow_version
import torch

sys.path.insert(0, str(ROOT))
from inference.server import Model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
BUNDLE = ROOT / 'research/runs/inference-cpu/ffhq1024/baseline-bundle'
EXPECTED_WEIGHTS = 'f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



started = time.monotonic()
model = Model(BUNDLE, threads=1, allow_unreviewed=True)
assert model.sha256 == EXPECTED_WEIGHTS
assert model.generator.img_resolution == 1024
assert model.metadata['truncation_psi'] == 1.0
assert next(model.generator.parameters()).device.type == 'cpu'
load_seconds = time.monotonic() - started
noise = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
         for name, value in model.generator.named_buffers() if name.endswith('noise_const')}
noise_sha = hashlib.sha256(json.dumps(noise, sort_keys=True).encode()).hexdigest()
captured = {}


def capture_mapping(module, inputs, output):
    captured['ws'] = output.detach().cpu().numpy().copy()


hook = model.generator.mapping.register_forward_hook(capture_mapping)


def generate(name, seed_hex):
    row_started = time.monotonic()
    seed = int.from_bytes(bytes.fromhex(seed_hex), 'big')
    z = np.random.default_rng(seed).standard_normal((1, model.generator.z_dim), dtype=np.float32)
    with torch.inference_mode():
        output = model.generator(torch.from_numpy(z), None, truncation_psi=1.0,
                                 noise_mode='const', force_fp32=True)
        pixels = ((output[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
    forward_seconds = time.monotonic() - row_started
    assert pixels.shape == (1024, 1024, 3)
    path = HERE / 'images' / f'{name}.png'
    Image.fromarray(pixels).save(path)
    latent_path = HERE / 'latents' / f'{name}.npz'
    ws = captured.pop('ws')
    np.savez(latent_path, z=z, w=ws)
    row = {'id': name, 'seed_hex': seed_hex, 'image': str(path.relative_to(HERE)),
           'image_sha256': sha(path), 'rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
           'latents': str(latent_path.relative_to(HERE)), 'latents_sha256': sha(latent_path),
           'source_path': str(path.relative_to(HERE)), 'latent_path': str(latent_path.relative_to(HERE)),
           'z_shape': list(z.shape), 'w_shape': list(ws.shape), 'ws_shape': list(ws.shape),
           'all_ws_equal_first_w': bool(np.array_equal(ws, np.repeat(ws[:, :1, :], ws.shape[1], axis=1))),
           'z_raw_sha256': hashlib.sha256(z.tobytes()).hexdigest(),
           'ws_raw_sha256': hashlib.sha256(ws.tobytes()).hexdigest(),
           'dimensions': [1024, 1024], 'forward_seconds': forward_seconds,
           'total_save_seconds': time.monotonic() - row_started}
    print(json.dumps(row), flush=True)
    return row, pixels


(HERE / 'images').mkdir(exist_ok=True)
(HERE / 'latents').mkdir(exist_ok=True)
rows = []
batch_started = time.monotonic()
for index in range(64):
    seed_hex = hashlib.sha256(f'priest-paired-source-v2-batch2:{index}'.encode('utf-8')).hexdigest()
    row, _ = generate(f'{index:03}', seed_hex)
    row['seed_derivation_index'] = index
    rows.append(row)
batch_seconds = time.monotonic() - batch_started
assert len({row['seed_hex'] for row in rows}) == 64
assert len({row['rgb_sha256'] for row in rows}) == 64

hook.remove()
contact = Image.new('RGB', (8 * 192, 8 * 214), (24, 24, 24))
draw = ImageDraw.Draw(contact)
for index, row in enumerate(rows):
    x, y = (index % 8) * 192, (index // 8) * 214
    with Image.open(HERE / row['image']) as im:
        contact.paste(im.resize((192, 192), Image.Resampling.LANCZOS), (x, y))
    draw.text((x + 5, y + 195), row['id'], fill='white')
contact.save(HERE / 'contact.png')
report = {
    'purpose': 'Second, unfiltered64-source batch for future offline edits; not a production seed catalog',
    'adapted_from': str(Path('research/data/ffhq-paired-sources32/generate.py')),
    'new_seed_prefix': 'priest-paired-source-v2-batch2:',
    'training_eligible': False, 'production_approved': False, 'is_priest_generator': False,
    'count': 64, 'seed_derivation': "SHA256(UTF8('priest-paired-source-v2-batch2:' + decimal zero-based index))",
    'latent_recipe': 'SeedSequence integer from full 32-byte big-endian seed; NumPy default_rng PCG64 standard_normal((1,512), dtype=float32)',
    'mapping_recipe': 'Capture actual mapping forward output during exact generator forward; NPZ w preserves all synthesis style inputs [1,18,512]',
    'noise_mode': 'const', 'truncation_psi': 1.0, 'force_fp32': True,
    'model': model.health(), 'model_metadata_sha256': sha(BUNDLE / 'model.json'),
    'source_provenance': json.loads((BUNDLE / 'source-provenance.json').read_text()),
    'constant_noise_raw_hashes': noise, 'constant_noise_manifest_sha256': noise_sha,
    'code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), ROOT / 'inference/server.py', ROOT / 'research/vendor/stylegan2-ada-pytorch/training/networks.py']},
    'runtime': {'platform': platform.platform(), 'python': sys.version, 'torch': torch.__version__,
                'numpy': np.__version__, 'pillow': pillow_version, 'libwebp': features.version('webp'),
                'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()},
    'memory_preflight_free_percent': free_percent, 'model_load_seconds': load_seconds,
    'batch64_seconds': batch_seconds, 'total_seconds': time.monotonic() - started,
    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'contact_sha256': sha(HERE / 'contact.png'), 'entries': rows,
}
(HERE / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ['entries', 'constant_noise_raw_hashes']}, indent=2), flush=True)
