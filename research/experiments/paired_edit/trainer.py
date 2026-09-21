"""Bounded research-only paired editing of an existing NVIDIA generator.

Pixel baseline, not a JoJoGAN/PTI reproduction. Vendor code is unmodified.
No adversarial training, output compositing, finite response catalog, or export.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import PIL
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
VENDOR = RESEARCH / 'vendor/stylegan2-ada-pytorch'
sys.path.insert(0, str(VENDOR))
from training.networks import Generator


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(to_cpu(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def grayscale(image):
    # Same differentiable, fixed operation for source, target and student.
    return (image * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


def masked_l1(image, target, mask):
    if image.shape != target.shape or mask.shape != (image.shape[0], 1, *image.shape[2:]):
        raise ValueError('Loss tensors or mask shapes differ')
    area = mask.flatten(1).sum(1) * image.shape[1]
    if not bool((area > 0).all()):
        raise ValueError('Every loss region must have positive area')
    return (((image - target).abs() * mask).flatten(1).sum(1) / area).mean()


def polygon_mask(polygons, resolution):
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('An explicit nonempty clothing_polygons list is required')
    canvas = Image.new('L', (resolution, resolution), 0)
    drawing = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError('Each normalized polygon needs at least three points')
        points = []
        for point in polygon:
            if len(point) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError('Polygon coordinates must be finite normalized x,y in [0,1]')
            points.append(tuple(round(v * (resolution - 1)) for v in point))
        drawing.polygon(points, fill=255)
    mask = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255)[None, None]
    if not 0 < float(mask.sum()) < resolution * resolution:
        raise ValueError('Clothing and protected regions must both be nonempty')
    return mask


def freeze_student(model):
    model.eval().requires_grad_(False)
    for resolution in model.synthesis.block_resolutions:
        if resolution > 32:
            model.synthesis._modules[f'b{resolution}'].requires_grad_(True)
    # Constant noise includes its scaling coefficients, not just noise buffers.
    for name, parameter in model.named_parameters():
        if name.endswith('noise_strength'):
            parameter.requires_grad_(False)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError('No trainable blocks: source resolution must exceed 32')
    return names


def frozen_state(model):
    return {**{name: p for name, p in model.named_parameters() if not p.requires_grad},
            **dict(model.named_buffers())}


def capture_rng(device, sampling, preservation):
    numpy_state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'mps': torch.mps.get_rng_state() if device == 'mps' else None,
            'sampling': sampling.get_state(), 'preservation': preservation.get_state()}


def restore_rng(state, device, sampling, preservation):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    sampling.set_state(state['sampling'])
    preservation.set_state(state['preservation'])
    if device == 'mps':
        if state['mps'] is None:
            raise ValueError('MPS RNG missing from checkpoint')
        torch.mps.set_rng_state(state['mps'])


def rgb_tensor(path, resolution, resize):
    with Image.open(path) as image:
        if image.width != image.height:
            raise ValueError(f'Expected square image without crop: {path}')
        image = image.convert('RGB')
        dimensions = list(image.size)
        if image.size != (resolution, resolution):
            if not resize:
                raise ValueError(f'Source must have native generator resolution: {path}')
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        array = np.array(image)
    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1, array, dimensions


def quantized(image):
    return ((image.detach().cpu()[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()


def load_pairs(manifest_path, source, source_max_error, w_atol):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['version'] != 1 or not manifest['pairs']:
        raise ValueError('Expected nonempty version-1 paired manifest')
    if manifest.get('production_approved', False):
        raise ValueError('This prototype refuses production-approved artifacts')
    records, provenance, seen = [], [], set()
    for item in manifest['pairs']:
        pair_id = item['id']
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError('Pair IDs must be nonempty and unique')
        seen.add(pair_id)
        split = item.get('split', 'train')
        if split not in ['train', 'validation']:
            raise ValueError('split must be train or validation')
        paths = {key: (manifest_path.parent / item[key]).resolve()
                 for key in ['latent_path', 'source_path', 'target_path']}
        if paths['source_path'].suffix.lower() != '.png':
            raise ValueError('Exact source provenance requires lossless PNG')
        with np.load(paths['latent_path'], allow_pickle=False) as latent:
            z = torch.from_numpy(latent['z'].copy())
            w = torch.from_numpy(latent['w'].copy())
        if z.ndim == 1:
            z = z[None]
        if w.ndim == 2:
            w = w[None]
        if z.dtype != torch.float32 or w.dtype != torch.float32:
            raise ValueError('z and w must be float32')
        if z.shape != (1, source.z_dim) or w.shape != (1, source.num_ws, source.w_dim):
            raise ValueError('Latent shape mismatch')
        if not torch.isfinite(z).all() or not torch.isfinite(w).all():
            raise ValueError('Nonfinite latent')
        with torch.no_grad():
            recomputed_w = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
            w_error = float((recomputed_w - w).abs().max())
            if w_error > w_atol:
                raise ValueError(f'{pair_id}: W mismatch {w_error}, allowed {w_atol}')
            original = source.synthesis(w, noise_mode='const', force_fp32=True)
        _, encoded, _ = rgb_tensor(paths['source_path'], source.img_resolution, resize=False)
        source_error = int(np.abs(quantized(original).astype(np.int16) - encoded.astype(np.int16)).max())
        if source_error > source_max_error:
            raise ValueError(f'{pair_id}: source uint8 mismatch {source_error}, allowed {source_max_error}')
        target, _, native_size = rgb_tensor(paths['target_path'], source.img_resolution, resize=True)
        mask = polygon_mask(item['clothing_polygons'], source.img_resolution)
        records.append({'id': pair_id, 'split': split, 'z': z, 'w': w,
                        'original': original.detach(), 'target': target, 'mask': mask})
        provenance.append({'id': pair_id, 'split': split, 'native_target_size': native_size,
                           'source_uint8_max_error': source_error, 'w_max_error': w_error,
                           'clothing_fraction': float(mask.mean()),
                           'clothing_polygons': item['clothing_polygons'],
                           'files': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}})
    if not any(record['split'] == 'train' for record in records):
        raise ValueError('No training pairs')
    return records, provenance


def update(student, source, optimizer, pairs, options, sampling, preservation, device):
    indices = torch.randint(len(pairs), (options['batch'],), generator=sampling).tolist()
    selected = [pairs[index] for index in indices]
    values = {name: torch.cat([pair[name] for pair in selected]).to(device)
              for name in ['w', 'original', 'target', 'mask']}
    optimizer.zero_grad(set_to_none=True)
    generated = grayscale(student.synthesis(values['w'], noise_mode='const', force_fp32=True))
    clothing = masked_l1(generated, grayscale(values['target']), values['mask'])
    protected = masked_l1(generated, grayscale(values['original']), 1 - values['mask'])
    paired = options['clothing_weight'] * clothing + options['protected_weight'] * protected
    if not torch.isfinite(paired):
        raise FloatingPointError('Nonfinite paired loss')
    paired.backward()
    metrics = {'pair_ids': [pair['id'] for pair in selected], 'clothing_l1': float(clothing.detach()),
               'protected_l1': float(protected.detach()), 'paired_loss': float(paired.detach()),
               'fresh_preservation_l1': 0.0}
    del generated, clothing, protected, paired, values
    if options['fresh_weight'] > 0:
        z = torch.randn(1, source.z_dim, generator=preservation).to(device)
        with torch.no_grad():
            w = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
            original = grayscale(source.synthesis(w, noise_mode='const', force_fp32=True))
        predicted = grayscale(student.synthesis(w, noise_mode='const', force_fp32=True))
        # Explicit geometric approximation, not a face-segmentation claim.
        height = max(1, round(source.img_resolution * options['upper_fraction']))
        fresh = (predicted[:, :, :height] - original[:, :, :height]).abs().mean()
        if not torch.isfinite(fresh):
            raise FloatingPointError('Nonfinite fresh-latent loss')
        (options['fresh_weight'] * fresh).backward()
        metrics['fresh_preservation_l1'] = float(fresh.detach())
    gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError('Missing or nonfinite student gradients')
    if any(p.grad is not None for p in source.parameters()):
        raise AssertionError('Frozen source acquired a gradient')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in student.parameters() if p.requires_grad):
        raise FloatingPointError('Nonfinite updated student')
    return metrics


def save_previews(student, pairs, fixed_z, run, step, device):
    # No dropout, random noise, train-mode mapping or augmentations are involved.
    with torch.no_grad():
        chosen = [next(pair for pair in pairs if pair['split'] == 'train')]
        chosen += [pair for pair in pairs if pair['split'] == 'validation'][:1]
        requests = [(f'pair-{index:03}-{pair["split"]}', pair['w'].to(device)) for index, pair in enumerate(chosen)]
        requests += [(f'unseen-{index:03}', student.mapping(z[None].to(device), None, truncation_psi=1,
                                                         skip_w_avg_update=True)) for index, z in enumerate(fixed_z)]
        for label, w in requests:
            rgb = student.synthesis(w, noise_mode='const', force_fp32=True)
            # Full generated image; never composite source/target pixels.
            mono = grayscale(rgb).repeat(1, 3, 1, 1)
            Image.fromarray(quantized(mono)).save(run / f'{step:06}-{label}.png')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--total-steps', type=int, default=10, help='Absolute total; explicitly override for longer research')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clothing-weight', type=float, default=1)
    parser.add_argument('--protected-weight', type=float, default=1)
    parser.add_argument('--fresh-weight', type=float, default=0)
    parser.add_argument('--upper-fraction', type=float, default=.75)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--preview-count', type=int, default=2)
    parser.add_argument('--source-max-uint8-error', type=int, default=0)
    parser.add_argument('--w-atol', type=float, default=0)
    parser.add_argument('--mps-memory-cap-gib', type=float)
    parser.add_argument('--production-approved', action='store_true', help='Always rejected; this is research only')
    parser.add_argument('--preflight-only', action='store_true', help='One-thread CPU source verification only; no student/backward')
    args = parser.parse_args()
    if args.production_approved:
        parser.error('Production approval/export is unavailable in this prototype')
    if min(args.total_steps, args.batch, args.threads, args.checkpoint_every) < 1 or args.preview_count < 0:
        parser.error('Counts must be positive (preview count may be zero)')
    if not all(math.isfinite(v) for v in [args.lr, args.clothing_weight, args.protected_weight, args.fresh_weight, args.upper_fraction, args.w_atol]):
        parser.error('Numeric options must be finite')
    if min(args.lr, args.clothing_weight, args.protected_weight) <= 0 or args.fresh_weight < 0 or not 0 < args.upper_fraction < 1:
        parser.error('Loss/LR options out of range')
    if not 0 <= args.source_max_uint8_error <= 255 or args.w_atol < 0:
        parser.error('Explicit source tolerance out of range')
    if args.device == 'mps' and (args.mps_memory_cap_gib is None or not 0 < args.mps_memory_cap_gib <= 24):
        parser.error('MPS requires an explicit allocation cap in (0,24] GiB')
    return args


def main():
    args = arguments()
    if args.preflight_only and (args.device != 'cpu' or args.threads != 1 or args.resume):
        raise ValueError('Preflight requires CPU, one thread and no resume')
    if args.run.exists() and any(args.run.iterdir()):
        raise ValueError('Use a new empty run directory, including for resume')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.device == 'mps':
        torch.mps.set_per_process_memory_fraction(args.mps_memory_cap_gib * 2**30 / torch.mps.recommended_max_memory())
    manifest = json.loads(args.manifest.read_text())
    bundle = (args.manifest.resolve().parent / manifest['source_bundle']).resolve()
    metadata = json.loads((bundle / 'model.json').read_text())
    weights = bundle / 'generator.safetensors'
    if metadata['schema_version'] != 1 or metadata['truncation_psi'] != 1:
        raise ValueError('Prototype requires schema 1, psi=1 exact source bundle')
    if sha256(weights) != metadata['weights_sha256']:
        raise ValueError('Source bundle checksum mismatch')
    source = Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
    source.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    if source.c_dim != 0 or source.img_channels != 3:
        raise ValueError('Only unconditional RGB sources supported')
    source_digest = state_digest(source.state_dict())
    # CPU verification precedes device transfer and any optimizer construction.
    verification_started = time.monotonic()
    pairs, pair_provenance = load_pairs(args.manifest, source, args.source_max_uint8_error, args.w_atol)
    if args.preflight_only:
        args.run.mkdir(parents=True, exist_ok=True)
        pressure = subprocess.run(['memory_pressure'], capture_output=True, text=True, check=True).stdout
        (args.run / 'memory-after.txt').write_text(pressure)
        result = {'passed': True, 'preflight_only': True, 'device': 'cpu', 'threads': 1,
                  'backward_or_optimizer_updates': 0, 'resolution': source.img_resolution,
                  'source_weights_sha256': sha256(weights), 'manifest_sha256': sha256(args.manifest),
                  'trainer_sha256': sha256(__file__), 'pairs': pair_provenance,
                  'source_state_unchanged': state_digest(source.state_dict()) == source_digest,
                  'source_max_uint8_error_allowed': args.source_max_uint8_error,
                  'w_atol_allowed': args.w_atol, 'verification_seconds': time.monotonic() - verification_started,
                  'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'production_approved': False}
        assert result['source_state_unchanged']
        (args.run / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    train_pairs = [pair for pair in pairs if pair['split'] == 'train']
    student = copy.deepcopy(source)
    trainable = freeze_student(student)
    frozen_digest = state_digest(frozen_state(student))
    source.to(args.device); student.to(args.device)
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr, betas=(.9, .999))
    sampling = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    preservation = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    preview_rng = torch.Generator(device='cpu').manual_seed(args.seed + 3)
    fixed_z = torch.randn(args.preview_count, source.z_dim, generator=preview_rng)
    options = {key: value for key, value in vars(args).items()
               if key not in ['manifest', 'run', 'resume', 'total_steps', 'checkpoint_every', 'production_approved', 'preflight_only']}
    provenance = {'manifest_sha256': sha256(args.manifest), 'source_weights_sha256': sha256(weights),
                  'source_metadata_sha256': sha256(bundle / 'model.json'), 'pairs': pair_provenance,
                  'source_files': {'trainer': sha256(__file__), **{name: sha256(VENDOR / name) for name in
                    ['training/networks.py', 'torch_utils/ops/conv2d_resample.py', 'torch_utils/ops/upfirdn2d.py',
                     'torch_utils/ops/bias_act.py', 'torch_utils/ops/conv2d_gradfix.py']}},
                  'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__}}
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['format'] != 'paired-edit-research-v1' or checkpoint['production_approved']:
            raise ValueError('Not an unapproved paired-edit research checkpoint')
        if checkpoint['options'] != options or checkpoint['provenance'] != provenance:
            raise ValueError('Exact resume requires matching options, source/data/code hashes and versions')
        if checkpoint['source_digest'] != source_digest or checkpoint['frozen_digest'] != frozen_digest:
            raise ValueError('Source/frozen state differs from checkpoint')
        student.load_state_dict(checkpoint['student'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        fixed_z = checkpoint['fixed_z']
        step = checkpoint['step']
        restore_rng(checkpoint['rng'], args.device, sampling, preservation)
    if step >= args.total_steps:
        raise ValueError('Total steps must exceed resumed completed steps')
    args.run.mkdir(parents=True, exist_ok=True)
    configuration = {'method': 'paired grayscale masked L1 pixel baseline', 'production_approved': False,
                     'resolution': source.img_resolution, 'train_pairs': len(train_pairs),
                     'validation_pairs': len(pairs) - len(train_pairs), 'total_steps': args.total_steps,
                     'resume': str(args.resume) if args.resume else None, 'options': options,
                     'trainable_parameters': trainable, 'provenance': provenance,
                     'freeze': 'mapping, all buffers/noise_strength, synthesis blocks <=32',
                     'loss_normalization': 'per-image region area, then batch mean; images in [-1,1]',
                     'fresh_region': 'top upper_fraction rows; not a detected facial region',
                     'noise_mode': 'const', 'truncation_psi': 1, 'output_compositing': False,
                     'native_target_resize': 'square LANCZOS to source resolution, no crop/warp'}
    (args.run / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    with (args.run / 'metrics.jsonl').open('w') as log:
        while step < args.total_steps:
            started = time.monotonic()
            metrics = update(student, source, optimizer, train_pairs, options, sampling, preservation, args.device)
            step += 1
            metrics.update(step=step, paired_images_seen=step * args.batch,
                           fresh_images_seen=step if args.fresh_weight > 0 else 0,
                           seconds=time.monotonic() - started)
            if step % args.checkpoint_every == 0 or step == args.total_steps:
                if state_digest(source.state_dict()) != source_digest or state_digest(frozen_state(student)) != frozen_digest:
                    raise AssertionError('Source or frozen student state changed')
                before = capture_rng(args.device, sampling, preservation)
                if args.preview_count:
                    save_previews(student, pairs, fixed_z, args.run, step, args.device)
                restore_rng(before, args.device, sampling, preservation)
                if state_digest(frozen_state(student)) != frozen_digest:
                    raise AssertionError('Preview mutated frozen buffers')
                atomic_save({'format': 'paired-edit-research-v1', 'production_approved': False,
                             'step': step, 'paired_images_seen': step * args.batch,
                             'fresh_images_seen': step if args.fresh_weight > 0 else 0,
                             'student': student.state_dict(), 'optimizer': optimizer.state_dict(),
                             'source_digest': source_digest, 'frozen_digest': frozen_digest,
                             'rng': before, 'fixed_z': fixed_z, 'options': options,
                             'provenance': provenance, 'init_kwargs': metadata['init_kwargs'],
                             'last_metrics': metrics}, args.run / 'resume.pt')
                metrics['checkpoint_verified_frozen_source'] = True
            log.write(json.dumps(metrics) + '\n'); log.flush()
            print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
