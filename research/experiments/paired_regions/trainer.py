"""Isolated research-only equal-area collar/rest extension.

Imports immutable paired_edit helpers; never resumes a broad-mask checkpoint.
No production export. Full images come from the student generator.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import numpy as np
import PIL
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
sys.path.insert(0, str(RESEARCH.parent))
from research.experiments.paired_edit import trainer as base

VENDOR = base.VENDOR
Generator = base.Generator
sha256 = base.sha256
state_digest = base.state_digest
atomic_save = base.atomic_save
grayscale = base.grayscale
masked_l1 = base.masked_l1
polygon_mask = base.polygon_mask
freeze_student = base.freeze_student
frozen_state = base.frozen_state
capture_rng = base.capture_rng
restore_rng = base.restore_rng
rgb_tensor = base.rgb_tensor
quantized = base.quantized
save_previews = base.save_previews


def source_hashes():
    # Include complete imported base helper module plus every vendored Python/
    # native implementation file; new or removed files also change provenance.
    files = {"regions_trainer": sha256(__file__), "paired_edit_helpers": sha256(base.__file__)}
    for directory in ['training', 'torch_utils', 'dnnlib']:
        for path in sorted((VENDOR / directory).rglob('*')):
            if path.is_file() and path.suffix in ['.py', '.cpp', '.cu', '.h', '.hpp']:
                files[str(path.relative_to(VENDOR))] = sha256(path)
    return files


def partition_masks(clothing, polygons, resolution):
    trace = polygon_mask(polygons, resolution)
    collar = trace * clothing
    rest = clothing - collar
    if not 0 < float(collar.sum()) < float(clothing.sum()):
        raise ValueError('Training collar intersection must be a nonempty strict subset of clothing; rest must be nonempty')
    return collar, rest, {'raw_trace_pixels': int(trace.sum()),
                         'collar_intersection_pixels': int(collar.sum()),
                         'rest_clothing_pixels': int(rest.sum()),
                         'trace_outside_clothing_pixels': int((trace-collar).sum()),
                         'trace_retained_fraction': float(collar.sum()/trace.sum()),
                         'collar_fraction_of_clothing': float(collar.sum()/clothing.sum())}


def partition_loss(generated, target, collar, rest):
    tab_l1 = masked_l1(generated, target, collar)
    rest_l1 = masked_l1(generated, target, rest)
    return .5 * tab_l1 + .5 * rest_l1, tab_l1, rest_l1


def load_pairs(manifest_path, source, source_max_error, w_atol):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['version'] != 2 or not manifest['pairs']:
        raise ValueError('Expected nonempty version-2 collar-region manifest')
    if manifest.get('production_approved', False):
        raise ValueError('This prototype refuses production-approved artifacts')
    if manifest.get('requires_trainer_feature') != 'equal_area_collar_partition_v1':
        raise ValueError('Manifest must require equal_area_collar_partition_v1')
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
        collar = rest = None
        region_info = None
        if split == 'train':
            collar, rest, region_info = partition_masks(mask, item.get('collar_polygons'), source.img_resolution)
        elif item.get('collar_polygons') is not None:
            # Optional validation annotation is measured, never used in update.
            collar, rest, region_info = partition_masks(mask, item['collar_polygons'], source.img_resolution)
        records.append({'collar': collar, 'rest': rest, 'id': pair_id, 'split': split, 'z': z, 'w': w,
                        'original': original.detach(), 'target': target, 'mask': mask})
        provenance.append({'collar_polygons': item.get('collar_polygons'), 'partition': region_info, 'id': pair_id, 'split': split, 'native_target_size': native_size,
                           'source_uint8_max_error': source_error, 'w_max_error': w_error,
                           'clothing_fraction': float(mask.mean()),
                           'clothing_polygons': item['clothing_polygons'],
                           'files': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}})
    if not any(record['split'] == 'train' for record in records):
        raise ValueError('No training pairs')
    return records, provenance

def update(student, source, optimizer, pairs, options, sampling, preservation, device):
    if not pairs or any(pair['split'] != 'train' for pair in pairs):
        raise ValueError('Only training pairs may be passed to update')
    indices = torch.randint(len(pairs), (options['batch'],), generator=sampling).tolist()
    selected = [pairs[index] for index in indices]
    values = {name: torch.cat([pair[name] for pair in selected]).to(device)
              for name in ['w', 'original', 'target', 'mask', 'collar', 'rest']}
    optimizer.zero_grad(set_to_none=True)
    generated = grayscale(student.synthesis(values['w'], noise_mode='const', force_fp32=True))
    clothing, collar, rest = partition_loss(generated, grayscale(values['target']), values['collar'], values['rest'])
    protected = masked_l1(generated, grayscale(values['original']), 1 - values['mask'])
    paired = options['clothing_weight'] * clothing + options['protected_weight'] * protected
    if not torch.isfinite(paired):
        raise FloatingPointError('Nonfinite paired loss')
    paired.backward()
    metrics = {'pair_ids': [pair['id'] for pair in selected], 'clothing_l1': float(clothing.detach()),
               'collar_l1': float(collar.detach()), 'rest_clothing_l1': float(rest.detach()),
               'protected_l1': float(protected.detach()), 'paired_loss': float(paired.detach()),
               'fresh_preservation_l1': 0.0}
    del generated, clothing, collar, rest, protected, paired, values
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
                  'trainer_sha256': sha256(__file__), 'source_files': source_hashes(), 'pairs': pair_provenance,
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
                  'source_files': source_hashes(),
                  'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__}}
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['format'] != 'paired-regions-research-v1' or checkpoint['production_approved']:
            raise ValueError('Only paired-regions research checkpoints can resume; broad-mask runs require a fresh initial run')
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
    configuration = {'method': 'paired grayscale equal-area collar/rest L1 baseline', 'production_approved': False,
                     'resolution': source.img_resolution, 'train_pairs': len(train_pairs),
                     'validation_pairs': len(pairs) - len(train_pairs), 'total_steps': args.total_steps,
                     'resume': str(args.resume) if args.resume else None, 'options': options,
                     'trainable_parameters': trainable, 'provenance': provenance,
                     'freeze': 'mapping, all buffers/noise_strength, synthesis blocks <=32',
                     'loss_normalization': 'clothing_weight*(0.5*collar_area_L1+0.5*rest_area_L1); protected/fresh unchanged; per-image then batch mean',
                     'collar_mask': 'raw trace intersect unchanged clothing; reject empty/full intersection',
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
                atomic_save({'format': 'paired-regions-research-v1', 'production_approved': False,
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
