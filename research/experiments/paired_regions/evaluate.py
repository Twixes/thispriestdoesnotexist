"""Evaluate completed collar-region research checkpoints, CPU only; no training changes."""
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.experiments.paired_edit import evaluate as shared

sha256 = shared.sha256
region_errors = shared.region_errors
latent_digest = shared.latent_digest
arguments = shared.arguments


def current_source_hashes(root=ROOT):
    """Mirror regions trainer provenance without importing torch or executing it."""
    root = Path(root)
    vendor = root / 'research/vendor/stylegan2-ada-pytorch'
    files = {
        'regions_trainer': sha256(root / 'research/experiments/paired_regions/trainer.py'),
        'paired_edit_helpers': sha256(root / 'research/experiments/paired_edit/trainer.py'),
    }
    for directory in ['training', 'torch_utils', 'dnnlib']:
        for path in sorted((vendor / directory).rglob('*')):
            if path.is_file() and path.suffix in ['.py', '.cpp', '.cu', '.h', '.hpp']:
                files[str(path.relative_to(vendor))] = sha256(path)
    return files


def verify_code_provenance(recorded, current):
    if recorded != current:
        missing = sorted(set(recorded) - set(current))
        added = sorted(set(current) - set(recorded))
        changed = sorted(key for key in set(recorded) & set(current) if recorded[key] != current[key])
        raise ValueError(f'Regions code provenance mismatch: missing={missing}, added={added}, changed={changed}')


def verify_metadata(metadata, checkpoint, configuration, source_manifest, weights_sha, metadata_sha):
    if checkpoint['format'] != 'paired-regions-research-v1' or checkpoint['production_approved'] is not False:
        raise ValueError('Expected an explicitly unapproved paired-regions research checkpoint; broad-mask checkpoints are incompatible')
    if metadata['schema_version'] != 1 or metadata['truncation_psi'] != 1:
        raise ValueError('Expected schema-1 psi-1 source bundle')
    if metadata['weights_sha256'] != weights_sha:
        raise ValueError('Source bundle checksum mismatch')
    if checkpoint['init_kwargs'] != metadata['init_kwargs']:
        raise ValueError('Checkpoint/source architecture mismatch')
    provenance = checkpoint['provenance']
    if provenance['source_weights_sha256'] != weights_sha or provenance['source_metadata_sha256'] != metadata_sha:
        raise ValueError('Checkpoint source provenance mismatch')
    if configuration['provenance'] != provenance or configuration['options'] != checkpoint['options']:
        raise ValueError('Run configuration/checkpoint provenance mismatch')
    if checkpoint['step'] != configuration['total_steps']:
        raise ValueError('Evaluation requires completed configured training, not an active intermediate checkpoint')
    if source_manifest['model']['model_sha256'] != weights_sha:
        raise ValueError('Evaluation sources were made by a different source model')


def main():
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty evaluation output directory')
    memory = subprocess.run(['memory_pressure'], text=True, capture_output=True, check=True).stdout
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', memory).group(1))
    if free < 25:
        raise RuntimeError(f'Defer: {free}% memory free is below 25%')
    # Heavy imports and checkpoint loading only occur after the explicit invocation/preflight.
    import torch
    import PIL
    from safetensors.torch import load_file
    from research.experiments.paired_regions.trainer import Generator, freeze_student, frozen_state, state_digest, grayscale, quantized

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    # Hash and deserialize the same open inode even if a path were atomically replaced.
    with args.checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        loaded = torch.load(stream, map_location='cpu', weights_only=True)
    student_state = loaded.pop('student')
    keys = ['format', 'production_approved', 'step', 'init_kwargs', 'provenance', 'options', 'source_digest', 'frozen_digest']
    checkpoint = {key: loaded[key] for key in keys}
    del loaded  # Release optimizer/RNG checkpoint data before allocating generator instances.
    gc.collect()
    if checkpoint['step'] != args.expected_step:
        raise ValueError('Checkpoint is not the explicitly requested completed step')
    configuration_path = args.checkpoint.parent / 'config.json'
    configuration = json.loads(configuration_path.read_text())
    metadata_path = args.source_bundle / 'model.json'
    metadata = json.loads(metadata_path.read_text())
    samples = json.loads(args.sources.read_text())
    weights = args.source_bundle / 'generator.safetensors'
    weights_sha, metadata_sha = sha256(weights), sha256(metadata_path)
    verify_metadata(metadata, checkpoint, configuration, samples, weights_sha, metadata_sha)
    verify_code_provenance(checkpoint['provenance']['source_files'], current_source_hashes())
    versions = {'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__}
    if checkpoint['provenance']['versions'] != versions:
        raise ValueError('Exact source rendering requires the recorded training library versions')

    source = Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
    source.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    if state_digest(source.state_dict()) != checkpoint['source_digest']:
        raise ValueError('Source state digest mismatch')
    freeze_student(source)
    if state_digest(frozen_state(source)) != checkpoint['frozen_digest']:
        raise ValueError('Frozen source state mismatch')
    source.requires_grad_(False)
    student = Generator(**checkpoint['init_kwargs']).cpu().eval()
    student.load_state_dict(student_state, strict=True)
    del student_state
    freeze_student(student)
    if state_digest(frozen_state(student)) != checkpoint['frozen_digest']:
        raise ValueError('Student changed a protected parameter or buffer')
    student.requires_grad_(False)
    for model in [source, student]:
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('Nonfinite generator state')
    student_digest = state_digest(student.state_dict())
    load_seconds = time.monotonic() - started

    training_latents = set()
    for pair in checkpoint['provenance']['pairs']:
        if pair['split'] != 'train':
            continue
        item = pair['files']['latent_path']
        path = Path(item['path'])
        if sha256(path) != item['sha256']:
            raise ValueError('Recorded training latent artifact changed')
        with np.load(path, allow_pickle=False) as latent:
            training_latents.add(latent_digest(latent['z']))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'host-memory-before.txt').write_text(memory)
    results, excluded, thumbnails, seen = [], [], [], set()
    with torch.inference_mode():
        for row in samples['entries']:  # Deliberately excludes the separate original training-seed reproduction.
            identity = row['id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', identity) or identity in seen:
                raise ValueError('Unsafe or duplicate sample id')
            seen.add(identity)
            latent_path = (args.sources.resolve().parent / row['latent_path']).resolve()
            source_path = (args.sources.resolve().parent / row['source_path']).resolve()
            if sha256(latent_path) != row['latents_sha256'] or sha256(source_path) != row['image_sha256']:
                raise ValueError(f'{identity}: sample file checksum mismatch')
            with np.load(latent_path, allow_pickle=False) as latent:
                z, w = latent['z'].copy(), latent['w'].copy()
            if z.dtype != np.float32 or w.dtype != np.float32 or z.shape != (1, source.z_dim) or w.shape != (1, source.num_ws, source.w_dim):
                raise ValueError(f'{identity}: latent schema mismatch')
            if not np.isfinite(z).all() or not np.isfinite(w).all():
                raise ValueError(f'{identity}: nonfinite latent')
            if latent_digest(z) != row['z_raw_sha256'] or latent_digest(w) != row['ws_raw_sha256']:
                raise ValueError(f'{identity}: latent tensor checksum mismatch')
            if latent_digest(z) in training_latents:
                excluded.append({'id': identity, 'reason': 'exact training latent'})
                continue
            z, w = torch.from_numpy(z), torch.from_numpy(w)
            mapped = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
            if not torch.equal(mapped, w):
                raise ValueError(f'{identity}: mapped W is not exact')
            row_started = time.monotonic()
            original = source.synthesis(w, noise_mode='const', force_fp32=True)
            with Image.open(source_path) as image:
                expected = np.asarray(image.convert('RGB'))
            if not np.array_equal(quantized(original), expected):
                raise ValueError(f'{identity}: source PNG does not match exact CPU forward')
            source_seconds = time.monotonic() - row_started
            student_started = time.monotonic()
            predicted = student.synthesis(w, noise_mode='const', force_fp32=True)
            student_seconds = time.monotonic() - student_started
            if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(predicted).all()):
                raise ValueError(f'{identity}: nonfinite inference')
            source_gray = grayscale(original)
            student_gray = grayscale(predicted)
            errors = region_errors((source_gray[0, 0].numpy() + 1) / 2, (student_gray[0, 0].numpy() + 1) / 2)
            paths = {}
            for label, pixels in [('source-gray', quantized(source_gray)[:, :, 0]),
                                  ('student-gray', quantized(student_gray)[:, :, 0]),
                                  ('student-rgb', quantized(predicted))]:
                path = args.output / f'{identity}-{label}.png'
                image = Image.fromarray(pixels)
                image.save(path)
                paths[label] = {'path': path.name, 'sha256': sha256(path), 'dimensions': list(image.size)}
                if label != 'student-rgb':
                    thumbnails.append(image.convert('RGB').resize((160, 160), Image.Resampling.LANCZOS))
            results.append({'id': identity, 'seed_hex': row['seed_hex'], 'source_path': str(source_path),
                            'source_png_sha256': row['image_sha256'], 'latent_path': str(latent_path),
                            'latent_npz_sha256': row['latents_sha256'], 'z_raw_sha256': row['z_raw_sha256'],
                            'w_raw_sha256': row['ws_raw_sha256'], 'source_quantized_exact': True,
                            'source_w_exact': True, 'source_seconds': source_seconds, 'student_seconds': student_seconds,
                            'metrics': errors, 'outputs': paths, 'production_approved': False})
            print(json.dumps({'id': identity, **errors}), flush=True)
            del original, predicted, source_gray, student_gray
    if not results:
        raise ValueError('No disjoint evaluation samples remain')
    if state_digest(source.state_dict()) != checkpoint['source_digest'] or state_digest(student.state_dict()) != student_digest:
        raise ValueError('Evaluation mutated a model state')
    contact = Image.new('RGB', (4 * 320, ((len(results) + 3) // 4) * 184 + 24), (24, 24, 24))
    draw = ImageDraw.Draw(contact)
    draw.text((4, 4), 'DIAGNOSTIC ONLY: each pair source grayscale | student grayscale; no output compositing', fill='white')
    for index, row in enumerate(results):
        x, y = (index % 4) * 320, (index // 4) * 184 + 24
        contact.paste(thumbnails[index * 2], (x, y)); contact.paste(thumbnails[index * 2 + 1], (x + 160, y))
        draw.text((x + 4, y + 163), row['id'] + ' source | student', fill='white')
    contact.save(args.output / 'comparison.png')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    report = {'purpose': 'Research-only collar-region student disjoint-latent comparison', 'production_approved': False,
              'checkpoint_path': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
              'checkpoint_step': checkpoint['step'], 'student_state_sha256': student_digest,
              'source_weights_sha256': weights_sha, 'source_metadata_sha256': metadata_sha,
              'source_state_sha256': checkpoint['source_digest'], 'frozen_state_sha256': checkpoint['frozen_digest'],
              'source_manifest_path': str(args.sources.resolve()), 'source_manifest_sha256': sha256(args.sources),
              'training_provenance': checkpoint['provenance'], 'init_kwargs': metadata['init_kwargs'],
              'runtime': {'platform': platform.platform(), 'python': sys.version, **versions, 'cpu_threads': 1, 'interop_threads': 1},
              'load_and_verify_seconds': load_seconds, 'total_seconds': time.monotonic() - started, 'peak_rss_bytes': rss,
              'memory_preflight_free_percent': free, 'count': len(results), 'excluded_training_latents': excluded,
              'original_training_seed_reproduction_excluded': 'Only source manifest entries are enumerated, never its reproduction field',
              'metrics_caveat': 'Unclipped luminance MAE on fixed top75/bottom25 regions; these are not detected face/clothing masks, identity scores, collar recognition, or perceptual quality measures. Background and hair contribute. Values can exceed1 if raw generator luminance is out of range.',
              'rendering': 'Actual source and student synthesis forwards; same exact source-mapped W verified against frozen student mapping; const noise and FP32. RGB student saved unchanged, grayscale is fixed display conversion, no output compositing.',
              'script_sha256': sha256(__file__), 'shared_evaluation_helpers_sha256': sha256(shared.__file__), 'contact_sha256': sha256(args.output / 'comparison.png'),
              'mean_metrics': {key: float(np.mean([row['metrics'][key] for row in results])) for key in results[0]['metrics']},
              'entries': results}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['count', 'total_seconds', 'peak_rss_bytes', 'mean_metrics']}, indent=2))


if __name__ == '__main__':
    main()
