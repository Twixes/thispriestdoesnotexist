"""CPU-only raw discriminator diagnostics; never a probability or quality metric."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/vendor/stylegan2-ada-pytorch'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summary(values):
    if not values:
        return {'count': 0}
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values),
            'population_stddev': statistics.pstdev(values),
            'positive_fraction': sum(x > 0 for x in values) / len(values),
            'zero_fraction': sum(x == 0 for x in values) / len(values),
            'sign_mean': sum((x > 0) - (x < 0) for x in values) / len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'research/runs/aligned110-frozen4/resume.pt')
    parser.add_argument('--training', type=Path, default=ROOT / 'research/alignment/collar-only/eyes42')
    parser.add_argument('--heldout', type=Path, default=ROOT / 'research/data/validation12/aligned256')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'research/data/validation12/alignment-manifest.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    pressure = subprocess.check_output(['memory_pressure'], text=True)
    free = re.search(r'System-wide memory free percentage: (\d+)%', pressure)
    if not free or int(free.group(1)) < 20:
        raise SystemExit('Defer: memory_pressure must report at least 20% free')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'memory-before.txt').write_text(pressure)
    train = sorted(args.training.glob('*.png'))
    held = sorted(args.heldout.glob('*.png'))
    if len(train) != 110 or len(held) != 12 or not args.manifest.is_file():
        raise SystemExit('Require complete training110, heldout12, and held-out manifest before evaluation')
    training_manifest_path = ROOT / 'research/alignment/collar-only/manifest.json'
    training_manifest = json.loads(training_manifest_path.read_text())
    held_manifest = json.loads(args.manifest.read_text())
    if held_manifest['accepted_training_manifest_sha256'] != digest(training_manifest_path):
        raise SystemExit('Held-out alignment does not refer to current accepted training transforms')
    training_entries = {entry['id']: entry for entry in training_manifest['entries']}
    held_entries = {entry['id']: entry for entry in held_manifest['entries']}
    for path in train:
        if digest(path) != training_entries[path.stem]['eyes42']['training_sha256']:
            raise SystemExit('Training image does not match alignment manifest')
    for path in held:
        entry = held_entries[path.stem]
        if digest(path) != entry['output_sha256'] or digest(ROOT / entry['source']) != entry['source_sha256']:
            raise SystemExit('Held-out image or its source does not match provenance')
    config_path = args.checkpoint.parent / 'config.json'
    config = json.loads(config_path.read_text())
    dataset_hash = hashlib.sha256(''.join(digest(p) for p in train).encode()).hexdigest()
    if dataset_hash != config['dataset_sha256'] or config['dataset_count'] != 110:
        raise SystemExit('Training data differ from recorded checkpoint configuration')
    overlap = {digest(p) for p in train} & {digest(p) for p in held}
    if overlap:
        raise SystemExit('Held-out aligned image duplicates a training image')
    base = ROOT / 'research/models/ffhq256.pkl'
    source_path = ROOT / 'research/sources.json'
    sources = json.loads(source_path.read_text())
    base_hash = digest(base)
    if base_hash != sources['pretrained']['sha256'] or base_hash != config['base_sha256']:
        raise SystemExit('Official base checkpoint hash mismatch')
    checkpoint_hash = digest(args.checkpoint)

    import numpy as np
    from PIL import Image
    import torch
    import legacy

    torch.set_num_threads(args.threads)
    torch.manual_seed(20260921)
    started = time.monotonic()
    # This is the hash-verified NVIDIA source and the task's own trusted local resume.
    # mmap avoids eagerly allocating the unused optimizer/generator tensors.
    # The trainer's resume includes a NumPy scalar that weights_only rejects.
    # Full deserialization is permitted only for this trusted, locally produced file.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['step'] != 1000 or checkpoint['freeze_d_layers'] != 4:
        raise SystemExit('Expected the paused step-1000 FreezeD-4 control checkpoint')
    with base.open('rb') as stream:
        networks = legacy.load_network_pkl(stream)
    source_d = networks['D']
    kwargs = copy.deepcopy(source_d.init_kwargs)
    kwargs['block_kwargs'] = dict(kwargs.get('block_kwargs', {}), freeze_layers=4)
    discriminator = type(source_d)(*source_d.init_args, **kwargs).cpu().eval().requires_grad_(False)
    discriminator.load_state_dict(checkpoint['D'], strict=True)
    checkpoint_step = checkpoint['step']
    checkpoint_aug_p = checkpoint['aug_p']
    del checkpoint, networks, source_d
    gc.collect()
    assert all(p.device.type == 'cpu' and not p.requires_grad for p in discriminator.parameters())

    def tensor(path):
        with Image.open(path) as im:
            if im.size != (256, 256) or im.mode != 'RGB':
                raise ValueError(f'Unexpected input format: {path}')
            array = np.array(im)
        return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1

    # These identical common anchors define the optional batch-four context.
    # Exclude those three targets from its training summary to avoid self-duplication.
    anchors = train[:3]
    anchor_tensor = torch.stack([tensor(p) for p in anchors])
    records = []
    with torch.inference_mode():
        for split, paths in [('training', train), ('heldout', held)]:
            for index, path in enumerate(paths):
                target = tensor(path).unsqueeze(0)
                singleton = float(discriminator(target, None, force_fp32=True).item())
                if not math.isfinite(singleton):
                    raise ValueError('Nonfinite discriminator output')
                padding = (training_entries[path.stem]['eyes42']['reflected_padding_fraction']
                           if split == 'training' else held_entries[path.stem]['reflected_padding_fraction'])
                row = {'split': split, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                       'batch_one_raw_logit': singleton, 'anchor_target': path in anchors,
                       'reflected_padding_fraction': padding,
                       'padding_stratum': 'above_5_percent' if padding > .05 else 'at_most_5_percent'}
                if path not in anchors:
                    batch = torch.cat([anchor_tensor, target], dim=0)
                    row['common_three_anchors_raw_logit'] = float(
                        discriminator(batch, None, force_fp32=True)[-1].item())
                    if not math.isfinite(row['common_three_anchors_raw_logit']):
                        raise ValueError('Nonfinite anchored discriminator output')
                records.append(row)
                print(json.dumps(row), flush=True)

    if digest(args.checkpoint) != checkpoint_hash:
        raise SystemExit('Checkpoint changed during evaluation; discard this incomplete result')
    protocols = {}
    for key in ('batch_one_raw_logit', 'common_three_anchors_raw_logit'):
        groups = {split: summary([r[key] for r in records if r['split'] == split and key in r])
                  for split in ('training', 'heldout')}
        groups['training_minus_heldout_mean'] = groups['training']['mean'] - groups['heldout']['mean']
        groups['padding_strata'] = {stratum: {
            split: summary([r[key] for r in records if r['split'] == split and key in r
                            and r['padding_stratum'] == stratum])
            for split in ('training', 'heldout')}
            for stratum in ('at_most_5_percent', 'above_5_percent')}
        protocols[key] = groups
    provenance_paths = [config_path, source_path, args.manifest, Path(__file__),
        ROOT / 'research/data/validation12/provenance.json',
        ROOT / 'research/data/validation12/prompts.json',
        ROOT / 'research/data/validation12/alignment-review.json',
        ROOT / 'research/data/validation12/alignment-review.md',
        ROOT / 'research/alignment/collar-only/manifest.json',
        ROOT / 'research/alignment/collar-only/review.json',
        ROOT / 'research/data/synthetic/manifest.json',
        ROOT / 'research/vendor/stylegan2-ada-pytorch/training/networks.py']
    report = {
        'checkpoint': str(args.checkpoint.relative_to(ROOT)), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_step': checkpoint_step, 'checkpoint_augmentation_probability': checkpoint_aug_p,
        'base_sha256': base_hash, 'base_source': sources['pretrained'],
        'training_dataset_sha256': dataset_hash,
        'reflected_padding_fraction_means': {
            split: statistics.mean(r['reflected_padding_fraction'] for r in records if r['split'] == split)
            for split in ('training', 'heldout')},
        'provenance_sha256': {str(p.relative_to(ROOT)): digest(p) for p in provenance_paths if p.is_file()},
        'protocol': {'device': 'cpu', 'threads': args.threads, 'dtype': 'float32',
            'mode': 'eval/inference_mode', 'augmentation': 'none',
            'normalization': 'unchanged 256x256 RGB pixels / 127.5 - 1',
            'batch_one_minibatch_std': 'variance zero; epsilon floor sqrt(1e-8)=0.0001',
            'common_anchor_paths': [str(p.relative_to(ROOT)) for p in anchors],
            'common_anchor_note': 'Same 3 training anchors + target at position 4; anchors excluded as scored targets.'},
        'summaries': protocols, 'per_image': records,
        'limits': ['Raw logits are not calibrated probabilities, aesthetic judgments, or production quality scores.',
            'No-augmentation eval and fixed minibatch context differ from training logs; numerical scales are not directly comparable.',
            'Twelve synthetic held-out portraits are a small nonrandom set, with possible generation/prompt/framing/domain mismatch.',
            'A train-heldout gap alone cannot establish memorization or diagnose the cause of poor generator quality.',
            'Held-out reflection/padding differs from training; strata are descriptive small subsets, not an adjusted causal estimate.',
            'Common anchors come from training and define only one chosen context; training summary excludes those 3 anchors.',
            'No optimizer steps, gradient computation, MPS use, or model approval occurred.'],
        'elapsed_seconds': time.monotonic() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'platform': platform.platform(), 'torch_version': torch.__version__,
    }
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(protocols), flush=True)


if __name__ == '__main__':
    main()
