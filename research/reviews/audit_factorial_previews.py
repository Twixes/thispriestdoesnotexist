"""Audit saved factorial previews without importing a model or changing a run.

Only complete evaluation.json snapshots are read. Numeric comparisons never
constitute visual approval. Output is a new directory outside the live run.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_snapshot(directory, step):
    path = directory / 'evaluation.json'
    data = json.loads(path.read_text())
    assert data['fork_step'] == step and data['parent_step'] == 600
    assert data['count'] == len(data['rows']) == 14
    assert data['source_compositing'] is False and data['rng_restored'] is True
    assert len({row['id'] for row in data['rows']}) == 14
    assert sum(row['split'] == 'train' for row in data['rows']) == 6
    for row in data['rows']:
        assert set(row['outputs']) == {'rgb', 'gray'}
        for key, output in row['outputs'].items():
            file = directory / output['path']
            assert file.parent == directory and digest(file) == output['sha256']
            with Image.open(file) as image:
                assert image.size == (1024, 1024)
                assert image.mode == output['mode'] == ('L' if key == 'gray' else 'RGB')
        assert all(math.isfinite(v) for v in row['metrics_minus1_to1'].values())
    return data, digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=[100, 200, 300], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a new output directory'
    assert args.output.resolve() != args.run.resolve()
    snapshots = {}
    provenance = {}
    for arm in 'ABCD':
        directory = args.run / arm / f'preview-{args.step:03}'
        if not (directory / 'evaluation.json').exists():
            continue
        snapshot, sha = read_snapshot(directory, args.step)
        snapshots[arm] = snapshot
        provenance[arm] = {'evaluation_sha256': sha, 'state_sha256': snapshot['state_sha256']}
    assert snapshots, 'No complete snapshots at this milestone'
    reference_ids = [(r['id'], r['split']) for r in next(iter(snapshots.values()))['rows']]
    for snapshot in snapshots.values():
        assert [(r['id'], r['split']) for r in snapshot['rows']] == reference_ids
    means = {}
    for arm, snapshot in snapshots.items():
        rows = [r['metrics_minus1_to1'] for r in snapshot['rows'] if r['split'] == 'train']
        means[arm] = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    deltas = {}
    for treatment, control in [('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'C')]:
        if treatment in means and control in means:
            deltas[f'{treatment}_minus_{control}'] = {key: means[treatment][key] - means[control][key] for key in means[control]}
    args.output.mkdir(parents=True)
    audit = {'milestone': args.step, 'available_arms': list(snapshots), 'all_four_snapshots_available': len(snapshots) == 4,
             'provenance': provenance, 'verified_native_png_count': len(snapshots) * 28,
             'train_means_minus1_to1': means, 'factor_differences': deltas,
             'per_image': {arm: [{'id': r['id'], 'split': r['split'], 'metrics': r['metrics_minus1_to1']} for r in s['rows']] for arm, s in snapshots.items()},
             'quality_approved': False, 'production_approved': False,
             'limitations': 'Only saved PNG hashes, formats and reported metrics audited. No model or checkpoint validation here. Numeric improvement does not establish collar structure, facial integrity, attractiveness, or unseen generalization. A complete experiment additionally requires successful terminal processes, matched sampling, source30 evaluation, and native visual review.'}
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    sheet = Image.new('RGB', (256 * len(snapshots), 280 * len(reference_ids)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for column, (arm, snapshot) in enumerate(snapshots.items()):
        directory = args.run / arm / f'preview-{args.step:03}'
        for index, row in enumerate(snapshot['rows']):
            x, y = column * 256, index * 280
            with Image.open(directory / row['outputs']['gray']['path']) as image:
                sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (x, y + 24))
            draw.text((x + 3, y + 5), f"{arm}{args.step} {row['id']} {row['split']}", fill='white')
    sheet.save(args.output / 'comparison.png')
    print(json.dumps({'arms': list(snapshots), 'means': means, 'differences': deltas}, indent=2))


if __name__ == '__main__':
    main()
