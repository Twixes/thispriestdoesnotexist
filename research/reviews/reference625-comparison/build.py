"""PNG-only native-pixel matched500/625 diagnostic sheets; never loads a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / 'research/runs/reference256-paper-b64/unseen32-000500'
FUTURE = ROOT / 'research/runs/reference256-paper-b64-resumed500-to625/matched32-000625'
VARIANTS = ('raw-psi1', 'ema-psi1', 'ema-psi07')
SEEDS = list(range(202609210000, 202609210032))
BASELINE_PINS = {
    'evaluation.json': '9fd2d1fdfdaa29cb9e691505385e252da75be884c63ea1d50cdf785001986eb5',
    'supervisor-result.json': 'e24b77482207f50ffab0d4fb13c28a054ac6f1c961270f143469e8fb426ae6ec',
    'launch.json': '89c7996484434f664750875ac993aa52f58f262b8d6266c378e7d174c70ff9e4',
    'latents.npz': '792c72cb2c3b59d0364a3af40c472fa9ae63f374f5b4039888506fb8847983c5',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def within(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), 'Input path escapes its evaluation directory')
    return path


def validate(directory, step):
    directory = directory.resolve()
    if step == 500:
        for name, expected in BASELINE_PINS.items():
            require(sha(directory / name) == expected, f'Pinned baseline changed: {name}')
    evaluation = read(directory / 'evaluation.json')
    supervisor = read(directory / 'supervisor-result.json')
    launch = read(directory / 'launch.json')
    require(evaluation['complete'] is True and supervisor['complete'] is True
            and supervisor['worker_exit_code'] == 0 and supervisor['error'] is None,
            f'Step{step} evaluation and supervisor must both succeed')
    require(evaluation['checkpoint_step'] == step and evaluation['images_seen_in_training'] == step * 64,
            'Wrong checkpoint counters')
    require(evaluation['checkpoint_sha256'] == launch['checkpoint_sha256'], 'Checkpoint launch/evaluation hash differs')
    require(evaluation['source_sha256'] == launch['script_sha256']
            and evaluation['pins_sha256'] == launch['pins_sha256'], 'Evaluation launch provenance differs')
    require(evaluation['input_hashes_before_and_after_equal'] is True, 'Evaluator inputs were not stable')
    require(evaluation['seeds'] == SEEDS, 'Different seed order')
    hashes = {name: sha(directory / name) for name in ['evaluation.json', 'supervisor-result.json', 'launch.json']}
    latent = within(directory, evaluation['latents']['path'])
    require(sha(latent) == evaluation['latents']['sha256'] == BASELINE_PINS['latents.npz'],
            'Matched comparison requires byte-identical saved500 NPZ')
    hashes[str(latent.relative_to(directory))] = sha(latent)
    with np.load(latent, allow_pickle=False) as saved:
        require(set(saved.files) == {'seeds', 'z', 'trainer_fixed_z'}, 'Unexpected latent archive')
        require(saved['seeds'].dtype == np.uint64 and np.array_equal(saved['seeds'], np.asarray(SEEDS, dtype=np.uint64)),
                'NPZ seeds differ from evaluation')
        for name, shape in [('z', (32,512)), ('trainer_fixed_z', (16,512))]:
            require(saved[name].shape == shape and saved[name].dtype == np.float32
                    and bool(np.isfinite(saved[name]).all()), 'Invalid saved latent array')
    expected = {(variant, i) for variant in VARIANTS for i in range(32)}
    images, entries, paths = {}, {}, set()
    require(len(evaluation['entries']) == 96, 'Need all96 per-run PNG entries')
    for entry in evaluation['entries']:
        key = (entry['variant'], entry['index'])
        require(key in expected and key not in entries, 'Unexpected/duplicate variant-index entry')
        require(entry['seed'] == SEEDS[entry['index']], 'Entry seed differs')
        path = within(directory, entry['path'])
        require(path.suffix.lower() == '.png' and path not in paths, 'Repeated or non-PNG image path')
        require(sha(path) == entry['sha256'], f'Image hash differs: {path.name}')
        with Image.open(path) as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256,256),
                    f'Expected native256 RGB PNG: {path.name}')
            image.load()
            images[key] = image.copy()
        paths.add(path); entries[key] = entry
        hashes[str(path.relative_to(directory))] = entry['sha256']
    require(set(entries) == expected, 'Missing variant/index')
    return {'directory': directory, 'evaluation': evaluation, 'hashes': hashes,
            'images': images, 'entries': entries}


def summary(run):
    return {'directory': str(run['directory'].relative_to(ROOT)),
            'checkpoint_step': run['evaluation']['checkpoint_step'], 'images_checked': len(run['images']),
            'checkpoint_sha256': run['evaluation']['checkpoint_sha256'],
            'native_dimensions': [256,256], 'mode': 'RGB', 'seeds': SEEDS,
            'latents_sha256': run['evaluation']['latents']['sha256'], 'inputs_sha256': run['hashes']}


def unchanged(run):
    for name, expected in run['hashes'].items():
        require(sha(run['directory'] / name) == expected, f'Input changed during comparison: {name}')


def build(before, after, output):
    require(before['evaluation']['seeds'] == after['evaluation']['seeds'], 'Seeds differ')
    require(before['evaluation']['latents'] == after['evaluation']['latents'], 'Latent manifests differ')
    require(before['evaluation']['checkpoint_sha256'] != after['evaluation']['checkpoint_sha256'],
            '500 and625 must not be the same checkpoint')
    require(output.is_relative_to(ROOT / 'research/reviews') and not output.exists(),
            'Use a new diagnostic directory below research/reviews; no overwrite')
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=14)
    pages = []
    for variant in VARIANTS:
        for page_index in range(4):
            canvas = Image.new('RGB', (1040,1200), (18,18,18))
            draw = ImageDraw.Draw(canvas)
            draw.text((4,8), f'{variant} | step500 LEFT / step625 RIGHT | page{page_index+1}/4 | native256 pixels',
                      fill=(240,240,240), font=font)
            tiles = []
            for slot in range(8):
                index = page_index * 8 + slot
                x = 4 + (slot % 2) * 520
                label_y = 40 + (slot // 2) * 288
                y = label_y + 24
                draw.text((x,label_y+2), f'index{index:02} | seed{SEEDS[index]} | {variant}',
                          fill=(220,220,220), font=font)
                for side, run, dx in [('500', before, 0), ('625', after, 256)]:
                    original = run['images'][(variant,index)]
                    box = (x+dx, y, x+dx+256, y+256)
                    canvas.paste(original, (x+dx,y))
                    require(canvas.crop(box).tobytes() == original.tobytes(), 'Paste changed native pixels')
                    tiles.append({'step': int(side), 'index': index, 'seed': SEEDS[index],
                        'variant': variant, 'box_xyxy': list(box),
                        'input': str((run['directory']/run['entries'][(variant,index)]['path']).relative_to(ROOT)),
                        'input_sha256': run['entries'][(variant,index)]['sha256']})
            file = output / f'{variant}-page{page_index+1}.png'
            with file.open('xb') as stream: canvas.save(stream, format='PNG')
            # Verify decoded output crops, not just the pre-encoding canvas.
            with Image.open(file) as decoded:
                require(decoded.mode == 'RGB' and decoded.size == (1040,1200), 'Output dimensions changed')
                for tile in tiles:
                    run = before if tile['step'] == 500 else after
                    require(decoded.crop(tile['box_xyxy']).tobytes() == run['images'][(variant,tile['index'])].tobytes(),
                            'Encoded page altered native pixels')
            pages.append({'path': file.name, 'sha256': sha(file), 'dimensions': [1040,1200], 'tiles': tiles})
    unchanged(before); unchanged(after)
    result = {'complete': True, 'artifact_type': 'diagnostic matched PNG montage; NOT model output',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(Path(__file__)),
        'before': summary(before), 'after': summary(after), 'page_count': 12, 'pages': pages,
        'native_tiles_verified_exact_after_png_decode': 192, 'inputs_unchanged': True,
        'resize_restoration_filtering_or_model_inference': False, 'production_approved': False}
    with (output / 'manifest.json').open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    return {'complete': True, 'output': str(output), 'pages': 12, 'exact_native_tiles': 192}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-baseline', action='store_true')
    mode.add_argument('--build', action='store_true')
    parser.add_argument('--evaluation625', type=Path, default=FUTURE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    baseline = validate(BASELINE, 500)
    if args.validate_baseline:
        require(args.output is None, 'Baseline validation writes no page output')
        unchanged(baseline)
        result = {'complete': True, 'mode': 'real500 baseline validation only', 'baseline': summary(baseline),
                  'script_sha256': sha(Path(__file__)), 'future625_checked': False, 'pages_created': 0,
                  'torch_imported': 'torch' in sys.modules, 'model_loaded': False}
    else:
        require(args.output is not None, '--build requires a new --output')
        future = validate(args.evaluation625, 625)
        result = build(baseline, future, args.output.resolve())
    require('torch' not in sys.modules, 'This tool must not import torch')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
