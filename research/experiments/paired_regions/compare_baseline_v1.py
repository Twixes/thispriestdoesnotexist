"""Compare saved PNGs from the controlled single-pair experiment; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SIZE = 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gray(path):
    with Image.open(path) as image:
        if image.width != image.height:
            raise ValueError('Square images required; no cropping')
        pixels = np.asarray(image.convert('RGB').resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float32)
    return pixels @ np.array([.299, .587, .114], dtype=np.float32)


def mask(polygons):
    canvas = Image.new('L', (SIZE, SIZE))
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon([tuple(round(value * (SIZE - 1)) for value in point) for point in polygon], fill=255)
    return np.asarray(canvas) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty output directory')
    manifest_path = ROOT / 'research/data/paired4-tabs/single-pair.json'
    manifest = json.loads(manifest_path.read_text())
    pair = manifest['pairs'][0]
    if len(manifest['pairs']) != 1 or pair['id'] != 'calibration-original':
        raise ValueError('This comparison is only the fixed single calibration pair')
    source_path = (manifest_path.parent / pair['source_path']).resolve()
    target_path = (manifest_path.parent / pair['target_path']).resolve()
    paths = {
        'Source (unadapted)': source_path,
        'Broad clothing loss': ROOT / f'research/runs/paired1024-cpu-200/{args.step:06}-pair-000-train.png',
        'Balanced collar/rest loss': ROOT / f'research/runs/paired-regions1024-cpu-200/{args.step:06}-pair-000-train.png',
        'Edited target (reference only)': target_path,
    }
    planes = {name: gray(path) for name, path in paths.items()}
    clothing = mask(pair['clothing_polygons'])
    tab = clothing & mask(pair['collar_polygons'])
    rest = clothing & ~tab
    if not tab.any() or not rest.any() or not (~clothing).any():
        raise ValueError('All comparison regions must be nonempty')
    source, target = planes['Source (unadapted)'], planes['Edited target (reference only)']
    measurements = {}
    for name, pixels in planes.items():
        measurements[name] = {
            'tab_mae_to_target_uint8': float(np.abs(pixels[tab] - target[tab]).mean()),
            'rest_clothing_mae_to_target_uint8': float(np.abs(pixels[rest] - target[rest]).mean()),
            'protected_mae_to_source_uint8': float(np.abs(pixels[~clothing] - source[~clothing]).mean()),
            'tab_mean_uint8': float(pixels[tab].mean()),
            'rest_clothing_mean_uint8': float(pixels[rest].mean()),
            'tab_minus_rest_mean_uint8': float(pixels[tab].mean() - pixels[rest].mean()),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    contact = Image.new('RGB', (4 * 512, 556), '#141414')
    draw = ImageDraw.Draw(contact)
    draw.text((8, 7), f'DIAGNOSTIC: same training seed at step {args.step}; target is imagegen reference, not model output', fill='white')
    for index, (name, pixels) in enumerate(planes.items()):
        draw.text((index * 512 + 8, 29), name, fill='white')
        thumbnail = Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8)).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
        contact.paste(thumbnail, (index * 512, 44))
    contact.save(args.output / 'comparison.png')
    report = {
        'step': args.step, 'production_approved': False, 'model_inference_performed': False,
        'scope': 'One training identity, saved post-update clipped PNGs. Not exact float-tensor loss, identity validation, collar recognition or unseen generalization.',
        'rendering': 'Fixed .299/.587/.114 grayscale; target resized with Lanczos; no crop/warp or model-output compositing. Contact sheet is diagnostic only.',
        'areas': {'tab_pixels': int(tab.sum()), 'rest_clothing_pixels': int(rest.sum()), 'protected_pixels': int((~clothing).sum())},
        'manifest_sha256': digest(manifest_path), 'script_sha256': digest(Path(__file__)),
        'inputs': {name: {'path': str(path), 'sha256': digest(path)} for name, path in paths.items()},
        'measurements': measurements,
        'comparison_sha256': digest(args.output / 'comparison.png'),
    }
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    main()
