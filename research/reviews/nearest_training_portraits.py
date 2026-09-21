"""Find pixel-nearest training portraits for visual memorization review.

This is a diagnostic, not an identity metric or proof of generalization.
It compares both orientations because training used mirror augmentation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(image):
    # Common aligned face region, excluding most architecture and clothing.
    return np.asarray(image.convert('L').crop((32, 8, 224, 176)).resize(
        (64, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--pattern', default='*.png')
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh output directory'
    targets = sorted(args.training.glob('*.png'))
    samples = sorted(args.samples.glob(args.pattern))
    assert len(targets) == 110 and len(samples) == 32
    references, arrays = [], []
    for path in targets:
        with Image.open(path) as source:
            assert source.size == (256, 256)
            for mirror in [False, True]:
                image = ImageOps.mirror(source) if mirror else source
                arrays.append(descriptor(image))
                references.append((path, mirror))
    arrays = np.stack(arrays)
    rows, pairs = [], []
    for path in samples:
        with Image.open(path) as image:
            assert image.size == (256, 256)
            distances = np.mean((arrays - descriptor(image)) ** 2, axis=(1, 2))
            index = int(np.argmin(distances))
            target, mirror = references[index]
            rows.append({'sample': str(path), 'sample_sha256': digest(path),
                         'nearest_training': str(target), 'training_sha256': digest(target),
                         'mirrored': mirror, 'face_crop_pixel_rmse': float(np.sqrt(distances[index]))})
            with Image.open(target) as training:
                pair = Image.new('RGB', (512, 256))
                pair.paste(image.convert('RGB'), (0, 0))
                pair.paste((ImageOps.mirror(training) if mirror else training).convert('RGB'), (256, 0))
                pairs.append(pair)
    args.output.mkdir(parents=True)
    for index, pair in enumerate(pairs):
        pair.save(args.output / f'{index:02}-generated-left-training-right.png')
    sheet = Image.new('RGB', (2048, 2048))
    for index, pair in enumerate(pairs):
        sheet.paste(pair, ((index % 4) * 512, (index // 4) * 256))
    sheet.save(args.output / 'comparison.png')
    report = {'method': 'RMSE of grayscale aligned face crop at64x64, both target orientations',
              'limitations': 'Pixel proximity is sensitive to pose and lighting. Neither low nor high distance establishes identity copying or novelty; review native pairs.',
              'production_approved': False, 'script_sha256': digest(Path(__file__)),
              'sample_count': len(rows), 'training_count': len(targets), 'results': rows}
    (args.output / 'matches.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'sample_count': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
