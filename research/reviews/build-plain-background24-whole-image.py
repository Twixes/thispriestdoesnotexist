#!/usr/bin/env python3
"""Build a research-only full-frame grayscale comparison; never edit sources."""
import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageDraw, features


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('research/data/plain-background24-whole-image'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise SystemExit('Refusing to overwrite an existing comparison directory.')
    sources = [(index, root / 'research/data' /
                ('plain-background-pilot8' if index <= 148 else 'plain-background-expansion16') /
                f'{index}.png') for index in range(141, 165)]
    before = {str(path.relative_to(root)): sha256(path) for _, path in sources}
    output.mkdir(parents=True)
    for resolution in (1024, 256):
        (output / str(resolution)).mkdir()
    contact = Image.new('RGB', (6 * 256, 4 * 280), '#181818')
    draw = ImageDraw.Draw(contact)
    entries = []
    for ordinal, (index, source) in enumerate(sources):
        relative = str(source.relative_to(root))
        with Image.open(source) as original:
            if original.width != original.height:
                raise ValueError(f'Expected square source, got {source}')
            original_mode = original.mode
            original_dimensions = list(original.size)
            # Deliberately preserve all source pixels geometrically: no EXIF
            # transpose, affine transform, crop, padding, reflection or restoration.
            gray_rgb = original.convert('L').convert('RGB')
        entry = {'id': index, 'source_path': relative,
                 'source_sha256': before[relative],
                 'source_dimensions': original_dimensions,
                 'source_mode': original_mode, 'outputs': {}}
        for resolution in (1024, 256):
            resized = gray_rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)
            destination = output / str(resolution) / f'{index}.png'
            resized.save(destination, format='PNG', optimize=False, compress_level=6)
            with Image.open(destination) as saved:
                saved.load()
                red, green, blue = saved.split()
                assert ImageChops.difference(red, green).getbbox() is None
                assert ImageChops.difference(red, blue).getbbox() is None
                assert saved.size == (resolution, resolution) and saved.mode == 'RGB'
            entry['outputs'][str(resolution)] = {
                'path': str(destination.relative_to(output)),
                'sha256': sha256(destination), 'dimensions': [resolution, resolution],
                'mode': 'RGB', 'exact_grayscale_channels': True,
            }
            if resolution == 256:
                left, top = (ordinal % 6) * 256, (ordinal // 6) * 280
                contact.paste(resized, (left, top))
                draw.text((left + 8, top + 260), str(index), fill='white')
        entries.append(entry)
    after = {str(path.relative_to(root)): sha256(path) for _, path in sources}
    assert before == after, 'A source changed during comparison generation.'
    contact_path = output / 'contact-256.png'
    contact.save(contact_path, format='PNG', optimize=False, compress_level=6)
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'preprocessing_candidate_pending_root_review',
        'accepted_training_dataset': False, 'model_output': False,
        'production_approved': False, 'count': len(entries),
        'method': 'Pillow original.convert(L).convert(RGB), then independent LANCZOS whole-image resize to1024 and256. Both outputs derive directly from the grayscale original;256 is not resized from1024.',
        'no_crop_affine_padding_reflection_inpainting': True,
        'source_hashes_unchanged_before_after': before == after,
        'source_hashes_after': after,
        'versions': {'python': sys.version, 'python_executable': sys.executable,
                     'pillow': PIL.__version__, 'zlib': features.version('zlib'),
                     'platform': platform.platform()},
        'script_path': str(Path(__file__).resolve().relative_to(root)),
        'script_sha256': sha256(Path(__file__)),
        'png_save_options': {'optimize': False, 'compress_level': 6},
        'contact': {'path': 'contact-256.png', 'sha256': sha256(contact_path),
                    'layout': '6 columns x4 rows, ascending IDs141–164; each256px portrait has24px label strip below.'},
        'entries': entries,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'count': len(entries),
                      'all_sources_unchanged': before == after,
                      'manifest_sha256': sha256(output / 'manifest.json')}))


if __name__ == '__main__':
    main()
