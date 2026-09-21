"""Diagnostic polygon overlay only; never edits input files or model output."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--mask-json', type=Path, required=True,
                        help='JSON object with clothing_polygons or a polygon-list JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    if not 32 <= args.size <= 2048:
        parser.error('Diagnostic size must be 32–2048')
    if args.output.exists() or args.output.resolve() in [args.source.resolve(), args.target.resolve()]:
        parser.error('Use a fresh diagnostic output path, never an input path')
    data = json.loads(args.mask_json.read_text())
    polygons = data['clothing_polygons'] if isinstance(data, dict) else data
    # Import validates exactly the same geometry as the trainer; no model loaded.
    from trainer import polygon_mask
    mask = polygon_mask(polygons, args.size)[0, 0].numpy()
    overlay = Image.new('RGBA', (args.size, args.size), (255, 30, 30, 0))
    overlay.putalpha(Image.fromarray((mask * 85).astype('uint8')))
    output = Image.new('RGB', (args.size * 2, args.size + 28), 'black')
    for index, path in enumerate([args.source, args.target]):
        with Image.open(path) as opened:
            if opened.width != opened.height:
                raise ValueError('No diagnostic crop: source and target must be square')
            original = opened.convert('RGBA').resize((args.size, args.size), Image.Resampling.LANCZOS)
        output.paste(Image.alpha_composite(original, overlay).convert('RGB'), (index * args.size, 28))
    ImageDraw.Draw(output).text((8, 8), 'SOURCE | red = clothing supervision; unshaded = protected source', fill='white')
    ImageDraw.Draw(output).text((args.size + 8, 8), 'EDIT TARGET | diagnostic overlay only', fill='white')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    info = {'production_approved': False, 'diagnostic_only': True, 'size': args.size,
            'clothing_fraction': float(mask.mean()), 'clothing_polygons': polygons,
            'inputs': {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in [('source', args.source), ('target', args.target), ('mask', args.mask_json)]}}
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
