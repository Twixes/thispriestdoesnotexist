"""Render the separate zero-reflection crop proposal; preserve all earlier data."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE / 'collar-only'
PROPOSAL = PARENT / 'eyes42-1024-zero-reflection-proposal.json'
OUT = PARENT / 'eyes42-1024-zero-reflection'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)
proposal = json.loads(PROPOSAL.read_text())
assert proposal['summary']['infeasible'] == 0
records = []
for row in proposal['entries']:
    source = ROOT / row['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
    pixels = np.asarray(Image.open(source).convert('RGB'))
    transform = np.asarray(row['matrix_corrected_source_to_output1024'])
    output = cv2.warpAffine(pixels, transform, (1024, 1024), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 255))
    reflected = cv2.warpAffine(pixels, transform, (1024, 1024), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REFLECT_101)
    # Identical pixels under different border rules prove the corrected render
    # has no contribution from the reflected or constant border extension.
    assert np.array_equal(output, reflected), row['id']
    target = OUT / f'{row["id"]}.png'
    Image.fromarray(output).save(target)
    records.append({**row, 'output': str(target.relative_to(ROOT)),
                    'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'output_rgb_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                    'output_bytes': target.stat().st_size, 'reflected_padding_fraction': 0,
                    'constant_vs_reflected_render_bitwise_equal': True})
manifest = {'version': 'collar-only-zero-reflection-original-to-1024-v1',
            'proposal_sha256': hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'opencv_version': cv2.__version__, 'pillow_version': pillow_version,
            'interpolation': 'cv2.INTER_LANCZOS4',
            'border': 'No border samples used, verified constant/reflected equality',
            'summary': proposal['summary'], 'entries': records}
(PARENT / 'eyes42-1024-zero-reflection-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
columns, thumb, label = 10, 128, 16
sheet = Image.new('RGB', (columns * thumb, ((len(records) + columns - 1) // columns) * (thumb + label)), (25, 25, 25))
draw = ImageDraw.Draw(sheet)
for index, row in enumerate(records):
    x, y = index % columns * thumb, index // columns * (thumb + label)
    with Image.open(ROOT / row['output']) as image:
        sheet.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
    draw.text((x + 2, y + thumb + 1), row['id'], fill='white')
sheet.save(PARENT / 'eyes42-1024-zero-reflection-contact.png')
selected = ['synthetic-122', 'synthetic-102', 'synthetic-099']
worst_shift = max(records, key=lambda row: row['eye_displacement_pixels'])['id']
if worst_shift not in selected: selected.append(worst_shift)
comparison = Image.new('RGB', (1024, len(selected) * 536), (25, 25, 25))
draw = ImageDraw.Draw(comparison)
for index, name in enumerate(selected):
    for column, variant in enumerate(['eyes42-1024', 'eyes42-1024-zero-reflection']):
        with Image.open(PARENT / variant / f'{name}.png') as image:
            comparison.paste(image.resize((512, 512), Image.Resampling.LANCZOS), (column * 512, index * 536))
        draw.text((column * 512 + 4, index * 536 + 514), name + ' ' + ('accepted geometry' if column == 0 else 'zero reflection'), fill='white')
comparison.save(PARENT / 'eyes42-1024-zero-reflection-comparison.png')
print(json.dumps({'count': len(records), 'total_bytes': sum(row['output_bytes'] for row in records),
                  'border_independence_verified': True, 'comparison_ids': selected}))
