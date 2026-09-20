"""Apply accepted eyes42 transforms directly to original-resolution source pixels."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE / 'collar-only'
SOURCE_MANIFEST = PARENT / 'manifest.json'
OUT = PARENT / 'eyes42-1024'
cv2.setNumThreads(2)
OUT.mkdir(exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def resize_coordinates(before, after):
    sx, sy = after[0] / before[0], after[1] / before[1]
    # Pixel centers use integer coordinates. PIL resizing maps image edges,
    # therefore x_out = (x_in + 0.5) * scale - 0.5.
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)

accepted = json.loads(SOURCE_MANIFEST.read_text())
assert len(accepted['entries']) == 110
records = []
for row in accepted['entries']:
    source = ROOT / row['source']
    assert digest(source) == row['source_sha256'], f'Changed source: {source}'
    original = Image.open(source).convert('RGB')
    dimensions = original.size
    assert dimensions == (1254, 1254), dimensions
    old_output = PARENT / 'eyes42' / f'{row["id"]}.png'
    assert digest(old_output) == row['eyes42']['training_sha256']
    reference = np.vstack([np.asarray(row['eyes42']['matrix_source256_to_output256']), [0, 0, 1]])
    source_to_reference = resize_coordinates(dimensions, (256, 256))
    reference_to_output = resize_coordinates((256, 256), (1024, 1024))
    transform = reference_to_output @ reference @ source_to_reference
    # Algebraically verify that the old and new transforms describe one geometry.
    points = np.array([[0, 0, 1], [1253, 1253, 1], [626.5, 626.5, 1], [300, 800, 1]], dtype=np.float64).T
    restored = np.linalg.inv(reference_to_output) @ transform @ points
    expected = reference @ source_to_reference @ points
    error = float(np.max(np.abs(restored - expected)))
    assert error < 1e-10
    pixels = cv2.warpAffine(np.asarray(original), transform[:2], (1024, 1024),
                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    target = OUT / f'{row["id"]}.png'
    Image.fromarray(pixels).save(target)
    mask = cv2.warpAffine(np.ones((dimensions[1], dimensions[0]), dtype=np.uint8),
                         transform[:2], (1024, 1024), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT)
    records.append({'id': row['id'], 'source': row['source'], 'source_sha256': row['source_sha256'],
                    'source_dimensions': list(dimensions), 'source_mode_decoded': 'RGB',
                    'accepted_256_output': str(old_output.relative_to(ROOT)),
                    'accepted_256_sha256': row['eyes42']['training_sha256'],
                    'matrix_source256_to_output256': reference[:2].tolist(),
                    'matrix_source1254_to_output1024': transform[:2].tolist(),
                    'output': str(target.relative_to(ROOT)), 'output_dimensions': [1024, 1024],
                    'output_sha256': digest(target), 'output_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                    'output_bytes': target.stat().st_size,
                    'geometry_roundtrip_max_error_in_reference_pixels': error,
                    'reflected_padding_fraction': float(1 - mask.mean()),
                    'accepted_collar_box_source256': row['collar_box_heuristic']})

manifest = {'version': 'collar-only-eyes42-original-to-1024-v1', 'count': len(records),
            'source_manifest': str(SOURCE_MANIFEST.relative_to(ROOT)),
            'source_manifest_sha256': digest(SOURCE_MANIFEST),
            'accepted_review_sha256': digest(PARENT / 'review.json'),
            'script_sha256': digest(Path(__file__)), 'opencv_version': cv2.__version__,
            'pillow_version': pillow_version, 'numpy_version': np.__version__,
            'settings': {'output_size': 1024, 'source_pixels': 'original 1254px RGB, no intermediate resize',
                         'interpolation': 'cv2.INTER_LANCZOS4', 'border': 'cv2.BORDER_REFLECT_101',
                         'transform': 'resize256to1024 @ accepted_affine256 @ resizeOriginalTo256',
                         'resize_coordinate_convention': '(pixel + 0.5) * output_size/input_size - 0.5',
                         'landmarks_redetected': False, 'collars_redetected': False},
            'entries': records}
(PARENT / 'eyes42-1024-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
columns, thumb, label = 10, 128, 16
sheet = Image.new('RGB', (columns * thumb, ((len(records) + columns - 1) // columns) * (thumb + label)), (25, 25, 25))
draw = ImageDraw.Draw(sheet)
for index, row in enumerate(records):
    x, y = index % columns * thumb, index // columns * (thumb + label)
    with Image.open(ROOT / row['output']) as image:
        sheet.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
    draw.text((x + 2, y + thumb + 1), row['id'], fill='white')
sheet.save(PARENT / 'eyes42-1024-contact.png')
print(json.dumps({'count': len(records), 'total_bytes': sum(r['output_bytes'] for r in records),
                  'source_dimensions': [1254, 1254], 'output_dimensions': [1024, 1024],
                  'unique_output_hashes': len({r['output_sha256'] for r in records}),
                  'manifest': str((PARENT / 'eyes42-1024-manifest.json').relative_to(ROOT))}))
