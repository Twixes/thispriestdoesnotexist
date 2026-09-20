"""Apply the existing 110-image eyes42 recipe to held-out inputs, without training."""
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
ROOT = RESEARCH.parent
ALIGNMENT = RESEARCH / 'alignment'
OUT = HERE / 'aligned256'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# analyze.py has top-level research writes. Reuse only its exact pure helpers,
# without importing/executing the data-processing part of that older script.
helper_source = ALIGNMENT / 'analyze.py'
helper_names = {'collar_box', 'contains_collar', 'metrics'}
helper_nodes = [node for node in ast.parse(helper_source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in helper_names]
assert {node.name for node in helper_nodes} == helper_names
helper_namespace = {'cv2': cv2, 'np': np}
exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(helper_source), 'exec'), helper_namespace)


def eyes42_transform(points, collar):
    # Same operations and ordering as alignment/collar_only.py; checked below
    # against every accepted training matrix before any validation output.
    points = np.array(points)
    left, right = sorted(points[:2], key=lambda p: p[0])
    eyes = (left + right) / 2
    mouth = points[3:5].mean(axis=0)
    eye_to_eye, eye_to_mouth = right - left, mouth - eyes
    axis = eye_to_eye - np.array([-eye_to_mouth[1], eye_to_mouth[0]])
    axis /= np.linalg.norm(axis)
    vertical = np.array([-axis[1], axis[0]])
    basis = np.stack([axis, vertical])
    if collar:
        x, y, w, h = collar
        guard = [np.array([x, y + h + 4]), np.array([x + w, y + h + 4])]
    else:
        guard = [mouth + vertical * np.linalg.norm(eye_to_mouth) * 1.8]
    depth = max(vertical @ (point - eyes) for point in guard)
    scale = .24 * 256 / np.linalg.norm(eye_to_eye)
    if depth > 0:
        scale = min(scale, (.96 - .42) * 256 / depth)
    offset = np.array([128, .42 * 256]) - basis @ eyes * scale
    return np.column_stack([basis * scale, offset])

accepted_path = ALIGNMENT / 'collar-only/manifest.json'
accepted = json.loads(accepted_path.read_text())
previous = {row['id']: row for row in json.loads((ALIGNMENT / 'manifest.json').read_text())['entries']}
errors = []
for row in accepted['entries']:
    matrix = eyes42_transform(previous[row['id']]['landmarks'], row['collar_box_heuristic'])
    errors.append(float(np.max(np.abs(matrix - row['eyes42']['matrix_source256_to_output256']))))
assert max(errors) < 1e-12
training_source_hashes = {row['source_sha256'] for row in accepted['entries']}
model = ALIGNMENT / 'models/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN.create(str(model), '', (256, 256), .75, .3, 5000)
inputs = sorted((HERE / 'source').glob('*.png'))
assert len(inputs) == 12
records = []
marked_images = []
for source in inputs:
    source_hash = digest(source)
    assert source_hash not in training_source_hashes, 'Validation input duplicates training bytes'
    with Image.open(source) as original:
        dimensions = original.size
        assert dimensions == (1254, 1254)
        rgb = np.array(original.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS))
    _, faces = detector.detect(rgb[:, :, ::-1].copy())
    if faces is None:
        raise RuntimeError(f'No confident face: {source.name}; no silent fallback output')
    face = max(faces, key=lambda value: value[2] * value[3] * value[-1])
    points = face[4:14].reshape(5, 2)
    collar = helper_namespace['collar_box'](rgb, face, points[:2].mean(axis=0), points[3:5].mean(axis=0))
    # Match stored JSON landmark precision from the original accepted workflow.
    matrix = eyes42_transform(np.asarray(points.tolist(), dtype=np.float64), collar)
    transformed = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    target = OUT / source.name
    Image.fromarray(transformed).save(target)
    coverage = cv2.warpAffine(np.ones((256, 256), dtype=np.uint8), matrix, (256, 256),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    records.append({'id': source.stem, 'split': 'validation_only', 'source': str(source.relative_to(ROOT)),
                    'source_sha256': source_hash, 'source_dimensions': list(dimensions),
                    'detector_confidence': float(face[-1]), 'face_box': face[:4].tolist(),
                    'landmarks_source256': points.tolist(), 'collar_box_heuristic': collar,
                    'collar_fallback_used': collar is None,
                    'matrix_source256_to_output256': matrix.tolist(),
                    'output': str(target.relative_to(ROOT)), 'output_dimensions': [256, 256],
                    'output_sha256': digest(target), 'metrics': helper_namespace['metrics'](face, matrix),
                    'detected_collar_fully_visible': helper_namespace['contains_collar'](matrix, collar),
                    'reflected_padding_fraction': float(1 - coverage.mean())})
    marked = Image.fromarray(rgb)
    draw = ImageDraw.Draw(marked)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red')
    if collar:
        x, y, w, h = collar
        draw.rectangle((x, y, x + w, y + h), outline='lime', width=2)
    marked_images.append(marked)

contact = Image.new('RGB', (4 * 256, 3 * 276), (25, 25, 25))
draw = ImageDraw.Draw(contact)
comparison = Image.new('RGB', (4 * 256, 6 * 276), (25, 25, 25))
compare_draw = ImageDraw.Draw(comparison)
for index, row in enumerate(records):
    aligned = Image.open(ROOT / row['output'])
    x, y = index % 4 * 256, index // 4 * 276
    contact.paste(aligned, (x, y)); draw.text((x + 4, y + 258), row['id'], fill='white')
    block_x, block_y = index % 2 * 512, index // 2 * 276
    comparison.paste(marked_images[index], (block_x, block_y))
    comparison.paste(aligned, (block_x + 256, block_y))
    compare_draw.text((block_x + 4, block_y + 258), row['id'] + ' detected / aligned', fill='white')
contact.save(HERE / 'alignment-contact.png')
comparison.save(HERE / 'alignment-comparison.png')
manifest = {'version': 'validation12-current-eyes42-v1', 'split': 'validation_only', 'count': len(records),
            'training_use_permitted': False, 'accepted_training_manifest_sha256': digest(accepted_path),
            'reference_collar_script_sha256': digest(ALIGNMENT / 'collar_only.py'),
            'reused_helper_source_sha256': digest(helper_source), 'reused_functions': sorted(helper_names),
            'script_sha256': digest(Path(__file__)), 'detector_sha256': digest(model),
            'opencv_version': cv2.__version__, 'numpy_version': np.__version__, 'pillow_version': pillow_version,
            'settings': {'resize': 'PIL LANCZOS 1254 to256 before detection and affine',
                         'detector': 'YuNet confidence0.75 nms0.3 topk5000; largest area timesconfidence',
                         'desired_interocular_fraction': .24, 'eye_y_fraction': .42,
                         'collar_bottom_guard_fraction': .96, 'collar_padding_source256_pixels': 4,
                         'interpolation': 'cv2.INTER_LANCZOS4', 'border': 'cv2.BORDER_REFLECT_101'},
            'accepted_matrix_regression': {'count': len(errors), 'maximum_absolute_difference': max(errors)},
            'source_hashes_disjoint_from_training_sources': True,
            'summary': {'collar_fallback_count': sum(row['collar_fallback_used'] for row in records),
                        'mean_reflected_padding_fraction': float(np.mean([row['reflected_padding_fraction'] for row in records])),
                        'max_reflected_padding_fraction': max(row['reflected_padding_fraction'] for row in records)},
            'entries': records}
(HERE / 'alignment-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'count': len(records), 'matrix_regression_max_error': max(errors), **manifest['summary']}))
