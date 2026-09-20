"""Compare two held-out framing calibrations using unchanged alignment recipes."""
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
ALIGN = RESEARCH / 'alignment'
OUT = HERE / 'candidates'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def helpers(path, names):
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

base = helpers(ALIGN / 'analyze.py', ['collar_box', 'transform_matrix', 'metrics', 'contains_collar'])
heldout = helpers(RESEARCH / 'data/validation12/align_validation.py', ['eyes42_transform'])
model = ALIGN / 'models/face_detection_yunet_2023mar.onnx'
detector = cv2.FaceDetectorYN.create(str(model), '', (256, 256), .75, .3, 5000)
inputs = json.loads((HERE / 'manifest.json').read_text())['images']
records = []
for row in inputs:
    source = HERE / row['file']
    assert sha(source) == row['sha256']
    with Image.open(source) as im:
        source_size = list(im.size)
        rgb = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS))
    _, faces = detector.detect(rgb[:, :, ::-1].copy())
    assert faces is not None
    face = max(faces, key=lambda f: f[2] * f[3] * f[-1])
    points = face[4:14].reshape(5, 2)
    collar = base['collar_box'](rgb, face, points[:2].mean(axis=0), points[3:5].mean(axis=0))
    matrices = {'raw': np.array([[1., 0., 0.], [0., 1., 0.]]),
                'eyes42': heldout['eyes42_transform'](np.array(points.tolist()), collar),
                'ffhq': base['transform_matrix'](face, 'ffhq', collar)}
    record = {'id': f'{row["id"]:03}', 'source': row['file'], 'source_sha256': sha(source),
              'source_dimensions': source_size, 'face_confidence': float(face[-1]),
              'landmarks_source256': points.tolist(), 'collar_box_heuristic': collar,
              'training_eligible': False, 'variants': {}}
    for name, matrix in matrices.items():
        output = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
        path = OUT / f'{record["id"]}-{name}.png'
        Image.fromarray(output).save(path)
        coverage = cv2.warpAffine(np.ones((256, 256), dtype=np.uint8), matrix, (256, 256),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        collar_fraction = None
        if collar:
            x, y, w, h = collar
            polygon = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]]) @ matrix.T
            area = cv2.contourArea(polygon.astype(np.float32))
            intersect, _ = cv2.intersectConvexConvex(polygon.astype(np.float32), np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32))
            collar_fraction = float(intersect / area) if area else None
        record['variants'][name] = {'matrix_source256_to_output256': matrix.tolist(),
                                   'metrics': base['metrics'](face, matrix),
                                   'reflected_padding_fraction': float(1 - coverage.mean()),
                                   'detected_collar_fully_visible': base['contains_collar'](matrix, collar),
                                   'detected_collar_area_fraction_inside_crop': collar_fraction,
                                   'output': str(path.relative_to(HERE)), 'output_sha256': sha(path)}
    records.append(record)
reference = json.loads((ALIGN / 'summary.json').read_text())['ffhq']['metrics']
reference_metrics = {name: reference[name]['mean'] for name in ['eye_y', 'interocular', 'mouth_y']}
training = json.loads((ALIGN / 'manifest.json').read_text())['entries']
training = [row for row in training if row['group'] != 'ffhq' and row['detected']]
training_metrics = {name: float(np.mean([row['metrics'][name] for row in training])) for name in reference_metrics}
result = {'purpose': 'Two-image geometry calibration only; excluded from all training',
          'script_sha256': sha(Path(__file__)), 'input_manifest_sha256': sha(HERE / 'manifest.json'),
          'helper_sources': {str(path.relative_to(RESEARCH)): sha(path) for path in [ALIGN / 'analyze.py', RESEARCH / 'data/validation12/align_validation.py']},
          'detector_sha256': sha(model), 'opencv_version': cv2.__version__, 'pillow_version': pillow_version,
          'source_reference_metrics': reference_metrics, 'existing110_raw_mean_metrics': training_metrics,
          'existing110_eyes42_metrics': json.loads((ALIGN / 'collar-only/manifest.json').read_text())['summary']['eyes42'],
          'entries': records}
(HERE / 'geometry.json').write_text(json.dumps(result, indent=2) + '\n')
canvas = Image.new('RGB', (3 * 384, 2 * 412), (25, 25, 25))
draw = ImageDraw.Draw(canvas)
for row_index, record in enumerate(records):
    for column, name in enumerate(['raw', 'eyes42', 'ffhq']):
        path = HERE / record['variants'][name]['output']
        with Image.open(path) as image:
            canvas.paste(image.resize((384, 384), Image.Resampling.LANCZOS), (column * 384, row_index * 412))
        draw.text((column * 384 + 4, row_index * 412 + 387), record['id'] + ' ' + name, fill='white')
canvas.save(HERE / 'geometry-comparison.png')
print(json.dumps({'reference': reference_metrics, 'existing110_raw': training_metrics,
 'images': [{'id': row['id'], 'collar': row['collar_box_heuristic'],
             'variants': {name: {'metrics': v['metrics'], 'reflection': v['reflected_padding_fraction'],
                                'collar_fraction': v['detected_collar_area_fraction_inside_crop']} for name, v in row['variants'].items()}}
            for row in records]}, indent=2))
