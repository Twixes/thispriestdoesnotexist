"""CPU-only geometry audit of three raw edits; no alignment or image mutation."""
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPER = HERE.parent / 'ffhq-clothing-edit1/measure.py'
MODEL = ROOT / 'research/alignment/models/face_detection_yunet_2023mar.onnx'
cv2.setNumThreads(1)
nodes = [node for node in ast.parse(HELPER.read_text()).body
         if isinstance(node, ast.FunctionDef) and node.name in ['sha', 'measure']]
assert len(nodes) == 2
namespace = {'ROOT': ROOT, 'hashlib': hashlib, 'Image': Image, 'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPER), 'exec'), namespace)
sha, measure = namespace['sha'], namespace['measure']
detector = cv2.FaceDetectorYN.create(str(MODEL), '', (256, 256), .75, .3, 5000)
prompts = json.loads((HERE / 'prompts.json').read_text())
sources_manifest_path = HERE.parent / 'ffhq-paired-sources64-b2/manifest.json'
sources = {row['id']: row for row in json.loads(sources_manifest_path.read_text())['entries']}


def compare(before, after, before_points, after_points):
    delta = after_points - before_points
    distances = np.linalg.norm(delta, axis=1)
    scale = after['interocular'] / before['interocular'] - 1
    return {'landmark_delta_xy': delta.tolist(), 'landmark_displacements': distances.tolist(),
            'max_landmark_displacement': float(distances.max()),
            'mean_landmark_displacement': float(distances.mean()),
            'eye_y_delta': after['eye_y'] - before['eye_y'],
            'mouth_y_delta': after['mouth_y'] - before['mouth_y'],
            'interocular_relative_change': scale,
            'landmark_tolerance_pass': bool(distances.max() <= .015),
            'interocular_tolerance_pass': bool(abs(scale) <= .05)}


canvas = Image.new('RGB', (768, len(prompts['records']) * 280), (24, 24, 24))
draw = ImageDraw.Draw(canvas)
entries, hashes = [], []
for index, row in enumerate(prompts['records']):
    source_path, edit_path = ROOT / row['source'], ROOT / row['output']
    assert edit_path.parent == HERE
    assert sha(source_path) == sources[row['id']]['image_sha256']
    assert sha(ROOT / row['latent_path']) == sources[row['id']]['latents_sha256']
    raw, raw_image, raw_points = measure(source_path, detector)
    gray, gray_image, gray_points = measure(source_path, detector, grayscale_control=True)
    edit_raw, edit_image, edit_raw_points = measure(edit_path, detector)
    edit_gray, _, edit_gray_points = measure(edit_path, detector, grayscale_control=True)
    entries.append({'id': row['id'], 'training_eligible': False, 'source_rgb': raw,
                    'source_grayscale': gray, 'edit_rgb': edit_raw, 'edit_grayscale': edit_gray,
                    'unadjusted_comparison': compare(raw, edit_raw, raw_points, edit_raw_points),
                    'matched_grayscale_comparison': compare(gray, edit_gray, gray_points, edit_gray_points),
                    'source_color_only_control': compare(raw, gray, raw_points, gray_points),
                    'chin_measurement': 'YuNet has no chin landmark; assessed visually, not numerically',
                    'collar_complete_in_frame': row['id'] == '058',
                    'collar_review': ('White Roman tab visible; lower edge cut by frame' if row['id'] in ['042', '059'] else 'Recognizable white Roman tab completely visible below chin')})
    for column, (image, label) in enumerate([(raw_image, 'source RGB'), (gray_image, 'source grayscale'), (edit_image, 'raw edit')]):
        x, y = column * 256, index * 280
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + 261), row['id'] + ' ' + label, fill='white')
    hashes.append({'id': row['id'], 'source_path': row['source'], 'source_sha256': sha(source_path),
                   'target_path': row['output'], 'target_sha256': sha(edit_path),
                   'latent_path': row['latent_path'], 'latent_sha256': sha(ROOT / row['latent_path']),
                   'training_eligible': False})
canvas.save(HERE / 'comparison.png')
result = {'purpose': 'Three built-in imagegen clothing-edit calibrations, not trained-priest-model outputs',
          'training_eligible': False, 'production_approved': False,
          'prospective_thresholds': {'max_landmark_euclidean_displacement': .015, 'max_absolute_interocular_relative_change': .05},
          'method': {'resize': 'RGB 256x256 Pillow LANCZOS; matched control then L to RGB on both inputs',
                     'cpu_threads': 1, 'detector': 'YuNet confidence .75, NMS .3, top_k 5000, largest area*confidence',
                     'alignment_warp': False, 'padding_or_reflection': False,
                     'landmark_order': ['eye_0', 'eye_1', 'nose', 'mouth_0', 'mouth_1']},
          'versions': {'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': pillow_version},
          'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(HELPER), 'detector_sha256': sha(MODEL),
          'entries': entries}
(HERE / 'geometry.json').write_text(json.dumps(result, indent=2) + '\n')
manifest = {'purpose': result['purpose'], 'training_eligible': False, 'production_approved': False,
            'path_resolution': 'Image and latent paths are repository-relative', 'entries': hashes,
            'source_manifest_sha256': sha(sources_manifest_path), 'prompts_sha256': sha(HERE / 'prompts.json'),
            'script_sha256': sha(Path(__file__)), 'geometry_sha256': sha(HERE / 'geometry.json'),
            'comparison_sha256': sha(HERE / 'comparison.png')}
(HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps([{'id': row['id'], 'matched': row['matched_grayscale_comparison'],
                  'unadjusted': row['unadjusted_comparison']} for row in entries], indent=2))
