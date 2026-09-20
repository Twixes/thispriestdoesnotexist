"""Measure a held-out clothing edit without alignment, warping, or model inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROMPT = HERE / 'prompt.json'
DETECTOR = ROOT / 'research/alignment/models/face_detection_yunet_2023mar.onnx'
cv2.setNumThreads(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path, detector, grayscale_control=False):
    with Image.open(path) as original:
        size = list(original.size)
        rgb = original.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    if grayscale_control:
        rgb = rgb.convert('L').convert('RGB')
    _, faces = detector.detect(np.asarray(rgb)[:, :, ::-1].copy())
    if faces is None:
        raise RuntimeError(f'No face detected in {path}')
    face = max(faces, key=lambda row: row[2] * row[3] * row[-1])
    points = face[4:14].reshape(5, 2).astype(np.float64) / 256
    result = {
        'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
        'dimensions': size, 'face_confidence': float(face[-1]),
        'grayscale_control': grayscale_control,
        'landmarks_normalized': points.tolist(),
        'eye_y': float(points[:2, 1].mean()),
        'mouth_y': float(points[3:, 1].mean()),
        'interocular': float(np.linalg.norm(points[0] - points[1])),
    }
    return result, rgb, points


prompt = json.loads(PROMPT.read_text())
source = ROOT / prompt['source']
assert sha(source) == prompt['source_sha256']
edited = HERE / '001.png'
detector = cv2.FaceDetectorYN.create(str(DETECTOR), '', (256, 256), .75, .3, 5000)
before, before_rgb, before_points = measure(source, detector)
after, after_rgb, after_points = measure(edited, detector)
gray_source, _, gray_points = measure(source, detector, grayscale_control=True)
delta = after_points - before_points
distances = np.linalg.norm(delta, axis=1)
scale_change = after['interocular'] / before['interocular'] - 1
result = {
    'purpose': 'Single built-in imagegen clothing-edit geometry calibration',
    'training_eligible': False, 'production_approved': False,
    'is_trained_priest_model_output': False,
    'script_sha256': sha(Path(__file__)), 'prompt_sha256': sha(PROMPT),
    'detector_sha256': sha(DETECTOR),
    'versions': {'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': pillow_version},
    'method': {'resize': 'Pillow RGB 256x256 LANCZOS', 'cpu_threads': 2,
               'detector': 'YuNet five landmarks, confidence .75, NMS .3, top_k 5000',
               'face_selection': 'largest area times confidence',
               'landmark_order': ['eye_0', 'eye_1', 'nose', 'mouth_0', 'mouth_1'],
               'alignment_warp': False, 'padding_or_reflection': False},
    'source': before, 'edit': after,
    'comparison': {
        'landmark_delta_xy': delta.tolist(),
        'landmark_euclidean_displacement': distances.tolist(),
        'max_landmark_euclidean_displacement': float(distances.max()),
        'mean_landmark_euclidean_displacement': float(distances.mean()),
        'max_absolute_coordinate_displacement': float(np.abs(delta).max()),
        'eye_y_delta': after['eye_y'] - before['eye_y'],
        'mouth_y_delta': after['mouth_y'] - before['mouth_y'],
        'interocular_relative_change': float(scale_change),
        'prospective_max_landmark_displacement': .015,
        'prospective_max_absolute_interocular_change': .05,
        'landmark_tolerance_pass': bool(distances.max() <= .015),
        'interocular_tolerance_pass': bool(abs(scale_change) <= .05),
        'interpretation': 'Detector geometry only; not proof of pixel identity or training quality',
    },
}
gray_edit_delta = after_points - gray_points
gray_edit_distances = np.linalg.norm(gray_edit_delta, axis=1)
gray_edit_scale = after['interocular'] / gray_source['interocular'] - 1
result['grayscale_source_control'] = {
    'method': 'Same source RGB resize, then Pillow L conversion back to RGB; in memory only',
    'source': gray_source,
    'raw_source_to_grayscale_max_landmark_displacement': float(np.linalg.norm(gray_points - before_points, axis=1).max()),
    'raw_source_to_grayscale_interocular_relative_change': gray_source['interocular'] / before['interocular'] - 1,
    'grayscale_source_to_edit_landmark_delta_xy': gray_edit_delta.tolist(),
    'grayscale_source_to_edit_landmark_euclidean_displacement': gray_edit_distances.tolist(),
    'grayscale_source_to_edit_max_landmark_displacement': float(gray_edit_distances.max()),
    'grayscale_source_to_edit_mean_landmark_displacement': float(gray_edit_distances.mean()),
    'grayscale_source_to_edit_interocular_relative_change': float(gray_edit_scale),
    'landmark_tolerance_pass': bool(gray_edit_distances.max() <= .015),
    'interocular_tolerance_pass': bool(abs(gray_edit_scale) <= .05),
    'interpretation': 'Color alone substantially shifts this detector; grayscale control reduces this confound but does not establish pixel identity',
}
canvas = Image.new('RGB', (512, 280), (24, 24, 24))
canvas.paste(before_rgb, (0, 0))
canvas.paste(after_rgb, (256, 0))
draw = ImageDraw.Draw(canvas)
draw.text((5, 262), 'Source, native crop at 256px', fill='white')
draw.text((261, 262), 'Clothing edit, native crop at 256px', fill='white')
comparison = HERE / 'comparison.png'
canvas.save(comparison)
result['comparison_image_sha256'] = sha(comparison)
(HERE / 'geometry.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
