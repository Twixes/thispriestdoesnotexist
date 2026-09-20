"""Find minimally shifted/scaled crops with no sampled pixels outside each source."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import scipy

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'collar-only'
manifest_path = PARENT / 'eyes42-1024-manifest.json'
manifest = json.loads(manifest_path.read_text())
landmarks = {row['id']: row for row in json.loads((HERE / 'manifest.json').read_text())['entries']}
records = []
for row in manifest['entries']:
    width, height = row['source_dimensions']
    matrix = np.asarray(row['matrix_source1254_to_output1024'])
    old_scale = float(np.linalg.norm(matrix[0, :2]))
    rotation = matrix[:, :2] / old_scale
    points = (np.asarray(landmarks[row['id']]['landmarks']) + .5) * np.array([width, height]) / 256 - .5
    eye = points[:2].mean(axis=0)
    target = matrix[:, :2] @ eye + matrix[:, 2]
    # Variables: scale, eye_x, eye_y, abs_dx, abs_dy, abs_dscale.
    inequalities, limits = [], []
    def add(coefficients, limit):
        inequalities.append(coefficients)
        limits.append(float(limit))
    source_margin = 4  # Lanczos4 taps need room beyond the sampled center.
    for corner in [[0, 0], [1023, 0], [0, 1023], [1023, 1023]]:
        rotated_corner = rotation.T @ corner
        for axis, bound in enumerate([width - 1 - source_margin, height - 1 - source_margin]):
            r = rotation.T[axis]
            add([source_margin - eye[axis], r[0], r[1], 0, 0, 0], rotated_corner[axis])
            add([eye[axis] - bound, -r[0], -r[1], 0, 0, 0], -rotated_corner[axis])
    def keep_point(point, lower, upper):
        displacement = rotation @ (point - eye)
        for axis in [0, 1]:
            coefficients = [displacement[axis], 0, 0, 0, 0, 0]
            coefficients[1 + axis] = 1
            add(coefficients, upper[axis])
            add([-c for c in coefficients], -lower[axis])
    for point in points:
        keep_point(point, [64, 64], [959, 959])
    collar = row['accepted_collar_box_source256']
    if collar:
        x, y, w, h = collar
        guard = np.array([[x, y], [x + w, y], [x, y + h + 4], [x + w, y + h + 4]])
        guard = (guard + .5) * np.array([width, height]) / 256 - .5
    else:
        mouth = points[3:5].mean(axis=0)
        vertical = rotation[1]
        guard = [mouth + vertical * np.linalg.norm(mouth - eye) * 1.8]
    for point in guard:
        keep_point(point, [16, 16], [1007, (.96 * 256 + .5) * 4 - .5])
    add([0, 1, 0, -1, 0, 0], target[0]); add([0, -1, 0, -1, 0, 0], -target[0])
    add([0, 0, 1, 0, -1, 0], target[1]); add([0, 0, -1, 0, -1, 0], -target[1])
    add([1, 0, 0, 0, 0, -1], old_scale); add([-1, 0, 0, 0, 0, -1], -old_scale)
    minimum_scale = (abs(rotation[0, 0]) + abs(rotation[0, 1])) * 1023 / (min(width, height) - 1 - 2 * source_margin)
    bounds = [(minimum_scale, old_scale * 2), (.40 * 1024, .60 * 1024), (.30 * 1024, .49 * 1024), (0, None), (0, None), (0, None)]
    exact_bounds = list(bounds)
    exact_bounds[1] = (target[0], target[0]); exact_bounds[2] = (target[1], target[1])
    exact = linprog([0, 0, 0, 0, 0, 1], A_ub=inequalities, b_ub=limits, bounds=exact_bounds, method='highs')
    near_bounds = list(bounds)
    near_bounds[2] = (.40 * 1024, .44 * 1024)
    near = linprog([0, 0, 0, 0, 0, 1], A_ub=inequalities, b_ub=limits, bounds=near_bounds, method='highs')
    # Lexicographically preserve eye height, then horizontal centering, then zoom.
    result = exact
    if not result.success:
        result = linprog([0, 0, 0, 0, 1, 0], A_ub=inequalities, b_ub=limits, bounds=bounds, method='highs')
        if result.success:
            bounds[4] = (0, max(0.0, float(result.x[4])) + 1e-4)
            result = linprog([0, 0, 0, 1, 0, 0], A_ub=inequalities, b_ub=limits, bounds=bounds, method='highs')
            if result.success:
                bounds[3] = (0, max(0.0, float(result.x[3])) + 1e-4)
                result = linprog([0, 0, 0, 0, 0, 1], A_ub=inequalities, b_ub=limits, bounds=bounds, method='highs')
    if not result.success:
        records.append({'id': row['id'], 'feasible': False, 'solver_message': result.message})
        continue
    scale, px, py = result.x[:3]
    transform = np.column_stack([scale * rotation, np.array([px, py]) - scale * rotation @ eye])
    inverse = np.linalg.inv(np.vstack([transform, [0, 0, 1]]))
    corners = (inverse @ np.array([[0, 0, 1], [1023, 0, 1], [0, 1023, 1], [1023, 1023, 1]]).T).T[:, :2]
    assert corners.min() >= source_margin - 1e-6
    assert (corners <= np.array([width - 1 - source_margin, height - 1 - source_margin]) + 1e-6).all()
    records.append({**row, 'feasible': True, 'exact_eye_position_feasible': bool(exact.success), 'eye_40_to_44_band_feasible': bool(near.success),
                    'matrix_corrected_source_to_output1024': transform.tolist(), 'inverse_output_corners': corners.tolist(),
                    'original_output_eye': target.tolist(), 'corrected_output_eye': [float(px), float(py)],
                    'scale_ratio_vs_accepted': float(scale / old_scale),
                    'eye_displacement_pixels': float(np.linalg.norm([px - target[0], py - target[1]])),
                    'eye_y_fraction': float(py / 1024), 'source_border_support_margin': source_margin})
summary = {'count': len(records), 'infeasible': sum(not r['feasible'] for r in records),
           'exact_eye_position_feasible': sum(r.get('exact_eye_position_feasible', False) for r in records),
           'eye_40_to_44_band_infeasible': sum(not r.get('eye_40_to_44_band_feasible', False) for r in records),
           'eyes_outside_40_to_44_percent': sum(not .4 <= r.get('eye_y_fraction', .42) <= .44 for r in records),
           'max_eye_displacement_pixels': max(r.get('eye_displacement_pixels', 0) for r in records),
           'eye_y_fraction_range': [min(r.get('eye_y_fraction', .42) for r in records), max(r.get('eye_y_fraction', .42) for r in records)]}
output = {'version': 'zero-reflection-affine-solver-v1', 'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scipy_version': scipy.__version__,
          'settings': {'rotation': 'unchanged', 'source_margin_pixels': 4, 'collar_guard': 'same detected box plus four reference pixels',
                       'maximum_collar_y_reference_fraction': .96, 'face_landmark_margin_output_pixels': 64,
                       'objective': 'exact eyes first; otherwise lexicographic absolute eye-Y change, eye-X change, then scale change'},
          'summary': summary, 'entries': records}
(PARENT / 'eyes42-1024-zero-reflection-proposal.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(summary, indent=2))
