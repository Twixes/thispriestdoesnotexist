"""Separate bounds-constrained native crop candidate set; never imports reference scripts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = [ROOT / 'research/alignment/solve_1024_bounds.py', ROOT / 'research/alignment/render_1024_bounds.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source256_to_native1024(matrix, dimensions):
    scale = 256 / np.asarray(dimensions, dtype=np.float64)
    source = np.eye(3)
    source[0, 0], source[1, 1] = scale
    source[:2, 2] = .5 * scale - .5
    dest = np.array([[4., 0, 1.5], [0, 4., 1.5], [0, 0, 1.]])
    return (dest @ np.vstack([matrix, [0, 0, 1]]) @ source)[:2]


def solve(row):
    width, height = row['source_dimensions']
    matrix = source256_to_native1024(np.asarray(row['matrix']), [width, height])
    old_scale = float(np.linalg.norm(matrix[0, :2]))
    rotation = matrix[:, :2] / old_scale
    points = (np.asarray(row['landmarks']) + .5) * np.array([width, height]) / 256 - .5
    eye = points[:2].mean(axis=0)
    target = matrix[:, :2] @ eye + matrix[:, 2]
    inequalities, limits = [], []

    def add(coefficients, limit):
        inequalities.append(coefficients)
        limits.append(float(limit))

    margin = 4
    for corner in [[0, 0], [1023, 0], [0, 1023], [1023, 1023]]:
        rotated_corner = rotation.T @ corner
        for axis, bound in enumerate([width - 1 - margin, height - 1 - margin]):
            r = rotation.T[axis]
            add([margin - eye[axis], r[0], r[1], 0, 0, 0], rotated_corner[axis])
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
    collar = row['collar_box_heuristic']
    if collar:
        x, y, w, h = collar
        guard = np.array([[x, y], [x + w, y], [x, y + h + 4], [x + w, y + h + 4]])
        guard = (guard + .5) * np.array([width, height]) / 256 - .5
    else:
        mouth = points[3:5].mean(axis=0)
        guard = np.array([mouth + rotation[1] * np.linalg.norm(mouth - eye) * 1.8])
    for point in guard:
        keep_point(point, [16, 16], [1007, (.96 * 256 + .5) * 4 - .5])
    add([0, 1, 0, -1, 0, 0], target[0]); add([0, -1, 0, -1, 0, 0], -target[0])
    add([0, 0, 1, 0, -1, 0], target[1]); add([0, 0, -1, 0, -1, 0], -target[1])
    add([1, 0, 0, 0, 0, -1], old_scale); add([-1, 0, 0, 0, 0, -1], -old_scale)
    minimum_scale = (abs(rotation[0, 0]) + abs(rotation[0, 1])) * 1023 / (min(width, height) - 1 - 2 * margin)
    bounds = [(minimum_scale, old_scale * 2), (.40 * 1024, .60 * 1024), (.30 * 1024, .49 * 1024), (0, None), (0, None), (0, None)]
    exact_bounds = list(bounds)
    exact_bounds[1] = (target[0], target[0]); exact_bounds[2] = (target[1], target[1])
    exact = linprog([0, 0, 0, 0, 0, 1], A_ub=inequalities, b_ub=limits, bounds=exact_bounds, method='highs')
    near_bounds = list(bounds); near_bounds[2] = (.40 * 1024, .44 * 1024)
    near = linprog([0, 0, 0, 0, 0, 1], A_ub=inequalities, b_ub=limits, bounds=near_bounds, method='highs')
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
        return {'id': row['id'], 'feasible': False, 'solver_message': result.message}
    scale, px, py = result.x[:3]
    transform = np.column_stack([scale * rotation, np.array([px, py]) - scale * rotation @ eye])
    inverse = np.linalg.inv(np.vstack([transform, [0, 0, 1]]))
    corners = (inverse @ np.array([[0, 0, 1], [1023, 0, 1], [0, 1023, 1], [1023, 1023, 1]]).T).T[:, :2]
    assert corners.min() >= margin - 1e-6
    assert (corners <= np.array([width - 1 - margin, height - 1 - margin]) + 1e-6).all()
    projected = points @ transform[:, :2].T + transform[:, 2]
    collar_projected = guard @ transform[:, :2].T + transform[:, 2]
    assert (projected >= 64 - 1e-5).all() and (projected <= 959 + 1e-5).all()
    assert (collar_projected >= 16 - 1e-5).all()
    assert (collar_projected <= np.array([1007, (.96 * 256 + .5) * 4 - .5]) + 1e-5).all()
    return {'id': row['id'], 'feasible': True, 'exact_eye_position_feasible': bool(exact.success),
            'eye_40_to_44_band_feasible': bool(near.success), 'matrix_source_to_output1024': transform.tolist(),
            'original_matrix_source_to_output1024': matrix.tolist(), 'inverse_output_corners': corners.tolist(),
            'original_output_eye1024': target.tolist(), 'corrected_output_eye1024': [float(px), float(py)],
            'eye_shift1024_xy': (np.array([px, py]) - target).tolist(),
            'eye_shift256_xy': ((np.array([px, py]) - target) / 4).tolist(),
            'eye_y_fraction': float(py / 1024), 'scale_ratio_vs_eyes42': float(scale / old_scale),
            'source_border_support_margin': margin, 'landmarks_output1024': projected.tolist(),
            'collar_guard_output1024': collar_projected.tolist(), 'collar_guard_fully_visible': True,
            'solver_message': result.message}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    started = time.monotonic()
    if args.output.exists():
        raise ValueError('Use a NEW output directory; preserve all prior sources and renders')
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['entries']
    assert [r['id'] for r in rows] == [str(i) for i in range(141, 149)]
    inputs = [args.manifest.resolve()] + REFERENCES + [ROOT / r['source'] for r in rows]
    inputs += [ROOT / r['output'] for r in rows]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    cv2.setNumThreads(1)
    records = []
    for row in rows:
        source = ROOT / row['source']
        assert sha(source) == row['source_sha256']
        assert row['source_dimensions'] == [1254, 1254]
        record = solve(row)
        record.update(source=row['source'], source_sha256=row['source_sha256'],
                      original_landmarks_source256=row['landmarks'], collar_box_source256=row['collar_box_heuristic'],
                      prior_reflected_padding_fraction=row['reflected_padding_fraction'])
        records.append(record)
    args.output.mkdir(parents=True)
    (args.output / 'native1024').mkdir(); (args.output / 'aligned256').mkdir()
    sheet = Image.new('RGB', (1024, 552), (25, 25, 25)); draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        if not record['feasible']:
            continue
        source = ROOT / record['source']
        with Image.open(source) as im:
            assert im.size == (1254, 1254)
            pixels = np.array(im.convert('L').convert('RGB'))
        transform = np.asarray(record['matrix_source_to_output1024'])
        rendered = cv2.warpAffine(pixels, transform, (1024, 1024), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 255))
        reflected = cv2.warpAffine(pixels, transform, (1024, 1024), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT_101)
        assert np.array_equal(rendered, reflected), record['id']
        assert np.array_equal(rendered[:, :, 0], rendered[:, :, 1]) and np.array_equal(rendered[:, :, 1], rendered[:, :, 2])
        native = args.output / 'native1024' / f'{record["id"]}.png'
        small = args.output / 'aligned256' / f'{record["id"]}.png'
        Image.fromarray(rendered).save(native)
        resized = Image.fromarray(rendered).resize((256, 256), Image.Resampling.LANCZOS)
        resized.save(small)
        x, y = i % 4 * 256, i // 4 * 276; sheet.paste(resized, (x, y)); draw.text((x + 4, y + 258), record['id'], fill='white')
        record.update(native_output=str(native.relative_to(ROOT)), native_sha256=sha(native),
                      output256=str(small.relative_to(ROOT)), output256_sha256=sha(small),
                      native_rgb_sha256=hashlib.sha256(rendered.tobytes()).hexdigest(),
                      reflected_padding_fraction=0, constant_vs_reflected_render_bitwise_equal=True,
                      native_grayscale_channels_equal=True, quality_approved=False)
    sheet.save(args.output / 'contact256.png')
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert before == after, 'Source/reference/aligned input mutated'
    feasible = [r for r in records if r['feasible']]
    summary = {'count': len(records), 'feasible': len(feasible), 'infeasible': len(records)-len(feasible),
               'exact_eye_position_feasible': sum(r['exact_eye_position_feasible'] for r in feasible),
               'eye_y_fraction_range': [min(r['eye_y_fraction'] for r in feasible), max(r['eye_y_fraction'] for r in feasible)],
               'max_abs_eye_shift256_xy': np.max(np.abs([r['eye_shift256_xy'] for r in feasible]), axis=0).tolist(),
               'border_independence_verified_count': sum(r.get('constant_vs_reflected_render_bitwise_equal', False) for r in records)}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'version': 'plain-background-pilot-zero-reflection-v1',
              'script_sha256': sha(Path(__file__)), 'reference_algorithms': {str(p.relative_to(ROOT)): sha(p) for p in REFERENCES},
              'method': 'Copied constrained LP from solve_1024_bounds.py; preserved rotation and lexicographic exact-eyes/eyeY/eyeX/scale objectives. No reference top-level code executed.',
              'pixel_coordinates': 'source256=(native+.5)*256/1254-.5; output1024=(output256+.5)*4-.5',
              'color': 'Every native source normalized Pillow L then RGB before warp, including146',
              'render': 'Native1254 to1024 cv2.INTER_LANCZOS4, source margin4; Pillow LANCZOS1024 to256',
              'border_test': 'Different constant magenta versus reflected101 border rules produce byte-identical1024 arrays; inverse sampled corners lie within4px source support margin.',
              'versions': {'opencv': cv2.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'pillow': pillow_version},
              'wall_seconds': time.monotonic()-started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'inputs_before': before, 'inputs_after': after, 'inputs_unchanged': True,
              'active_dataset_changed': False, 'quality_approved': False, 'production_approved': False,
              'summary': summary, 'entries': records}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
