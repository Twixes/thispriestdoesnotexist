"""Prepare a separate eight-image candidate set with the existing eyes42 recipe.

Never changes the110-image training dataset or any running experiment input.
Face/collar detection is only preprocessing; visual acceptance is separate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = ROOT / 'research/alignment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monochrome', action='store_true', help='Normalize candidate training pixels to grayscale RGB')
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh candidate output directory'
    paths = [args.source / f'{i}.png' for i in range(141, 149)]
    assert all(p.is_file() for p in paths), 'All eight original candidates must exist'
    namespace = {'np': np, 'cv2': cv2}
    helper = ALIGNMENT / 'analyze.py'
    recipe = ROOT / 'research/data/validation12/align_validation.py'
    load_functions(helper, {'collar_box', 'contains_collar', 'metrics'}, namespace)
    load_functions(recipe, {'eyes42_transform'}, namespace)
    manifest = json.loads((ALIGNMENT / 'collar-only/manifest.json').read_text())
    previous = {row['id']: row for row in json.loads((ALIGNMENT / 'manifest.json').read_text())['entries']}
    # Ensure the extracted recipe reproduces all existing transforms exactly.
    for row in manifest['entries']:
        matrix = namespace['eyes42_transform'](previous[row['id']]['landmarks'], row['collar_box_heuristic'])
        assert np.max(np.abs(matrix - row['eyes42']['matrix_source256_to_output256'])) < 1e-12
    cv2.setNumThreads(1)
    detector_path = ALIGNMENT / 'models/face_detection_yunet_2023mar.onnx'
    detector = cv2.FaceDetectorYN.create(str(detector_path), '', (256, 256), .75, .3, 5000)
    args.output.mkdir(parents=True)
    records = []
    sheet = Image.new('RGB', (1024, 552), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        digest = sha(path)
        with Image.open(path) as original:
            assert original.width == original.height and original.width >= 1024
            dimensions = original.size
            converted = original.convert('L').convert('RGB') if args.monochrome else original.convert('RGB')
            rgb = np.array(converted.resize((256, 256), Image.Resampling.LANCZOS))
        _, faces = detector.detect(rgb[:, :, ::-1].copy())
        record = {'id': path.stem, 'source': str(path), 'source_sha256': digest,
                  'source_dimensions': dimensions, 'visual_approval': False,
                  'training_status': 'candidate_only_not_added_to_training'}
        if faces is None or len(faces) != 1:
            record['preprocessing_failure'] = 'Expected exactly one confident face'
            records.append(record)
            continue
        face = faces[0]
        points = np.asarray(face[4:14].reshape(5, 2).tolist(), dtype=np.float64)
        collar = namespace['collar_box'](rgb, face, points[:2].mean(axis=0), points[3:5].mean(axis=0))
        matrix = namespace['eyes42_transform'](points, collar)
        aligned = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT_101)
        coverage = cv2.warpAffine(np.ones((256, 256), dtype=np.uint8), matrix, (256, 256),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        target = args.output / path.name
        Image.fromarray(aligned).save(target)
        record.update(output=str(target), output_sha256=sha(target), landmarks=points.tolist(),
                      detector_confidence=float(face[-1]), collar_box_heuristic=collar,
                      collar_fallback_used=collar is None, matrix=matrix.tolist(),
                      detected_collar_fully_visible=namespace['contains_collar'](matrix, collar),
                      reflected_padding_fraction=float(1 - coverage.mean()),
                      metrics=namespace['metrics'](face, matrix))
        assert sha(path) == digest, 'Source changed during preprocessing'
        records.append(record)
        x, y = index % 4 * 256, index // 4 * 276
        sheet.paste(Image.fromarray(aligned), (x, y))
        draw.text((x + 4, y + 258), path.stem, fill='white')
    sheet.save(args.output / 'contact.png')
    report = {'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(helper),
              'recipe_sha256': sha(recipe), 'detector_sha256': sha(detector_path),
              'existing110_matrices_reproduced': True, 'active_dataset_changed': False,
              'color_preprocessing': 'Pillow L then RGB' if args.monochrome else 'original RGB',
              'opencv_version': cv2.__version__, 'entries': records,
              'quality_approved': False, 'production_approved': False}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'prepared': sum('output' in r for r in records)}))


if __name__ == '__main__':
    main()
