"""Measure collar surrounds in completed paired PNGs; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from compare_baseline import mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior diagnostics; use a new output path')
    report_path = args.evaluation / 'evaluation.json'
    evaluation = json.loads(report_path.read_text())
    manifest_path = Path(evaluation['manifest_path'])
    if sha(manifest_path) != evaluation['manifest_sha256']:
        raise ValueError('Evaluated manifest changed')
    pairs = {p['id']: p for p in json.loads(manifest_path.read_text())['pairs']}
    records = []
    for entry in evaluation['entries']:
        pair = pairs[entry['id']]
        if pair['split'] != entry['split']:
            raise ValueError('Split mismatch')
        if not pair.get('collar_polygons'):
            continue
        planes, inputs = {}, {}
        for name in ['source-gray', 'student-gray', 'target-gray']:
            artifact = entry['outputs'][name]
            path = args.evaluation / artifact['path']
            if sha(path) != artifact['sha256']:
                raise ValueError('Evaluated image changed')
            with Image.open(path) as im:
                if im.size != (1024, 1024) or im.mode != 'L':
                    raise ValueError('Expected native grayscale evaluation PNG')
                planes[name] = np.array(im, dtype=np.float32)
            inputs[name] = artifact
        clothing = mask(pair['clothing_polygons'])
        raw_tab = mask(pair['collar_polygons'])
        rings = []
        for radius in [5, 15, 30]:
            dilated = np.asarray(Image.fromarray(raw_tab.astype(np.uint8) * 255)
                                 .filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0
            ring = dilated & ~raw_tab & clothing
            if not ring.any():
                raise ValueError('Empty collar exterior ring')
            rings.append({'radius_px': radius, 'pixels': int(ring.sum()),
                          'protected_pixels': int((ring & ~clothing).sum()),
                          'mean_luminance_uint8': {n: float(p[ring].mean()) for n, p in planes.items()},
                          'target_mae_uint8': {n: float(np.abs(p[ring] - planes['target-gray'][ring]).mean())
                                               for n, p in planes.items()}})
        records.append({'id': entry['id'], 'split': entry['split'], 'rings': rings, 'inputs': inputs})
    if not records:
        raise ValueError('No annotated pairs')
    report = {'checkpoint_step': evaluation['checkpoint_step'],
              'checkpoint_sha256': evaluation['checkpoint_sha256'],
              'evaluation_sha256': sha(report_path), 'manifest_sha256': sha(manifest_path),
              'script_sha256': sha(Path(__file__)),
              'mask_helper_sha256': sha(Path(__file__).with_name('compare_baseline.py')),
              'model_inference_performed': False, 'production_approved': False,
              'method': 'Square dilation outside raw traced tab, intersected with original clothing; radii5/15/30px.',
              'limitations': 'Clipped post-update grayscale PNGs, not unclipped training tensors. '
                             'Manually traced boundaries may be inset or imperfect. '
                             'A ring may include bright collar antialiasing; target means describe actual pixels. '
                             'Training reconstruction errors do not establish unseen generalization or a causal fix.',
              'records': records}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ['id', 'split', 'rings']} for r in records], indent=2))


if __name__ == '__main__':
    main()
