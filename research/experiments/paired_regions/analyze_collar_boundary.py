"""PNG-only diagnosis of collar-adjacent error; no model changes or inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from compare_baseline import gray, mask

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/runs/paired-regions1024-cpu-200/boundary-diagnosis.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve existing diagnostic results')
    manifest_path = ROOT / 'research/data/paired4-tabs/single-pair.json'
    pair = json.loads(manifest_path.read_text())['pairs'][0]
    clothing = mask(pair['clothing_polygons'])
    raw_tab = mask(pair['collar_polygons'])
    tab = raw_tab & clothing
    paths = {
        'source': (manifest_path.parent / pair['source_path']).resolve(),
        'target': (manifest_path.parent / pair['target_path']).resolve(),
        'broad200': ROOT / 'research/runs/paired1024-cpu-200/000200-pair-000-train.png',
        'balanced200': ROOT / 'research/runs/paired-regions1024-cpu-200/000200-pair-000-train.png',
    }
    planes = {name: gray(path) for name, path in paths.items()}
    records = []
    for radius in (5, 15, 30):
        dilated = np.asarray(Image.fromarray(raw_tab.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(2*radius+1))) > 0
        ring = dilated & ~raw_tab & clothing
        if not ring.any():
            raise ValueError('Empty collar exterior ring')
        records.append({'radius_px': radius, 'shape':'square/Chebyshev dilation', 'pixels':int(ring.sum()),
            'outside_clothing_pixels':int((ring & ~clothing).sum()),
            'mean_luminance_uint8':{name:float(pixels[ring].mean()) for name,pixels in planes.items()},
            'mae_to_target_uint8':{name:float(np.abs(pixels[ring]-planes['target'][ring]).mean()) for name,pixels in planes.items()}})
    report = {'production_approved':False, 'model_inference_performed':False,
              'scope':'Single training identity at step200. Three preselected exterior collar bands clipped to existing clothing protection; no face pixels reclassified. Target remains imagegen reference.',
              'limitations':'Post-update clipped PNG grayscale/resizing. Bands are manually traced geometry, not semantic collar ground truth; errors do not prove a causal loss fix or generalization.',
              'tab_pixels':int(tab.sum()), 'rings':records,
              'inputs':{name:{'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()},
              'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256':hashlib.sha256((Path(__file__).parent/'compare_baseline.py').read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
