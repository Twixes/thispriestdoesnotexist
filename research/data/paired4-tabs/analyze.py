"""PNG-only white-tab audit and proposed manifest; no generator inference."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
SIZE = 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask(polygons):
    image = Image.new('L', (SIZE, SIZE), 0)
    drawing = ImageDraw.Draw(image)
    for polygon in polygons:
        drawing.polygon([tuple(round(v * (SIZE - 1)) for v in point) for point in polygon], fill=255)
    return np.asarray(image) > 0


def pixels(path):
    with Image.open(path) as image:
        if image.width != image.height:
            raise ValueError('Square inputs required, no crop allowed')
        rgb = np.asarray(image.convert('RGB').resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float32)
    return (rgb * np.array([.299, .587, .114], dtype=np.float32)).sum(2)


def main():
    parent = RESEARCH / 'data/paired4/manifest.json'
    original = json.loads(parent.read_text())
    proposed = copy.deepcopy(original)
    # Reject the existing version-1 trainer rather than silently ignore a new loss.
    proposed['version'] = 2
    proposed['purpose'] = 'PROPOSED optional equal-area collar objective; unsupported by the current v1 trainer'
    proposed['requires_trainer_feature'] = 'equal_area_collar_partition_v1'
    proposed['optional_objective'] = '0.5*MAE(collar intersection clothing)+0.5*MAE(rest clothing); protected/fresh losses unchanged'
    traces = {
        'calibration-original': [[[.370, .923], [.484, .931], [.472, .986], [.452, .993], [.400, .994], [.357, .987]]],
        '000': [[[.437, .948], [.490, .956], [.539, .955], [.587, .946], [.588, 1], [.435, 1]]],
        '030': [[[.505, .966], [.563, .967], [.629, .955], [.635, 1], [.509, 1]]]
    }
    measurements, regions = [], {}
    diagnostics = Image.new('RGB', (3 * 512, 540), '#111111')
    training = [item for item in proposed['pairs'] if item['split'] == 'train']
    for index, record in enumerate(training):
        pair_id = record['id']
        record['collar_polygons'] = traces[pair_id]
        clothing = mask(record['clothing_polygons'])
        traced = mask(record['collar_polygons'])
        collar = traced & clothing
        rest = clothing & ~collar
        assert collar.any() and rest.any() and not np.any(collar & ~clothing)
        regions[pair_id] = (clothing, traced, collar, rest)
        sizes = {'id': pair_id, 'clothing_pixels': int(clothing.sum()), 'traced_tab_pixels': int(traced.sum()),
                 'effective_tab_pixels': int(collar.sum()), 'rest_clothing_pixels': int(rest.sum()),
                 'tab_fraction_of_clothing': float(collar.sum() / clothing.sum()),
                 'trace_retained_inside_unchanged_clothing_mask': float(collar.sum() / traced.sum()),
                 'equal_partition_tab_per_pixel_multiplier_vs_broad': float(.5 * clothing.sum() / collar.sum()),
                 'equal_partition_rest_per_pixel_multiplier_vs_broad': float(.5 * clothing.sum() / rest.sum())}
        measurements.append(sizes)
        target_path = (HERE / record['target_path']).resolve()
        with Image.open(target_path) as image:
            image = image.convert('RGBA').resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        layer = Image.new('RGBA', (SIZE, SIZE), (255, 210, 0, 0))
        layer.putalpha(Image.fromarray(collar.astype(np.uint8) * 100))
        image = Image.alpha_composite(image, layer)
        draw = ImageDraw.Draw(image)
        for polygon in record['clothing_polygons']:
            points = [tuple(round(v * (SIZE - 1)) for v in point) for point in polygon]
            draw.line(points + [points[0]], fill=(0, 220, 255, 255), width=3)
        for polygon in record['collar_polygons']:
            points = [tuple(round(v * (SIZE - 1)) for v in point) for point in polygon]
            draw.line(points + [points[0]], fill=(0, 255, 80, 255), width=3)
        image.convert('RGB').save(HERE / f'{pair_id}-tab-overlay.png')
        diagnostics.paste(image.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), (index * 512, 28))
        ImageDraw.Draw(diagnostics).text((index * 512 + 8, 8), f'{pair_id}: cyan clothing / green tab / yellow intersection', fill='white')
    diagnostics.save(HERE / 'tab-contact-sheet.png')
    # Existing split, face/clothing masks, paths and held-out entry are immutable.
    for before, after in zip(original['pairs'], proposed['pairs']):
        assert {k: v for k, v in after.items() if k != 'collar_polygons'} == before
    assert proposed['pairs'][-1] == original['pairs'][-1]
    (HERE / 'manifest-proposed.json').write_text(json.dumps(proposed, indent=2) + '\n')
    (HERE / 'tab-regions.json').write_text(json.dumps({'coordinate_system': 'manual normalized x,y target white-tab traces',
        'effective_mask': 'intersection with unchanged clothing mask; never extend into protected face',
        'polygons': traces, 'measurements': measurements}, indent=2) + '\n')
    original_pair = original['pairs'][0]
    target_path = (HERE / original_pair['target_path']).resolve()
    target = pixels(target_path)
    clothing, traced, collar, rest = regions['calibration-original']
    samples, missing = [], []
    for step in [150, 175, 200]:
        path = RESEARCH / f'runs/paired1024-cpu-200/{step:06}-pair-000-train.png'
        if not path.exists():
            missing.append(step)
            continue
        generated = pixels(path)
        error = np.abs(generated - target)
        areas = {}
        for name, region in [('full_traced_tab', traced), ('effective_tab', collar), ('rest_clothing', rest), ('broad_clothing', clothing)]:
            areas[name] = {'mae_uint8': float(error[region].mean()), 'mae_on_minus1_plus1_scale': float(error[region].mean() / 127.5),
                           'target_mean_uint8': float(target[region].mean()), 'generated_mean_uint8': float(generated[region].mean())}
        samples.append({'step': step, 'path': str(path.relative_to(RESEARCH.parent)), 'sha256': digest(path), 'regions': areas,
                        'effective_tab_fraction_of_broad_absolute_error': float(error[collar].sum() / error[clothing].sum()),
                        'effective_tab_contribution_to_broad_mae_uint8': float(error[collar].sum() / clothing.sum()),
                        'rest_contribution_to_broad_mae_uint8': float(error[rest].sum() / clothing.sum()),
                        'proposed_equal_partition_mae_uint8': .5 * (areas['effective_tab']['mae_uint8'] + areas['rest_clothing']['mae_uint8'])})
    result = {'method': 'PNG-only RGB->LANCZOS1024->fixed .299/.587/.114 grayscale; absolute error in 0..255',
              'not_exact_training_tensor_loss': 'Rendered previews are post-update, uint8-clipped; training uses unclipped float tensors before each update.',
              'model_inference_performed': False, 'training_performed': False,
              'original_manifest_unchanged_sha256': digest(parent), 'analysis_sha256': digest(Path(__file__)),
              'target_sha256': digest(target_path), 'region_definitions': measurements,
              'samples': samples, 'missing_steps': missing,
              'proposed_manifest_sha256': digest(HERE / 'manifest-proposed.json'),
              'original_clothing_masks_and_heldout_record_unchanged': True}
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
