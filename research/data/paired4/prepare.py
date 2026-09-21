"""Assemble existing pairs and hand-drawn masks; no model forward or image edit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / 'manifest.json').exists():
        raise SystemExit('Refusing to overwrite prepared candidate evidence')
    original = RESEARCH / 'data/ffhq-clothing-edit1/clothing-mask.json'
    original_mask = json.loads(original.read_text())['clothing_polygons']
    # Manual normalized coordinates from native source/target inspection.
    # These are approximate clothing boundaries, not automatic segmentation.
    masks = {
        'calibration-original': original_mask,
        '000': [[[0, .838], [.181, .800], [.218, .724], [.236, .692],
                 [.245, .795], [.265, .880], [.310, .929], [.390, .965],
                 [.485, .978], [.560, .975], [.640, .948], [.702, .906],
                 [.736, .845], [.748, .791], [.774, .825], [1, .932], [1, 1], [0, 1]]],
        '030': [[[0, 1], [.330, .895], [.367, .944], [.415, .972],
                 [.477, .982], [.550, .976], [.650, .944], [.750, .893],
                 [.830, .840], [.866, .806], [.889, .854], [.918, .891],
                 [1, .918], [1, 1]]],
        '028': [[[0, .995], [.226, .904], [.240, .866], [.255, .884],
                 [.300, .933], [.350, .963], [.430, .990], [.490, 1],
                 [.580, 1], [.645, .972], [.700, .921], [.734, .865],
                 [.746, .839], [.782, .881], [.803, .932], [1, 1], [0, 1]]]
    }
    records = []
    for pair_id in ['calibration-original', '000', '030', '028']:
        source_id = 'reproduction-sample002' if pair_id == 'calibration-original' else pair_id
        target = '../ffhq-clothing-edit1/001.png' if pair_id == 'calibration-original' else f'../ffhq-clothing-edits3/{pair_id}.png'
        record = {'id': pair_id, 'split': 'validation' if pair_id == '028' else 'train',
                  'latent_path': f'../ffhq-paired-sources32/latents/{source_id}.npz',
                  'source_path': f'../ffhq-paired-sources32/images/{source_id}.png',
                  'target_path': target, 'clothing_polygons': masks[pair_id]}
        records.append(record)
        mask_file = HERE / f'{pair_id}-mask.json'
        mask_file.write_text(json.dumps({'id': pair_id, 'split': record['split'],
            'coordinate_system': 'normalized x,y in unchanged square frames',
            'method': 'manual native-size source/edit inspection; approximate clothing/lower-neck boundary',
            'original_mask_sha256': digest(original) if pair_id == 'calibration-original' else None,
            'clothing_polygons': masks[pair_id]}, indent=2) + '\n')
        process = subprocess.run([sys.executable, str(RESEARCH / 'experiments/paired_edit/mask_preview.py'),
            '--source', str((HERE / record['source_path']).resolve()),
            '--target', str((HERE / record['target_path']).resolve()),
            '--mask-json', str(mask_file), '--output', str(HERE / f'{pair_id}-overlay.png'), '--size', '768'],
            capture_output=True, text=True, check=True)
        (HERE / f'{pair_id}-overlay.log').write_text(process.stdout + process.stderr)
    manifest = {'version': 1, 'purpose': 'Prospective fresh paired experiment: three training seeds, one held-out seed',
                'production_approved': False, 'candidate_pending_root_review': True,
                'source_bundle': '../../runs/inference-cpu/ffhq1024/baseline-bundle', 'pairs': records}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    provenance = {'manifest_sha256': digest(HERE / 'manifest.json'), 'prepare_sha256': digest(Path(__file__)),
                  'train_ids': ['calibration-original', '000', '030'], 'validation_ids': ['028'],
                  'source_weights_sha256': 'f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d',
                  'original_v2_mask_reused_exactly': masks['calibration-original'] == original_mask,
                  'original_v2_mask_file_sha256': digest(original),
                  'heavy_model_preflight_run': False, 'training_run': False,
                  'source_and_target_modified': False,
                  'files': {record['id']: {key: {'path': record[key], 'sha256': digest(HERE / record[key])}
                            for key in ['latent_path', 'source_path', 'target_path']} for record in records}}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'manifest': str(HERE / 'manifest.json'), 'train': 3, 'validation': 1}))


if __name__ == '__main__':
    main()
