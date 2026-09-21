"""Recorded manual overlay-review corrections; no model/image-content edits."""
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
    if (HERE / 'manifest-draft1.json').exists():
        raise SystemExit('Refusing to repeat recorded mask revision')
    manifest_path = HERE / 'manifest.json'
    (HERE / 'manifest-draft1.json').write_bytes(manifest_path.read_bytes())
    (HERE / 'provenance-draft1.json').write_bytes((HERE / 'provenance.json').read_bytes())
    manifest = json.loads(manifest_path.read_text())
    changes = {
        '000': [[[0, .838], [.181, .800], [.218, .724], [.238, .692],
                 [.254, .758], [.262, .835], [.281, .886], [.320, .935],
                 [.400, .966], [.490, .978], [.560, .975], [.640, .948],
                 [.702, .906], [.723, .839], [.733, .779], [.774, .825],
                 [1, .932], [1, 1], [0, 1]]],
        '030': [[[.160, 1], [.330, .895], [.367, .944], [.415, .972],
                 [.477, .982], [.550, .976], [.650, .944], [.750, .893],
                 [.830, .840], [.866, .806], [.889, .854], [.918, .891],
                 [1, .918], [1, 1]]]
    }
    notes = {'000': 'Cover narrow original white-shirt edges alongside neck; preserve lower chin.',
             '030': 'Exclude bottom-left x<.16 from clothing mask to protect inherited partial second person.'}
    for record in manifest['pairs']:
        pair_id = record['id']
        if pair_id not in changes:
            continue
        old = HERE / f'{pair_id}-mask.json'
        (HERE / f'{pair_id}-mask-draft1.json').write_bytes(old.read_bytes())
        mask = json.loads(old.read_text())
        mask['clothing_polygons'] = changes[pair_id]
        mask['revision_note'] = notes[pair_id]
        old.write_text(json.dumps(mask, indent=2) + '\n')
        record['clothing_polygons'] = changes[pair_id]
        process = subprocess.run([sys.executable, str(RESEARCH / 'experiments/paired_edit/mask_preview.py'),
            '--source', str((HERE / record['source_path']).resolve()),
            '--target', str((HERE / record['target_path']).resolve()), '--mask-json', str(old),
            '--output', str(HERE / f'{pair_id}-overlay-v2.png'), '--size', '768'],
            capture_output=True, text=True, check=True)
        (HERE / f'{pair_id}-overlay-v2.log').write_text(process.stdout + process.stderr)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    provenance_path = HERE / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance.update(manifest_sha256=digest(manifest_path), refine_masks_sha256=digest(Path(__file__)),
                      manual_refinement_notes=notes,
                      final_overlays={record['id']: f'{record["id"]}-overlay' + ('-v2' if record['id'] in changes else '') + '.png'
                                      for record in manifest['pairs']})
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
