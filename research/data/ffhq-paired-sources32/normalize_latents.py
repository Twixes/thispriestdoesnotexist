"""Normalize initial NPZ names to trainer z/w schema without model or image changes."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = HERE / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
original_manifest_sha = sha(manifest_path)
changes = []
for row in manifest['entries'] + [manifest['reproduction']]:
    path = HERE / row['latents']
    old_hash = sha(path)
    assert old_hash == row['latents_sha256']
    with np.load(path, allow_pickle=False) as arrays:
        z = arrays['z'].copy()
        w = arrays['ws'].copy()
    assert z.shape == (1, 512) and w.shape == (1, 18, 512)
    assert hashlib.sha256(z.tobytes()).hexdigest() == row['z_raw_sha256']
    assert hashlib.sha256(w.tobytes()).hexdigest() == row['ws_raw_sha256']
    assert sha(HERE / row['image']) == row['image_sha256']
    np.savez(path, z=z, w=w)
    row['latents_sha256'] = sha(path)
    row['w_shape'] = list(w.shape)
    row['source_path'] = row['image']
    row['latent_path'] = row['latents']
    changes.append({'id': row['id'], 'old_npz_sha256': old_hash, 'new_npz_sha256': sha(path)})
manifest['mapping_recipe'] = 'Capture actual mapping forward output during exact generator forward; NPZ w preserves all synthesis style inputs [1,18,512]'
manifest['path_resolution'] = 'source_path and latent_path are relative to this manifest directory'
manifest['schema_normalization'] = {
    'script_sha256': sha(Path(__file__)), 'original_manifest_sha256': original_manifest_sha,
    'actual_generation_script': 'generate-executed-v1.py',
    'actual_generation_script_sha256': sha(HERE / 'generate-executed-v1.py'),
    'future_generation_script': 'generate.py', 'future_generation_script_sha256': sha(HERE / 'generate.py'),
    'explanation': 'Renamed previously captured full ws array to trainer key w; unchanged z/W tensor hashes and PNG hashes verified. No new model forwards.',
    'changes': changes,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print('Normalized and verified 33 exact captured latent files, no image changes')
