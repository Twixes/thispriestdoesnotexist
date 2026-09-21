"""Derive the disjoint fixed evaluation manifest; NumPy/file hashes only, no models."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'ffhq-paired-sources32/manifest.json'
TRAINING = HERE / 'manifest-proposed-v2.json'
EXPECTED_TRAINING_SHA = '9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def z_digest(path):
    with np.load(path, allow_pickle=False) as latent:
        z = latent['z']
        if z.dtype != np.float32 or z.shape != (1, 512) or not np.isfinite(z).all():
            raise ValueError(f'Invalid latent: {path}')
        return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()


def main():
    if sha(TRAINING) != EXPECTED_TRAINING_SHA:
        raise ValueError('Expected the reviewed v2 training manifest; review a changed split explicitly')
    source = json.loads(SOURCE.read_text())
    training = json.loads(TRAINING.read_text())
    train_rows = [row for row in training['pairs'] if row['split'] == 'train']
    validation_rows = [row for row in training['pairs'] if row['split'] == 'validation']
    if len(train_rows) != 6 or [row['id'] for row in validation_rows] != ['028']:
        raise ValueError('Expected six train identities and held-out028')
    train_latents = {z_digest((HERE / row['latent_path']).resolve()) for row in train_rows}
    validation_latents = {z_digest((HERE / row['latent_path']).resolve()) for row in validation_rows}
    if len(train_latents) != 6 or train_latents & validation_latents:
        raise ValueError('Repeated train seed or validation overlap')
    entries, excluded = [], []
    for row in source['entries']:
        latent_path = (SOURCE.parent / row['latent_path']).resolve()
        image_path = (SOURCE.parent / row['source_path']).resolve()
        if sha(latent_path) != row['latents_sha256'] or sha(image_path) != row['image_sha256']:
            raise ValueError('Original source artifact changed')
        digest = z_digest(latent_path)
        if digest != row['z_raw_sha256']:
            raise ValueError('Recorded source Z digest mismatch')
        if digest in train_latents:
            excluded.append({'id': row['id'], 'reason': 'Exact Z used for training', 'z_raw_sha256': digest})
            continue
        item = copy.deepcopy(row)
        for field in ['image', 'source_path']:
            item[field] = os.path.relpath(image_path, HERE)
        for field in ['latents', 'latent_path']:
            item[field] = os.path.relpath(latent_path, HERE)
        entries.append(item)
    if len(entries) != 30 or [row['id'] for row in excluded] != ['000', '030']:
        raise ValueError('Unexpected evaluation split; expected30 after excluding000/030')
    if not validation_latents <= {row['z_raw_sha256'] for row in entries}:
        raise ValueError('Held-out028 missing')
    if len({row['z_raw_sha256'] for row in entries}) != 30:
        raise ValueError('Duplicate evaluation Z')
    retained = ['seed_derivation', 'latent_recipe', 'mapping_recipe', 'noise_mode', 'truncation_psi',
                'force_fp32', 'model', 'model_metadata_sha256', 'constant_noise_raw_hashes',
                'constant_noise_manifest_sha256', 'runtime']
    result = {key: copy.deepcopy(source[key]) for key in retained}
    result.update({'purpose': 'Disjoint source30 evaluation for reviewed paired7 v2; no new generated data',
                   'training_eligible': False, 'production_approved': False, 'is_priest_generator': False,
                   'count': 30, 'entries': entries, 'path_resolution': 'All entry paths relative to this manifest',
                   'selection_provenance': {'original_manifest_path': os.path.relpath(SOURCE, HERE),
                       'original_manifest_sha256': sha(SOURCE), 'training_manifest_path': TRAINING.name,
                       'training_manifest_sha256': sha(TRAINING), 'script_sha256': sha(Path(__file__)),
                       'excluded_training_entries': excluded, 'retained_validation_ids': ['028'],
                       'training_latent_sha256': sorted(train_latents),
                       'separate_original_training_seed_reproduction_omitted': True,
                       'no_model_inference': True}})
    output = HERE / 'evaluation-source30.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'sha256': sha(output), 'count': len(entries), 'excluded': excluded}))


if __name__ == '__main__':
    main()
