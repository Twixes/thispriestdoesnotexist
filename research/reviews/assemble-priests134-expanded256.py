#!/usr/bin/env python3
"""Assemble a prospective dataset fork using byte-exact image copies only."""
import argparse
import datetime
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import PIL
from PIL import Image, ImageChops


ALIGNED_DIGEST = '3a6e258a9b9ddae8d415b70c3be540ff39cf44e8a37d80ab71540058f1ee0553'
WHOLE_MANIFEST_SHA256 = '334ad5bfa22c72baab1d7a24298616483d125a9fccbc3a7816b1b3b2ef59067d'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trainer_digest(directory):
    # Exact algorithm read from archived reference trainer lines200–206.
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*.png')):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def image_properties(path):
    with Image.open(path) as image:
        image.load()
        if image.size != (256, 256) or image.mode != 'RGB':
            raise ValueError(f'Expected RGB256x256: {path}, {image.mode}, {image.size}')
        red, green, blue = image.split()
        return {'mode': image.mode, 'dimensions': list(image.size),
                'exact_grayscale_channels':
                ImageChops.difference(red, green).getbbox() is None and
                ImageChops.difference(red, blue).getbbox() is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('research/data/priests134-expanded256'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise SystemExit('Refusing to overwrite an existing dataset directory.')
    aligned = root / 'research/alignment/collar-only/eyes42'
    whole = root / 'research/data/plain-background24-whole-image'
    archived_trainer = root / 'research/runs/reference256-paper-b64/train-source.py'
    assert trainer_digest(aligned) == ALIGNED_DIGEST, 'Original110 digest mismatch'
    assert sha256(whole / 'manifest.json') == WHOLE_MANIFEST_SHA256
    whole_manifest = json.loads((whole / 'manifest.json').read_text())
    old_files = sorted(aligned.rglob('*.png'))
    assert len(old_files) == 110
    proposed = [(source, 'aligned110--' + source.name, 'aligned110', None)
                for source in old_files]
    new_records = sorted(whole_manifest['entries'], key=lambda record: record['id'])
    assert [record['id'] for record in new_records] == list(range(141, 165))
    for record in new_records:
        source = whole / record['outputs']['256']['path']
        assert sha256(source) == record['outputs']['256']['sha256']
        proposed.append((source, f"whole24--{record['id']}.png", 'whole24', record))
    assert len(proposed) == len({name for _, name, _, _ in proposed}) == 134
    before = {str(source.relative_to(root)): sha256(source) for source, _, _, _ in proposed}
    # Validate all inputs before creating the destination; never decode/re-encode copies.
    for source, _, group, _ in proposed:
        properties = image_properties(source)
        if group == 'whole24':
            assert properties['exact_grayscale_channels']
    output.mkdir(parents=True)
    entries = []
    for source, filename, group, original_record in proposed:
        destination = output / filename
        shutil.copyfile(source, destination)
        properties = image_properties(destination)
        relative = str(source.relative_to(root))
        assert sha256(destination) == before[relative]
        entry = {'group': group, 'source_path': relative,
                 'source_sha256': before[relative], 'output_path': filename,
                 'output_sha256': sha256(destination), 'byte_exact_copy': True,
                 **properties}
        if original_record is not None:
            entry['original_generation_path'] = original_record['source_path']
            entry['original_generation_sha256'] = original_record['source_sha256']
            entry['generated_id'] = original_record['id']
        entries.append(entry)
    after = {str(source.relative_to(root)): sha256(source) for source, _, _, _ in proposed}
    assert before == after
    assert trainer_digest(aligned) == ALIGNED_DIGEST
    assert sorted(path.name for path in output.glob('*.png')) == sorted(e['output_path'] for e in entries)
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'prospective_expanded_dataset_fork_input_only',
        'accepted_training_dataset': False, 'training_launched': False,
        'same_dataset_continuation': False, 'model_output': False,
        'count': 134, 'groups': {'aligned110': 110, 'whole24': 24},
        'assembly': 'Byte-exact shutil.copyfile only; no decoding/re-encoding, grayscale conversion, resize or geometry change during this assembly.',
        'geometric_mixture': '110 previously accepted collar-only eyes42 aligned crops, unchanged, plus24 whole-original grayscale/LANCZOS256 portraits with unnormalized source eye positions, head sizes and tilts. This is an intentional changed-data fork candidate, not an exact same-dataset continuation.',
        'source_digest_algorithm': 'SHA256 of sorted recursive PNG relative-path UTF-8 bytes followed by each PNG file bytes, repeated without separators; matches archived reference trainer lines200–206.',
        'original110_dataset_sha256': ALIGNED_DIGEST,
        'original110_digest_verified_before_and_after': True,
        'expanded_dataset_sha256': trainer_digest(output),
        'digest_algorithm_source': str(archived_trainer.relative_to(root)),
        'digest_algorithm_source_sha256': sha256(archived_trainer),
        'whole24_manifest_path': str((whole / 'manifest.json').relative_to(root)),
        'whole24_manifest_sha256': WHOLE_MANIFEST_SHA256,
        'all134_source_hashes_unchanged_before_after': before == after,
        'source_hashes_after': after,
        'all134_outputs_inspected_for_shape_mode_and_hash': True,
        'new24_exact_grayscale_channels_verified': True,
        'filename_policy': 'aligned110--<original filename> and whole24--<generated ID>.png; collision-checked. Dataset digest necessarily includes these new names.',
        'script_path': str(Path(__file__).resolve().relative_to(root)),
        'script_sha256': sha256(Path(__file__)),
        'versions': {'python': sys.version, 'pillow': PIL.__version__,
                     'platform': platform.platform()},
        'entries': entries,
        'limitations': ['No model loading, training, held-out scoring or quality claim.',
                       'No image filtering or automatic duplicate removal.',
                       'This candidate changes image count, filenames, digest and framing distribution; any use must be an explicitly declared dataset fork.'],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'count': len(entries),
                      'original110_digest': ALIGNED_DIGEST,
                      'expanded_dataset_sha256': manifest['expanded_dataset_sha256'],
                      'manifest_sha256': sha256(output / 'manifest.json')}))


if __name__ == '__main__':
    main()
