"""Reconstruct local inference weights from the byte-exact Git LFS chunks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(data)
    return h.hexdigest()


def restore(model_dir):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'provenance.json').read_text())
    for entry in manifest['files']:
        path = model_dir / entry['path']
        if not path.exists() and entry.get('chunks'):
            temporary = path.with_name(path.name + '.partial')
            with temporary.open('wb') as out:
                for chunk in entry['chunks']:
                    part = model_dir / chunk['path']
                    if part.stat().st_size != chunk['bytes'] or digest(part) != chunk['sha256']:
                        raise ValueError(f'Invalid LFS part: {part}')
                    with part.open('rb') as stream:
                        shutil.copyfileobj(stream, out, length=8 * 1024 ** 2)
            if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                raise ValueError(f'Reconstructed weight mismatch: {path}')
            temporary.replace(path)
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Weight/config mismatch: {path}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps({'verified_files': len(restore(args.model_dir)['files'])}))
