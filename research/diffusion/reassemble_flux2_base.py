"""Verify/reconstruct official undistilled klein base4B files; no torch import or inference."""
import hashlib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'flux2-klein-base-4b'
MODEL = 'black-forest-labs/FLUX.2-klein-base-4B'
REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
BUFFER_BYTES = 8 * 1024 ** 2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(BUFFER_BYTES), b''):
            h.update(data)
    return h.hexdigest()


def verified_model_dir(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'provenance.json').read_text())
    if not manifest['complete'] or manifest['model'] != MODEL or manifest['revision'] != REVISION:
        raise ValueError('Unexpected or incomplete model assembly')
    for entry in manifest['files']:
        target = model_dir / entry['path']
        if not target.exists() and entry['chunks']:
            temporary = target.with_name(target.name + '.partial')
            try:
                with temporary.open('wb') as output:
                    for chunk in entry['chunks']:
                        part = model_dir / chunk['path']
                        if part.stat().st_size != chunk['bytes'] or digest(part) != chunk['sha256']:
                            raise ValueError(f'Weight part mismatch: {part}')
                        with part.open('rb') as source:
                            for data in iter(lambda: source.read(BUFFER_BYTES), b''):
                                output.write(data)
                if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                    raise ValueError(f'Reassembled weight mismatch: {target}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
            raise ValueError(f'Model file mismatch: {target}')
    return model_dir


if __name__ == '__main__':
    print(json.dumps({'verified_model_dir': str(verified_model_dir()), 'model_loaded': False}))
