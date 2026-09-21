"""Verify/reconstruct and load the pinned local full-UNet SDXL-Lightning 2-step model.

This file does not perform inference at import time. The two-step scheduler is
part of the model definition; one-step or four-step calls are not equivalent.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'sdxl-lightning-2step'
BUFFER_BYTES = 8 * 1024 ** 2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(BUFFER_BYTES), b''):
            h.update(data)
    return h.hexdigest()


def verify_and_reconstruct(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'provenance.json').read_text())
    if not manifest['complete'] or manifest['revision'] != 'c9a24f48e1c025556787b0c58dd67a091ece2e44':
        raise ValueError('Unexpected or incomplete assembly')
    for entry in manifest['files']:
        target = model_dir / entry['path']
        if not target.exists() and entry['chunks']:
            temporary = target.with_name(target.name + '.partial')
            try:
                with temporary.open('wb') as output:
                    for chunk in entry['chunks']:
                        part = model_dir / chunk['path']
                        if part.stat().st_size != chunk['bytes'] or digest(part) != chunk['sha256']:
                            raise ValueError(f'Invalid weight part: {part}')
                        with part.open('rb') as source:
                            for data in iter(lambda: source.read(BUFFER_BYTES), b''):
                                output.write(data)
                if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                    raise ValueError('Reassembled weight does not match upstream SHA256')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
            raise ValueError(f'Assembly file mismatch: {target}')
    return manifest


def load_pipeline(device='cpu', dtype=None, model_dir=MODEL_DIR):
    """Root caller chooses dtype/device and owns resource limits; no generation here."""
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    verify_and_reconstruct(model_dir)
    if dtype is None:
        dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        variant='fp16',
        local_files_only=True,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing='trailing',
        prediction_type='epsilon',
    )
    return pipe.to(device)


if __name__ == '__main__':
    manifest = verify_and_reconstruct()
    print(json.dumps({'complete': True, 'verified_files': len(manifest['files']), 'model_loaded': False}))
