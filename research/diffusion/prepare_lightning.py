"""Assemble pinned official SDXL-Lightning 2-step components without loading a model."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models' / 'sdxl-lightning-2step'
BASE = ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b')
LIGHTNING = ('ByteDance/SDXL-Lightning', 'c9a24f48e1c025556787b0c58dd67a091ece2e44')
CHUNK_BYTES = 1024 ** 3
BUFFER_BYTES = 8 * 1024 ** 2


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(BUFFER_BYTES), b''):
            h.update(data)
    return h.hexdigest()


def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Size mismatch: {path}')
    expected = entry.get('lfs', {}).get('sha256')
    actual = digest(path)
    if expected:
        if actual != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
    elif digest(path, git_blob=True) != entry['blobId']:
        raise ValueError(f'Git blob SHA1 mismatch: {path}')
    return actual


def split_file(path):
    chunks = []
    with path.open('rb') as source:
        index = 0
        while source.tell() < path.stat().st_size:
            part = path.with_name(path.name + f'.weights-part-{index:03}')
            temporary = part.with_name(part.name + '.partial')
            remaining = min(CHUNK_BYTES, path.stat().st_size - source.tell())
            h = hashlib.sha256()
            size = remaining
            with temporary.open('wb') as target:
                while remaining:
                    data = source.read(min(BUFFER_BYTES, remaining))
                    if not data:
                        raise ValueError('Unexpected EOF while splitting')
                    target.write(data)
                    h.update(data)
                    remaining -= len(data)
            temporary.replace(part)
            chunks.append({'path': str(part.relative_to(DEST)), 'bytes': size, 'sha256': h.hexdigest()})
            index += 1
    return chunks


def fetch(spec):
    repo, revision, entry, target_name = spec
    target = DEST / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['rfilename']}"
    reuse = None
    if not target.exists():
        expected = entry.get('lfs', {}).get('sha256')
        if expected:
            for family in ('sdxl-turbo', 'sdxl-inpaint', 'sd-turbo'):
                source = ROOT / 'models' / family / entry['rfilename']
                if source.exists() and source.stat().st_size == entry['size'] and digest(source) == expected:
                    # Copy instead of linking so later experiments cannot mutate another model.
                    shutil.copyfile(source, target)
                    reuse = {'path': str(source.relative_to(ROOT)), 'method': 'copy', 'verified_upstream_sha256': expected}
                    break
        if not target.exists():
            temporary = target.with_name(target.name + '.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                for data in iter(lambda: response.read(BUFFER_BYTES), b''):
                    output.write(data)
            verify(temporary, entry)
            temporary.replace(target)
    sha = verify(target, entry)
    chunks = split_file(target) if target.stat().st_size > 2 * 1024 ** 3 else []
    print(f'Verified {target_name}: {target.stat().st_size} bytes' + (' (reused)' if reuse else ''), flush=True)
    return {'path': target_name, 'bytes': target.stat().st_size, 'sha256': sha, 'chunks': chunks, 'source': {'model': repo, 'revision': revision, 'path': entry['rfilename'], 'url': url, 'upstream_entry': entry}, 'local_reuse': reuse}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, (repo, revision) in [('base', BASE), ('lightning', LIGHTNING)]:
        url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        if data['sha'] != revision:
            raise ValueError('Revision mismatch')
        metadata[name] = data
        (DEST / f'{name}-upstream-metadata.json').write_text(json.dumps(data, indent=2) + '\n')
    specs = []
    component_dirs = ('scheduler/', 'tokenizer/', 'tokenizer_2/', 'text_encoder/', 'text_encoder_2/', 'vae/')
    for entry in metadata['base']['siblings']:
        name = entry['rfilename']
        needed = name in ('model_index.json', 'unet/config.json', 'LICENSE.md', 'README.md') or (name.startswith(component_dirs) and (name.endswith(('.json', '.txt')) or name.endswith('.fp16.safetensors')))
        if needed:
            specs.append((*BASE, entry, 'sources/base-' + name if name in ('LICENSE.md', 'README.md') else name))
    for entry in metadata['lightning']['siblings']:
        name = entry['rfilename']
        if name == 'sdxl_lightning_2step_unet.safetensors':
            specs.append((*LIGHTNING, entry, 'unet/diffusion_pytorch_model.fp16.safetensors'))
        elif name in ('LICENSE.md', 'README.md'):
            specs.append((*LIGHTNING, entry, 'sources/lightning-' + name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, specs))
    weight = DEST / 'unet/diffusion_pytorch_model.fp16.safetensors'
    with weight.open('rb') as stream:
        header_length = int.from_bytes(stream.read(8), 'little')
        if header_length > 10 * 1024 ** 2:
            raise ValueError('Unreasonable safetensors header')
        header = json.loads(stream.read(header_length))
    tensors = [value for key, value in header.items() if key != '__metadata__']
    dtypes = sorted({value['dtype'] for value in tensors})
    manifest = {
        'complete': True,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': LIGHTNING[0], 'revision': LIGHTNING[1],
        'base_model': BASE[0], 'base_revision': BASE[1],
        'checkpoint_kind': 'official full UNet, not LoRA',
        'checkpoint_name': 'sdxl_lightning_2step_unet.safetensors',
        'inference': {'steps': 2, 'guidance_scale': 0, 'scheduler': 'EulerDiscreteScheduler', 'timestep_spacing': 'trailing', 'prediction_type': 'epsilon', 'native_resolution': 1024},
        'license': 'openrail++', 'files': files,
        'unet_header': {'tensor_count': len(tensors), 'dtypes': dtypes, 'header_bytes': header_length},
        'oversized_weights': 'Concatenate chunks in listed order and verify full SHA256 before loading; reconstructed full UNet is ignored by research/diffusion/.gitignore.',
        'model_load_or_inference_performed': False,
    }
    if dtypes != ['F16']:
        raise ValueError(f'Unexpected UNet dtypes: {dtypes}')
    (DEST / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Assembly complete; model has not been loaded or executed.', flush=True)


if __name__ == '__main__':
    main()
