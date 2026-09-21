"""Download only pinned official distilled FLUX.2 klein 4B, with <=1GiB LFS parts."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models' / 'flux2-klein-4b'
MODEL = 'black-forest-labs/FLUX.2-klein-4B'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
PART_BYTES = 1024 ** 3
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
    sha = digest(path)
    expected = entry.get('lfs', {}).get('sha256')
    if expected and sha != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    if not expected and digest(path, git_blob=True) != entry['blobId']:
        raise ValueError(f'Git blob SHA1 mismatch: {path}')
    return sha


def split_weight(path):
    parts = []
    with path.open('rb') as source:
        index = 0
        while source.tell() < path.stat().st_size:
            size = min(PART_BYTES, path.stat().st_size - source.tell())
            remaining = size
            part = path.with_name(path.name + f'.weights-part-{index:03}')
            temporary = part.with_name(part.name + '.partial')
            h = hashlib.sha256()
            with temporary.open('wb') as target:
                while remaining:
                    data = source.read(min(BUFFER_BYTES, remaining))
                    if not data:
                        raise ValueError('Unexpected EOF while splitting')
                    target.write(data)
                    h.update(data)
                    remaining -= len(data)
            temporary.replace(part)
            parts.append({'path': str(part.relative_to(DEST)), 'bytes': size, 'sha256': h.hexdigest()})
            index += 1
    return parts


def fetch(entry):
    name = entry['rfilename']
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
    if not path.exists():
        temporary = path.with_name(path.name + '.partial')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as target:
            for data in iter(lambda: response.read(BUFFER_BYTES), b''):
                target.write(data)
        verify(temporary, entry)
        temporary.replace(path)
    sha = verify(path, entry)
    parts = split_weight(path) if path.suffix == '.safetensors' and entry['size'] > PART_BYTES else []
    print(f'Verified {name}: {entry["size"]} bytes, {len(parts)} LFS parts', flush=True)
    return {'path': name, 'bytes': entry['size'], 'sha256': sha, 'chunks': parts, 'source_url': url, 'upstream_entry': entry}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata_url = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Upstream revision mismatch')
    entries = [e for e in metadata['siblings'] if e['rfilename'] not in ('.gitattributes', 'flux-2-klein-4b.safetensors') and not e['rfilename'].endswith('.jpg')]
    total_bytes = sum(e['size'] for e in entries)
    extra_parts_bytes = sum(e['size'] for e in entries if e['rfilename'].endswith('.safetensors') and e['size'] > PART_BYTES)
    free_bytes = shutil.disk_usage(DEST).free
    needed_bytes = total_bytes + extra_parts_bytes + 20 * 1024 ** 3
    if free_bytes < needed_bytes:
        raise RuntimeError(f'Insufficient disk headroom: free={free_bytes}, required={needed_bytes}')
    ignores = ['# Reconstructed originals; pinned byte-identical parts are stored in Git LFS.']
    ignores += ['/' + e['rfilename'] for e in entries if e['rfilename'].endswith('.safetensors') and e['size'] > PART_BYTES]
    (DEST / '.gitignore').write_text('\n'.join(ignores) + '\n')
    (DEST / 'upstream-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (DEST / 'download-plan.json').write_text(json.dumps({'model': MODEL, 'revision': REVISION, 'metadata_url': metadata_url, 'file_count': len(entries), 'total_download_bytes': total_bytes, 'additional_part_bytes': extra_parts_bytes, 'free_bytes_before': free_bytes, 'required_with_20gib_headroom': needed_bytes, 'maximum_network_threads': 3, 'selected_files': [e['rfilename'] for e in entries]}, indent=2) + '\n')
    print(f'Downloading {len(entries)} files, {total_bytes} bytes; free disk {free_bytes} bytes.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, entries))
    manifest = {'complete': True, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': MODEL, 'revision': REVISION, 'license': metadata.get('cardData', {}).get('license'), 'checkpoint_kind': 'official distilled 4B transformer, not undistilled training base', 'files': files, 'inference_reference': {'pipeline': 'Flux2KleinPipeline', 'dtype': 'bfloat16', 'steps': 4, 'guidance_scale': 1.0, 'resolution': 1024}, 'oversized_weights': 'Originals exceeding1GiB are ignored in this model directory; concatenate manifest parts in order and verify full SHA256 before loading.', 'model_load_or_inference_performed': False}
    (DEST / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Download complete. Model has not been loaded or executed.', flush=True)


if __name__ == '__main__':
    main()
