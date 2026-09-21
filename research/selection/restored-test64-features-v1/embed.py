"""CPU1 CLIP features for every attempted image in completed GFPGAN diagnostics."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIMIT_RSS = 6 * 1024**3
LIMIT_SECONDS = 600
ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def guard(deadline):
    require(time.time() < deadline, 'Ten-minute deadline exceeded')
    require(peak_rss() <= LIMIT_RSS, '6GiB worker peak RSS exceeded')


def verify_pins():
    pins = read(HERE / 'embed-pins.json')
    require(platform.python_version() == pins['python_version'] and platform.machine() == pins['machine'],
            'Python/architecture differs from pinned selection environment')
    for name, version in pins['packages'].items():
        require(importlib.metadata.version(name) == version, f'Package version differs: {name}')
    for name, digest in pins['files_sha256'].items():
        require(sha(ROOT / name) == digest, f'Pinned source/weight changed: {name}')
    vendor = ROOT / pins['vendor']
    actual = {str(p.relative_to(ROOT)) for p in (vendor/'clip').rglob('*')
              if p.is_file() and p.suffix in {'.py', '.gz'}}
    expected = {name for name in pins['files_sha256'] if name.startswith(pins['vendor']+'/')
                and Path(name).suffix in {'.py', '.gz'}}
    require(actual == expected, 'CLIP source inventory changed')
    return pins


def inventory(inputs):
    """Completed cohorts only. Failed restorations remain rows, never dropped."""
    cohorts, rows = [], []
    require(len(set(inputs)) == len(inputs), 'Duplicate input directory')
    for cohort_number, directory in enumerate(inputs):
        result, supervisor = read(directory/'result.json'), read(directory/'supervisor-result.json')
        require(result['complete'] and supervisor['complete'] and supervisor['worker_exit_code'] == 0
                and supervisor.get('error') is None, f'Incomplete GFPGAN cohort: {directory}')
        records = result['records']
        require(len(records) == result['attempted'], 'Attempted count differs from records')
        require(len({entry['index'] for entry in records}) == len(records), 'Duplicate cohort index')
        errors = sum(entry['error'] is not None for entry in records)
        require(errors == result['errors'] and len(records)-errors == result['successful_restorations'],
                'Restoration denominator/counters disagree')
        bound = {name: sha(directory/name) for name in ['result.json', 'supervisor-result.json', 'inputs.json']}
        cohorts.append({'number': cohort_number, 'directory': str(directory), 'cohort': result['cohort'],
                        'attempted': len(records), 'file_sha256': bound})
        for entry in records:
            index = entry['index']
            require(type(index) is int and index >= 0, 'Invalid restoration index')
            folder = directory/f'{index:03}'
            require(read(folder/'record.json') == entry, 'Per-image record differs from completed result')
            row = {'feature_row': len(rows), 'cohort_number': cohort_number, 'index': index,
                   'input_directory': str(directory), 'record_sha256': sha(folder/'record.json'),
                   'original_source_path': entry['path'], 'original_source_sha256': entry['sha256'],
                   'restoration_error': entry['error'], 'webp_path': None, 'webp_sha256': None}
            if entry['error'] is None:
                file = folder/'restored-1024.webp'
                digest = entry['outputs']['restored-1024.webp']
                require(sha(file) == digest, f'Final WebP changed/missing: {file}')
                row.update(webp_path=str(file), webp_sha256=digest)
            rows.append(row)
    return {'cohorts': cohorts, 'rows': rows}


def benchmark_module(pins):
    spec = importlib.util.spec_from_file_location('fixed_clip_prompt_definitions', ROOT/pins['benchmark'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    launch = read(args.output/'launch.json')
    require(os.getppid() == launch['supervisor_pid'] and args.nonce == launch['nonce'], 'Worker ownership mismatch')
    require(sha(Path(__file__)) == launch['source_sha256'], 'Worker source changed')
    require(any(args.input) and [str(p) for p in args.input] == launch['inputs'], 'Input arguments differ')
    require(all(os.environ.get(k) == v for k, v in ENV.items()), 'CPU thread environment mismatch')
    deadline = launch['deadline_unix']
    guard(deadline)
    pins = verify_pins()
    require(sha(HERE/'embed-pins.json') == launch['pins_sha256'], 'Pins changed after launch')
    data = inventory(args.input)
    require(data == launch['inventory'], 'Input provenance changed after launch')
    benchmark = benchmark_module(pins)
    prompts_by_category = benchmark.PROMPTS
    require(all(len(v) == 3 for v in prompts_by_category.values()), 'Fixed prompt grouping changed')
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT/pins['vendor']))
    import clip
    require(Path(clip.__file__).resolve().is_relative_to(ROOT/pins['vendor']), 'Wrong CLIP import')
    load_started = time.monotonic()
    model, preprocess = clip.load(str(ROOT/pins['weights']), device='cpu', jit=False)
    model.float().eval().requires_grad_(False)
    load_seconds = time.monotonic()-load_started
    guard(deadline)
    prompts = [p for values in prompts_by_category.values() for p in values]
    features = np.full((len(data['rows']), 512), np.nan, dtype=np.float32)
    valid = np.zeros(len(features), dtype=np.bool_)
    rows = []
    with torch.inference_mode():
        text = torch.cat([model.encode_text(clip.tokenize(prompts[i:i+4])) for i in range(0, len(prompts), 4)])
        require(bool(torch.isfinite(text).all()) and bool((text.norm(dim=-1) > 0).all()), 'Invalid text features')
        text = text / text.norm(dim=-1, keepdim=True)
        for original in data['rows']:
            guard(deadline)
            row = dict(original, embedding_error=None, feature_valid=False, embedding_index=None)
            started = time.monotonic()
            if row['restoration_error'] is not None:
                row['embedding_error'] = 'Not encoded: original restoration failed; original error retained'
            else:
                try:
                    path = Path(row['webp_path'])
                    require(sha(path) == row['webp_sha256'], 'WebP changed before decode')
                    with Image.open(path) as image:
                        require(image.format == 'WEBP' and image.size == (1024, 1024), 'Expected final1024 WebP')
                        rgb = image.convert('RGB')
                        row['decoded_rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                        tensor = preprocess(rgb).unsqueeze(0)
                    feature = model.encode_image(tensor)
                    require(tuple(feature.shape) == (1, 512) and bool(torch.isfinite(feature).all()), 'Invalid image embedding')
                    norm = feature.norm(dim=-1, keepdim=True)
                    require(float(norm) > 0, 'Zero image embedding')
                    feature = feature / norm
                    similarities = (feature @ text.T).squeeze(0).tolist()
                    category = {name: sum(similarities[i*3:i*3+3])/3 for i, name in enumerate(prompts_by_category)}
                    features[row['feature_row']] = feature[0].cpu().numpy()
                    valid[row['feature_row']] = True
                    row.update(feature_valid=True, embedding_index=row['feature_row'], prompt_cosines=dict(zip(prompts, similarities)),
                        category_mean_cosines=category,
                        adult_male_margin=category['adult_male']-max(category['adult_woman'], category['child']),
                        headwear_margin=category['headwear']-category['bare_head'],
                        appearance_argmax=max(['adult_male', 'adult_woman', 'child'], key=category.__getitem__))
                    del tensor, feature
                except Exception as exc:
                    features[row['feature_row']] = np.nan
                    valid[row['feature_row']] = False
                    row.update(feature_valid=False, embedding_index=None, embedding_error=repr(exc))
            row['preprocess_encode_seconds'] = time.monotonic()-started
            rows.append(row)
            with (args.output/'rows.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps({k: row[k] for k in ['feature_row', 'index', 'feature_valid', 'embedding_error']}), flush=True)
            guard(deadline)
    require(len(rows) == len(data['rows']), 'Denominator changed')
    require(np.isfinite(features[valid]).all() and np.isnan(features[~valid]).all(), 'Invalid masked feature representation')
    require(np.allclose(np.linalg.norm(features[valid], axis=1), 1, atol=1e-5), 'Embeddings not unit normalized')
    np.savez(args.output/'features.npz', features=features, valid=valid,
             feature_row=np.arange(len(rows), dtype=np.int64), text_features=text.cpu().numpy())
    guard(deadline)
    require(inventory(args.input) == data, 'Input hashes changed during embedding')
    require(verify_pins() == pins and sha(HERE/'embed-pins.json') == launch['pins_sha256']
            and sha(Path(__file__)) == launch['source_sha256'], 'Source/weights changed during extraction')
    guard(deadline)
    write(args.output/'result.json', {'complete': True, 'attempted': len(rows),
        'encoded': int(valid.sum()), 'restoration_errors': sum(r['restoration_error'] is not None for r in rows),
        'embedding_errors_after_successful_restoration': sum(r['restoration_error'] is None and not r['feature_valid'] for r in rows),
        'cohorts': data['cohorts'], 'rows': rows, 'features_sha256': sha(args.output/'features.npz'),
        'row_mapping': 'NPZfeatures row i equals metadata feature_row i; failures are NaN512 rows with valid=false, never omitted',
        'source_sha256': launch['source_sha256'], 'pins_sha256': launch['pins_sha256'],
        'clip_weights_sha256': pins['files_sha256'][pins['weights']], 'clip_commit': pins['clip_commit'],
        'device': 'cpu', 'threads': 1, 'image_batch': 1, 'text_batch_max': 4,
        'preprocess': str(preprocess), 'image_source': 'Full frame decoded final restored-1024.webp; square resize/center crop retains full frame',
        'prompts': prompts_by_category, 'score_definition': 'Same benchmark_clip mean of three cosine similarities; margins are differences, not probabilities',
        'model_load_seconds': load_seconds, 'peak_rss_bytes': peak_rss(),
        'no_training_or_selection': True, 'quality_approved': False, 'production_approved': False})


def supervise(args):
    started = time.time()
    deadline = started+LIMIT_SECONDS
    require(not args.output.exists(), 'Output must be new; no overwrite or automatic retry')
    pins = verify_pins()
    data = inventory(args.input)
    pressure = subprocess.check_output(['memory_pressure'], text=True, timeout=10)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    require(match is not None and int(match[1]) >= 25, 'Require >=25% free memory before CLIP load')
    guard(deadline)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['embed.py', 'embed-pins.json']:
        shutil.copyfile(HERE/name, args.output/name)
    nonce = uuid.uuid4().hex
    write(args.output/'launch.json', {'supervisor_pid': os.getpid(), 'nonce': nonce,
        'started_unix': started, 'deadline_unix': deadline, 'source_sha256': sha(Path(__file__)),
        'pins_sha256': sha(HERE/'embed-pins.json'), 'inputs': [str(p) for p in args.input],
        'inventory': data, 'pins': pins, 'memory': {'free_percent': int(match[1]), 'raw': pressure},
        'sampled_rss_limit': LIMIT_RSS, 'absolute_seconds_limit': LIMIT_SECONDS})
    command = [sys.executable, str(Path(__file__)), '--_worker', '--nonce', nonce, '--output', str(args.output)]
    for path in args.input:
        command.extend(['--input', str(path)])
    process, failure, peak = None, None, 0
    try:
        with (args.output/'worker.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **ENV},
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                require(time.time() < deadline, 'Ten-minute supervisor deadline exceeded')
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True, timeout=2)
                if sample.returncode == 0 and sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip())*1024)
                    require(peak <= LIMIT_RSS, '6GiB sampled worker RSS exceeded')
                time.sleep(.1)
            require(process.returncode == 0, f'Worker exited {process.returncode}; partial evidence retained')
            result = read(args.output/'result.json')
            require(result['complete'] and result['attempted'] == len(data['rows']), 'Incomplete extraction')
            require(time.time() < deadline and result['peak_rss_bytes'] <= LIMIT_RSS, 'Final resource limit exceeded')
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        write(args.output/'supervisor-result.json', {'complete': failure is None, 'error': failure,
            'worker_exit_code': process.returncode if process is not None else None,
            'wall_seconds': time.time()-started, 'peak_sampled_worker_rss_bytes': peak,
            'training_or_selection_performed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True, help='Completed GFPGAN result directory; repeat to concatenate')
    parser.add_argument('--output', type=Path, required=True, help='New extraction output directory')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--nonce', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.input = [p.resolve() for p in args.input]
    args.output = args.output.resolve()
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        worker(args)
    else:
        require(args.nonce is None, 'Internal worker nonce is not a launch override')
        supervise(args)


if __name__ == '__main__':
    main()
