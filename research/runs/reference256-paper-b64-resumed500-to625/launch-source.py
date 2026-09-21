"""Proposed one-shot reference500-to625 continuation; no model imports here."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'research/runs/reference256-paper-b64'
OUTPUT = ROOT / 'research/runs/reference256-paper-b64-resumed500-to625'
PHASES = ROOT / 'research/experiments/reference_phases'
FACTORIAL = ROOT / 'research/runs/paired-factorial-600'
MAX_SECONDS = 7200


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def dataset_digest():
    data = ROOT / 'research/alignment/collar-only/eyes42'
    files = sorted(data.rglob('*.png'))
    sha = hashlib.sha256()
    for path in files:
        sha.update(str(path.relative_to(data)).encode())
        sha.update(path.read_bytes())
    return len(files), sha.hexdigest()


def preflight():
    pins = read(HERE / 'pins.json')
    for path, expected in pins['files_sha256'].items():
        require(digest(ROOT / path) == expected, f'Pinned input changed: {path}')
    require(not OUTPUT.exists(), 'New output already exists; no overwrite or automatic retry')
    spec = importlib.util.spec_from_file_location('resume_guard', PHASES / 'resume_preflight.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    result = guard.check(ROOT, ORIGINAL, PHASES / 'resume-provenance-reference256-paper-b64.json', guard.environment())
    count, data_sha = dataset_digest()
    require(count == 110 and data_sha == pins['dataset_sha256'], 'Dataset changed')
    require('torch' not in sys.modules, 'Preflight must not import torch')
    return {'provenance': result, 'pinned_files': len(pins['files_sha256']),
            'dataset_count': count, 'dataset_sha256': data_sha, 'torch_imported': False}


def command():
    return [str(ROOT / 'research/.venv/bin/python'), '-u', str(PHASES / 'trainer.py'),
            '--run', str(OUTPUT), '--resume', str(ORIGINAL / 'resume-000500.pt'),
            '--data', str(ROOT / 'research/alignment/collar-only/eyes42'),
            '--base', str(ROOT / 'research/models/ffhq256.pkl'), '--device', 'mps', '--threads', '2',
            '--batch', '64', '--microbatch', '8', '--pl-batch-shrink', '2', '--mirror',
            '--freeze-d-layers', '0', '--lr', '0.0025', '--r1-gamma', '1', '--ema-kimg', '20',
            '--augment-p', '0', '--ada-target', '0.6', '--ada-kimg', '100', '--seed', '20260920',
            '--steps', '625', '--checkpoint-every', '125', '--snapshot-every', '125']


def execution_guard():
    comparison = read(FACTORIAL / 'comparison.json')
    recovery = read(FACTORIAL / 'recovery-CD.launch.json')
    require(comparison['complete'] and comparison['arms'] == list('ABCD'), 'Factorial incomplete')
    require(recovery['complete'] and recovery['exit_code'] == 0, 'Factorial recovery not terminal-success')
    for arm in 'CD':
        result = read(FACTORIAL / arm / 'complete.json')
        require(result['complete'] and result['actual_optimizer_calls'] == 300, f'{arm} incomplete')
    processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True, timeout=15)
    for line in processes.splitlines():
        require(not any(marker in line for marker in ('reference_phases/trainer.py', 'paired_factorial/runner.py')),
                'A reference/factorial model process is still running')
    pressure = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    require(match is not None and int(match.group(1)) >= 35, 'Defer: at least 35% free memory required')
    return pressure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true', help='Only stdlib hashes/metadata; no output/run creation')
    mode.add_argument('--execute', action='store_true', help='Deliberately launch once after root review and factorial completion')
    args = parser.parse_args()
    checked = preflight()
    if args.check_only:
        print(json.dumps({'checks': checked, 'command': command(), 'training_launched': False}, indent=2))
        return
    pressure = execution_guard()
    # Preserve the numerical environment checked above; no launch-only overrides.
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    OUTPUT.mkdir(exist_ok=False)
    for source, target in [('trainer.py', 'train-source.py'), ('smoke.py', 'adapters-source.py')]:
        (OUTPUT / target).write_bytes((PHASES / source).read_bytes())
    (OUTPUT / 'launch-source.py').write_bytes(Path(__file__).read_bytes())
    (OUTPUT / 'launch-pins.json').write_bytes((HERE / 'pins.json').read_bytes())
    started = time.monotonic()
    record = {'started_unix': time.time(), 'checks': checked, 'command': command(),
              'memory_before': pressure, 'maximum_wall_seconds': MAX_SECONDS,
              'start_step': 500, 'stop_step': 625, 'additional_updates': 125,
              'source_sha256': digest(Path(__file__)), 'complete': False,
              'automatic_retry': False, 'production_approved': False}
    write(OUTPUT / 'launch.json', record)
    try:
        with (OUTPUT / 'train.log').open('x') as stream:
            process = subprocess.Popen(command(), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record['pid'] = process.pid
                write(OUTPUT / 'launch.json', record)
                code = process.wait(timeout=max(0, MAX_SECONDS - (time.monotonic() - started)))
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                record['child_exit_code'] = process.returncode
        require(code == 0, f'Trainer exited {code}; no retry')
        completed = read(OUTPUT / 'completed.json')
        require(completed['step'] == 625 and completed['images_seen'] == 40000, 'Wrong completion counters')
        record['outputs_sha256'] = {name: digest(OUTPUT / name) for name in
            ['resume.pt', 'generator-000625.pt', 'samples-000625-ema.png',
             'samples-000625-raw.png', 'samples-000625-untruncated.png', 'config.json']}
        require(read(OUTPUT / 'config.json')['recipe'] == read(ORIGINAL / 'config.json')['recipe'], 'Recipe changed')
        # These stream hashes establish original input preservation, not tensor health.
        for path, expected in read(HERE / 'pins.json')['files_sha256'].items():
            require(digest(ROOT / path) == expected, f'Original input changed during run: {path}')
        require(dataset_digest() == (110, checked['dataset_sha256']), 'Dataset changed during run')
        record['complete'] = True
        record['numeric_and_visual_review_pending'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record.update(ended_unix=time.time(), wall_seconds=time.monotonic() - started)
        write(OUTPUT / 'launch.json', record)
    print('Reached step 625; independent numeric and visual review still required', flush=True)


if __name__ == '__main__':
    main()
