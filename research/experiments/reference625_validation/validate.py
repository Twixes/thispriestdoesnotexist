"""Bounded numeric validation and immutable capture of trusted completed625 output."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'research/runs/reference256-paper-b64-resumed500-to625'
ORIGINAL = ROOT / 'research/runs/reference256-paper-b64'
PINS = HERE / 'pins.json'
EVIDENCE = RUN / 'validation-000625'
PRESERVED = RUN / 'resume-000625.pt'
COPYING = RUN / 'resume-000625.pt.copying'
MAX_SECONDS = 300
MAX_RSS = 4 * 1024**3
CHUNK_ELEMENTS = 262144
DATASET_SHA = '3a6e258a9b9ddae8d415b70c3be540ff39cf44e8a37d80ab71540058f1ee0553'
OUTPUT_NAMES = ('resume.pt', 'generator-000625.pt', 'samples-000625-ema.png',
                'samples-000625-raw.png', 'samples-000625-untruncated.png', 'config.json')
INPUT_NAMES = OUTPUT_NAMES + ('completed.json', 'launch.json', 'train-source.py',
                             'adapters-source.py', 'launch-source.py', 'launch-pins.json')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def static_checks():
    pins = read(PINS)
    require(sys.version == pins['python_version'], 'Pinned Python version changed')
    for name, expected in pins['versions'].items():
        require(importlib.metadata.version(name) == expected, f'Pinned package changed: {name}')
    for name, expected in pins['files'].items():
        require(sha(ROOT / name) == expected, f'Pinned source changed: {name}')
    return pins


def check_completion(completed, launch, config, baseline):
    require(completed['step'] == 625 and completed['images_seen'] == 40000,
            'Expected completed625/40000 counters')
    require(launch['complete'] and launch['child_exit_code'] == 0 and
            launch['start_step'] == 500 and launch['stop_step'] == 625,
            'Continuation supervisor must be terminal-success500-to625')
    require(set(launch['outputs_sha256']) == set(OUTPUT_NAMES), 'Incomplete launch output hash set')
    for key in ('recipe', 'architecture', 'trainer_sha256', 'frozen_d_tensor_names',
                'torch_version', 'sampler', 'deviations'):
        require(config[key] == baseline[key], f'Continuation config changed: {key}')
    recipe = config['recipe']
    require(recipe['dataset_sha256'] == DATASET_SHA and recipe['dataset_count'] == 220 and
            recipe['batch'] == 64 and recipe['mirror'], 'Only original110 mirrored dataset is eligible')
    arguments = config['arguments']
    require(set(arguments) == set(baseline['arguments']), 'Unexpected argument schema')
    for key in set(arguments) - {'run', 'resume', 'steps', 'data', 'base'}:
        require(arguments[key] == baseline['arguments'][key], f'Continuation argument changed: {key}')
    require(arguments['steps'] == 625, 'Declared stop must be625')
    for key, path in [('run', RUN), ('resume', ORIGINAL / 'resume-000500.pt'),
                      ('data', ROOT / 'research/alignment/collar-only/eyes42'),
                      ('base', ROOT / 'research/models/ffhq256.pkl')]:
        require((ROOT / arguments[key]).resolve() == path.resolve(), f'Unexpected argument path: {key}')


def preservation_refusal():
    require(not PRESERVED.exists() and not PRESERVED.is_symlink(), 'Refuse existing preserved625 file')
    require(not COPYING.exists() and not COPYING.is_symlink(), 'Refuse existing partial copying file')


def input_hashes():
    for name in INPUT_NAMES:
        path = RUN / name
        require(path.is_file() and not path.is_symlink(), f'Missing or symlink input: {name}')
    return {name: sha(RUN / name) for name in INPUT_NAMES}


def preflight():
    pins = static_checks()
    preservation_refusal()
    completed, launch, config = [read(RUN / name) for name in ('completed.json', 'launch.json', 'config.json')]
    check_completion(completed, launch, config, read(ORIGINAL / 'config.json'))
    hashes = input_hashes()
    for name, expected in launch['outputs_sha256'].items():
        require(hashes[name] == expected, f'Completed output hash changed: {name}')
    require(hashes['train-source.py'] == config['trainer_sha256'] == sha(ORIGINAL / 'train-source.py'),
            'Trainer source mismatch')
    require(hashes['adapters-source.py'] == sha(ORIGINAL / 'adapters-source.py'), 'Adapter source mismatch')
    launcher = ROOT / 'research/experiments/reference_resume500_625'
    require(hashes['launch-source.py'] == launch['source_sha256'] == sha(launcher / 'launch.py'),
            'Continuation launcher source mismatch')
    require(hashes['launch-pins.json'] == sha(launcher / 'pins.json'), 'Continuation launcher pins mismatch')
    return pins, hashes, config, launch


def active_model_processes(process_text, trainer_pid, own_pids):
    markers = ('reference_phases/trainer.py', 'paired_factorial/runner.py',
               'research/train.py', 'reference_unseen/evaluate.py',
               'reference625_evaluation/evaluate.py', 'paired_edit/trainer.py',
               'paired_regions/trainer.py', 'paired_surround/runner.py',
               'paired_capacity/runner.py', 'inference/service.py',
               'reference625_validation/validate.py')
    found = []
    for line in process_text.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        if pid in own_pids:
            continue
        other_research_model = ('research/' in command and
            re.search(r'/(?:train(?:er)?|evaluate(?:_[^/\s]*)?|eval(?:_[^/\s]*)?|benchmark(?:_[^/\s]*)?|runner|service)\.py(?:\s|$)', command))
        if pid == trainer_pid or any(marker in command for marker in markers) or other_research_model:
            found.append({'pid': pid, 'command': command})
    return found


def execution_guard(launch):
    require(sys.platform == 'darwin', 'Memory/RSS policy is reviewed for macOS only')
    processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True, timeout=15)
    busy = active_model_processes(processes, int(launch['pid']), {os.getpid(), os.getppid()})
    require(not busy, f'Defer while model training/evaluation process is active: {busy}')
    # Explicit PID check avoids treating a missing ps observation as terminal.
    try:
        os.kill(int(launch['pid']), 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Recorded training PID is still live; defer')
    pressure = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    require(match is not None and int(match.group(1)) >= 35, 'Defer: at least35% free memory required')
    return {'free_percent': int(match.group(1)), 'raw': pressure,
            'recorded_trainer_pid_absent': True, 'other_known_model_processes': busy}


def capture_after_validation(expected_sha):
    preservation_refusal()
    # Exclusive temporary creation and atomic hard-link publication never replace
    # an existing final filename. Both files are on the same filesystem/directory.
    with (RUN / 'resume.pt').open('rb') as source, COPYING.open('xb') as target:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    require(sha(COPYING) == expected_sha == sha(RUN / 'resume.pt'), 'Copy/source digest changed')
    os.link(COPYING, PRESERVED)
    directory_fd = os.open(RUN, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
        require(sha(PRESERVED) == expected_sha, 'Preserved copy hash mismatch')
        COPYING.unlink()
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return expected_sha


def worker(deadline, nonce):
    started = time.monotonic()
    record = read(EVIDENCE / 'launch.json')
    require(record['nonce'] == nonce and record['deadline_unix'] == deadline and
            record['script_sha256'] == sha(Path(__file__)) and record['pins_sha256'] == sha(PINS),
            'Worker must match supervisor provenance')
    require(time.time() < deadline <= time.time() + MAX_SECONDS, 'Invalid original deadline')
    pins, before, config, training_launch = preflight()
    require(before == record['hashes_before'], 'Inputs changed after supervisor preflight')
    pressure = execution_guard(training_launch)
    write_new(EVIDENCE / 'worker-preflight.json', pressure)
    stop = threading.Event()
    def watchdog():
        while not stop.wait(.05):
            if time.time() >= deadline or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS:
                try:
                    write_new(EVIDENCE / 'watchdog-failure.json', {'reason': 'five-minute deadline or4GiB observed peakRSS exceeded',
                              'production_approved': False})
                finally:
                    os._exit(70)
    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Trusted own trainer schema includes NumPy/Python RNG metadata.
        checkpoint = torch.load(RUN / 'resume.pt', map_location='cpu', mmap=True, weights_only=False)
        generator = torch.load(RUN / 'generator-000625.pt', map_location='cpu', mmap=True, weights_only=True)
        require(checkpoint['format_version'] == 1, 'Unexpected resume schema')
        require(checkpoint['batch_idx'] == generator['step'] == 625, 'Wrong tensor checkpoint step')
        require(checkpoint['images_seen'] == generator['images_seen'] == 40000, 'Wrong image presentations')
        require(checkpoint['sampler'] == {'seed': config['recipe']['seed'], 'consumed': 40000}, 'Wrong sampler counters')
        require(checkpoint['recipe'] == generator['recipe'] == config['recipe'], 'Tensor recipe mismatch')
        require(checkpoint['config'] == generator['config'] == config, 'Tensor config mismatch')
        require(checkpoint['fixed_z'].shape == (16, 512) and checkpoint['fixed_z'].dtype == torch.float32,
                'Unexpected trainer fixed_z schema')
        for key in ('G', 'D', 'G_ema'):
            require(bool(checkpoint[key]), f'Empty model state: {key}')
        for key in ('G_opt', 'D_opt'):
            require(bool(checkpoint[key]['state']) and bool(checkpoint[key]['param_groups']), f'Empty Adam state: {key}')
        tensors = []
        def walk(value, path):
            if torch.is_tensor(value):
                require(value.device.type == 'cpu' and value.is_contiguous(), f'Noncontiguous/nonCPU tensor: {path}')
                flat = value.view(-1)
                if value.is_floating_point() or value.is_complex():
                    for offset in range(0, flat.numel(), CHUNK_ELEMENTS):
                        require(bool(torch.isfinite(flat[offset:offset+CHUNK_ELEMENTS]).all()), f'Nonfinite tensor: {path}')
                tensors.append({'path': path, 'shape': list(value.shape), 'dtype': str(value.dtype),
                                'elements': value.numel(), 'finite': True})
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(item, path + '/' + str(key))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    walk(item, path + '/' + str(index))
            elif isinstance(value, float):
                require(math.isfinite(value), f'Nonfinite scalar: {path}')
        walk(checkpoint, 'resume')
        walk(generator, 'generator')
        require(set(checkpoint['G_ema']) == set(generator['G_ema']), 'EMA export keys differ')
        comparisons = []
        for key, left in checkpoint['G_ema'].items():
            right = generator['G_ema'][key]
            require(left.shape == right.shape and left.dtype == right.dtype, f'EMA shape/dtype mismatch: {key}')
            a, b = left.view(-1), right.view(-1)
            for offset in range(0, a.numel(), CHUNK_ELEMENTS):
                require(torch.equal(a[offset:offset+CHUNK_ELEMENTS], b[offset:offset+CHUNK_ELEMENTS]), f'EMA mismatch: {key}')
            comparisons.append({'name': key, 'elements': left.numel(), 'equal': True})
        require(input_hashes() == before and static_checks() == pins, 'Inputs changed during numeric validation')
        execution_guard(training_launch)
        require(time.time() < deadline and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= MAX_RSS,
                'Resource budget exhausted before capture')
        copied_sha = capture_after_validation(before['resume.pt'])
        after = input_hashes()
        require(before == after, 'Inputs changed during preservation; inspect preserved file, do not retry')
        require(sha(Path(__file__)) == record['script_sha256'] and sha(PINS) == record['pins_sha256'],
                'Validator source/pins changed')
        result = {'validation': 'passed', 'production_approved': False, 'quality_approved': False,
                  'reviewed_utc': datetime.now(timezone.utc).isoformat(),
                  'resume_batch_idx': 625, 'generator_step': 625, 'images_seen': 40000,
                  'sampler': checkpoint['sampler'], 'finite_tensors': len(tensors),
                  'ema_equal_tensors': len(comparisons), 'nonfinite_tensors': [],
                  'scope': 'CPU mmap finiteness/export equality and byte-exact preservation only; no model construction or inference',
                  'load_policy': {'resume': 'CPU mmap=True weights_only=False trusted own output',
                                  'generator': 'CPU mmap=True weights_only=True'},
                  'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                  'finite_chunk_elements': CHUNK_ELEMENTS, 'torch_version': str(torch.__version__),
                  'memory_preflight': pressure, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'duration_seconds': time.monotonic() - started, 'hashes_before': before, 'hashes_after': after,
                  'artifacts_unchanged': before == after, 'preserved_checkpoint': str(PRESERVED.relative_to(ROOT)),
                  'preserved_checkpoint_sha256': copied_sha, 'preserved_copy_matches_validated_source': True,
                  'script_sha256': record['script_sha256'], 'pins_sha256': record['pins_sha256'],
                  'tensors': tensors, 'ema_comparisons': comparisons,
                  'limits': ['Numeric validity is not image-quality approval.',
                             'PNG artifacts are hash-checked, not regenerated.',
                             'Observed RSS threshold is sampled, not an OS hard allocation guarantee.']}
        write_new(EVIDENCE / 'validation.json', result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('tensors', 'ema_comparisons')}), flush=True)
    finally:
        stop.set()
        thread.join(timeout=1)


def supervise():
    started = time.time()
    deadline = started + MAX_SECONDS
    require(not EVIDENCE.exists(), 'Refuse existing validation evidence directory; no automatic retry')
    _, before, _, training_launch = preflight()
    pressure = execution_guard(training_launch)
    EVIDENCE.mkdir(exist_ok=False)
    nonce = uuid.uuid4().hex
    write_new(EVIDENCE / 'launch.json', {'started_unix': started, 'deadline_unix': deadline, 'nonce': nonce,
              'script_sha256': sha(Path(__file__)), 'pins_sha256': sha(PINS), 'hashes_before': before,
              'memory_preflight': pressure, 'rss_abort_bytes': MAX_RSS, 'production_approved': False})
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    process, failure, peak = None, None, 0
    try:
        with (EVIDENCE / 'worker.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--execute', '--_worker',
                '--_deadline', str(deadline), '--_nonce', nonce], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                require(time.time() < deadline, 'Five-minute supervisor deadline exceeded')
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True,
                    text=True, timeout=min(2, max(.01, deadline-time.time())))
                if sample.returncode == 0 and sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip()) * 1024)
                    require(peak <= MAX_RSS, '4GiB sampled worker RSS exceeded')
                time.sleep(.1)
            require(process.returncode == 0, f'Worker failed: {process.returncode}; preserve evidence, no retry')
            require(time.time() < deadline and (EVIDENCE / 'validation.json').exists(), 'Missing result or deadline exceeded')
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        write_new(EVIDENCE / 'supervisor-result.json', {'complete': failure is None,
            'worker_exit_code': process.returncode if process is not None else None,
            'error': failure, 'wall_seconds': time.time()-started, 'deadline_unix': deadline,
            'peak_sampled_worker_rss_bytes': peak, 'preserved_file_exists': PRESERVED.exists(),
            'partial_copy_exists': COPYING.exists(), 'production_approved': False,
            'failure_policy': 'Retain all evidence/final/partial files; no overwrite or automatic retry.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-static', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--_nonce', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_static:
        pins = static_checks()
        print(json.dumps({'static_files': len(pins['files']), 'torch_imported': 'torch' in sys.modules,
                          'checkpoint_loaded': False}))
        return
    if args.check_only:
        _, hashes, _, launch = preflight()
        print(json.dumps({'hashes': hashes, 'guard': execution_guard(launch), 'checkpoint_loaded': False}))
        return
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        require(args._deadline is not None and args._nonce is not None, 'Worker requires supervisor token')
        worker(args._deadline, args._nonce)
    else:
        require(args._deadline is None and args._nonce is None, 'Cannot override original deadline')
        supervise()


if __name__ == '__main__':
    main()
