"""Guarded, explicit native1024 NADA-style 10-to-100 text-direction probe; stdlib supervisor."""
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
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAX_SECONDS = 1200
MAX_RSS = 12 * 1024**3
HANDSHAKE_SECONDS = 5
NUMERIC_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify_pins():
    pins = json.loads((HERE / 'pins.json').read_text())
    if platform.python_version() != pins['python_version'] or platform.machine() != pins['machine']:
        raise RuntimeError('Python or architecture differs from preparation')
    for package, version in pins['selection_versions'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Package version changed: {package}')
    for path, expected in pins['files_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f'Pinned input/source changed: {path}')
    for directory, expected in pins['inventories'].items():
        files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob('*')
                       if p.is_file() and p.suffix in {'.py', '.so', '.cpp', '.cu', '.h', '.hpp', '.gz', '.pem'})
        if files != expected:
            raise RuntimeError(f'Pinned source inventory changed: {directory}')
    return pins


def memory_guard():
    if sys.platform != 'darwin':
        raise RuntimeError('This sampled RSS/memory_pressure guard is macOS-specific')
    pressure = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    if match is None or int(match[1]) < 35:
        raise RuntimeError('Defer: at least 35% free memory required')
    return {'free_percent': int(match[1]), 'raw': pressure}


def concurrency_guard(owned_pid=None):
    processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True, timeout=15)
    for line in processes.splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid.isdigit() and int(pid) not in {os.getpid(), owned_pid} and re.search(r'(^|/|\s)python[\w.]*\s', command):
            if 'research/' in command and any(word in command for word in
                ('trainer.py', 'train.py', 'runner.py', 'evaluate', 'benchmark', 'recover-factorial', 'verify-checkpoint', 'reference625_validation/', 'nada_clean24_overlap/supervise.py')):
                raise RuntimeError(f'Defer: another known research model process is active (PID {pid})')


def publish_child(output, value):
    # A worker must never see an incompletely written JSON readiness record.
    destination = output / 'child.json'
    if destination.exists():
        raise RuntimeError('Child readiness record already exists')
    temporary = output / 'child.json.tmp'
    write_new(temporary, value)
    os.replace(temporary, destination)


def worker_preflight(output, deadline, nonce):
    launch = json.loads((output / 'launch.json').read_text())
    if launch['nonce'] != nonce or launch['deadline_unix'] != deadline or launch['runner_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Worker must match its supervised invocation')
    if time.time() >= deadline or deadline > time.time() + MAX_SECONDS:
        raise RuntimeError('Expired or invalid original deadline')
    readiness_deadline = min(deadline, time.time() + HANDSHAKE_SECONDS)
    while not (output / 'child.json').exists():
        if os.getppid() != launch['supervisor_pid']:
            raise RuntimeError('Worker lost its supervisor before readiness')
        if time.time() >= readiness_deadline:
            raise RuntimeError('Supervisor child readiness timed out')
        time.sleep(.01)
    child = json.loads((output / 'child.json').read_text())
    if (child['pid'] != os.getpid() or child['supervisor_pid'] != os.getppid()
            or child['supervisor_pid'] != launch['supervisor_pid'] or child['nonce'] != nonce):
        raise RuntimeError('Child readiness PID/parent/nonce mismatch')
    if set(p.name for p in output.iterdir()) != {'launch.json', 'worker.log', 'child.json'}:
        raise RuntimeError('Worker refuses existing research results')
    if any(os.environ.get(k) != v for k, v in NUMERIC_ENV.items()):
        raise RuntimeError('Worker CPU environment differs')
    pins = verify_pins()
    if sha(HERE / 'pins.json') != launch['pins_sha256']:
        raise RuntimeError('Pins changed after launch')
    pressure = memory_guard()
    return launch, pins, pressure


def worker(output, deadline, nonce):
    launch, pins, pressure = worker_preflight(output, deadline, nonce)
    write_new(output / 'worker-preflight.json', pressure)
    for filename in ['runner.py', 'worker.py', 'pins.json', 'inputs.json', 'latents.npz']:
        shutil.copyfile(HERE / filename, output / ('source-' + filename if filename.endswith('.py') else filename))
    stop = threading.Event()
    def watchdog():
        while not stop.wait(.05):
            reason = None
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS:
                reason = '12GiB observed peak RSS exceeded'
            if time.time() >= deadline:
                reason = 'Original twenty-minute absolute deadline expired'
            if reason:
                try:
                    write_new(output / 'watchdog-failure.json', {'reason': reason})
                finally:
                    os._exit(70)
    watch = threading.Thread(target=watchdog, daemon=True)
    watch.start()
    try:
        spec = importlib.util.spec_from_file_location('nada_worker', HERE / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(ROOT, output, pins, restore_only=launch['restore_only'])
        if time.time() >= deadline or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS:
            raise RuntimeError('Resources exceeded before successful completion')
        if verify_pins() != pins or sha(HERE / 'pins.json') != launch['pins_sha256']:
            raise RuntimeError('Pinned sources/inputs changed during smoke')
        write_new(output / 'worker-provenance-complete.json', {'pins_unchanged': True,
            'production_approved': False, 'deadline_unix': deadline})
    finally:
        stop.set()
        watch.join(timeout=1)


def supervise(output, restore_only=False):
    if not output.is_relative_to(ROOT / 'research/runs') or output.exists():
        raise RuntimeError('Choose a new directory below research/runs; no overwrites/retries')
    started = time.time()
    deadline = started + MAX_SECONDS
    concurrency_guard()
    pressure = memory_guard()
    pins = verify_pins()
    if 'torch' in sys.modules:
        raise RuntimeError('Supervisor must not import torch')
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    launch = {'started_unix': started, 'deadline_unix': deadline, 'nonce': nonce,
              'supervisor_pid': os.getpid(),
              'runner_sha256': sha(Path(__file__)), 'worker_sha256': sha(HERE / 'worker.py'),
              'pins_sha256': sha(HERE / 'pins.json'), 'memory_preflight': pressure,
              'rss_abort_bytes': MAX_RSS, 'sampled_not_hard_allocation_cap': True,
              'numeric_environment': NUMERIC_ENV, 'maximum_updates': 0 if restore_only else 90, 'start_step': 10,
              'end_step': 10 if restore_only else 100, 'restore_only': restore_only,
              'automatic_retry': False, 'production_approved': False}
    write_new(output / 'launch.json', launch)
    environment = os.environ.copy()
    environment.update(NUMERIC_ENV)
    command = [sys.executable, str(Path(__file__)), '--execute', '--output', str(output),
               '--_worker', '--_deadline', str(deadline), '--_nonce', nonce]
    process = None
    failure = None
    peak = 0
    next_process_check = 0.
    try:
        with (output / 'worker.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            # Finally below applies immediately after Popen, including write failure.
            publish_child(output, {'pid': process.pid, 'supervisor_pid': os.getpid(),
                                   'nonce': nonce, 'command': command})
            while process.poll() is None:
                if time.monotonic() >= next_process_check:
                    concurrency_guard(owned_pid=process.pid)
                    next_process_check = time.monotonic() + 2
                if time.time() >= deadline:
                    raise RuntimeError('Twenty-minute absolute deadline expired')
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                    capture_output=True, text=True, timeout=min(2, max(.01, deadline-time.time())))
                if sample.returncode == 0 and sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip()) * 1024)
                    if peak > MAX_RSS:
                        raise RuntimeError('12GiB observed worker RSS exceeded')
                time.sleep(.1)
            if process.returncode != 0:
                raise RuntimeError(f'Worker exited {process.returncode}; partial evidence retained, no retry')
            if time.time() >= deadline or not (output / 'worker-provenance-complete.json').exists():
                raise RuntimeError('Incomplete provenance or expired deadline')
            result = json.loads((output / 'worker-result.json').read_text())
            expected_updates, expected_step = (0, 10) if restore_only else (90, 100)
            if (not result['complete'] or result['updates'] != expected_updates
                    or result['optimizer_calls'] != expected_updates or result['total_step'] != expected_step
                    or result['restore_only'] != restore_only):
                raise RuntimeError('Incorrect completion counters')
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
        write_new(output / 'supervisor-result.json', {'complete': failure is None,
            'worker_exit_code': process.returncode if process is not None else None,
            'error': failure, 'wall_seconds': time.time()-started, 'deadline_unix': deadline,
            'peak_sampled_worker_rss_bytes': peak, 'production_approved': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--restore-only', action='store_true', help='Supervised exact step10 restoration and previews; zero updates')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--_nonce', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        if args._worker or args._deadline is not None or args._nonce is not None or args.restore_only:
            parser.error('No worker flags in check-only mode')
        pins = verify_pins()
        print(json.dumps({'status': 'pass', 'pinned_files': len(pins['files_sha256']),
                          'torch_imported': 'torch' in sys.modules, 'training_launched': False}))
        return
    if args.output is None:
        parser.error('--execute requires a new --output path')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'research/runs'):
        parser.error('Output must be below research/runs')
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        if args.restore_only:
            parser.error('Worker restore mode comes from authenticated launch metadata')
        if args._deadline is None or args._nonce is None:
            parser.error('Internal worker requires supervised provenance')
        worker(output, args._deadline, args._nonce)
    else:
        if args._deadline is not None or args._nonce is not None:
            parser.error('The original deadline cannot be overridden')
        supervise(output, restore_only=args.restore_only)


if __name__ == '__main__':
    main()
