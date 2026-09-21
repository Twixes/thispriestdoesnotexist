"""Optional overlap policy around byte-identical, pinned NADA smoke worker inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN_SHA = '7bf6ab6513a6ecfb54f9219fd138c53f57388e9f9316e179efc9ed5a73745855'
MAX_SECONDS = 1200
MAX_RSS = 12 * 1024**3
PRESSURE_INTERVAL = 2
PRESSURE_MAX_GAP = 5


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_frozen():
    path = HERE / 'runner.py'
    if sha(path) != FROZEN_SHA:
        raise RuntimeError('Frozen worker-entry supervisor source changed')
    spec = importlib.util.spec_from_file_location('frozen_nada_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_setup(base):
    pins = base.verify_pins()
    policy = json.loads((HERE / 'policy.json').read_text())
    for copied, item in policy['original_source_copies'].items():
        if sha(HERE / copied) != item['sha256'] or sha(ROOT / item['source']) != item['sha256']:
            raise RuntimeError(f'Original/copy byte identity differs: {copied}')
        if pins['files_sha256'].get(str((HERE / copied).relative_to(ROOT))) != item['sha256']:
            raise RuntimeError(f'Actual copied input is not pinned: {copied}')
    argv = policy['allowed_reference_argv']
    required = {'--device': 'mps', '--run': str(ROOT / 'research/runs/reference256-paper-b64-resumed500-to625')}
    if argv[:3] != [str(ROOT / 'research/.venv/bin/python'), '-u', str(ROOT / 'research/experiments/reference_phases/trainer.py')]:
        raise RuntimeError('Policy does not name the exact reference trainer')
    if any(argv[argv.index(flag)+1] != value for flag, value in required.items()):
        raise RuntimeError('Policy reference device/run differs')
    if policy['allowed_reference_command_sha256'] != hashlib.sha256(policy['allowed_reference_command'].encode()).hexdigest():
        raise RuntimeError('Policy command hash mismatch')
    if shlex.split(policy['allowed_reference_command']) != argv:
        raise RuntimeError('Policy argv/command mismatch')
    return pins, policy


def processes():
    text = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,lstart=,command='], text=True, timeout=1)
    result = []
    for line in text.splitlines():
        fields = line.strip().split(None, 8)
        if len(fields) != 9:
            raise RuntimeError('Process monitor returned an incomplete record')
        result.append({'pid': int(fields[0]), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                       'start_time': ' '.join(fields[3:8]), 'command': fields[8]})
    return result


def validate_processes(rows, policy, allowed_pid, owned_pid=None, require_reference=True):
    if allowed_pid != policy['allowed_reference_pid']:
        raise RuntimeError('Explicit allowed PID differs from pinned current reference')
    reference = None
    for item in rows:
        pid, command = item['pid'], item['command']
        if pid == allowed_pid:
            if (shlex.split(command) != policy['allowed_reference_argv']
                    or item['start_time'] != policy['allowed_reference_start_time']
                    or hashlib.sha256(command.encode()).hexdigest() != policy['allowed_reference_command_sha256']):
                raise RuntimeError('Allowed PID command/start identity differs; refuse PID reuse or another run')
            reference = item
            continue
        if pid in {os.getpid(), owned_pid}:
            continue
        if re.search(r'(^|/|\s)python[\w.]*\s', command) and 'research/' in command:
            if any(word in command for word in ('trainer.py', 'train.py', 'runner.py', 'evaluate', 'benchmark', 'recover-factorial', 'verify-checkpoint')):
                raise RuntimeError(f'Another known heavy research process is active (PID {pid})')
    if reference is None and require_reference:
        raise RuntimeError('Allowed reference PID is not running')
    return {'reference_present': reference is not None, 'reference': reference}


def pressure():
    started = time.monotonic()
    raw = subprocess.check_output(['memory_pressure'], text=True, timeout=1)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', raw)
    if match is None or not 0 <= int(match[1]) <= 100:
        raise RuntimeError('System pressure monitor could not parse free percentage')
    return {'observed_unix': time.time(), 'monotonic': time.monotonic(),
            'free_percent': int(match[1]), 'command_seconds': time.monotonic()-started, 'raw': raw}


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()


def checked_pressure(log, minimum, last=None):
    observation = pressure()
    append(log, observation)
    if last is not None and observation['monotonic'] - last['monotonic'] > PRESSURE_MAX_GAP:
        raise RuntimeError('Pressure monitor observation gap exceeded five seconds')
    if observation['free_percent'] < minimum:
        raise RuntimeError(f'System free memory {observation["free_percent"]}% is below {minimum}%')
    return observation


def stop_owned(process, protected_pid, protected_pgid):
    # No signal path ever targets the allowed reference PID or its process group.
    if process.pid in {protected_pid, protected_pgid}:
        raise RuntimeError('Owned child unexpectedly overlaps protected reference identity')
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(output, allowed_pid):
    base = load_frozen()
    started = time.time()
    deadline = started + MAX_SECONDS
    pins, policy = verify_setup(base)
    allowed = validate_processes(processes(), policy, allowed_pid)
    if not output.is_relative_to(ROOT / 'research/runs') or output.exists():
        raise RuntimeError('Use a new research/runs directory; no overwrite or automatic retry')
    records = output.with_name(output.name + '-supervision')
    if records.exists():
        raise RuntimeError('Supervision evidence directory already exists')
    # Fresh pressure after all source/pin checks; model processes have not started.
    initial = pressure()
    if initial['free_percent'] < 55:
        raise RuntimeError('Overlap requires at least55% free memory initially')
    if time.time() >= deadline:
        raise RuntimeError('Original twenty-minute deadline already expired')
    output.mkdir(parents=True, exist_ok=False)
    records.mkdir(exist_ok=False)
    append(records / 'pressure.jsonl', initial)
    base.write_new(records / 'policy.json', policy)
    base.write_new(records / 'provenance.json', {'overlap_supervisor_sha256': sha(Path(__file__)),
        'overlap_pins_sha256': sha(HERE / 'pins.json'), 'original_source_copies': policy['original_source_copies'],
        'actual_copied_input_paths_pinned': True, 'worker_recipe_changed': False})
    nonce = uuid.uuid4().hex
    launch = {'started_unix': started, 'deadline_unix': deadline, 'nonce': nonce,
        'supervisor_pid': os.getpid(), 'runner_sha256': sha(HERE / 'runner.py'),
        'worker_sha256': sha(HERE / 'worker.py'), 'pins_sha256': sha(HERE / 'pins.json'),
        'memory_preflight': initial, 'rss_abort_bytes': MAX_RSS,
        'sampled_not_hard_allocation_cap': True, 'numeric_environment': base.NUMERIC_ENV,
        'maximum_updates': 10, 'automatic_retry': False, 'production_approved': False,
        'overlap_supervisor_sha256': sha(Path(__file__)), 'allowed_reference': allowed['reference'],
        'allowed_reference_command_sha256': policy['allowed_reference_command_sha256'],
        'system_pressure_policy': {'initial_min': 55, 'running_min': 35, 'interval_seconds': 2, 'maximum_gap_seconds': 5},
        'supervision_records': str(records.relative_to(ROOT))}
    base.write_new(output / 'launch.json', launch)
    environment = os.environ.copy()
    environment.update(base.NUMERIC_ENV)
    command = [sys.executable, str(HERE / 'runner.py'), '--execute', '--output', str(output),
               '--_worker', '--_deadline', str(deadline), '--_nonce', nonce]
    process = None
    failure = None
    peak = 0
    last = initial
    next_observation = time.monotonic()
    try:
        with (output / 'worker.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            base.publish_child(output, {'pid': process.pid, 'supervisor_pid': os.getpid(),
                                        'nonce': nonce, 'command': command})
            while process.poll() is None:
                if time.time() >= deadline:
                    raise RuntimeError('Twenty-minute absolute deadline expired')
                if time.monotonic() - last['monotonic'] > PRESSURE_MAX_GAP:
                    raise RuntimeError('Pressure monitor became stale beyond five seconds')
                if time.monotonic() >= next_observation:
                    last = checked_pressure(records / 'pressure.jsonl', 35, last)
                    state = validate_processes(processes(), policy, allowed_pid, process.pid, require_reference=False)
                    append(records / 'process-observations.jsonl', {'observed_unix': time.time(), **state})
                    next_observation = last['monotonic'] + PRESSURE_INTERVAL
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True,
                    text=True, timeout=min(.5, max(.01, deadline-time.time())))
                if sample.returncode == 0 and sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip()) * 1024)
                    if peak > MAX_RSS:
                        raise RuntimeError('12GiB observed NADA RSS exceeded')
                elif process.poll() is None:
                    raise RuntimeError('NADA RSS monitor failed')
                time.sleep(.1)
            if process.returncode != 0:
                raise RuntimeError(f'Worker exited {process.returncode}; no retry')
            if time.time() >= deadline or not (output / 'worker-provenance-complete.json').exists():
                raise RuntimeError('Incomplete worker provenance or expired deadline')
            result = json.loads((output / 'worker-result.json').read_text())
            if not result['complete'] or result['updates'] != 10 or result['optimizer_calls'] != 10:
                raise RuntimeError('Incorrect worker completion counters')
            if base.verify_pins() != pins:
                raise RuntimeError('Input/source provenance changed during overlap')
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if process is not None:
            stop_owned(process, allowed_pid, allowed['reference']['pgid'])
        result = {'complete': failure is None, 'worker_exit_code': process.returncode if process else None,
            'error': failure, 'wall_seconds': time.time()-started, 'deadline_unix': deadline,
            'peak_sampled_worker_rss_bytes': peak, 'last_pressure': last,
            'allowed_reference_pid': allowed_pid, 'reference_signaled': False,
            'pressure_log': str((records/'pressure.jsonl').relative_to(ROOT)), 'production_approved': False}
        base.write_new(output / 'supervisor-result.json', result)
        base.write_new(records / 'supervisor-result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--allowed-reference-pid', type=int, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.check_only:
        base = load_frozen()
        pins, policy = verify_setup(base)
        allowed = validate_processes(processes(), policy, args.allowed_reference_pid)
        observed = pressure()
        print(json.dumps({'source_checks': 'pass', 'pinned_files': len(pins['files_sha256']),
            'allowed_reference': allowed, 'pressure': observed,
            'initial_pressure_requirement_met': observed['free_percent'] >= 55,
            'torch_imported': 'torch' in sys.modules, 'training_launched': False}, indent=2))
        return
    if args.output is None:
        parser.error('--execute requires a new --output')
    supervise(args.output.resolve(), args.allowed_reference_pid)


if __name__ == '__main__':
    main()
