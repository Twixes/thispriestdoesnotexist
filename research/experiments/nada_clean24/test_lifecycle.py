"""Exercise proposed continuation cleanup using temporary fixtures and harmless sleep groups only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent / 'runner.py'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def wait_for(path, seconds=5):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Timed out waiting for dummy fixture {path.name}')
        time.sleep(.01)


def dummy(directory):
    # Both processes are harmless, time-bounded sleeps in the recovery-owned group.
    grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    dump(directory / 'dummy.json', {'pid': os.getpid(), 'pgid': os.getpgrp(), 'grandchild_pid': grandchild.pid})
    time.sleep(20)
    grandchild.wait()


def scenario(name, directory):
    spec = importlib.util.spec_from_file_location('reviewed_recovery', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = directory / 'research/runs/fake-run'
    module.ROOT = directory.resolve()
    module.verify_pins = lambda: {'fixture': 'No model files or imports'}
    module.concurrency_guard = lambda: None
    module.memory_guard = lambda: {'fixture_free_percent': 99}
    module.MAX_SECONDS = 1.0 if name == 'deadline' else 10
    if name == 'rss': module.MAX_RSS = 1
    real_popen = subprocess.Popen

    def dummy_popen(command, **kwargs):
        assert '--_worker' in command and kwargs['start_new_session'] is True
        process = real_popen([sys.executable, str(Path(__file__).resolve()), '--dummy', str(directory)], **kwargs)
        wait_for(directory / 'dummy.json')
        dump(directory / 'spawn-ready.json', {'pid': process.pid, 'real_command_was_replaced': True})
        return process

    module.subprocess = SimpleNamespace(Popen=dummy_popen, run=subprocess.run, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT)
    if name == 'write_failure':
        original_write = module.write_new
        failed = False

        def fail_after_spawn(path, value):
            nonlocal failed
            if path.name == 'child.json.tmp' and not failed:
                failed = True
                raise OSError('Injected record-write failure after dummy spawn')
            return original_write(path, value)

        module.write_new = fail_after_spawn
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.argv = [str(SOURCE), '--execute', '--output', str(run)]
    module.main()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    started = time.monotonic()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    results = []
    for name in ['write_failure', 'sigint', 'sigterm', 'deadline', 'rss']:
        with tempfile.TemporaryDirectory(prefix='priest-recovery-lifecycle-') as temporary:
            directory = Path(temporary)
            process = None
            ids = None
            tick = time.monotonic()
            try:
                with (directory / 'coordinator.log').open('w') as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--scenario', name, str(directory)], stdout=log, stderr=subprocess.STDOUT)
                    wait_for(directory / 'spawn-ready.json')
                    ids = json.loads((directory / 'dummy.json').read_text())
                    assert ids['pid'] == ids['pgid']
                    if name in ['sigint', 'sigterm']:
                        process.send_signal(signal.SIGINT if name == 'sigint' else signal.SIGTERM)
                    exit_code = process.wait(timeout=8)
                until = time.monotonic() + 3
                while time.monotonic() < until and (alive(ids['pid']) or alive(ids['grandchild_pid']) or group_exists(ids['pgid'])):
                    time.sleep(.02)
                absent = {k: not alive(ids[k]) for k in ['pid', 'grandchild_pid']}
                no_group = not group_exists(ids['pgid'])
                record = json.loads((directory / 'research/runs/fake-run/supervisor-result.json').read_text())
                assert all(absent.values()) and no_group, (name, ids, absent, no_group)
                assert exit_code != 0 and record['complete'] is False
                assert record['worker_exit_code'] == -signal.SIGKILL
                assert not (directory / 'research/runs/fake-run/worker-result.json').exists()
                assert not (directory / 'research/runs/fake-run/checkpoint-010.pt').exists()
                if name == 'sigterm':
                    assert exit_code == 143 and 'SystemExit(143)' in record['error']
                if name == 'sigint':
                    assert 'KeyboardInterrupt' in record['error']
                if name == 'write_failure':
                    assert 'Injected record-write failure' in record['error']
                if name == 'deadline':
                    assert 'deadline expired' in record['error']
                if name == 'rss':
                    assert 'RSS exceeded' in record['error']
                results.append({'scenario': name, 'passed': True, 'wall_seconds': time.monotonic()-tick,
                    'coordinator_exit_code': exit_code, 'dummy_pids': ids, 'dummy_processes_absent': absent,
                    'dummy_group_absent': no_group, 'child_reaped_returncode': record['worker_exit_code'],
                    'recorded_error': record['error'], 'only_one_dummy_spawned': True, 'no_checkpoint_or_completion_published': True,
                    'coordinator_log': (directory / 'coordinator.log').read_text()})
            finally:
                # Test-only safety net. A passing assertion above means this never kills anything.
                if ids and group_exists(ids['pgid']):
                    os.killpg(ids['pgid'], signal.SIGKILL)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    evidence = {'passed': True, 'tested_source': str(SOURCE.relative_to(ROOT)), 'source_sha256': digest,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_seconds': time.monotonic()-started, 'source_unchanged': True,
        'real_launcher_commands_executed': False, 'real_model_imports_or_loads': False,
        'fixture_policy': 'Temporary fake run/provenance, fake memory guard; Popen substitutes only harmless Python sleep group. Continuation main/cleanup/wait/signal handlers are real unmodified code.',
        'results': results}
    output = Path(__file__).resolve().parent / 'lifecycle-evidence.json'
    if output.exists():
        raise RuntimeError('Do not overwrite prior test evidence')
    dump(output, evidence)
    print(json.dumps({'passed': True, 'scenarios': [r['scenario'] for r in results], 'wall_seconds': evidence['wall_seconds']}))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--dummy':
        dummy(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == '--scenario':
        scenario(sys.argv[2], Path(sys.argv[3]))
    else:
        main()
