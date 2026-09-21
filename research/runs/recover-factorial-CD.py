"""Reviewed one-shot completion of unstarted C/D, retaining the original deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/paired-factorial-600'
EXPERIMENT = ROOT / 'research/experiments/paired_factorial'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def own_hashes():
    paths = list(EXPERIMENT.glob('*.py')) + [EXPERIMENT / 'parent-pins.json']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def memory_guard():
    text = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if not match or int(match.group(1)) < 35:
        raise RuntimeError('Defer: unchanged 35% free-memory threshold not met')
    return text


def validate_completed(arm, expected):
    result = read(RUN / arm / 'complete.json')
    assert result['complete'] and result['smoke'] is False
    assert result['updates_per_trajectory'] == result['actual_optimizer_calls'] == 300
    assert result['full_checkpoint_roundtrip'] and result['source_and_frozen_checks_passed']
    assert result['final_source30_count'] == 30 and result['source_files'] == expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required; nothing loaded or launched')
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    record_path = RUN / 'recovery-CD.launch.json'
    if record_path.exists() or (RUN / 'comparison.json').exists():
        raise RuntimeError('Recovery/result exists; no automatic retry or overwrite')
    for arm in 'CD':
        assert not (RUN / arm).exists() and not (RUN / f'{arm}.log').exists()
    original = read(Path(str(RUN) + '.launch.json'))
    assert original['exit_code'] == 1
    original_log = Path(str(RUN) + '.log').read_text()
    assert 'Defer: at least 35% free memory required' in original_log
    expected = read(RUN / 'source-hashes.json')
    assert own_hashes() == expected == original['source_files']
    deadline = read(RUN / 'launch.json')['deadline_unix']
    assert time.time() < deadline
    results = [validate_completed(arm, expected) for arm in 'AB']
    assert results[0]['schedule'] == results[1]['schedule']
    record = {'started_unix': time.time(), 'original_deadline_unix': deadline,
              'original_supervisor_exit': 1, 'arms_to_launch': ['C', 'D'],
              'source_files': expected, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'memory_threshold_percent': 35, 'worker_rss_limit_bytes': 10 * 1024**3,
              'automatic_retry': False, 'completed_arms_rerun': False,
              'production_approved': False, 'children': []}
    write(record_path, record)
    try:
        for arm in 'CD':
            assert own_hashes() == expected
            pressure = memory_guard()
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError('Original four-hour deadline expired; no extension')
            command = ['/usr/bin/time', '-l', 'caffeinate', '-i', str(ROOT / 'research/.venv/bin/python'),
                       str(EXPERIMENT / 'runner.py'), '--execute', '--output', str(RUN / arm),
                       '--worker', arm, '--deadline', str(deadline), '--source-hashes', str(RUN / 'source-hashes.json')]
            child = {'arm': arm, 'command': command, 'memory_before': pressure, 'started_unix': time.time()}
            record['children'].append(child)
            with (RUN / f'{arm}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    child['wrapper_pid'] = process.pid
                    write(record_path, record)
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Original total deadline expired; terminated only this recovery child group')
                finally:
                    # The child owns a new session. Clean up even if its wrapper
                    # exited first, launch-record writing failed, or this parent
                    # received SIGINT/SIGTERM; never leave an orphan training job.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    child.update(exit_code=process.returncode, ended_unix=time.time())
            write(record_path, record)
            if code != 0:
                raise RuntimeError(f'{arm} exited {code}; no retry')
            results.append(validate_completed(arm, expected))
        assert own_hashes() == expected
        assert len({r['initial_student_state_sha256'] for r in results}) == 1
        assert len({r['initial_rng_sha256'] for r in results}) == 1
        assert all(r['schedule'] == results[0]['schedule'] for r in results)
        assert time.time() <= deadline
        comparison = {'complete': True, 'smoke': False, 'arms': list('ABCD'),
                      'initial_weights_and_sampling_matched': True, 'source_files': expected,
                      'execution_status': 'completed_after_between_arm_memory_deferral',
                      'original_supervisor_exit': 1, 'original_deadline_unix': deadline,
                      'quality_approved': False, 'production_approved': False,
                      'results': [{k: r[k] for k in ['arm', 'peak_rss_bytes', 'wall_seconds', 'final_student_state_sha256']} for r in results]}
        write(RUN / 'comparison.json', comparison)
        record.update(complete=True, exit_code=0, ended_unix=time.time())
    except BaseException as exc:
        record.update(complete=False, exit_code=1, error=repr(exc), ended_unix=time.time())
        write(record_path, record)
        raise
    write(record_path, record)
    print('C/D completed under original deadline', flush=True)


if __name__ == '__main__':
    main()
