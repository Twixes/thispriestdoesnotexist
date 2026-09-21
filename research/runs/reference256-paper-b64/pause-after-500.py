"""Pause this exact live reference run only after a durable provisional500 copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RUN = Path(__file__).resolve().parent
PID = 82281


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    if not parser.parse_args().execute:
        parser.error('Explicit --execute required; no signal sent')
    output = RUN / 'pause-for-factorial-after500.json'
    assert not output.exists(), 'A pause request exists; inspect actual process before any further action'
    candidate = RUN / 'resume-000500-candidate.pt'
    capture = json.loads((RUN / 'resume-000500-candidate.json').read_text())
    assert capture['status'] == 'UNVALIDATED_CANDIDATE'
    assert capture['checkpoint_step_verified'] is False
    candidate_hash = digest(candidate)
    assert candidate_hash == capture['candidate_sha256'] == digest(RUN / 'resume.pt')
    metrics = json.loads((RUN / 'metrics.jsonl').read_text().splitlines()[-1])
    assert 500 <= metrics['step'] < 625
    assert (RUN / 'generator-000500.pt').is_file()
    command = subprocess.check_output(['ps', '-p', str(PID), '-o', 'command='], text=True).strip()
    assert 'research/experiments/reference_phases/trainer.py ' in command
    assert ' --run research/runs/reference256-paper-b64 ' in command
    assert ' --steps 15625 ' in command
    record = {'requested_unix': time.time(), 'pid': PID, 'verified_command': command,
              'signal': 'SIGINT', 'reason': 'Preserve500 and release MPS memory to finish remaining C/D within original factorial deadline and unchanged memory guards.',
              'candidate_sha256': candidate_hash, 'candidate_numeric_validation_pending': True,
              'last_completed_metrics_step': metrics['step'], 'retained_checkpoint_expected_step': 500,
              'completed_updates_after_retained_checkpoint': metrics['step'] - 500,
              'may_discard_one_additional_in_flight_update': True,
              'production_approved': False, 'terminal_exit_pending': True,
              'pause_script_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(record, indent=2) + '\n')
    os.kill(PID, signal.SIGINT)
    record['signal_sent_unix'] = time.time()
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
