"""Launch the bounded six-identity experiment with saved command and memory evidence."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'research/runs/paired-regions1024-sixpair-600'
MANIFEST = ROOT / 'research/data/paired7/manifest-proposed-v2.json'
EXPECTED_MANIFEST = '9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6'


def main():
    if RUN.exists() or RUN.with_suffix('.log').exists():
        raise ValueError('Fresh run only; never overwrite or restart an existing run')
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Reviewed manifest changed')
    preflight = json.loads((MANIFEST.parent / 'source-preflight-v2/preflight.json').read_text())
    if not preflight['passed'] or preflight['manifest_sha256'] != EXPECTED_MANIFEST:
        raise ValueError('Missing matching seven-pair source preflight')
    pressure = subprocess.run(['memory_pressure'], capture_output=True, text=True, check=True).stdout
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1))
    RUN.with_suffix('.memory-before.txt').write_text(pressure)
    if free < 25:
        raise RuntimeError(f'Fresh memory guard requires 25%, observed {free}%')
    command = ['/usr/bin/time', '-l', 'caffeinate', '-i', str(ROOT / 'research/.venv/bin/python'), '-u', '-c',
               'import runpy, torch; torch.set_num_interop_threads(1); runpy.run_path("research/experiments/paired_regions/trainer.py", run_name="__main__")',
               '--manifest', str(MANIFEST.relative_to(ROOT)), '--run', str(RUN.relative_to(ROOT)),
               '--device', 'cpu', '--threads', '2', '--total-steps', '600', '--batch', '1',
               '--lr', '0.0001', '--clothing-weight', '1', '--protected-weight', '1', '--fresh-weight', '1',
               '--upper-fraction', '0.75', '--seed', '20260921', '--checkpoint-every', '100', '--preview-count', '4']
    record = {'command': command, 'cwd': str(ROOT), 'free_memory_percent': free, 'manifest_sha256': EXPECTED_MANIFEST,
              'launch_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fresh_original_source': True, 'production_approved': False, 'interop_threads': 1,
              'purpose': 'Test six-identity generalization with unchanged region-balanced loss and frozen face-generating layers. 028 stays held out.',
              'planned_updates': 600, 'estimated_minutes_from_short_cpu_benchmark': 46.2,
              'estimate_caveat': 'Additional verification, checkpoint and preview overhead; concurrent MPS runs may change throughput.'}
    record_path = RUN.with_suffix('.launch.json')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    started = time.monotonic()
    with RUN.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['wrapper_child_pid'] = child.pid
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        exit_code = child.wait()
    record.update(exit_code=exit_code, elapsed_seconds=time.monotonic()-started)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Training terminal exit {exit_code}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
