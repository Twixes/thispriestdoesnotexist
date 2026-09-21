"""Launch the reviewed bounded probe and retain actual process exit evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'research/runs/paired-surround-600-probe'
PINS = {
    'runner.py': '9fe816ee02f6f341f7dac25f6562dc8e011b394b256c30c4d3b527af5b768ed9',
    'objective.py': 'e601a3119c3096c6e222f50c68edb103e6e91f28a353cc01e47eaba67660b10d',
    'parent-pins.json': 'f92f34d7ed5d8b923e49238d850e2b1979a4271064eb4e70fe6e3065d84b0cfa',
}


def main():
    if RUN.exists() or RUN.with_suffix('.log').exists():
        raise ValueError('Never overwrite or restart an existing probe')
    for name, expected in PINS.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reviewed {name} changed')
    command = ['/usr/bin/time', '-l', 'caffeinate', '-i', sys.executable, '-u',
               str(HERE / 'runner.py'), '--execute', '--output', str(RUN)]
    record = {'command': command, 'cwd': str(ROOT), 'source_sha256': PINS,
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'purpose': 'Two sequential50-update objective forks from the same completed600 checkpoint.',
              'production_approved': False, 'started_unix': time.time()}
    record_path = RUN.with_suffix('.launch.json')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    started = time.monotonic()
    with RUN.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['wrapper_child_pid'] = child.pid
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        code = child.wait()
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-started)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Probe supervisor terminal exit {code}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
