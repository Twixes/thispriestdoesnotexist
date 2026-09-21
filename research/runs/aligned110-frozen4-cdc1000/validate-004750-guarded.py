"""Run the existing mmap checkpoint validator with memory/integrity evidence."""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import runpy
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    output = RUN/'checkpoint-004750-validation.json'
    integrity_path = RUN/'checkpoint-004750-validation-integrity.json'
    if output.exists() or integrity_path.exists():
        raise SystemExit('Refusing to overwrite completed evidence')
    pressure = subprocess.check_output(['memory_pressure'], text=True)
    (RUN/'checkpoint-004750-memory-before.txt').write_text(pressure)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure)[1])
    if free < 25:
        raise SystemExit(f'DEFER: {free}% free is below the required 25%; no torch/model load')
    started = time.perf_counter()
    validator = ROOT/'research/reviews/check_checkpoint.py'
    files = [RUN/'resume.pt', RUN/'generator-004750.pt', validator]
    before = {str(p.relative_to(ROOT)):sha(p) for p in files}
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.argv = [str(validator), '--resume', str(RUN/'resume.pt'), '--generator',
                str(RUN/'generator-004750.pt'), '--step', '4750', '--output', str(output)]
    with (RUN/'checkpoint-004750-validation.log').open('w') as log, contextlib.redirect_stdout(log):
        runpy.run_path(str(validator), run_name='__main__')
    after = {str(p.relative_to(ROOT)):sha(p) for p in files}
    if before != after:
        raise RuntimeError('Checkpoint/generator/code changed during guarded validation; do not archive as verified')
    result = {
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'step':4750,'memory_preflight_free_percent':free,'device':'cpu',
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'mmap':True,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'wall_seconds':time.perf_counter()-started,'before_sha256':before,'after_sha256':after,
        'artifacts_unchanged':True,'wrapper_sha256':sha(Path(__file__)),
        'inference_or_training_performed':False,'active_processes_changed':False,
        'note':'Root reports CDC stopped after step4750; before/after hashes still establish stability only across this validation.',
    }
    integrity_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
