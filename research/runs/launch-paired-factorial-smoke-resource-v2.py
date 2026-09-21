"""Root-reviewed resource-v2 real-D smoke launch; records terminal result, never retries."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
run=root/'research/runs/paired-factorial600-smoke-10g'
meta=Path(str(run)+'.launch.json');log=Path(str(run)+'.log')
if run.exists() or meta.exists() or log.exists():raise SystemExit('Output exists; do not silently retry')
exp=root/'research/experiments/paired_factorial'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in exp.glob('*.py')};hashes['parent-pins.json']=hashlib.sha256((exp/'parent-pins.json').read_bytes()).hexdigest()
command=['/usr/bin/time','-l','caffeinate','-i',str(root/'research/.venv/bin/python'),str(exp/'runner.py'),'--execute','--smoke','--output',str(run)]
record={'command':command,'cwd':str(root),'started_unix':time.time(),'source_files':hashes,'purpose':'Two independent one-update arm-D replicas, diagnostic and ordinary, both from original600. No full factorial permission inferred from this smoke.','production_approved':False}
with log.open('w') as stream:
 p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
 record['wrapper_child_pid']=p.pid;meta.write_text(json.dumps(record,indent=2)+'\n')
 code=p.wait()
record.update(exit_code=code,elapsed_seconds=time.time()-record['started_unix']);meta.write_text(json.dumps(record,indent=2)+'\n')
print('Smoke terminal exit',code,flush=True)
raise SystemExit(code)
