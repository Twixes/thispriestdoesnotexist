"""Root-authorized early stop after a complete periodic checkpoint; never restart."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
run=Path(__file__).resolve().parent
pid=77688
report=run/'stop-after-4750.json'
if report.exists():raise SystemExit('Already recorded; refusing repeated signal')
for _ in range(36):
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if p.returncode or 'research/train.py --device mps' not in p.stdout or '--run aligned110-frozen4-cdc1000' not in p.stdout:
        raise SystemExit('Expected live process missing; no signal sent')
    lines=(run/'metrics.jsonl').read_text().splitlines()
    last=json.loads(lines[-1])['step']
    g=run/'generator-004750.pt'; resume=run/'resume.pt'
    if last>=4760 and g.exists() and resume.stat().st_mtime>=g.stat().st_mtime and not (run/'resume.tmp.pt').exists():
        with resume.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
        os.kill(pid,signal.SIGINT)
        report.write_text(json.dumps({'pid':pid,'signal':'SIGINT','last_logged_step':last,'expected_checkpoint_step':4750,'resume_sha256_before_signal':h,'sent_unix':time.time(),'reason':'Root and independent native reviews through4500 show persistent facial artifacts and narrow variety; stop unchanged CDC recipe, retain reference run.','quality_approved':False,'checkpoint_validation_pending':True},indent=2)+'\n')
        print('Sent SIGINT after4750 checkpoint; numeric validation pending',flush=True)
        break
    time.sleep(5)
else:raise SystemExit('Watch window expired; no signal sent, no restart')
