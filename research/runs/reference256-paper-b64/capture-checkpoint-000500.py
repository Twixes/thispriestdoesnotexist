"""Provisional disk-only capture; DOES NOT establish the schema or numeric checkpoint step."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

RUN=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def observation():
    metrics=json.loads((RUN/'metrics.jsonl').read_text().splitlines()[-1])
    assert 500<=metrics['step']<625, 'Outside the save-order capture window; no checkpoint-step inference allowed'
    required=[RUN/name for name in ['resume.pt','generator-000500.pt','samples-000500-ema.png','samples-000500-raw.png','samples-000500-untruncated.png']]
    assert all(path.is_file() for path in required), 'Missing milestone files'
    assert not (RUN/'generator-000625.pt').exists(), 'Next durable export already exists'
    assert (RUN/'resume.pt').stat().st_mtime_ns <= (RUN/'generator-000500.pt').stat().st_mtime_ns, 'Live resume newer than500 export'
    return {'metrics_step':metrics['step'],'images_seen':metrics['images_seen'],
            'hashes':{path.name:digest(path) for path in required}}


def main():
    target=RUN/'resume-000500-candidate.pt';temporary=RUN/'resume-000500-candidate.pt.copying';metadata=RUN/'resume-000500-candidate.json'
    if any(p.exists() for p in [target,temporary,metadata]):raise RuntimeError('Refuse overwrite; inspect any previous capture manually')
    before=observation()
    shutil.copyfile(RUN/'resume.pt',temporary)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    copied=digest(temporary);after=observation()
    assert before['hashes']==after['hashes'] and copied==before['hashes']['resume.pt'], 'Artifacts changed during provisional copy'
    os.replace(temporary,target)
    fd=os.open(RUN,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    assert digest(target)==copied
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'status':'UNVALIDATED_CANDIDATE',
            'production_approved':False,'numeric_validation_performed':False,'checkpoint_step_verified':False,
            'candidate':target.name,'candidate_sha256':copied,'before':before,'after':after,
            'claim':'Exact stable live-resume bytes captured in checkpoint500 save-order window; tensor/schema step not inspected',
            'capture_script_sha256':digest(Path(__file__))}
    metadata.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
