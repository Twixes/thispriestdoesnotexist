"""Verify every public deployed file, including hydrated Git LFS image bytes."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
paths=[p for p in (ROOT/'public').rglob('*') if p.is_file() and not p.name.startswith('_')]
def verify(path):
    relative=path.relative_to(ROOT/'public').as_posix()
    route='' if relative=='index.html' else relative
    remote=subprocess.check_output(['curl','-fLsS','--max-time','40','https://thispriestdoesnotexist.com/'+route])
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    actual=hashlib.sha256(remote).hexdigest()
    if expected!=actual:raise RuntimeError(f'Mismatch: {relative}')
    return {'file':relative,'sha256':actual,'bytes':len(remote)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(verify,paths))
commit=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()
report={'commit':commit,'site':'https://thispriestdoesnotexist.com','ci':'Cloudflare Workers Builds','branch':'master','verified_files':results}
(ROOT/f'research/runs/deployment-{commit}.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'All {len(results)} live assets match local files; Git LFS hydration verified.')
