"""Preparation only: freeze local source, package metadata, official assets and cohorts."""
import hashlib, importlib.metadata, json, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
files={}
def add(p):files[str(p.relative_to(ROOT))]=sha(p)
for p in BASE.glob('*.py'):add(p)
for p in BASE.glob('*.in'):add(p)
for p in (BASE/'vendor').rglob('*'):
 if p.is_file() and (p.suffix=='.py' or 'LICENSE' in p.name or p.name=='requirements.txt'):add(p)
for x in json.loads((BASE/'asset-downloads.json').read_text()):
 p=BASE/x['path'];assert sha(p)==x['sha256'];add(p)
for name in ['basicsr','facexlib','torchvision']:
 dist=importlib.metadata.distribution(name)
 for rel in dist.files:
  p=Path(dist.locate_file(rel))
  if p.suffix=='.py' or 'LICENSE' in p.name or p.name in ('METADATA','WHEEL'):add(p)
for rel in ['research/reviews/reference625-matched32-visual-review.json','research/runs/reference256-paper-b64-resumed500-to625/matched32-000625/evaluation.json','research/runs/reference256-paper-b64-resumed500-to625/matched32-000625/supervisor-result.json','research/runs/reference256-paper-b64-resumed500-to625/matched32-000625/latents.npz']:
 add(ROOT/rel)
for p in (ROOT/'research/runs/reference256-paper-b64-resumed500-to625/matched32-000625').glob('raw-psi1-[0-9][0-9][0-9].png'):add(p)
for name in ['recipe.json','pins.json','source-worker.py','source-runner.py','latents.npz','inputs.json']:
 add(ROOT/'research/runs/nada-clean24-cpu-continue500'/name)
packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()}
pins={'python_version':sys.version,'packages':dict(sorted(packages.items())),'files':dict(sorted(files.items())),'gfpgan_commit':'7552a7791caad982045a7bbe5634bbf1cd5c8679','nada_prospective_only':{'run':'research/runs/nada-clean24-cpu-continue500','step':500,'indices':list(range(8)),'kind':'student-gray','expected_complete':False,'note':'Future checkpoint SHA supplied independently at launch; actual successful supervisor/worker plus all8 reload-verified preview hashes required.'},'model_load_or_inference_performed':False}
p=BASE/'pins.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(pins,indent=2)+'\n')
(BASE/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n')
print(json.dumps({'files':len(files),'packages':len(packages),'pins_sha256':sha(p),'torch_imported':'torch' in sys.modules}))
