"""Bind manually observed PNG reviews to immutable image/evaluation hashes; no model imports."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def save(arm,step,notes,summary,contact_count=14):
 d=ROOT/arm/f'preview-{step:03}';e=json.loads((d/'evaluation.json').read_text())
 assert e['fork_step']==step and e['count']==contact_count
 rows=[]
 for r in e['rows']:
  im=d/r['outputs']['gray']['path'];assert sha(im)==r['outputs']['gray']['sha256']
  rows.append({'id':r['id'],'split':r['split'],'image_sha256':sha(im),'review_resolution':'native1024' if r['id'] in notes else 'contact256','visual_note':notes.get(r['id'],'Viewed in full contact sheet; no broad anatomical collapse apparent at contact resolution. Native detail not assessed at this milestone.')})
 out={'arm':arm,'fork_step':step,'parent_step':600,'reviewed_unix':time.time(),'review_kind':'PNG-only; no inference','evaluation_sha256':sha(d/'evaluation.json'),'student_state_sha256':e['state_sha256'],'contact_sha256':sha(d/'contact.png'),'contact_count':contact_count,'native_review_ids':list(notes),'summary':summary,'new_broad_face_collapse_observed':False,'production_approved':False,'limits':'Small fixed panel. Contact-level coherence does not establish native photographic quality, identity preservation or distributional diversity. Held-out cases remain excluded from training.','rows':rows}
 p=ROOT/'reviews'/f'{arm}-{step:03}.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(out,indent=2)+'\n');print(p)
if __name__=='__main__':
 spec=json.loads(Path(sys.argv[1]).read_text());save(**spec)
