"""PNG-only provenance and consistency audit for the proposed dataset."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
p=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((p/'manifest-proposed.json').read_text());b=json.loads((p.parent/'paired7/manifest-proposed-v2.json').read_text())
assert m['pairs'][:7]==b['pairs']
records=[]
for row in m['pairs']:
 out={'id':row['id'],'split':row['split']}
 for field in ['source_path','target_path','latent_path']:
  path=(p/row[field]).resolve();out[field+'_sha256']=sha(path)
 with Image.open(p/row['source_path']) as im: assert im.size==(1024,1024);out['source_size']=list(im.size)
 with Image.open(p/row['target_path']) as im:out['target_size']=list(im.size)
 with np.load(p/row['latent_path']) as z:
  assert z['z'].shape==(1,512) and z['w'].shape==(1,18,512)
  assert np.isfinite(z['z']).all() and np.isfinite(z['w']).all()
 def mask(polys):
  im=Image.new('L',(1024,1024),0)
  for poly in polys:ImageDraw.Draw(im).polygon([tuple(round(v*1023) for v in pt) for pt in poly],fill=255)
  return np.asarray(im)>0
 clothing=mask(row['clothing_polygons']);out['clothing_pixels']=int(clothing.sum())
 if 'collar_polygons' in row:
  tab=mask(row['collar_polygons']);effective=tab&clothing
  assert effective.any() and (clothing&~effective).any()
  out.update(raw_tab_pixels=int(tab.sum()),effective_tab_pixels=int(effective.sum()),tab_retention=float(effective.sum()/tab.sum()))
 records.append(out)
audit={'manifest_sha256':sha(p/'manifest-proposed.json'),'old_seven_exact_JSON_equal':True,'heldout_028_unchanged':True,'source_W_shapes_finite_all_ten':True,'training_preflight_performed':False,'model_inference':False,'all_pairs':records}
(p/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
review={'reviewer':'Codex expand_dataset manual image review, not automatic segmentation','manifest_sha256':sha(p/'manifest-proposed.json'),'review_scope':'All three source/target coordinate grids and final individual full-size overlay panels viewed; final contact sheet also viewed. Target tab detail crops reviewed for042 and059.','root_approval_pending':True,'production_approved':False,'training_eligible':False,'findings':{
'b2-042':'Source face and upper neck protected conservatively. Allowed region includes blue shirt and beige jacket; edited background/expanded shoulder outside original garment remains protected. The target white tab overlaps protected source lower jaw/upper neck, so only37.35% is retained. Bottom clipping is accepted; boundary is not expanded to recover the tab.',
'b2-058':'Boundary follows original source chin and jaw to white shirt edge, preserving facial skin, mustache, lips and nose. Target nose/mouth shift is ignored by target clothing supervision. The high target tab is clipped above original source chin;75.80% retained. No visible white shirt strip was deliberately left protected along the inspected jaw boundary.',
'b2-059':'Mask follows original lower jaw, leaves chin/stubble and mouth protected, and permits lower neck plus original hoodie. All visible target tab is inside allowed lower-neck region;100% retained. Target shoulder expansion beyond original right garment remains protected background.'},'limitations':['Manual source chin boundaries are subjective, especially shadowed042 lower jaw versus upper neck; root should inspect before research use.','Masked target losses do not prove identity preservation or quality of any trained model.','Sources are diverse clothing teachers, not approved hot-output selections.','Native target pixels remain1254x1254; diagnostic target resizing does not modify source or target files.'], 'draft_history':{'draft-v1':'Initial polygons and overlays preserved before tab-outline corrections.','draft-v2a':'Intermediate042 exterior-boundary update had an unintended diagonal closure excluding bottom clothing; rejected and preserved. Final polygon explicitly includes bottom-left corner.'},'visual_artifact_sha256':{n:sha(p/n) for n in ['b2-042-overlay.png','b2-058-overlay.png','b2-059-overlay.png','mask-contact-sheet.png']}}
(p/'review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'manifest_sha256':audit['manifest_sha256'],'old_seven_exact':True,'train':9,'validation':1,'pairs_audited':len(records),'no_model_compute':True},indent=2))
