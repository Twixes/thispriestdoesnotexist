"""Manual source skin/cloth boundary refinement; preserve all v1 evidence."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
SIZE=1024
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=HERE/'manifest-proposed.json'
original=json.loads(base.read_text())
updated=copy.deepcopy(original)
old=next(p for p in original['pairs'] if p['id']=='b2-055')
new=next(p for p in updated['pairs'] if p['id']=='b2-055')
# Native 1024-pixel hand-traced points. Keep the outer garment boundary unchanged;
# follow just outside the visible skin at the cheeks and jaw, not the old sweater.
left_outer=[[0,.755],[.17,.68],[.20,.59],[.225,.55]]
right_outer=[[.771,.566],[.81,.642],[1,.67],[1,1],[0,1]]
boundary_pixels=[[242,598],[246,637],[252,676],[267,738],[289,791],[319,835],[367,881],[419,916],[465,936],[513,943],[562,940],[613,924],[647,895],[678,860],[701,820],[719,780],[733,733],[743,683],[750,631],[755,580]]
new['clothing_polygons']=[left_outer+[[x/1023,y/1023] for x,y in boundary_pixels]+right_outer]
def mask(polys):
 im=Image.new('L',(SIZE,SIZE),0)
 for poly in polys:ImageDraw.Draw(im).polygon([tuple(round(x*(SIZE-1)) for x in pt) for pt in poly],fill=255)
 return np.asarray(im)>0
m0=mask(old['clothing_polygons']);m1=mask(new['clothing_polygons']);tab=mask(new['collar_polygons'])
added=m1&~m0;removed=m0&~m1;effective=m1&tab
assert added.any(), 'Boundary must include previously protected cloth'
for before,after in zip(original['pairs'],updated['pairs']):
 if before['id']!='b2-055':assert before==after
 else:assert {k:v for k,v in before.items() if k!='clothing_polygons'}=={k:v for k,v in after.items() if k!='clothing_polygons'}
canvas=Image.new('RGB',(2048,1072),'#171717')
for index,key in enumerate(['source_path','target_path']):
 im=Image.open((HERE/new[key]).resolve()).convert('RGBA').resize((SIZE,SIZE),Image.Resampling.LANCZOS)
 layer=np.zeros((SIZE,SIZE,4),np.uint8);layer[m1]=[255,40,60,60];layer[added]=[0,180,255,140];layer[effective]=[255,220,0,115]
 im=Image.alpha_composite(im,Image.fromarray(layer))
 d=ImageDraw.Draw(im)
 for polys,color in [(old['clothing_polygons'],(255,0,230,255)),(new['clothing_polygons'],(0,255,255,255)),(new['collar_polygons'],(0,255,80,255))]:
  for poly in polys:
   pts=[tuple(round(x*(SIZE-1)) for x in pt) for pt in poly];d.line(pts+[pts[0]],fill=color,width=2)
 canvas.paste(im.convert('RGB'),(index*1024,48))
 ImageDraw.Draw(canvas).text((index*1024+8,15),f"055 v2 {key}: cyan=new boundary, magenta=old; blue=added clothing; yellow=effective tab",fill='white')
canvas.save(HERE/'b2-055-overlay-v2.png')
# Same three-pair contact layout; other native diagnostics are reused unchanged.
contact=Image.new('RGB',(1024,1608),'#151515')
for i,name in enumerate(['b2-055-overlay-v2.png','b2-051-overlay.png','b2-020-overlay.png']):
 with Image.open(HERE/name) as im:contact.paste(im.resize((1024,536),Image.Resampling.LANCZOS),(0,i*536))
contact.save(HERE/'mask-contact-sheet-v2.png')
(HERE/'manifest-proposed-v2.json').write_text(json.dumps(updated,indent=2)+'\n')
report={'base_manifest_sha256':sha(base),'candidate_manifest_sha256':sha(HERE/'manifest-proposed-v2.json'),'script_sha256':sha(Path(__file__)),'changed_pair':'b2-055','changed_field_only':'clothing_polygons','all_other_pairs_and_tab_polygons_unchanged':True,'old_clothing_pixels':int(m0.sum()),'new_clothing_pixels':int(m1.sum()),'previously_protected_pixels_added_to_clothing':int(added.sum()),'old_clothing_pixels_removed':int(removed.sum()),'left_side_added_pixels':int(added[:,:450].sum()),'central_chin_added_pixels':int(added[:,450:580].sum()),'right_side_added_pixels':int(added[:,580:].sum()),'raw_tab_pixels':int(tab.sum()),'old_effective_tab_pixels':int((m0&tab).sum()),'new_effective_tab_pixels':int(effective.sum()),'raw_tab_retained_fraction':float(effective.sum()/tab.sum()),'native_manual_boundary_points':boundary_pixels,'source_native_sha256':sha((HERE/new['source_path']).resolve()),'mask_method':'Manual visual trace just outside source skin/cloth boundary; no automatic skin classification. Added pixels reported geometrically, not claimed semantic ground truth.','model_compute':False,'training':False}
(HERE/'refinement-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
