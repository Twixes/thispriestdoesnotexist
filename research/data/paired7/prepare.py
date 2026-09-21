"""Prospective manual seven-pair masks and PNG diagnostics; no model inference."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent/'paired4-tabs/manifest-proposed.json'
SIZE=1024
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads(BASE.read_text())
manifest=copy.deepcopy(original)
manifest['purpose']='Prospective seven-pair region-loss research dataset: six training seeds, one unchanged held-out seed'
manifest['candidate_pending_root_review']=True
manifest['training_eligible']=False
polygons={
'055': {
'clothing_polygons': [[[0,.755],[.17,.68],[.20,.59],[.225,.55],[.24,.65],[.26,.73],[.29,.80],[.345,.86],[.415,.902],[.495,.926],[.555,.918],[.61,.893],[.661,.854],[.700,.80],[.727,.725],[.750,.641],[.747,.586],[.771,.566],[.81,.642],[1,.67],[1,1],[0,1]]],
'collar_polygons': [[[.465,.915],[.513,.916],[.579,.904],[.571,.961],[.551,.973],[.498,.975],[.463,.971]]]},
'051': {
'clothing_polygons': [[[0,.855],[.11,.861],[.17,.805],[.181,.746],[.209,.798],[.258,.865],[.325,.93],[.407,.962],[.49,.970],[.55,.956],[.608,.924],[.638,.888],[.655,.843],[.683,.898],[.75,.933],[1,1],[0,1]]],
'collar_polygons': [[[.395,.948],[.457,.958],[.524,.956],[.522,1],[.389,1]]]},
'020': {
'clothing_polygons': [[[0,.59],[.146,.570],[.165,.59],[.184,.616],[.21,.646],[.24,.735],[.277,.810],[.341,.881],[.421,.94],[.481,.965],[.53,.965],[.58,.937],[.615,.89],[.65,.824],[.69,.75],[.736,.677],[.77,.602],[.792,.552],[.792,.469],[.815,.388],[1,.46],[1,1],[0,1]]],
'collar_polygons': [[[.422,.921],[.456,.938],[.491,.943],[.523,.934],[.517,.963],[.507,.970],[.472,.975],[.44,.973],[.408,.965]]]},
}
def mask(poly):
 im=Image.new('L',(SIZE,SIZE),0)
 for points in poly: ImageDraw.Draw(im).polygon([tuple(round(v*(SIZE-1)) for v in p) for p in points],fill=255)
 return np.asarray(im)>0

# Both manifest directories are sibling directories; relative paths remain exact.
for before,after in zip(original['pairs'],manifest['pairs']):
 assert before==after
 for field in ['source_path','target_path','latent_path']:
  assert (BASE.parent/before[field]).resolve()==(HERE/after[field]).resolve()
assert (BASE.parent/original['source_bundle']).resolve()==(HERE/manifest['source_bundle']).resolve()
source_manifest=HERE.parent/'ffhq-paired-sources64-b2/manifest.json'
source_entries={p['id']:p for p in json.loads(source_manifest.read_text())['entries']}
records=[]
for ident in ['055','051','020']:
 row={'id':f'b2-{ident}','split':'train','source_path':f'../ffhq-paired-sources64-b2/images/{ident}.png','latent_path':f'../ffhq-paired-sources64-b2/latents/{ident}.npz','target_path':f'../ffhq-clothing-edits-b2/{ident}.png',**polygons[ident]}
 manifest['pairs'].append(row)
 m,t=mask(row['clothing_polygons']),mask(row['collar_polygons'])
 c=m&t
 assert c.any() and (m&~c).any()
 source=(HERE/row['source_path']).resolve(); target=(HERE/row['target_path']).resolve(); latent=(HERE/row['latent_path']).resolve()
 assert sha(source)==source_entries[ident]['image_sha256']
 assert sha(latent)==source_entries[ident]['latents_sha256']
 diagnostic=Image.new('RGB',(SIZE*2,SIZE+48),'#151515')
 for index,path in enumerate([source,target]):
  im=Image.open(path).convert('RGBA').resize((SIZE,SIZE),Image.Resampling.LANCZOS)
  layer=np.zeros((SIZE,SIZE,4),dtype=np.uint8)
  layer[m]=[255,40,60,65]
  layer[c]=[255,220,0,120]
  im=Image.alpha_composite(im,Image.fromarray(layer))
  d=ImageDraw.Draw(im)
  for field,color in [('clothing_polygons',(0,220,255,255)),('collar_polygons',(0,255,80,255))]:
   for pts in row[field]:
    pts=[tuple(round(v*(SIZE-1)) for v in p) for p in pts]
    d.line(pts+[pts[0]],fill=color,width=3)
  diagnostic.paste(im.convert('RGB'),(index*SIZE,48))
  ImageDraw.Draw(diagnostic).text((index*SIZE+12,14),f"{row['id']} {'SOURCE' if index==0 else 'TARGET'} | red: clothing; cyan: boundary; green: raw tab; yellow: effective tab",fill='white')
 diagnostic.save(HERE/f"{row['id']}-overlay.png")
 records.append({'id':row['id'],'source_sha256':sha(source),'target_sha256':sha(target),'latent_sha256':sha(latent),'clothing_pixels':int(m.sum()),'raw_tab_pixels':int(t.sum()),'effective_tab_pixels':int(c.sum()),'rest_clothing_pixels':int((m&~c).sum()),'raw_tab_retained_fraction':float(c.sum()/t.sum()),'effective_tab_fraction_of_clothing':float(c.sum()/m.sum()),'face_protection':'Manual source jaw contour; collar is intersected with clothing, never expands the clothing boundary.'})
assert manifest['pairs'][:4]==original['pairs']
assert sum(p['split']=='train' for p in manifest['pairs'])==6
assert [p for p in manifest['pairs'] if p['split']=='validation']==[p for p in original['pairs'] if p['split']=='validation']
assert len({str((HERE/p['latent_path']).resolve()) for p in manifest['pairs']})==7
(HERE/'manifest-proposed.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'regions.json').write_text(json.dumps({'coordinate_system':'Manually traced normalized x,y; all masks rasterized at1024 with round(coord*1023).','effective_collar':'raw tab intersection clothing','new_pairs':records},indent=2)+'\n')
(HERE/'provenance.json').write_text(json.dumps({'base_manifest_sha256':sha(BASE),'new_source_manifest_sha256':sha(source_manifest),'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(HERE/'manifest-proposed.json'),'prior_four_entries_byte_equivalent_as_json_objects':True,'prior_paths_resolve_to_same_files':True,'same_source_bundle':True,'heldout028_unchanged':True,'model_compute':False,'training':False},indent=2)+'\n')
contact=Image.new('RGB',(1024,3*536),'#151515')
for i,ident in enumerate(['055','051','020']):
 with Image.open(HERE/f'b2-{ident}-overlay.png') as im: contact.paste(im.resize((1024,536),Image.Resampling.LANCZOS),(0,i*536))
contact.save(HERE/'mask-contact-sheet.png')
print(json.dumps(records,indent=2))
