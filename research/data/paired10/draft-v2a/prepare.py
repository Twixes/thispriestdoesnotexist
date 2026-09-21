"""Manual prospective paired10 annotations; PNG diagnostics only, no model inference."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'paired7/manifest-proposed-v2.json'
SIZE=1024
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Hand-traced in source pixel coordinates on 1024px originals. Tab coordinates
# refer to the target rescaled only for diagnostic display (no training asset edit).
raw={
'042':{'clothing':[[0,966],[25,950],[55,900],[118,883],[122,853],[147,880],[197,908],[261,938],[330,965],[400,989],[451,1006],[491,1013],[531,1006],[571,990],[603,968],[618,938],[612,980],[637,1023]],'tab':[[454,975],[475,978],[510,980],[545,978],[575,970],[570,994],[561,1023],[450,1023]]},
'058':{'clothing':[[143,1023],[171,970],[234,910],[314,857],[388,814],[415,863],[437,899],[459,921],[486,934],[521,943],[550,960],[585,965],[643,965],[681,949],[722,921],[771,886],[820,850],[868,805],[909,759],[944,715],[959,683],[968,665],[1023,723],[1023,1023]],'tab':[[560,939],[585,948],[618,951],[654,947],[670,1006],[573,1019]]},
'059':{'clothing':[[0,887],[73,841],[118,824],[158,837],[191,869],[218,884],[243,907],[286,931],[344,951],[411,966],[480,978],[535,979],[590,966],[646,945],[678,917],[698,889],[729,909],[757,944],[785,987],[809,1023],[0,1023]],'tab':[[434,985],[465,990],[500,992],[534,990],[556,988],[557,1000],[555,1023],[429,1023]]},
}
(HERE/'raw-polygon-proposals.json').write_text(json.dumps({'coordinate_system':'1024x1024 pixel coordinates; manual annotation, not automatic segmentation','pairs':raw},indent=2)+'\n')
original=json.loads(BASE.read_text()); manifest=copy.deepcopy(original)
manifest.update(purpose='PROPOSED paired10 region-loss dataset: nine training seeds and unchanged held-out 028',candidate_pending_root_review=True,training_eligible=False,production_approved=False)
def mask(polygons):
 im=Image.new('L',(SIZE,SIZE),0)
 for p in polygons: ImageDraw.Draw(im).polygon([tuple(round(v*1023) for v in pt) for pt in p],fill=255)
 return np.asarray(im)>0
entries={p['id']:p for p in json.loads((HERE.parent/'ffhq-paired-sources64-b2/manifest.json').read_text())['entries']}
records=[]
for ident,coords in raw.items():
 row={'id':f'b2-{ident}','split':'train','source_path':f'../ffhq-paired-sources64-b2/images/{ident}.png','latent_path':f'../ffhq-paired-sources64-b2/latents/{ident}.npz','target_path':f'../ffhq-clothing-edits-b3/{ident}.png','clothing_polygons':[[[x/1023,y/1023] for x,y in coords['clothing']]],'collar_polygons':[[[x/1023,y/1023] for x,y in coords['tab']]]}
 manifest['pairs'].append(row); m,t=mask(row['clothing_polygons']),mask(row['collar_polygons']); c=m&t
 source=(HERE/row['source_path']).resolve();target=(HERE/row['target_path']).resolve();latent=(HERE/row['latent_path']).resolve()
 assert sha(source)==entries[ident]['image_sha256'] and sha(latent)==entries[ident]['latents_sha256']
 assert c.any() and (m&~c).any()
 panel=Image.new('RGB',(2048,1072),'#151515')
 for index,path in enumerate([source,target]):
  im=Image.open(path).convert('RGBA').resize((1024,1024),Image.Resampling.LANCZOS)
  rgba=np.zeros((1024,1024,4),dtype=np.uint8); rgba[m]=[255,40,60,65];rgba[c]=[255,220,0,120]
  im=Image.alpha_composite(im,Image.fromarray(rgba));draw=ImageDraw.Draw(im)
  for field,color in [('clothing_polygons',(0,220,255,255)),('collar_polygons',(0,255,80,255))]:
   for poly in row[field]:
    pts=[tuple(round(v*1023) for v in pt) for pt in poly];draw.line(pts+[pts[0]],fill=color,width=3)
  panel.paste(im.convert('RGB'),(index*1024,48));ImageDraw.Draw(panel).text((index*1024+12,14),f'{row["id"]} {"SOURCE" if index==0 else "TARGET"} | red clothing; cyan boundary; green raw tab; yellow effective tab',fill='white')
 panel.save(HERE/f'b2-{ident}-overlay.png')
 records.append({'id':row['id'],'source_sha256':sha(source),'target_sha256':sha(target),'latent_sha256':sha(latent),'clothing_pixels':int(m.sum()),'raw_tab_pixels':int(t.sum()),'effective_tab_pixels':int(c.sum()),'rest_clothing_pixels':int((m&~c).sum()),'protected_pixels':int((~m).sum()),'raw_tab_retained_fraction':float(c.sum()/t.sum()),'effective_tab_fraction_of_clothing':float(c.sum()/m.sum())})
assert manifest['pairs'][:7]==original['pairs']
for row in original['pairs']:
 for field in ['source_path','target_path','latent_path']: assert (BASE.parent/row[field]).resolve()==(HERE/row[field]).resolve()
assert (BASE.parent/original['source_bundle']).resolve()==(HERE/manifest['source_bundle']).resolve()
assert sum(p['split']=='train' for p in manifest['pairs'])==9
assert [p for p in manifest['pairs'] if p['split']=='validation']==[p for p in original['pairs'] if p['split']=='validation']
assert len({(HERE/p['latent_path']).resolve() for p in manifest['pairs']})==10
(HERE/'manifest-proposed.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'regions.json').write_text(json.dumps({'rasterization':'round(normalized_coordinate * 1023), PIL polygon fill; 1024x1024','effective_tab':'raw tab intersection clothing; face remains protected','new_pairs':records},indent=2)+'\n')
(HERE/'provenance.json').write_text(json.dumps({'base_manifest_sha256':sha(BASE),'manifest_sha256':sha(HERE/'manifest-proposed.json'),'script_sha256':sha(Path(__file__)),'root_edit_review_sha256':sha(HERE.parent/'ffhq-clothing-edits-b3/root-review.json'),'old_seven_records_exact_JSON_equal':True,'prior_paths_unchanged':True,'same_source_bundle':True,'heldout028_unchanged':True,'distinct_latents':10,'split_counts':{'train':9,'validation':1},'model_compute':False,'training':False,'all_new_source_and_latent_hashes_match_source_manifest':True},indent=2)+'\n')
contact=Image.new('RGB',(1024,1608),'#151515')
for i,ident in enumerate(raw):
 with Image.open(HERE/f'b2-{ident}-overlay.png') as im:contact.paste(im.resize((1024,536),Image.Resampling.LANCZOS),(0,i*536))
contact.save(HERE/'mask-contact-sheet.png')
print(json.dumps(records,indent=2))
