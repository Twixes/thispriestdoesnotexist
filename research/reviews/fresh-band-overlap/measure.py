"""Deterministic saved-polygon audit, with no torch imports or model/image inference."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RES=1024
BAND_HEIGHT=max(1,round(RES*.75))
BAND=np.zeros((RES,RES),dtype=bool);BAND[:BAND_HEIGHT]=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'current6':'research/data/paired7/manifest-proposed-v2.json','proposed9':'research/data/paired10/manifest-proposed.json'}
expected={'current6':'9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6','proposed9':'8d0660f1ccf5115a356e9e6b479e4d225c1e9993402fc71575b0fcb9ed8dcff1'}
def mask(polys):
 canvas=Image.new('L',(RES,RES),0)
 for poly in polys:ImageDraw.Draw(canvas).polygon([tuple(round(v*(RES-1)) for v in point) for point in poly],fill=255)
 return np.asarray(canvas)>0
cohorts={}
for cohort,path in paths.items():
 p=ROOT/path;assert sha(p)==expected[cohort],f'Input changed: {p}'
 manifest=json.loads(p.read_text());rows=[];regions_by_name={n:[] for n in ['clothing','raw_tab','effective_tab','rest_clothing','protected']}
 for pair in manifest['pairs']:
  if pair['split']!='train':continue
  clothing=mask(pair['clothing_polygons']);raw=mask(pair['collar_polygons']);tab=clothing&raw
  regions={'clothing':clothing,'raw_tab':raw,'effective_tab':tab,'rest_clothing':clothing&~tab,'protected':~clothing}
  row={'id':pair['id'],'regions':{}}
  for name,m in regions.items():
   area=int(m.sum());overlap=int((m&BAND).sum());regions_by_name[name].append(m)
   row['regions'][name]={'pixels':area,'fraction_of_image':area/(RES*RES),'band_overlap_pixels':overlap,'fraction_of_region_in_band':overlap/area,'fraction_of_band_covered':overlap/int(BAND.sum())}
  tab_fraction=row['regions']['effective_tab']['fraction_of_region_in_band'];rest_fraction=row['regions']['rest_clothing']['fraction_of_region_in_band']
  row['normalized_spatial_loss_mass_in_band']={'tab_component':.5*tab_fraction,'rest_component':.5*rest_fraction,'paired_clothing_total':.5*(tab_fraction+rest_fraction),'paired_protected_component':row['regions']['protected']['fraction_of_region_in_band'],'fresh_band_at_clothing_coordinates':row['regions']['clothing']['fraction_of_band_covered']}
  rows.append(row)
 n=len(rows);summary={}
 for name,masks in regions_by_name.items():
  total=sum(int(m.sum()) for m in masks);overlap=sum(int((m&BAND).sum()) for m in masks);union=np.logical_or.reduce(masks)
  summary[name]={'sum_region_pixels_across_seeds':total,'sum_overlap_pixels_across_seeds':overlap,'pooled_region_fraction_in_band':overlap/total,'mean_fraction_of_region_in_band':float(np.mean([r['regions'][name]['fraction_of_region_in_band'] for r in rows])),'mean_fraction_of_band_covered':overlap/(n*int(BAND.sum())),'union_region_pixels':int(union.sum()),'union_band_overlap_pixels':int((union&BAND).sum()),'union_fraction_of_band_covered':float((union&BAND).sum()/BAND.sum())}
 means={name:float(np.mean([r['normalized_spatial_loss_mass_in_band'][name] for r in rows])) for name in rows[0]['normalized_spatial_loss_mass_in_band']}
 cohorts[cohort]={'manifest_path':path,'manifest_sha256':sha(p),'train_count':n,'excluded_validation_ids':[r['id'] for r in manifest['pairs'] if r['split']!='train'],'per_pair':rows,'aggregate':summary,'uniform_seed_mean_normalized_spatial_loss_mass_in_band':means}
config=ROOT/'research/runs/paired-regions1024-sixpair-600/config.json'
cfg=json.loads(config.read_text())
assert cfg['provenance']['manifest_sha256']==expected['current6']
assert cfg['options']['upper_fraction']==.75
result={'resolution':RES,'upper_fraction':.75,'band_height':BAND_HEIGHT,'band_rows_inclusive':[0,BAND_HEIGHT-1],'band_pixels':int(BAND.sum()),'rasterization':'PIL polygon(fill=255), round(coord*(resolution-1)); mirrors paired_edit.polygon_mask without importing model code','source_files':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/experiments/paired_edit/trainer.py',ROOT/'research/experiments/paired_regions/trainer.py',config]},'active_run_loss_options':{k:cfg['options'][k] for k in ['clothing_weight','protected_weight','fresh_weight','upper_fraction']},'interpretation':'Geometric support only across saved training masks. Fresh preservation samples different independent random z/W images; these are not same-example opposing targets. Spatial loss mass assumes unit absolute error and ignores Jacobian/gradient magnitude. No causal failure inference.','cohorts':cohorts,'model_or_image_inference':False}
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
for cohort,data in cohorts.items():
 print(cohort,data['manifest_sha256'])
 for r in data['per_pair']:
  c=r['regions']['clothing'];t=r['regions']['effective_tab'];print(r['id'],c['pixels'],c['band_overlap_pixels'],round(100*c['fraction_of_region_in_band'],3),t['pixels'],t['band_overlap_pixels'],round(100*r['normalized_spatial_loss_mass_in_band']['paired_clothing_total'],3))
 print('normalized mass',json.dumps(data['uniform_seed_mean_normalized_spatial_loss_mass_in_band']))
