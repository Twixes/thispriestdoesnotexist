"""Package manual visual labels with source hashes; no model or score imports."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
batches={'v1':'research/data/ffhq-paired-sources32','b2':'research/data/ffhq-paired-sources64-b2'}
native_current={'v1':{'000','004','005','009','016','017','023','024','025','028','030'},'b2':{'002','013','021'}}
native_previous={'v1':set(),'b2':{'001','007','012','015','020','025','040','042','050','051','055','058','059','060','061'}}
manifest={k:json.loads((ROOT/v/'manifest.json').read_text()) for k,v in batches.items()}
by_id={k:{r['id']:r for r in m['entries']} for k,m in manifest.items()}
rows=[]
with (HERE/'annotations.tsv').open() as stream:
 for a in csv.DictReader(stream,delimiter='\t'):
  key,ident=a['batch'],a['id'];source=by_id[key][ident]
  source_path=ROOT/batches[key]/source['source_path']
  checksum=sha(source_path)
  assert checksum==source['image_sha256']
  labels={k:{'y':'yes','n':'no','u':'uncertain'}[a[k]] for k in ['adult','masculine','headwear','extra_people','usable_neck','facial_artifact_concerns','hot_candidate']}
  apparent=('yes' if labels['adult']==labels['masculine']=='yes' else 'no' if 'no' in [labels['adult'],labels['masculine']] else 'uncertain')
  scope='contact_and_native_current_review' if ident in native_current[key] else 'contact_current_and_native_prior_same_source_review' if ident in native_previous[key] else 'contact_only'
  hot_assessed=labels['adult']==labels['masculine']=='yes'
  assert labels['hot_candidate']!='yes' or hot_assessed
  rows.append({'id':f'{key}-{ident}','batch':key,'source_id':ident,'source_path':str(source_path.relative_to(ROOT)),'source_sha256':checksum,'seed_hex':source['seed_hex'],'review_scope':scope,'apparent_adult':labels.pop('adult'),'apparent_masculine_presentation':labels.pop('masculine'),'apparent_adult_male':apparent,**labels,'hot_assessment_performed':hot_assessed,'visual_note':a['note'],'production_approved':False})
assert len(rows)==96 and len({r['id'] for r in rows})==96
for key,count in [('v1',32),('b2',64)]:
 assert {r['source_id'] for r in rows if r['batch']==key}=={f'{i:03}' for i in range(count)}
summary={k:dict(Counter(r[k] for r in rows)) for k in ['review_scope','apparent_adult','apparent_adult_male','headwear','extra_people','usable_neck','facial_artifact_concerns','hot_candidate']}
strict=[r['id'] for r in rows if r['apparent_adult_male']=='yes' and r['headwear']=='no' and r['extra_people']=='no' and r['usable_neck']=='yes' and r['facial_artifact_concerns']=='no' and r['hot_candidate']=='yes']
result={'schema_version':1,'purpose':'Subjective visual research annotations for studying rejection samplers; not validated human ground truth or production approval','annotator':'Single Codex visual reviewer; manually authored labels, not a CLIP score or automated demographic classifier','label_date':'2026-09-21','source_path_base':'repository root','source_count':96,'sources_preserved':True,'production_approved':False,'scope':'Exactly indices000..031 from v1 and000..063 from b2; excludes extra reproduction-sample002 calibration record','score_blinding':'No CLIP scores, ranking results, model predictions, or source-selection JSON were loaded for this annotation pass. Reviewer had prior visual familiarity with some sources and edits, so this is not a blinded independent human study. Labels refer only to original PNGs.','model_compute_performed':False,'definitions':{'apparent_adult':'yes only when visual appearance is clearly adult; no for clearly childlike; uncertain for youthful or ambiguous cases. No numeric age claim.','apparent_adult_male':'Visual eligibility proxy: conjunction of clearly adult and masculine presentation; not a claim about a real gender identity.','headwear':'Visible cap, hat, scarf, helmet, crown or substantial head/hair ornament; ordinary hair and glasses alone are not headwear.','extra_people':'Another visible person, face/body fragment or recognizable crowd, including partial background people. Ambiguous blur or ownership is uncertain.','usable_neck':'Enough visible lower-neck/clothing area for a plausible collar without recropping or moving the face; yes/no/uncertain.','facial_artifact_concerns':'yes for visible facial anatomy/rendering concerns; no means no obvious concern at reviewed native size, not guaranteed perfect; contact-only fine detail is uncertain. Normal wrinkles, asymmetry and tooth spacing are not defects by themselves.','hot_candidate':'Subjective appeal of clearly adult masculine face for a calendar-style aesthetic, independent of hat/extra-person composition. For ineligible age/presentation, no means not a candidate, not unattractive; uncertain means no adult attractiveness judgment or reviewer uncertainty.','hot_assessment_performed':'false whenever age or masculine presentation is not yes; no attractiveness assessment of minors or age-ambiguous subjects.'},'provenance':{key:{'manifest_path':batches[key]+'/manifest.json','manifest_sha256':sha(ROOT/batches[key]/'manifest.json'),'contact_path':batches[key]+'/contact.png','contact_sha256':sha(ROOT/batches[key]/'contact.png')} for key in batches},'annotation_tsv_sha256':sha(HERE/'annotations.tsv'),'packaging_script_sha256':sha(Path(__file__)),'summary':summary,'strict_all_labels_yes_no_example_ids':strict,'labels':rows}
(HERE/'labels.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'count':len(rows),'summary':summary,'strict_example_ids':strict,'labels_sha256':sha(HERE/'labels.json')},indent=2))
