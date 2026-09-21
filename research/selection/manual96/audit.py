"""Independent read-only schema/provenance audit; never reruns label packager."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=HERE/'labels.json';before=sha(path)
data=json.loads(path.read_text());rows=data['labels'];checks=[]
assert data['schema_version']==1 and data['source_count']==len(rows)==96
assert data['production_approved'] is False and data['model_compute_performed'] is False
assert len({r['id'] for r in rows})==96 and len({r['source_sha256'] for r in rows})==96
fields=['apparent_adult','apparent_masculine_presentation','apparent_adult_male','headwear','extra_people','usable_neck','facial_artifact_concerns','hot_candidate']
scopes={'contact_only','contact_and_native_current_review','contact_current_and_native_prior_same_source_review'}
with (HERE/'annotations.tsv').open() as f: tsv={f"{r['batch']}-{r['id']}":r for r in csv.DictReader(f,delimiter='\t')}
assert set(tsv)=={r['id'] for r in rows}
map_fields={'adult':'apparent_adult','masculine':'apparent_masculine_presentation','headwear':'headwear','extra_people':'extra_people','usable_neck':'usable_neck','facial_artifact_concerns':'facial_artifact_concerns','hot_candidate':'hot_candidate'}
source_hashes=0
for batch,count in [('v1',32),('b2',64)]:
 selected=[r for r in rows if r['batch']==batch]
 assert {r['source_id'] for r in selected}=={f'{i:03}' for i in range(count)}
 provenance=data['provenance'][batch]
 mp=ROOT/provenance['manifest_path'];assert sha(mp)==provenance['manifest_sha256']
 cp=ROOT/provenance['contact_path'];assert sha(cp)==provenance['contact_sha256']
 originals={r['id']:r for r in json.loads(mp.read_text())['entries']}
 for r in selected:
  assert all(r[f] in {'yes','no','uncertain'} for f in fields)
  assert r['review_scope'] in scopes and isinstance(r['visual_note'],str) and r['visual_note']
  assert r['production_approved'] is False
  source=originals[r['source_id']];p=(ROOT/r['source_path']).resolve()
  assert p.is_relative_to(ROOT.resolve()) and p.suffix=='.png'
  assert p==(mp.parent/source['source_path']).resolve()
  assert sha(p)==r['source_sha256']==source['image_sha256'];source_hashes+=1
  assert r['seed_hex']==source['seed_hex']
  conjunction='yes' if r['apparent_adult']==r['apparent_masculine_presentation']=='yes' else 'no' if 'no' in [r['apparent_adult'],r['apparent_masculine_presentation']] else 'uncertain'
  assert r['apparent_adult_male']==conjunction
  assert r['hot_assessment_performed']==(conjunction=='yes')
  assert r['hot_candidate']!='yes' or r['hot_assessment_performed']
  for a,b in map_fields.items():assert r[b]=={'y':'yes','n':'no','u':'uncertain'}[tsv[r['id']][a]]
  assert r['visual_note']==tsv[r['id']]['note']
for key,expected in data['summary'].items():assert dict(Counter(r[key] for r in rows))==expected
strict=[r['id'] for r in rows if r['apparent_adult_male']=='yes' and r['headwear']=='no' and r['extra_people']=='no' and r['usable_neck']=='yes' and r['facial_artifact_concerns']=='no' and r['hot_candidate']=='yes']
assert strict==data['strict_all_labels_yes_no_example_ids']
assert sha(HERE/'annotations.tsv')==data['annotation_tsv_sha256']
assert sha(HERE/'build_labels.py')==data['packaging_script_sha256']
assert sha(path)==before
result={'passed':True,'audit_scope':'Independent implementation verifies stored schema, CSV consistency, completeness, summaries, source/manifest/contact/code hashes. It does not independently relabel images or validate subjective judgments.','labels_sha256':before,'audit_script_sha256':sha(Path(__file__)),'records':96,'source_png_hashes_verified':source_hashes,'source_manifests_verified':2,'contact_hashes_verified':2,'unique_ids_and_source_hashes':96,'adult_male_counts':dict(Counter(r['apparent_adult_male'] for r in rows)),'review_scope_counts':dict(Counter(r['review_scope'] for r in rows)),'strict_candidate_ids':strict,'csv_rows_match_json':True,'stored_counts_match_recalculation':True,'labels_unchanged':True,'model_compute':False,'image_edits':False}
(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
