"""Package manual original-PNG shortlist; no model scoring or image generation."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
labels_path=HERE.parent/'manual96/labels.json'
labels={r['id']:r for r in json.loads(labels_path.read_text())['labels']}
specs=[
('b2-042','priority','Bald head, glasses, broad smile, darker skin tone, three-quarter pose and textured jacket provide useful clothing/identity variation. Face appears clearly adult and anatomically plausible; no obvious hat or other person.','Neck is partly angled toward bottom edge; retain natural tooth spacing and exact glasses/face. This is a clothing teacher, not an approved hot output.'),
('b2-058','priority','Clearly mature adult masculine face, grey mustache, receding slicked hair, serious three-quarter pose, jacket/shirt/tie. Different pose and facial-hair case; no obvious other person or hat.','Collar room is restricted beneath chin but existing tie/shirt is visible. Do not redraw mustache or jaw to enlarge collar. Not selected for hot appearance.'),
('b2-059','priority','Clearly mature adult masculine face, short grey hair, small mustache, glasses and open smile. Visible neck and casual layered clothing; no obvious headwear or extra people.','Glasses reflections and natural teeth need preservation; garment/collar reaches frame edge. Subjective hot eligibility remains uncertain.'),
('b2-050','conditional_tight_neck','Clearly mature adult bald masculine face, glasses, gentle smile, textured blue shirt; clean foliage background with no visible person/headwear.','Very little front neck remains beneath chin; only edit if a partial collar can be placed without moving face. Less useful than priority three; not hot-qualified.'),
('b2-060','conditional_tight_neck','Clearly mature adult masculine face, short hair, glasses, neutral expression and dark jacket/white shirt, against an uncluttered blurred background.','Chin/neck are close to lower frame; a small partial tab may be possible, but do not move jaw or recrop. Subjective hot eligibility remains uncertain.'),
]
records=[]
for ident,status,reason,caveat in specs:
 r=labels[ident];p=ROOT/r['source_path'];assert sha(p)==r['source_sha256']
 assert r['apparent_adult_male']=='yes' and r['headwear']=='no' and r['extra_people']=='no'
 records.append({'id':ident,'status':status,'source_path':r['source_path'],'source_absolute_path':str(p),'source_sha256':sha(p),'native_png_inspected_this_pass':True,'reason':reason,'caveat':caveat,'manual96_hot_candidate':r['hot_candidate'],'production_approved':False,'generation_authorized_by_this_file':False,'generation_owner':('root_in_progress_do_not_duplicate' if ident=='b2-059' else 'not_assigned'),'root_native_suitability_review':('suitable_clothing_teacher_not_hot_approval' if ident in ['b2-042','b2-059','b2-060'] else 'pending')})
reserved=[]
for ident,note in [('v1-017','Attractive adult full-beard/glasses case, but beard leaves very little front collar room.'),('v1-024','Mature adult curly grey hair, glasses, stubble and three-quarter pose; neck visible.'),('v1-025','Mature adult short upright hair, no glasses, three-quarter pose and open striped shirt; useful neck.')]:
 r=labels[ident];p=ROOT/r['source_path'];assert sha(p)==r['source_sha256']
 reserved.append({'id':ident,'source_path':r['source_path'],'source_absolute_path':str(p),'source_sha256':sha(p),'purpose':'Reserved for possible additional edited validation; not new training','native_png_inspected_this_pass':True,'note':note})
excluded=['v1-000','v1-028','v1-030','b2-055','b2-051','b2-020','v1-004','calibration-original']
assert not ({r['id'] for r in records}&set(excluded))
result={'purpose':'Prospective original-PNG sources for clothing-only teachers; root selection review required before imagegen','training_shortlist_count':5,'priority_count':3,'conditional_count':2,'no_forced_sixth_candidate':True,'manual96_labels_sha256':sha(labels_path),'script_sha256':sha(Path(__file__)),'source_images_unchanged':True,'imagegen_or_model_inference':False,'active_training_manifest_changed':False,'candidates':records,'v1_reserved_validation':reserved,'excluded_existing_or_ambiguous_ids':excluded,'other_exclusions':[{'id':'b2-040','native_reviewed_this_pass':True,'reason':'Chin is cropped by lower frame; insufficient safe front collar area.'},{'id':'b2-013','reason':'Cheek microphone is an avoidable accessory; preserve for later hard case rather than force sixth candidate.'},{'id':'b2-021','reason':'Sunglasses prevent inspecting eyes; avoid as current teacher.'}],'aesthetic_limitation':'Most proposed teachers are mature-looking faces with glasses, and are not hot-approved. They provide clothing variation while source-face supervision preserves identity. Final output hot eligibility must be separate; this is not an attractiveness training set or representative demographic sample.'}
(HERE/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
for name,rs,cols in [('training-contact.png',records,3),('reserved-validation-contact.png',reserved,3)]:
 height=((len(rs)+cols-1)//cols)*548;canvas=Image.new('RGB',(cols*512,height),'#191919');d=ImageDraw.Draw(canvas)
 for i,r in enumerate(rs):
  x,y=(i%cols)*512,(i//cols)*548
  with Image.open(ROOT/r['source_path']) as im:canvas.paste(im.resize((512,512),Image.Resampling.LANCZOS),(x,y))
  d.text((x+10,y+521),r['id']+' | '+r.get('status','reserved validation'),fill='white')
 canvas.save(HERE/name)
print(json.dumps({'priority':[r['id'] for r in records if r['status']=='priority'],'conditional':[r['id'] for r in records if r['status']!='priority'],'reserved_validation':[r['id'] for r in reserved],'selection_sha256':sha(HERE/'selection.json')},indent=2))
