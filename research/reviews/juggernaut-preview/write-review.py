"""Bind the first-four native1024 visual judgments to retained image artifacts."""
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'research/diffusion/runs/juggernaut-hyper-tcd4-native1024-v2-tiled'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
notes=[
'Structurally coherent face and eyes, with a clear clerical tab and no hat. Conventionally attractive, gentle smiling composition. At native1024 the skin is very smoothly shaded under repetitive etched detail, and hair has regular fine striping; photographic texture is uncertain. Strong resemblance to001.',
'Clear collar, no hat, attractive angled portrait. Viewer-right iris is a flat green patch with weak/unclear pupil detail compared with the other eye: facial integrity uncertain. Smooth rendered skin and regular hair/skin texture make photographic quality uncertain. Face is strongly similar to000 despite changed pose.',
'Coherent eyes and facial structure, clear white tab, no hat. Balanced formal portrait, adequate calendar appeal. Enlarged glossy irises and regular etched/mesh texture across smooth skin give a rendered appearance at native1024; photographic texture uncertain. Strong resemblance to003.',
'Coherent face and eyes, no hat, recognizable collar with an odd notch at the top of the white tab. Adequate formal portrait appeal. Smooth shading with patterned/etched skin and hair detail makes photographic texture uncertain. Very similar facial family to002 despite changed hair/pose.',
]
rows=[]
for i,note in enumerate(notes):
 p=RUN/f'{i:03d}';r=json.loads((p/'record.json').read_text());assert r['status']=='complete';assert sha(p/'native.png')==r['files']['native.png']
 rows.append({'index':i,'seed':r['seed'],'native_path':str((p/'native.png').relative_to(ROOT)),'native_sha256':sha(p/'native.png'),'record_sha256':sha(p/'record.json'),'face_integrity':'uncertain' if i==1 else 'pass','photographic_texture':'uncertain','recognizable_clerical_clothing':'pass','single_adult_male_appearance':'pass','no_hat':'pass','calendar_composition_and_appeal':'strong' if i<2 else 'adequate','clothing_artifact':i==3,'joint_quality_pass':False,'within_review_face_family':'pair00-01' if i<2 else 'pair02-03','notes':note})
d={'reviewer':'heldout_report agent','method':'Each of000–003 individually opened as native1024 PNG using view_image detail original. No selections or model reruns. This review does not cover004–007 or the512 run.','protocol_sha256':sha(RUN/'protocol.json'),'runtime_sha256':sha(RUN/'runtime.json'),'model_provenance_sha256':sha(RUN/'model-provenance.json'),'vae_tiling':{'enabled':True,'sample_tile':512,'latent_tile':64,'overlap':.25},'tiling_causation_note':'Patterned texture is observed; this review cannot attribute it specifically to tiled decoding without a matched successful untiled reference.','rows':rows,'summary':{'reviewed':4,'face_integrity_pass':3,'face_integrity_uncertain':1,'photographic_texture_pass':0,'photographic_texture_uncertain':4,'recognizable_clergy':4,'no_hats':4,'joint_quality_pass':0,'uncertain_counts_as_not_passed':True,'plausible_within_review_families':2,'largest_within_review_family':2,'teacher_copying_assessed':False,'decision':'Potentially useful prior for recognizable clergy and portrait appeal, but these native1024 outputs do not yet demonstrate the requested photographic quality or diversity. A separate512 comparison can assess resolution tradeoffs; not production approval.'},'post_trained_by_this_project':False,'production_approved':False,'server_latency_proven':False}
(OUT/'review-first4.json').write_text(json.dumps(d,indent=2)+'\n')
print('Saved hash-bound first-four native review')
