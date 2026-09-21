"""Record the completed native six-step first-four visual review, without inference."""
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'research/diffusion/runs/juggernaut-hyper-tcd6-cfg1p5-native1024-v1-tiled'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((OUT/'review-first4.json').read_text())
notes=[
'Coherent facial structure and eyes, natural smiling pose and clear collar. Skin now has irregular pores, stubble and wrinkles rather than the earlier mesh-like detail over smooth shading. Hair remains rather finely etched. Strong calendar appeal. Photographic detail improves over four-step; strong resemblance to001 persists.',
'Clear collar and no hat. The previous flat green viewer-right iris is resolved, but the viewer-left iris/pupil has an angular dark structure that remains uncertain at native1024. Pores, stubble and wrinkles are more photographic than four-step. Strong portrait appeal, but face-integrity uncertainty prevents joint pass. Strong resemblance to000 persists.',
'Coherent eyes and facial structure, clear collar and no hat. More irregular pores/stubble and facial detail make texture more photographic than four-step, despite a somewhat glossy forehead and etched curls. Adequate formal calendar appeal. Close facial-family resemblance to003.',
'Coherent eyes and face, clear collar and no hat. Skin pores and stubble improve photographic detail, although hair still looks finely etched. The earlier notch is gone; two overlapping white collar layers remain but clergy appearance is clear. Adequate formal appeal. Very close facial-family resemblance to002.',
]
rows=[]
for i,note in enumerate(notes):
 p=RUN/f'{i:03d}';r=json.loads((p/'record.json').read_text());assert r['status']=='complete' and r['steps']==6 and r['guidance_scale']==1.5
 assert sha(p/'native.png')==r['files']['native.png']
 before=prior['rows'][i];assert before['seed']==r['seed']
 rows.append({'index':i,'seed':r['seed'],'native_path':str((p/'native.png').relative_to(ROOT)),'native_sha256':sha(p/'native.png'),'record_sha256':sha(p/'record.json'),'fourstep_native_sha256':before['native_sha256'],'face_integrity':'uncertain' if i==1 else 'pass','photographic_texture':'pass','recognizable_clerical_clothing':'pass','single_adult_male_appearance':'pass','no_hat':'pass','calendar_composition_and_appeal':'strong' if i<2 else 'adequate','joint_quality_pass':i!=1,'photographic_quality_vs_fourstep':'win','calendar_appeal_vs_fourstep':'tie','within_review_face_family':before['within_review_face_family'],'notes':note})
d={'reviewer':'heldout_report agent','method':'All four native1024 PNGs000–003 individually viewed with view_image detail original and compared with the previously reviewed native four-step images. No selections, test access or model reruns.','protocol_sha256':sha(RUN/'protocol.json'),'runtime_sha256':sha(RUN/'runtime.json'),'model_provenance_sha256':sha(RUN/'model-provenance.json'),'prior_fourstep_review_sha256':sha(OUT/'review-first4.json'),'configuration':{'steps':6,'guidance_scale':1.5,'eta':.3,'scheduler':'TCD','resolution':1024,'tiled_vae':True},'comparison_limits':'Both step count and CFG changed from the four-step reference, so gains cannot be attributed to either change alone. Tiled VAE is shared. Same prompts and seeds are a configuration comparison, not proof of identical identities. No claim that tiling caused remaining etched texture.','rows':rows,'summary':{'reviewed':4,'face_integrity_pass':3,'face_integrity_uncertain':1,'photographic_texture_pass':4,'recognizable_clergy':4,'no_hats':4,'joint_quality_pass':3,'uncertain_counts_as_not_passed':True,'paired_photographic_quality':{'wins':4,'ties':0,'losses':0},'paired_calendar_appeal':{'wins':0,'ties':4,'losses':0},'plausible_within_review_families':2,'largest_within_review_family':2,'conclusion':'A visible photographic-detail improvement over four-step in this half. Eye uncertainty001 and strong paired family resemblance remain; these four development cases do not demonstrate production-level reliability or unseen-latent diversity.'},'post_trained_by_this_project':False,'production_approved':False,'server_latency_proven':False}
(OUT/'review-sixstep-first4.json').write_text(json.dumps(d,indent=2)+'\n')
print('Saved six-step first-four comparative native review')
