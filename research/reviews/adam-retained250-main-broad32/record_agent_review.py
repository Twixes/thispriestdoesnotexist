"""Record actual agent observations from all8 contact sheets and6 native follow-ups.

This writes observations entered after visual inspection; it performs no model-based
scoring, classification, nearest-neighbor analysis or image manipulation.
"""
from pathlib import Path
import datetime,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RUN=ROOT/'research/runs/adam-native1024-retained250-main100-broad32-v1'
manifest=json.loads((RUN/'manifest.json').read_text());template=json.loads((HERE/'annotation-template.json').read_text())
# Observations by preregistered index, from actual visual inspection.
notes=[
'Adult-looking smiling face with cropped neighboring person; raw cooler and more etched, EMA warmer.',
'Apparent child; raw hair has odd multicolored sheen and grayer face; EMA more natural-colored.',
'Young-looking subject with flower hair accessory; adulthood uncertain. Raw face pale/cyan; EMA warmer.',
'Apparent child with curly hair; raw cooler with stronger shiny strands, coherent facial structure.',
'Adult-looking face substantially obscured by distorted foreground object/hand in both arms; raw cooling adds to the awkward image.',
'Young-looking face, adulthood uncertain; raw shiny hair and pale/cyan face, EMA warmer. No clear structural break.',
'Apparent child in red shirt, no collar; raw less warm, structure coherent.',
'Older adult-looking face with gray hair; raw skin more gray/etched, EMA more naturally shaded.',
'Adult-looking face with partial neighbor; raw forehead/nose sheen more metallic and colors cooler.',
'Adult-looking subject wearing graduation-like headwear and glasses; raw paler/cooler, coherent face.',
'Adult-looking face with glasses and ordinary light shirt; raw skin and hair cool/gray, no clerical insert.',
'Adult-looking subject with hand/object near cheek; both arms contain cyan object/paint-like smears, raw more strongly cyan.',
'Adult-looking subject in sunglasses and casual clothing; raw grayer/bluer with hard skin highlights.',
'Adult-looking smiling face; raw pale/cool with etched fine texture, EMA warm, coherent.',
'Adult-looking face with possible yellow headwear behind hair; cannot distinguish hat from background decoration confidently. Raw cooler and shinier.',
'Adult-looking bearded face; raw distinctly gray/blue against warm EMA, ordinary shirt not clerical.',
'Adult-looking bearded face in harsh side sunlight; raw deeper cool shadows and silvery highlights, coherent structure.',
'Adult-looking face with lower-frame black object in both arms; raw grayer and mildly rougher, no collar.',
'Young-looking blond-haired subject, adulthood uncertain; raw paler/cooler with etched hair, coherent features.',
'Adult-looking glasses wearer speaking into microphone; raw cooling and harder skin texture, coherent.',
'Young-looking red-haired subject, adulthood uncertain; native review confirms raw chin/neck framing shifts and a large sharp lower-frame ornament versus EMA. Local concern, not facial collapse.',
'Adult-looking smiling face; raw blue cast and strong cheek highlights versus warmer EMA.',
'Adult-looking curly-haired face in harsh sunlight; both have hard shadows, raw more gray/etched.',
'Adult-looking smiling face with jewelry; raw cooler with stronger forehead/cheek sheen, coherent features.',
'Adult-looking smiling bearded face; raw blue/cyan skin and deeper beard shadows, EMA warmer.',
'Apparent infant wrapped in blanket; raw gray/cyan shading, coherent face. No attractiveness assessment.',
'Adult-looking face in costume/headdress; native review confirms raw blue paint/marks spread over eyes/nose/cheeks compared with cleaner EMA. Local photographic regression, face still readable.',
'Adult-looking smiling face; raw grayer with harder fine skin texture than EMA.',
'Adult-looking face wearing tall dark headwear; raw cool/dark face, coherent eyes and mouth, no clerical collar.',
'Adult-looking face in ordinary jacket/shirt; raw cooler and slightly harsher skin contrast, no collar.',
'Adult-looking face against curtain; raw less warm and more etched, coherent anatomy.',
'Adult-looking face under dark headwear beside another person; native review confirms raw scratch-like dark chin patch and stronger unnatural skin texture versus cleaner EMA.'
]
adult=[True,False,None,False,True,None,False,True,True,True,True,True,True,True,True,True,True,True,None,True,None,True,True,True,True,False,True,True,True,True,True,True]
hat={9:True,14:None,26:True,28:True,31:True}
concerns={4:'Distorted foreground object/hand obscures lower face in both arms.',11:'Ambiguous smeared hand/object/color near cheek in both arms.'}
for row in template['rows']:
 i=row['index'];arm=row['arm'];row['status']='reviewed';row['apparent_adult']=adult[i];row['priest_clothing_and_collar']=False;row['hat_present']=hat.get(i,False)
 row['gross_artifacts']=concerns.get(i,'No obvious gross structural failure at contact-sheet scale; this is not a native-detail clearance.')
 if arm=='raw' and i in (20,26,31):row['gross_artifacts']={20:'Enlarged sharp lower-frame ornament and shifted chin framing; local concern confirmed natively.',26:'Spreading blue marks around eyes/nose/cheeks relative to EMA; localized detail degradation confirmed natively.',31:'Dark scratch-like chin patch and exaggerated skin texture relative to EMA; confirmed natively.'}[i]
 row['photographic_coherence']=('Recognizable face retained, but systematically cooler/desaturated and more etched than paired EMA; see sample-specific notes.' if arm=='raw' else 'Generally coherent warmer facial rendering; sample-specific source-like artifacts remain, see notes. EMA is not an untouched source control.')
 row['repetition_or_nearest_training_image_concern']='No obvious repeated identity across this32-latent contact review. No specific training-copy match identified visually; no nearest-neighbor computation or exhaustive paired training comparison performed.'
 row['notes']=notes[i]+' Review method: '+('contact sheet plus untouched native raw/EMA pair.' if i in (20,26,31) else 'entire-frame512px contact thumbnails; original images retained.')
template['review_status']='reviewed_with_limits';template['reviewer']='continuation_preview';template['reviewed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(HERE/'annotations-agent-main100.json').write_text(json.dumps(template,indent=2)+'\n')
record={'reviewer':'continuation_preview','reviewed_at_utc':template['reviewed_at_utc'],'source_checkpoint_sha256':manifest['checkpoint_sha256'],
 'render_manifest_sha256':hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),'n_preregistered_latents':32,'images_reviewed':64,
 'contact_sheets':json.loads((HERE/'contact-sheets-manifest.json').read_text())['outputs'],
 'native_followups':[{'path':str((RUN/f'{arm}-{i:03}.png').relative_to(ROOT)),'sha256':hashlib.sha256((RUN/f'{arm}-{i:03}.png').read_bytes()).hexdigest()} for i in (20,26,31) for arm in ('raw','ema')],
 'viewed_images':[{'path':str((RUN/im['path']).relative_to(ROOT)),'sha256':im['sha256'],'method':'native_and_contact' if im['index'] in (20,26,31) else '512px_whole_frame_contact'} for im in manifest['images']],
 'domain':{'clear_priest_clothing_and_collar_raw':0,'clear_priest_clothing_and_collar_ema':0,'denominator_each_arm':32,'obvious_apparent_minors_indices':[1,3,6,25],'adulthood_uncertain_indices':[2,5,18,20],'definite_headwear_indices':[9,26,28,31],'headwear_uncertain_indices':[14],'attractiveness_rated':False},
 'photographic_assessment':'Most faces remain anatomically coherent and identities varied. Raw has systematic cooling/desaturation, harsher skin contrast and etched highlights versus EMA across the batch. Local raw regressions are especially visible at20/26/31.004 and011 have conspicuous awkward foreground/paint-like forms in both arms. No widespread dense-grain catastrophe or structural mode collapse observed. EMA comparison does not establish which defects were present in the original unadapted source.',
 'native_followup_conclusion':'020 shows a larger sharp lower-frame ornament and chin framing change, without facial collapse.026 has substantially more spread blue facial markings and softened eyes than EMA.031 has a new-looking scratch-like dark chin patch and harsh skin texture relative to EMA. These are genuine local quality concerns; EMA is not an original-source causal control.',
 'recommendation':'No broad catastrophic damage requiring immediate reset is evident. Support only bounded exact-state continuation with a mandatory150 visual decision, including review of these vulnerable samples if practicable. Stop/reassess if gray/etched shading becomes dominant, facial structures break, or these local defects spread. This is not permission to run blindly to250 and not model-quality approval.',
 'quality_accepted':False,'production_approved':False,'server_latency_proven':False,
 'limitations':'64 images from32 preregistered latents; most assessed through512px full-frame thumbnails that the tool displayed slightly reduced, six also inspected natively. No attractiveness ratings, biometric identity tests, nearest-neighbor metric, population quality estimate or causal source comparison. No native model loaded by reviewer.'}
(HERE/'agent-review-main100.json').write_text(json.dumps(record,indent=2)+'\n')
print({'rows_annotated':len(template['rows']),'review_sha256':hashlib.sha256((HERE/'agent-review-main100.json').read_bytes()).hexdigest()})
