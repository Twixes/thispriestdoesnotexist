"""Record completed tool-observed evaluations and native PNG review; no model imports."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
paired=json.loads((RUN/'evaluation-pairs7/evaluation.json').read_text())
unseen=json.loads((RUN/'evaluation30/evaluation.json').read_text())
launch=json.loads(RUN.with_suffix('.launch.json').read_text())
metrics=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
assert launch['exit_code']==0 and metrics[-1]['step']==600 and metrics[-1]['checkpoint_verified_frozen_source']
assert paired['checkpoint_sha256']==unseen['checkpoint_sha256']==sha(RUN/'resume.pt')
assert paired['count']==7 and unseen['count']==30 and not unseen['excluded_training_latents']
verified={}
for name,report in [('evaluation-pairs7',paired),('evaluation30',unseen)]:
    count=0
    for row in report['entries']:
        for output in row['outputs'].values():
            assert sha(RUN/name/output['path'])==output['sha256'];count+=1
    verified[name]=count
assert verified=={'evaluation-pairs7':28,'evaluation30':90}
for label in ['source-gray','student-gray']:
    assert sha(RUN/'evaluation-pairs7'/f'028-{label}.png')==sha(RUN/'evaluation30'/f'028-{label}.png')

paired_notes={
'calibration-original':('Coherent detailed source face, eyes, mouth and hair preserved.','Broad bright patch across the lower neck and residual light folded shirt. No clean rectangular tab or continuous clerical band.','not_convincing'),
'000':('Recognizable coherent source face; source sunlight/squint retained.','Shirt has darkened substantially, but the whitened area follows a broad curved neck/chin region rather than the target rectangular tab.','not_convincing'),
'030':('Eyes, expression and upper face remain coherent; preexisting second person at left remains.','Strong dark pitted/scratch-like texture across the right neck and a white patch at the lower center. This is a substantial new neck artifact, not acceptable clerical fabric.','not_convincing'),
'028':('Held-out eyes, smile and beard remain coherent and recognizable.','Ordinary white shirt edges remain under the beard. No recognizable new central tab. Small neck space is a caveat, not an exemption from visible priest appearance.','not_convincing'),
'b2-055':('Glasses, gaze and facial features remain coherent relative to source.','Original light high neckline has become smeared glossy dark cloth with a wide ragged white band below the chin. It lacks the target crisp central tab and clean fabric.','not_convincing'),
'b2-051':('Face, eyes, expression and fine wrinkles remain recognizable and coherent.','Central white area under the chin is the closest collar-like training result, but its side boundary blends into rough neck texture and surrounding garment lacks the clean target structure. Partial learning, not a convincing quality pass.','partial_ambiguous'),
'b2-020':('Smile, eyes and face remain coherent under original harsh light; original crowd context retained.','Garment darkens but a white patch merges with the underside of the chin; the remaining tiny tab is not clearly recognizable as the target clerical collar. Face/mask geometry conflict was known.','not_convincing')}
paired_records=[]
for row in paired['entries']:
    face,clothing,status=paired_notes[row['id']]
    paired_records.append({'id':row['id'],'split':row['split'],'native_source_student_target_reviewed':True,
                           'face':face,'clothing':clothing,'collar_assessment':status,'production_approved':False,
                           'training_reconstruction_only':row['split']=='train','metrics':row['metrics'],
                           'images':row['outputs']})
paired_review={'reviewed_at_utc':now,'checkpoint_sha256':paired['checkpoint_sha256'],
               'checkpoint_step':600,'production_approved':False,'scope':'All seven source/student/target triples individually viewed at native 1024; contact also reviewed. No new model inference for visual review.',
               'summary':'Faces are generally recognizable and coherent. Several training garments darken, but white patches, incomplete collar structure and neck/fabric artifacts remain. Training 030 has pronounced new neck pitting; b2-055 has smeared cloth. b2-051 shows partial collar-like learning, not a convincing quality pass. Holdout 028 has no recognizable tab.',
               'records':paired_records,'no_aggregate_train_holdout_score':True,
               'metrics_caveat':paired['metrics_caveat'],
               'recommendation':'Do not deploy or automatically extend this run. Low collar-region L1 did not guarantee a coherent black collar surrounding the white tab. Use the recorded surround/boundary diagnosis to define one separately bounded ablation; preserve source faces and compare held-out sources. Training fits must never count as unseen successes.'}
(RUN/'evaluation-pairs7/visual-review.json').write_text(json.dumps(paired_review,indent=2)+'\n')

unseen_notes={
'017':('Adult masculine, bareheaded; beard and glasses.','Source face, glasses and beard remain coherent. Ordinary dark neckline persists; no visible priest tab. Beard leaves little neck space.',True),
'024':('Adult masculine, bareheaded; glasses.','Face and smile remain coherent. Additional dark coarse texture emphasizes lower neck creases; only a thin original shirt edge is visible. No recognizable clerical collar.',True),
'025':('Adult masculine, bareheaded.','Angled source face and eyes remain coherent. Conventional striped shirt is still visible with some darkening, without a white clerical tab.',True),
'028':('Adult masculine, bareheaded; beard.','Same byte-identical source/student PNGs as the paired holdout review: ordinary white shirt edges remain, no central tab. Face and beard coherent.',True),
'004':('Age uncertain; additional person in original background. Excluded from eligible adult/hot success counts.','Face remains coherent. Shirt gains darker coarse texture at the right shoulder/neckline, but remains conventional clothing without a tab. Diagnostic preservation case only.',False)}
unseen_records=[]
for identity,(eligibility,observation,adult) in unseen_notes.items():
    row=next(row for row in unseen['entries'] if row['id']==identity)
    unseen_records.append({'id':identity,'native_source_student_reviewed':True,'manual_eligibility_note':eligibility,
                           'count_in_eligible_adult_subset':adult,'observation':observation,
                           'recognizable_clerical_collar':False,'production_approved':False,
                           'metrics':row['metrics'],'images':row['outputs']})
reference=ROOT/'research/data/ffhq-clothing-edits-validation/025.png'
unseen_review={'reviewed_at_utc':now,'checkpoint_sha256':unseen['checkpoint_sha256'],
               'checkpoint_step':600,'production_approved':False,'scope':'All 30 source/student pairs reviewed on contact; 017, 024, 025, 004 individually viewed at native 1024, and 028 native paired views verified byte-identical across both evaluators.',
               'summary':'No convincing clerical collar on the four clearly adult masculine native-reviewed unseen identities. Faces are mostly preserved, while unrestricted sampling retains women, children and hats. Source 004 is age-uncertain and never counted as an eligible-adult/hot success.',
               'eligible_adult_subset_ids':['017','024','025','028'],'eligible_adult_subset_collar_successes':0,
               'records':unseen_records,'excluded_training_source_ids':['000','030'],
               'contact_findings':['No obvious whole-cohort facial identity collapse in this small fixed contact; this is not an identity/diversity guarantee.','Women, children and headwear remain; eligibility sampling is unsolved.','Clothing changes can add rough neck/shirt texture, without a recognizable collar.'],
               'qualitative025_clothing_reference':{'path':str(reference.relative_to(ROOT)),'sha256':sha(reference),
                  'native_reviewed':True,'training_input':False,'exact_face_ground_truth':False,
                  'observation':'The separately generated reference shows a clear white tab inside black clerical clothing, unlike student 025, whose striped shirt remains. Its face/chin changed, so comparison is qualitative clothing only; original source remains facial ground truth.'},
               'metrics_caveat':unseen['metrics_caveat'],
               'recommendation':'Do not deploy and do not count training 000/030 as generalization. The six-pair run has not demonstrated collar generalization on this native subset. Further work requires an explicitly evaluated method change; no automatic identical extension.'}
(RUN/'evaluation30/visual-review.json').write_text(json.dumps(unseen_review,indent=2)+'\n')

commands={
'evaluation-pairs7':'research/.venv/bin/python research/experiments/paired_regions/evaluate_pairs.py --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle --checkpoint research/runs/paired-regions1024-sixpair-600/resume.pt --manifest research/data/paired7/manifest-proposed-v2.json --expected-step 600 --output research/runs/paired-regions1024-sixpair-600/evaluation-pairs7 > research/runs/paired-regions1024-sixpair-600/evaluation-pairs7.log 2>&1',
'evaluation30':'research/.venv/bin/python research/experiments/paired_regions/evaluate.py --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle --checkpoint research/runs/paired-regions1024-sixpair-600/resume.pt --sources research/data/paired7/evaluation-source30.json --expected-step 600 --output research/runs/paired-regions1024-sixpair-600/evaluation30 > research/runs/paired-regions1024-sixpair-600/evaluation30.log 2>&1'}
for name,report,session in [('evaluation-pairs7',paired,77421),('evaluation30',unseen,35300)]:
    invocation={'recorded_at_utc':now,'cwd':str(ROOT),'command':commands[name],'exec_session':session,
                'exit_code':0,'checkpoint_sha256':report['checkpoint_sha256'],
                'evaluator_script_sha256':report['script_sha256'],
                'actual_evaluator_memory_preflight_free_percent':report['memory_preflight_free_percent'],
                'sequential_order':1 if name=='evaluation-pairs7' else 2,
                'verified_output_pngs':verified[name],'no_concurrent_evaluation_jobs':True,
                'training_terminal_evidence':{'observed_original_pids_absent':[98309,98300],
                    'final_step':600,'launcher_exit_code':launch['exit_code'],
                    'launcher_record_sha256':sha(RUN.with_suffix('.launch.json'))},
                'no_restart':True,'production_approved':False}
    (RUN/name/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    (RUN/name/'host-memory-after-review.txt').write_text('System-wide memory free percentage: 31%\nObserved during PNG-only review after both evaluators exited; other agent validators were already authorized, so this is not an isolated post-process measurement.\n')
    (RUN/name/'README.md').write_text(f'''# Completed step 600 {name}

The evaluator exited 0 after the original six-pair training and launcher exited 0 at step 600. It ran sequentially with the other evaluator, never as a duplicate job. Exact invocation and process/memory evidence are in `invocation.json`.

{report['count']} images evaluated in {report['total_seconds']:.2f} seconds; peak native macOS RSS {report['peak_rss_bytes']/1024**3:.2f} GiB. The actual in-process memory preflight reported {report['memory_preflight_free_percent']}% free. All {verified[name]} native output PNG hashes were verified. Source PNG/W, full provenance and frozen-state checks passed. Checkpoint SHA256: `{report['checkpoint_sha256']}`.

`visual-review.json` records the explicit native review and limitations. `comparison.png` is a diagnostic contact sheet; images are independently synthesized, never composited into outputs. TRAIN identities are reconstruction examples, never unseen successes. Source 004 remains age-uncertain and excluded from eligible-adult/hot success counts.

Faces remain generally coherent, but convincing collar generalization is not demonstrated and neck/clothing artifacts remain. No production approval, deployment, new training or automatic continuation follows from these checks. See the other evaluation directory for complementary paired/unseen evidence. The native Mac RSS is not a Linux hosting benchmark.
''')
completion={'recorded_at_utc':now,'training_exit_code':0,'training_elapsed_seconds':launch['elapsed_seconds'],
            'final_step':600,'checkpoint_sha256':paired['checkpoint_sha256'],'student_state_sha256':paired['student_state_sha256'],
            'actual_training_pair_counts':dict(collections.Counter(identity for row in metrics for identity in row['pair_ids'])),
            'evaluations':[{'directory':name,'exit_code':0,'count':report['count'],'seconds':report['total_seconds'],
                'peak_rss_bytes':report['peak_rss_bytes'],'memory_preflight_free_percent':report['memory_preflight_free_percent'],
                'output_png_hashes_verified':verified[name]} for name,report in [('evaluation-pairs7',paired),('evaluation30',unseen)]],
            'no_model_compute_remaining':True,'hosting_agent_notified_after_both_exits':True,
            'production_approved':False,'no_training_restart_or_input_mutation':True,
            'finding':'Coherent faces but insufficient clerical clothing; artifacts on supervised necks and no collar successes among native-reviewed unseen adult subset.'}
(RUN/'evaluation-completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print(json.dumps(completion,indent=2))
