"""Materialize recorded native-image review labels; performs no model inference."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

base_path = OUT / 'review-base-first12.json'
base = json.loads(base_path.read_text())
notes20 = [
    'Near-identical facial detail, pose and coherent eyes; slight collar change. No photographic or calendar gain.',
    'Natural angled headshot retained with tiny skin and collar differences. No clear gain.',
    'Tiny chain/color and collar changes; legitimate clerical clothing and coherent face retained.',
    'Same high-contrast, somewhat crunchy facial texture and coherent eyes; no new defect or clear gain.',
    'Plain arm has a natural toothier smile. Collar shifts toward a black shirt with white tab; trigger remains close to base. No clear quality gain.',
    'Slight face-width and collar changes; natural skin and coherent eyes retained.',
    'Expressive silver-haired portrait retained; small-tab styling changes without photographic improvement.',
    'Tight crop continues to hide the collar front; a white edge alone remains uncertain evidence of clergy. Coherent face retained.',
    'Four-step skin detail remains better than the one-step output, but adapter is essentially tied with four-step base.',
    'Plain viewer-right iris remains uncertain. Trigger face is coherent; irregular overlapping white extension below the collar persists.',
    'Clearer tab styling and black shirt variation, but no photographic gain; chin stubble still misses the no-beard request.',
    'Detailed forehead, beard and coherent eyes retained, with a slight pose change. No clear photographic or calendar gain.',
]
notes100 = [
    'Smoother, less wrinkled forehead and slightly darker but coherent eyes. Still a plausible strong portrait; no clear gain over base/20.',
    'Subtle crop, eye and smile changes with softer skin detail. Coherent face and visible collar retained; no clear gain.',
    'Smoother face and reshaped collar while skin irregularities and coherent eyes remain. No clear photographic or calendar improvement.',
    'Trigger face becomes broader and more frontal; high-contrast texture persists but remains photographic. Eyes and broad collar are coherent.',
    'Face and forehead become smoother and the trigger smile more neutral, but fine wrinkles and skin detail remain plausible. Adequate appeal retained.',
    'Softer skin with visible pigment and stubble retained; coherent eyes and collar. No clear quality or calendar gain.',
    'Pose and expression change substantially in trigger arm, with textured skin/hair and a clear tab. Still strong appeal; not a clear paired win.',
    'Coherent face and detailed beard retained. Front of collar still cropped away, so clergy evidence remains uncertain.',
    'Some smoothing, but coherent eyes and irregular skin details remain. No clear gain over four-step base/20.',
    'Plain viewer-right iris becomes a flat malformed blue-black region: fail for facial integrity versus baseline uncertainty. Trigger eyes remain coherent and its former overlapping white collar extension is removed: a limited clothing-quality win, not improved facial/calendar appeal.',
    'Coherent face and photographic detail retained. Trigger collar now has an overlapping/doubled white strip above the tab: a clothing-quality loss, although priest clothing remains recognizable.',
    'Detailed skin, hair and beard with coherent gaze and visible collar. Pose changes without a clear photographic or calendar improvement.',
]

for step, notes in [(20, notes20), (100, notes100)]:
    run = ROOT / f'research/diffusion/runs/sdxl-pilot-step{step}-development-4step-v2-cached'
    result = json.loads((run / 'result.json').read_text())
    assert result['complete'] is True and result['steps'] == 4
    rows = []
    for i, source in enumerate(base['rows']):
        row = copy.deepcopy(source)
        for arm in row['arms']:
            assert sha(ROOT / arm['native_path']) == arm['native_sha256']
        for prior in source['arms']:
            arm = copy.deepcopy(prior)
            arm['arm'] = prior['arm'].replace('base-', 'adapter-')
            folder = run / source['case_id'] / arm['arm']
            record_path = folder / 'record.json'
            record = json.loads(record_path.read_text())
            native = folder / 'native.png'
            assert sha(native) == record['files']['native.png']['sha256']
            assert record['steps'] == 4 and record['seed'] == source['seed']
            assert record['protocol_sha256'] == base['protocol_sha256']
            for key in ['prompt_embeds', 'pooled_prompt_embeds']:
                assert record['cached_conditioning']['tensors'][key]['values_sha256'] == prior['cached_conditioning']['tensors'][key]['values_sha256']
            arm.update(native_path=str(native.relative_to(ROOT)), native_sha256=sha(native), record_sha256=sha(record_path), cached_conditioning=record['cached_conditioning'])
            if step == 100 and source['case_id'] == 'dev-01':
                if arm['arm'] == 'adapter-plain':
                    arm.update(facial_integrity=False, face_integrity_label='fail', joint_pass=False)
                else:
                    arm['clothing_artifact'] = False
            if step == 100 and source['case_id'] == 'dev-02' and arm['arm'] == 'adapter-trigger':
                arm['clothing_artifact'] = True
            row['arms'].append(arm)
        row['notes'] = notes[i]
        quality = 'win' if step == 100 and source['case_id'] == 'dev-01' else 'loss' if step == 100 and source['case_id'] == 'dev-02' else 'tie'
        row['adapter_trigger_vs_base_trigger_photographic_priest_quality'] = quality
        row['adapter_trigger_vs_base_trigger_calendar_appeal'] = 'tie'
        row['adapter_plain_vs_base_plain_quality'] = 'loss' if step == 100 and source['case_id'] == 'dev-01' else 'tie'
        row['material_adapter_trigger_quality_regression'] = False
        row['plain_face_uncertain_to_fail'] = step == 100 and source['case_id'] == 'dev-01'
        if step == 100:
            row['adapter_trigger_vs_step20_trigger_photographic_priest_quality'] = quality
            row['adapter_trigger_vs_step20_trigger_calendar_appeal'] = 'tie'
        rows.append(row)
    report = {
        'reviewer': 'heldout_report agent',
        'method': 'All 24 native 512px RGB adapter images individually inspected with view_image. All 24 native base images were individually inspected in the preceding base review; their labels and hashes are reused here. All 24 step20 images were also individually inspected before step100. No image selection, model rerun or threshold tuning.',
        'reviewed_cases': 12, 'reviewed_images': 48, 'new_adapter_images_reviewed': 24,
        'inference_steps': 4, 'training_updates': step, 'cached_prompts': True,
        'protocol_sha256': base['protocol_sha256'], 'prior_base_review_sha256': sha(base_path),
        'preview_manifest_sha256_at_review': sha(OUT / 'manifest.json'),
        'run_result_sha256': sha(run / 'result.json'),
        'cached_conditioning_verification_sha256': sha(OUT / 'cached-conditioning-verification.json'),
        'timing_note': 'Frozen FP16 text conditioning was cached before model sampling. Startup encoding is excluded from generation timing; embedding transfer is included. This review makes no GPU-server latency claim.',
        'production_approved': False, 'rows': rows,
        'summary': {
            'adapter_trigger_joint_pass': 11, 'adapter_plain_joint_pass': 10,
            'base_trigger_joint_pass': 11, 'base_plain_joint_pass': 10,
            'adapter_trigger_face_integrity_pass': 12, 'adapter_plain_face_integrity_pass': 11,
            'base_trigger_face_integrity_pass': 12,
            'paired_trigger_quality': {'wins': 0 if step == 20 else 1, 'ties': 12 if step == 20 else 10, 'losses': 0 if step == 20 else 1},
            'paired_trigger_calendar_appeal': {'wins': 0, 'ties': 12, 'losses': 0},
            'material_trigger_adaptation_regressions': 0,
            'plain_face_uncertain_to_fail': 0 if step == 20 else 1,
            'within_half_plausible_families': 9, 'largest_within_half_family': 2,
            'uncertain_counts_as_not_passed': True,
            'full24_family_assignment_not_performed': True, 'teacher_copying_not_assessed': True,
            'recommendation': 'No general LoRA quality or calendar benefit in this half. Four-step texture improvement comes primarily from inference configuration. This half alone does not decide full development eligibility; combine with the separately reviewed last12. No production or held-out-test approval.',
        },
    }
    if step == 100:
        report['prior_step20_review_sha256'] = sha(OUT / 'review-step20-first12.json')
        report['summary']['step20_trigger_joint_pass'] = 11
    for arm in ['base-plain','base-trigger','adapter-plain','adapter-trigger']:
        assert sum(a['joint_pass'] for r in rows for a in r['arms'] if a['arm'] == arm) == (10 if arm.endswith('plain') else 11)
    write(f'review-step{step}-first12.json', report)

write('review-first12-summary.json', {
    'reviewer': 'heldout_report agent', 'case_ids': [r['case_id'] for r in base['rows']],
    'native_images_individually_reviewed': 72,
    'reports': [{'path': name, 'sha256': sha(OUT / name)} for name in ['review-base-first12.json','review-step20-first12.json','review-step100-first12.json']],
    'inference_steps': 4,
    'joint_pass': {'base_plain':10,'base_trigger':11,'step20_plain':10,'step20_trigger':11,'step100_plain':10,'step100_trigger':11},
    'trigger_face_integrity_pass': {'base':12,'step20':12,'step100':12},
    'conclusion': 'Four-step inference improves texture over one-step, but training gives no general quality or calendar benefit in the first12. Step100 fixes one trigger collar extension and introduces another; dev01 plain iris worsens from uncertain to malformed. All horizons retain the legacy07 collar-crop failure. This report covers only the first12; full-run eligibility requires the other half and all protocol gates.',
    'production_approved': False, 'server_timing_claim': False,
})
print('Wrote step20/100 first12 native reviews and hash-bound summary; 72 unique native images reviewed.')
