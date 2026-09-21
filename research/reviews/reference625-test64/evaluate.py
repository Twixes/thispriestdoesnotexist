#!/usr/bin/env python3
"""Audit frozen predictions against existing blind labels; never fit or rescore.

Run with Python 3 from any directory. All dependencies are in the standard library.
Outputs are deterministic. Missing LFS objects or changed inputs fail the audit.
"""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COHORT = Path('research/restoration/runs/reference625-test64-v1')
SCORES = Path('research/selection/restored-test64-scores-v1/result.json')
CALIBRATION = Path('research/selection/restored-calibration32-scores-v1/threshold-selection.json')
FIELDS = ('adult', 'male', 'bareheaded', 'single_subject', 'plausible_collar',
          'no_major_artifacts', 'photographic', 'appealing')


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: {path}: {actual} != {expected}')
    return actual


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    scores = read(SCORES)
    calibration = read(CALIBRATION)
    protocol_path = Path('research/experiments/restored_selection/protocol.json')
    verify(protocol_path, scores['protocol_sha256'])
    require(scores['complete'] and scores['split'] == 'test', 'Not complete test scores')
    require(scores['label_sources'] == [], 'Test predictions already contain labels')
    require(calibration['test_data_used'] is False, 'Threshold used test data')
    require(scores['fixed_threshold'] == calibration['selected_threshold'] == 0.9,
            'Frozen threshold differs')
    require(scores['protocol_sha256'] == calibration['protocol_sha256'], 'Protocol differs')
    require(scores['frozen_model'] == calibration['model'], 'Frozen model differs')
    model = Path(scores['frozen_model']['directory'])
    verify(model / 'result.json', scores['frozen_model']['result_sha256'])
    verify(model / 'weights.npz', scores['frozen_model']['weights_sha256'])
    verify(CALIBRATION.parent / 'result.json', calibration['calibration_scores_sha256'])
    provenance = scores['provenance']
    features = Path(provenance['embedding_directory'])
    verify(features / 'result.json', provenance['result_sha256'])
    verify(features / 'supervisor-result.json', provenance['supervisor_sha256'])
    verify(features / 'features.npz', provenance['features_sha256'])
    generation_path = Path('research/runs/reference625-restoration-test64/evaluation.json')
    verify(generation_path, provenance['generation_sha256'])
    verify(COHORT / 'launch.json', provenance['restoration_launch_sha256'])
    generation = read(generation_path)
    require(generation['complete'] and generation['split'] == 'test', 'Not a completed test cohort')
    require(generation['count'] == scores['attempted'] == 64, 'Unexpected denominator')
    require(generation['seed_base'] == scores['protocol']['test']['seed_base'], 'Test seeds differ')
    embedded = read(features / 'result.json')
    for name, expected in embedded['cohorts'][0]['file_sha256'].items():
        verify(COHORT / name, expected)
    label_paths = sorted(p.relative_to(ROOT) for p in (ROOT / COHORT).glob('labels-*.json'))
    labels = {}
    label_sources = []
    resemblance = []
    for path in label_paths:
        doc = read(path)
        label_sources.append({'path': str(path), 'sha256': digest(path),
                              'reviewer': doc['reviewer'],
                              'blind_review_evidence': doc.get('scope',
                                  {'scores_hidden': doc.get('scores_hidden'),
                                   'native1024_viewed_individually': doc.get('native1024_viewed_individually')})})
        if doc.get('family_resemblance'):
            resemblance.append({'path': str(path), 'observation': doc['family_resemblance']})
        for entry in doc['entries']:
            index = entry['index']
            require(index not in labels, f'Duplicate label {index}')
            require(entry['acceptable'] == all(entry[field] is True for field in FIELDS),
                    f'Label/criteria mismatch {index}')
            labels[index] = (entry, path, doc['reviewer'])
    notes = COHORT / 'family-resemblance-032-047.txt'
    resemblance.append({'path': str(notes), 'sha256': digest(notes),
                        'observation': (ROOT / notes).read_text().strip()})
    require(set(labels) == set(range(64)), 'Incomplete labels')
    require([p['index'] for p in scores['predictions']] == list(range(64)), 'Prediction coverage differs')
    rows = []
    output_files_verified = 0
    require(len(generation['entries']) == len(embedded['rows']) == 64, 'Source/feature coverage differs')
    for prediction, source, feature in zip(scores['predictions'], generation['entries'], embedded['rows']):
        index = prediction['index']
        require(source['index'] == feature['index'] == index, 'Row order differs')
        entry, path, reviewer = labels[index]
        directory = COHORT / f'{index:03d}'
        record_path = directory / 'record.json'
        record_hash = verify(record_path, entry['record_sha256'])
        require(record_hash == feature['record_sha256'], f'Feature record differs {index}')
        record = read(record_path)
        require(record['index'] == index, f'Record index differs {index}')
        require(record['sha256'] == source['sha256'] == prediction['original_source_sha256']
                == feature['original_source_sha256'], f'Source hash differs {index}')
        verify(generation_path.parent / source['path'], source['sha256'])
        require(source['seed'] == generation['seed_base'] + index, f'Seed sequence differs {index}')
        for name, expected in record['outputs'].items():
            verify(directory / name, expected)
            output_files_verified += 1
        if record['error'] is None:
            png_hash = record['outputs']['restored-1024.png']
            require(entry.get('reviewed_image_sha256', entry.get('sha256')) == png_hash,
                    f'Reviewed PNG differs {index}')
            webp_hash = record['outputs']['restored-1024.webp']
            require(prediction['webp_sha256'] == feature['webp_sha256'] == webp_hash,
                    f'WebP hash differs {index}')
            if 'final_webp_sha256' in entry:
                require(entry['final_webp_sha256'] == webp_hash, f'Review WebP differs {index}')
            require(prediction['valid'] and feature['feature_valid'], f'Unexpected invalid feature {index}')
            require(prediction['accepted'] == (prediction['score'] >= scores['fixed_threshold']),
                    f'Threshold decision differs {index}')
        else:
            require(not entry['acceptable'] and not prediction['accepted'] and not prediction['valid']
                    and prediction['score'] is None, f'Error escaped rejection {index}')
        rows.append({'index': index, 'seed': source['seed'], 'score': prediction['score'],
                     'accepted': prediction['accepted'], 'manual_acceptable': entry['acceptable'],
                     'restoration_error': record['error'], 'reviewer': reviewer,
                     'label_path': str(path), 'record_sha256': record_hash,
                     'reason': entry['reason']})
    accepted = [r for r in rows if r['accepted']]
    true_accepts = [r['index'] for r in accepted if r['manual_acceptable']]
    false_accepts = [r['index'] for r in accepted if not r['manual_acceptable']]
    positive = sum(r['manual_acceptable'] for r in rows)
    errors = [r['index'] for r in rows if r['restoration_error']]
    report = {
        'complete': True, 'production_approved': False,
        'conclusion': 'FAILED: frozen selector accepts visually defective portraits; production quality and identity diversity are not established.',
        'fixed_threshold': scores['fixed_threshold'], 'attempted': len(rows),
        'restoration_successes': len(rows) - len(errors), 'restoration_error_indices': errors,
        'manual_acceptable_all_attempts': positive,
        'accepted_indices': [r['index'] for r in accepted],
        'true_accept_indices': true_accepts, 'false_accept_indices': false_accepts,
        'confusion_matrix_all_attempts': {'true_positive': len(true_accepts), 'false_positive': len(false_accepts),
                                        'false_negative': positive - len(true_accepts),
                                        'true_negative': len(rows) - positive - len(false_accepts)},
        'acceptance_rate': len(accepted) / len(rows),
        'precision': len(true_accepts) / len(accepted), 'recall': len(true_accepts) / positive,
        'false_accept_share_of_deliveries': len(false_accepts) / len(accepted),
        'acceptable_deliveries_per_attempt': len(true_accepts) / len(rows),
        'hash_audit': {'all_record_files': len(rows), 'all_record_output_files': output_files_verified,
                       'all_original_generation_images': len(rows), 'passed': True},
        'sources': [{'path': str(p), 'sha256': digest(p)} for p in
                    [SCORES, CALIBRATION, protocol_path, generation_path, features / 'result.json',
                     features / 'features.npz', model / 'result.json', model / 'weights.npz',
                     Path(__file__).relative_to(ROOT)]],
        'label_sources': label_sources, 'family_resemblance_observations': resemblance,
        'limitations': [
            'No retraining, rescoring, threshold tuning, relabeling, or test-set filtering performed.',
            'Labels are subjective agent visual reviews performed with classifier scores hidden; they are not independent human production approval.',
            'PNG review images and final WebP outputs are separately hash-bound; this audit does not claim byte or perceptual equivalence between lossy WebP and PNG.',
            'Small test cohort and one reviewer per portrait; no inter-rater reliability measurement.',
            'Family resemblance is a visual observation, not an identity-recognition measurement. Unique hashes do not prove novel identities.',
            'These diagnostic timing/memory measurements do not establish a viable production hosting budget.'
        ], 'rows': rows}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    failures = '\n'.join(f"- **{r['index']:03d}** (score {r['score']:.6f}): {r['reason']}"
                         for r in accepted if not r['manual_acceptable'])
    markdown = f'''# Frozen reference625 restoration selector: held-out test

**The selector fails the required visual-quality gate.** At its preselected threshold of 0.9,
{len(accepted)} of 64 fresh attempts pass, but {len(false_accepts)} of those deliveries have visible eye defects.
No model, threshold, labels, or predictions were changed after examining this test.

| Measure | Result |
|---|---:|
| Attempts, including failures | 64 |
| Successful restorations | {64-len(errors)} |
| Manually acceptable across all attempts | {positive}/64 |
| Automatically accepted | {len(accepted)}/64 ({100*len(accepted)/64:.2f}%) |
| Accepted and manually acceptable | {len(true_accepts)}/{len(accepted)} ({100*len(true_accepts)/len(accepted):.2f}% precision) |
| False accepts among deliveries | {len(false_accepts)}/{len(accepted)} ({100*len(false_accepts)/len(accepted):.2f}%) |
| Acceptable portraits retained | {len(true_accepts)}/{positive} ({100*len(true_accepts)/positive:.2f}% recall) |
| Acceptable deliveries per attempt | {len(true_accepts)}/64 ({100*len(true_accepts)/64:.2f}%) |

Accepted indices: {', '.join(f'{r["index"]:03d}' for r in accepted)}.
Restoration failures {', '.join(f'{i:03d}' for i in errors)} remain in the denominator and are forced rejects.

{failures}

The calibration cohort had five accepts with zero observed false accepts at this same threshold.
The independent test contradicts an inference that the calibrated threshold guarantees defect-free output.
The two failed accepted portraits must remain in the review preview and archived results.

## Diversity and interpretation

Reviewers described recurring face families in 023/024/028, 016/026/030, and 017/020/025;
037/038, 035/045, and 032/044 also share facial templates. Hair, beard, pose and expression
vary, but this does not establish the broad identity diversity requested for deployment.
These are subjective visual observations, not face-recognition measurements. Unique image hashes
are provenance evidence only.

The 64 existing score-hidden visual labels are unchanged. They were produced by four agent
reviewers, one per portrait, rather than independent human production approval. Small sample
size, subjective eye-quality judgments and no inter-rater assessment limit the estimate.
Acceptability also does not prove parity with thispersondoesnotexist.

## Reproduce and inspect

Run `python3 research/reviews/reference625-test64/evaluate.py` from the repository checkout.
The standard-library script verifies every source image, all 64 restoration records, all
{output_files_verified} record-listed outputs, reviewed PNG and scored WebP hashes, the feature archive,
the frozen weights and calibration selection. Any missing LFS object or changed hash fails.
It evaluates frozen predictions and never executes a model.

[Machine-readable report](report.json) contains all 64 decisions, reasons, source paths and
SHA-256 references. Original labels remain in
[the restoration cohort](../../restoration/runs/reference625-test64-v1/).
The [frozen scores](../../selection/restored-test64-scores-v1/result.json) and
[calibration selection](../../selection/restored-calibration32-scores-v1/threshold-selection.json)
remain unchanged. [Non-blocking preview](../latest-preview/index.html) contains every automatic accept.

Production approval remains **false**. This report does not establish service latency,
Linux memory use, monthly operating cost, or readiness for release.
'''
    (OUT / 'README.md').write_text(markdown)
    print(json.dumps({k: report[k] for k in ['conclusion', 'attempted', 'accepted_indices',
                                           'false_accept_indices', 'confusion_matrix_all_attempts', 'hash_audit']}, indent=2))


if __name__ == '__main__':
    main()
