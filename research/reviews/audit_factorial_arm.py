"""Audit completed saved arm artifacts without loading model weights."""
import argparse
import collections
import json
from pathlib import Path
from PIL import Image
from audit_factorial_previews import digest, read_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=list('ABCD'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.run / args.arm
    complete = json.loads((root / 'complete.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    assert complete['complete'] is True and complete['smoke'] is False
    assert complete['updates_per_trajectory'] == complete['actual_optimizer_calls'] == 300
    assert complete['source_and_frozen_checks_passed'] and complete['full_checkpoint_roundtrip']
    assert complete['final_source30_count'] == 30
    assert complete['initial_student_state_sha256'] == config['parent_student_state_sha256']
    metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    assert len(metrics) == len(complete['schedule']) == 300
    for i, (metric, draw) in enumerate(zip(metrics, complete['schedule'])):
        assert metric['fork_step'] == i + 1 and metric['parent_step'] == 600
        assert all(metric[key] == value for key, value in draw.items())
    counts = collections.Counter(row['pair_id'] for row in complete['schedule'])
    assert set(counts) == set(config['train_ids']) and set(counts.values()) == {50}
    assert [row['fork_step'] for row in complete['milestones']] == [0, 100, 200, 300]
    checkpoints = []
    snapshots = []
    for milestone in complete['milestones']:
        checkpoint = root / milestone['checkpoint']
        assert digest(checkpoint) == milestone['sha256']
        snapshot, sha = read_snapshot(root / f"preview-{milestone['fork_step']:03}", milestone['fork_step'])
        assert snapshot['state_sha256'] == milestone['student_sha256']
        checkpoints.append({'path': str(checkpoint), 'sha256': milestone['sha256']})
        snapshots.append({'step': milestone['fork_step'], 'evaluation_sha256': sha})
    source_dir = root / 'source30-final'
    source = json.loads((source_dir / 'evaluation.json').read_text())
    assert source['count'] == len(source['rows']) == 30
    assert sorted(source['excluded_training_ids']) == ['000', '030']
    assert source['source_compositing'] is False and source['rng_restored'] is True
    assert {row['id'] for row in source['rows']} == {f'{i:03}' for i in range(32)} - {'000', '030'}
    assert len({row['z_sha256'] for row in source['rows']}) == 30
    verified = 112
    for row in source['rows']:
        assert row['source_uint8_max_error'] == row['w_max_error'] == 0
        for field in ['outputs', 'source_outputs']:
            assert set(row[field]) == {'rgb', 'gray'}
            for key, output in row[field].items():
                file = source_dir / output['path']
                assert file.parent == source_dir and digest(file) == output['sha256']
                with Image.open(file) as image:
                    assert image.size == (1024, 1024)
                    assert image.mode == output['mode'] == ('L' if key == 'gray' else 'RGB')
                verified += 1
    assert verified == 232
    report = {'arm': args.arm, 'artifact_audit_passed': True,
              'completed_worker_record_sha256': digest(root / 'complete.json'),
              'config_sha256': digest(root / 'config.json'), 'checkpoint_hashes': checkpoints,
              'preview_evaluations': snapshots, 'source30_evaluation_sha256': digest(source_dir / 'evaluation.json'),
              'pair_counts': dict(counts), 'verified_native_pngs': verified,
              'worker_reported_full_checkpoint_roundtrip': True,
              'wall_seconds': complete['wall_seconds'], 'peak_rss_bytes': complete['peak_rss_bytes'],
              'production_approved': False,
              'limitations': 'Does not load weights or independently repeat training/roundtrip checks. Native visual quality and cross-arm matching require separate review. Completion here means one worker, not the four-arm experiment.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'native_pngs': verified, 'checkpoint_count': len(checkpoints), 'pair_counts': dict(counts)}))


if __name__ == '__main__':
    main()
