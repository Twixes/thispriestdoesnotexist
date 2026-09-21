#!/usr/bin/env python3
"""Progressive review builder. Read finalized artifacts only; never load a model."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'research/runs/adam-native1024-output-rank1-fixed-offsets100-v1'
STEPS = [0, 10, 25, 50, 100]


def artifact(path, data=None):
    data = path.read_bytes() if data is None else data
    return {'path': str(path.relative_to(ROOT)), 'url': os.path.relpath(path, HERE),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def image_record(path, expected_hash):
    data = path.read_bytes()
    item = artifact(path, data)
    assert item['sha256'] == expected_hash, f'Image hash mismatch: {path}'
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (1024, 1024) and im.format == 'PNG', path
        im.verify()
    return {'status': 'complete', **item, 'width': 1024, 'height': 1024}


def collect(run=RUN):
    # The source worker atomically writes JSON via .partial then rename.
    # Never inspect logs, metrics, checkpoint tensors, or .partial paths.
    protocol_data = (run / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_data)
    assert protocol['preview_count'] == 4 and protocol['preview_steps'] == STEPS
    sources = {'protocol.json': artifact(run / 'protocol.json', protocol_data)}
    protocol_hash = sources['protocol.json']['sha256']
    assert protocol['conv_layout'] == 'output_rank1' and protocol['iterations'] == 100
    assert protocol['checkpoint_steps'] == [100]
    policy = protocol['fixed_offset_policy']
    assert policy == {'offset_count': 34, 'noise_count': 17, 'synthesis_bias_count': 17,
                      'offset_value': 0, 'optimizer_excluded': True, 'raw_and_ema_fixed': True,
                      'original_strengths_and_biases_unchanged': True}
    for name in ['probe_output_rank1_fixed_offsets100.py', 'modulation.py']:
        record = artifact(run / name)
        assert record['sha256'] == protocol['pins'][f'research/experiments/adam_native/{name}']
        sources[name] = record
    policy_record = artifact(run / 'fixed-offset-policy.json')
    sources['fixed-offset-policy.json'] = policy_record
    frozen_policy = json.loads((run / 'fixed-offset-policy.json').read_text())
    assert len(frozen_policy['offset_names']) == len(set(frozen_policy['offset_names'])) == 34
    z_path = run / 'eval-z.npz'
    sources['eval-z.npz'] = artifact(z_path)
    assert sources['eval-z.npz']['sha256'] == protocol['eval_z_source_sha256']
    baseline_data = (run / 'baseline-manifest.json').read_bytes()
    baseline = json.loads(baseline_data)
    sources['baseline-manifest.json'] = artifact(run / 'baseline-manifest.json', baseline_data)
    assert baseline['complete'] is True and baseline['count'] == 4
    assert baseline['protocol_sha256'] == protocol_hash
    assert baseline['eval_z_sha256'] == sources['eval-z.npz']['sha256']
    assert len(baseline['images']) == 4
    assert {im['path'] for im in baseline['images']} == {f'baseline-{i:03}.png' for i in range(4)}
    baseline_images = {im['path']: image_record(run / im['path'], im['sha256']) for im in baseline['images']}

    rows = [{'index': i, 'images': {'baseline': {'status': 'pending'},
            **{f'{arm}-{step}': {'status': 'pending'} for step in STEPS for arm in ['raw', 'ema']}}}
            for i in range(4)]
    for row in rows:
        row['images']['baseline'] = baseline_images[f"baseline-{row['index']:03}.png"]
    complete_steps = []
    snapshots = []
    for step in STEPS:
        folder = run / f'step-{step:03}'
        marker = folder / 'manifest.json'
        checkpoint_marker = run / f'checkpoint-{step:03}.json'
        if not marker.exists():
            snapshots.append({'step': step, 'status': 'pending'})
            continue
        marker_data = marker.read_bytes()
        snap = json.loads(marker_data)
        assert snap['complete'] is True and snap['step'] == step
        assert snap['protocol_sha256'] == protocol_hash and snap['rng_unchanged'] is True
        assert snap['raw_and_ema_offsets_fixed'] is True
        sources[f'step-{step:03}/manifest.json'] = artifact(marker, marker_data)
        if step == 100 and checkpoint_marker.exists():
            checkpoint_data = checkpoint_marker.read_bytes()
            checkpoint = json.loads(checkpoint_data)
            assert checkpoint['complete'] is True and checkpoint['step'] == step
            assert checkpoint['protocol_sha256'] == protocol_hash
            assert checkpoint['snapshot_manifest_sha256'] == sources[f'step-{step:03}/manifest.json']['sha256']
            assert checkpoint['model_optimizer_rng_path_state_restored_exactly'] is True
            sources[f'checkpoint-{step:03}.json'] = artifact(checkpoint_marker, checkpoint_data)
        z_path = run / 'eval-z.npz'
        sources['eval-z.npz'] = artifact(z_path)
        assert snap['eval_z_sha256'] == sources['eval-z.npz']['sha256']
        expected = {f'{arm}-{i:03}.png' for arm in ['raw', 'ema'] for i in range(4)}
        assert len(snap['images']) == 8 and {im['path'] for im in snap['images']} == expected
        records = {im['path']: image_record(folder / im['path'], im['sha256']) for im in snap['images']}
        for row in rows:
            i = row['index']
            for arm in ['raw', 'ema']:
                row['images'][f'{arm}-{step}'] = records[f'{arm}-{i:03}.png']
            if step == 0:
                # Initial equality marker is written before the complete step-0 marker.
                # Bind each source baseline directly to the step-0 image hash, too.
                assert records[f'raw-{i:03}.png']['sha256'] == records[f'ema-{i:03}.png']['sha256']
                row['images']['baseline'] = image_record(run / f'baseline-{i:03}.png', records[f'raw-{i:03}.png']['sha256'])
        complete_steps.append(step)
        snapshots.append({'step': step, 'status': 'complete', 'manifest_sha256': sources[f'step-{step:03}/manifest.json']['sha256']})
    result = None
    result_path = run / 'result.json'
    if result_path.exists():
        result_data = result_path.read_bytes()
        result = json.loads(result_data)
        assert result['complete'] is True and result['iterations'] == 100
        assert result['protocol_sha256'] == protocol_hash
        assert complete_steps == STEPS
        assert result['raw_and_ema_offsets_fixed'] is True and result['stability_test_only'] is True
        assert result['fisher_estimated'] is False and result['main_adaptation_run'] is False
        assert result['production_approved'] is False and result['server_latency_proven'] is False
        assert result['g_optimizer_steps'] == 125 and result['d_optimizer_steps'] == 107
        assert 'checkpoint-100.json' in sources
        sources['result.json'] = artifact(result_path, result_data)
    termination = None
    termination_path = run / 'termination.json'
    if termination_path.exists():
        raw = termination_path.read_bytes()
        termination = json.loads(raw)
        assert termination['status'] in ['stopped_after_visual_review', 'stopped_by_user', 'stopped_for_safety']
        assert termination['supervisor_and_worker_absent'] is True
        assert result is None
        sources['termination.json'] = artifact(termination_path, raw)
    supervisor = None
    supervisor_path = run / 'supervisor.json'
    if supervisor_path.exists():
        data = supervisor_path.read_bytes()
        supervisor = json.loads(data)
        sources['supervisor.json'] = artifact(supervisor_path, data)
        assert supervisor['protocol_sha256'] == protocol_hash
        if supervisor['complete']:
            assert supervisor['failure'] is None and result is not None
    if termination:
        run_status = 'stopped'
    elif supervisor and not supervisor['complete']:
        run_status = 'failed'
    elif supervisor and supervisor['complete']:
        run_status = 'complete'
    elif result:
        run_status = 'awaiting_supervisor'
    else:
        run_status = 'in_progress_or_awaiting_terminal_record'
    count = sum(im['status'] == 'complete' for row in rows for im in row['images'].values())
    return {
        'schema_version': 1, 'title': 'Native fixed-offset stability test',
        'run': str(run.relative_to(ROOT)), 'run_complete': run_status == 'complete',
        'termination': termination, 'supervisor': supervisor, 'run_status': run_status,
        'built_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'complete_steps': complete_steps, 'default_step': max(complete_steps, default=0),
        'snapshots': snapshots, 'available_images': count, 'expected_images': 44,
        'unfiltered': True, 'preview_seed': protocol['preview_seed'], 'sampling': protocol['sampling'],
        'scope': 'Fresh-source output_rank1 stability test; 34 noise/synthesis-bias offsets fixed. Not full AdAM, Fisher estimation, main adaptation or a quality candidate.',
        'quality_accepted': False, 'production_approved': False, 'server_latency_proven': False,
        'sources': sources, 'builder': artifact(Path(__file__)), 'template': artifact(HERE / 'template.html'), 'rows': rows,
    }


def main():
    manifest = collect()
    data = json.dumps(manifest, indent=2) + '\n'
    page = (HERE / 'template.html').read_text().replace('__DATA__', json.dumps(manifest).replace('<', '\\u003c'))
    # Avoid serving partially written preview outputs during a rebuild.
    for name, contents in [('manifest.json', data), ('index.html', page)]:
        temp = HERE / (name + '.partial')
        temp.write_text(contents)
        temp.replace(HERE / name)
    print(json.dumps({'complete_steps': manifest['complete_steps'], 'images': manifest['available_images'],
                      'manifest_sha256': artifact(HERE / 'manifest.json')['sha256']}))


if __name__ == '__main__':
    main()
