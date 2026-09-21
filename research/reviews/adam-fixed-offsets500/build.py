#!/usr/bin/env python3
"""Static, nonblocking review of finalized continuation artifacts; no model execution."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'research/runs/adam-native1024-output-rank1-fixed-offsets100-v1'
RUN = ROOT / 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1'
STEPS = [100, 250, 500]
# Reuse the existing native PNG validator and complete parent-run audit.
HELPER = HERE.parent / 'adam-fixed-offsets100/build.py'
spec = importlib.util.spec_from_file_location('parent_preview', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def artifact(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'url': os.path.relpath(path, HERE),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def rebase(record):
    return {**record, 'url': os.path.relpath(ROOT / record['path'], HERE)}


def collect():
    parent = helper.collect(PARENT)
    assert parent['run_complete'] and parent['complete_steps'] == [0, 10, 25, 50, 100]
    protocol = json.loads((RUN / 'protocol.json').read_text())
    sources = {'parent/' + k: rebase(v) for k, v in parent['sources'].items()}
    sources['parent-review-builder'] = artifact(HELPER)
    def source(name):
        record = artifact(RUN / name)
        sources[name] = record
        return json.loads((RUN / name).read_text()) if name.endswith('.json') else record
    source('protocol.json')
    ph = sources['protocol.json']['sha256']
    assert protocol['preview_count'] == 4 and protocol['preview_steps'] == STEPS
    assert protocol['iterations'] == 500 and protocol['start_iteration'] == 100
    assert protocol['checkpoint_steps'] == [250, 500] and protocol['conv_layout'] == 'output_rank1'
    parent_protocol = json.loads((PARENT / 'protocol.json').read_text())
    assert protocol['fixed_offset_policy'] == parent_protocol['fixed_offset_policy']
    assert protocol['parent_protocol_sha256'] == sources['parent/protocol.json']['sha256']
    for name in ['continue_output_rank1_fixed_offsets500.py', 'modulation.py']:
        source(name)
        assert sources[name]['sha256'] == protocol['pins']['research/experiments/adam_native/' + name]
    source('parent-fixed-offsets100.py')
    assert sources['parent-fixed-offsets100.py']['sha256'] == protocol['parent_source_sha256']
    for name in ['protocol.json', 'checkpoint-100.json', 'result.json', 'supervisor.json', 'step-100/manifest.json', 'fixed-offset-policy.json', 'eval-z.npz']:
        assert sources['parent/' + name]['sha256'] == protocol['pins'][str((PARENT / name).relative_to(ROOT))]
    policy = source('fixed-offset-policy.json')
    assert policy == json.loads((PARENT / 'fixed-offset-policy.json').read_text())
    source('eval-z.npz')
    assert sources['eval-z.npz']['sha256'] == protocol['eval_z_source_sha256'] == sources['parent/eval-z.npz']['sha256']
    restoration = source('restoration.json')
    assert restoration['complete'] and restoration['iterations'] == 100
    assert restoration['parent_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    assert restoration['restored_state_sha256'] == protocol['parent_state_sha256']
    assert restoration['optimizer_steps'] == {'g': 125, 'd': 107}
    assert restoration['original_frozen_references_exact'] and restoration['raw_and_ema_offsets_fixed']
    rows = [{'index': row['index'], 'images': {'baseline': rebase(row['images']['baseline']),
             **{f'{arm}-{step}': {'status': 'pending'} for step in STEPS for arm in ['raw', 'ema']}}}
            for row in parent['rows']]
    complete_steps, snapshots = [], []
    for step in STEPS:
        folder = RUN / f'step-{step:03}'
        if not (folder / 'manifest.json').exists():
            snapshots.append({'step': step, 'status': 'pending'})
            continue
        snap = source(f'step-{step:03}/manifest.json')
        assert snap['complete'] and snap['step'] == step and snap['protocol_sha256'] == ph
        assert snap['rng_unchanged'] and snap['raw_and_ema_offsets_fixed']
        assert snap['eval_z_sha256'] == sources['eval-z.npz']['sha256']
        expected = {f'{arm}-{i:03}.png' for arm in ['raw', 'ema'] for i in range(4)}
        assert len(snap['images']) == 8 and {im['path'] for im in snap['images']} == expected
        images = {im['path']: rebase(helper.image_record(folder / im['path'], im['sha256'])) for im in snap['images']}
        for row in rows:
            for arm in ['raw', 'ema']:
                name = f"{arm}-{row['index']:03}.png"
                if step == 100:
                    assert snap['parent_reproduction_exact']
                    assert images[name]['sha256'] == protocol['parent_preview_png_sha256'][name]
                    assert images[name]['sha256'] == parent['rows'][row['index']]['images'][f'{arm}-100']['sha256']
                    # Display the archived parent image; verified child reproduction remains in sources.
                    row['images'][f'{arm}-{step}'] = rebase(parent['rows'][row['index']]['images'][f'{arm}-100'])
                else:
                    row['images'][f'{arm}-{step}'] = images[name]
        checkpoint_name = f'checkpoint-{step:03}.json'
        if step != 100 and (RUN / checkpoint_name).exists():
            checkpoint = source(checkpoint_name)
            assert checkpoint['complete'] and checkpoint['step'] == step and checkpoint['protocol_sha256'] == ph
            assert checkpoint['snapshot_manifest_sha256'] == sources[f'step-{step:03}/manifest.json']['sha256']
            assert checkpoint['model_optimizer_rng_path_state_restored_exactly']
        complete_steps.append(step)
        snapshots.append({'step': step, 'status': 'complete', 'manifest_sha256': sources[f'step-{step:03}/manifest.json']['sha256']})
    result = source('result.json') if (RUN / 'result.json').exists() else None
    if result:
        assert result['complete'] and result['iterations'] == 500 and result['start_iteration'] == 100
        assert result['protocol_sha256'] == ph and complete_steps == STEPS
        assert result['g_optimizer_steps'] == 625 and result['d_optimizer_steps'] == 532
        assert result['raw_and_ema_offsets_fixed'] and result['stability_test_only']
        assert not any(result[k] for k in ['fisher_estimated', 'main_adaptation_run', 'production_approved', 'server_latency_proven'])
        assert all(f'checkpoint-{step:03}.json' in sources for step in [250, 500])
    supervisor = source('supervisor.json') if (RUN / 'supervisor.json').exists() else None
    if supervisor:
        assert supervisor['protocol_sha256'] == ph
        if supervisor['complete']:
            assert supervisor['failure'] is None and result is not None
    termination = source('termination.json') if (RUN / 'termination.json').exists() else None
    if termination:
        assert termination['status'] in ['stopped_after_visual_review', 'stopped_by_user', 'stopped_for_safety']
        assert termination['supervisor_and_worker_absent'] and result is None
    status = ('stopped' if termination else 'failed' if supervisor and not supervisor['complete'] else
              'complete' if supervisor else 'awaiting_supervisor' if result else 'in_progress_or_awaiting_terminal_record')
    return {'schema_version': 1, 'title': 'Native fixed-offset continuation: 100 to 500',
            'run': str(RUN.relative_to(ROOT)), 'parent_run': str(PARENT.relative_to(ROOT)),
            'run_complete': status == 'complete', 'run_status': status, 'supervisor': supervisor, 'termination': termination,
            'built_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'complete_steps': complete_steps, 'default_step': max(complete_steps, default=100), 'snapshots': snapshots,
            'available_images': sum(im['status'] == 'complete' for row in rows for im in row['images'].values()),
            'expected_images': 28, 'unfiltered': True, 'preview_seed': protocol['preview_seed'], 'sampling': protocol['sampling'],
            'scope': 'Exact 100-to-500 continuation; 34 offsets fixed. Prototype only, not a priest-generation candidate, full AdAM, Fisher estimation or main adaptation.',
            'quality_accepted': False, 'production_approved': False, 'server_latency_proven': False,
            'sources': sources, 'builder': artifact(Path(__file__)), 'template': artifact(HERE / 'template.html'), 'rows': rows}


def main():
    manifest = collect()
    page = (HERE / 'template.html').read_text().replace('__DATA__', json.dumps(manifest).replace('<', '\\u003c'))
    for name, contents in [('manifest.json', json.dumps(manifest, indent=2) + '\n'), ('index.html', page)]:
        temp = HERE / (name + '.partial')
        temp.write_text(contents)
        temp.replace(HERE / name)
    print(json.dumps({'complete_steps': manifest['complete_steps'], 'images': manifest['available_images'],
                      'manifest_sha256': artifact(HERE / 'manifest.json')['sha256']}))


if __name__ == '__main__':
    main()
