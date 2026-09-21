#!/usr/bin/env python3
"""Build a static, hash-verified diagnostic review without loading model weights."""
import hashlib
import html
import io
import json
import os
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'research/runs/adam-native1024-noise-ablation100-v1'
PARENT = ROOT / 'research/runs/adam-native1024-probing500-v1'
ARMS = ['unchanged_raw', 'reset_b1024_conv0_offset', 'reset_all_noise_offsets']
LABELS = {
    'baseline': ('Original FFHQ', 'Before probing; same fixed latent.'),
    'unchanged_raw': ('Unchanged raw G · 100', 'Exact reproduction of the archived checkpoint-100 PNG.'),
    'reset_b1024_conv0_offset': ('Reset one noise offset', 'Only the learned b1024.conv0 noise-strength offset is zero.'),
    'reset_all_noise_offsets': ('Reset all 17 noise offsets', 'Every learned noise-strength offset is zero; other state is unchanged.'),
}


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
    return {**item, 'width': 1024, 'height': 1024}


def collect():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = artifact(path, raw)
        return json.loads(raw)

    protocol = read(RUN / 'protocol.json')
    protocol_hash = sources[str((RUN / 'protocol.json').relative_to(ROOT))]['sha256']
    assert protocol['arms'] == ARMS and protocol['latent_count'] == 4
    assert protocol['resolution'] == 1024 and protocol['conv_layout'] == 'source_flattened'
    assert protocol['checkpoint_step'] == 100 and protocol['checkpoint_state'] == 'G'
    assert protocol['no_training'] is True and protocol['no_image_editing_or_filtering'] is True
    assert protocol['render'] == {'truncation_psi': 1, 'noise_mode': 'const', 'force_fp32': True, 'fused_modconv': False}
    result = read(RUN / 'result.json')
    supervisor = read(RUN / 'supervisor.json')
    loaded = read(RUN / 'loaded-state.json')
    read(RUN / 'runtime.json')
    for record in [result, supervisor]:
        assert record['complete'] is True and record['protocol_sha256'] == protocol_hash
    assert supervisor['failure'] is None
    assert result['original_state_restored_exact'] is True
    assert result['unchanged_raw_parent_pngs_exact'] is True
    assert result['production_approved'] is False and result['server_latency_proven'] is False
    assert loaded['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert loaded['layout'] == protocol['conv_layout']
    noise_names = set(loaded['noise_offsets'])
    assert len(noise_names) == protocol['all_noise_offset_count'] == 17
    assert protocol['single_offset'] in noise_names
    for name in ['noise_ablation.py', 'modulation.py']:
        entry = artifact(RUN / name)
        assert entry['sha256'] == protocol['pins'][f'research/experiments/adam_native/{name}']
        sources[entry['path']] = entry

    parent_protocol = read(PARENT / 'protocol.json')
    assert sources[str((PARENT / 'protocol.json').relative_to(ROOT))]['sha256'] == protocol['parent_protocol_sha256']
    checkpoint = read(PARENT / 'checkpoint-100.json')
    assert checkpoint['complete'] is True and checkpoint['step'] == 100
    assert checkpoint['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert checkpoint['protocol_sha256'] == protocol['parent_protocol_sha256']
    assert checkpoint['model_optimizer_rng_path_state_restored_exactly'] is True
    snapshots = {step: read(PARENT / f'step-{step:03}/manifest.json') for step in [0, 100]}
    assert sources[str((PARENT / 'step-100/manifest.json').relative_to(ROOT))]['sha256'] == checkpoint['snapshot_manifest_sha256']
    z = artifact(RUN / 'eval-z.npz')
    parent_z = artifact(PARENT / 'eval-z.npz')
    sources[z['path']], sources[parent_z['path']] = z, parent_z
    assert z['sha256'] == parent_z['sha256']
    for step, snap in snapshots.items():
        assert snap['complete'] is True and snap['step'] == step and snap['rng_unchanged'] is True
        assert snap['protocol_sha256'] == protocol['parent_protocol_sha256']
        assert snap['eval_z_sha256'] == z['sha256']
        assert len(snap['images']) == 8
        assert {im['path'] for im in snap['images']} == {f'{arm}-{i:03}.png' for arm in ['raw', 'ema'] for i in range(4)}
    parent_images = {step: {im['path']: im['sha256'] for im in snap['images']} for step, snap in snapshots.items()}
    rows = [{'index': i, 'images': {}} for i in range(4)]
    result_arms = {arm['arm']: arm for arm in result['arms']}
    assert len(result['arms']) == len(result_arms) == 3 and set(result_arms) == set(ARMS)
    assert len(result['images']) == 12
    result_images = {(im['arm'], im['index']): im for im in result['images']}
    assert set(result_images) == {(arm, i) for arm in ARMS for i in range(4)}
    for arm in ARMS:
        marker = read(RUN / arm / 'manifest.json')
        assert marker == result_arms[arm]
        assert marker['complete'] is True and marker['protocol_sha256'] == protocol_hash
        assert marker['all_other_state_exact'] is True
        expected_changes = set() if arm == ARMS[0] else ({protocol['single_offset']} if arm == ARMS[1] else noise_names)
        for key in ['declared_resets', 'actual_changed_state_names']:
            assert len(marker[key]) == len(expected_changes) and set(marker[key]) == expected_changes
        assert len(marker['images']) == 4 and {im['index'] for im in marker['images']} == set(range(4))
        for im in marker['images']:
            i = im['index']
            assert im == result_images[(arm, i)] and im['path'] == f'{arm}/{i:03}.png'
            row = rows[i]
            latent_hash = im['latent_tensor_sha256']
            assert row.setdefault('latent_tensor_sha256', latent_hash) == latent_hash
            row['images'][arm] = image_record(RUN / im['path'], im['sha256'])
            if arm == ARMS[0]:
                assert im['parent_png_exact'] is True
                assert im['sha256'] == parent_images[100][f'raw-{i:03}.png'] == protocol['archived_raw_png_sha256'][f'raw-{i:03}.png']
                source_image = image_record(PARENT / f'step-100/raw-{i:03}.png', im['sha256'])
                sources[source_image['path']] = source_image
    for row in rows:
        i = row['index']
        expected = parent_images[0][f'raw-{i:03}.png']
        assert expected == parent_images[0][f'ema-{i:03}.png']
        row['images']['baseline'] = image_record(PARENT / f'baseline-{i:03}.png', expected)
    # Bind the relevant frozen parent metadata to the ablation protocol pins.
    for path, entry in sources.items():
        if path in protocol['pins']:
            assert entry['sha256'] == protocol['pins'][path], f'Pinned input mismatch: {path}'
    return {
        'schema_version': 1, 'title': 'AdAM noise-offset ablation · checkpoint 100',
        'run': str(RUN.relative_to(ROOT)), 'run_complete': True,
        'scope': 'Diagnostic only; source-layout probing control, not a priest-generation candidate.',
        'conv_layout': protocol['conv_layout'], 'checkpoint_state': 'G', 'checkpoint_step': 100,
        'checkpoint_sha256': protocol['checkpoint_sha256'],
        'checkpoint_verification': 'Completed checkpoint marker and run load record agree; builder does not reread model weights.',
        'no_new_training': True, 'unfiltered': True, 'image_editing': False,
        'production_approved': False, 'server_latency_proven': False,
        'exact_raw_parent_reproduction': True, 'available_images': 16, 'ablation_images': 12,
        'sampling': protocol['render'], 'preview_seed': parent_protocol['preview_seed'],
        'arms': ARMS, 'rows': rows, 'sources': sources, 'builder': artifact(Path(__file__)),
    }


def render(manifest):
    rows = []
    for row in manifest['rows']:
        cards = []
        for arm in ['baseline', *ARMS]:
            image = row['images'][arm]
            label, description = LABELS[arm]
            url = html.escape(image['url'], quote=True)
            cards.append(f'<article><h3>{label}</h3><p>{description}</p><a href="{url}" target="_blank" rel="noopener"><img src="{url}" width="1024" height="1024" alt="Latent {row["index"]:03} — {label}" loading="lazy"></a><small>Native 1024 × 1024 PNG · <code>{image["sha256"][:16]}</code></small></article>')
        rows.append(f'<section><h2>Latent {row["index"]:03}</h2><div class="grid">{"".join(cards)}</div><p class="hash">Same latent in each ablation arm: <code>{row["latent_tensor_sha256"]}</code></p></section>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AdAM noise-offset ablation · checkpoint 100</title>
<style>html{color-scheme:dark}body{margin:0;background:#101012;color:#eee;font:15px/1.5 system-ui,sans-serif}main{max-width:1800px;margin:auto;padding:28px}h1{font-size:28px;line-height:1.2}h2{margin:32px 0 12px}h3{font-size:16px;margin:0}a{color:#a8cfff}p{max-width:1100px}.scope{border-left:3px solid #e8b963;padding:10px 16px;background:#22201b}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}article{min-width:0}article p{font-size:13px;color:#bbb;min-height:40px;margin:6px 0 12px}img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#000}small,.hash{font-size:11px;color:#aaa;overflow-wrap:anywhere}code{font-family:ui-monospace,monospace}.meta{color:#bbb}footer{border-top:1px solid #333;margin-top:28px;padding-top:16px}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){main{padding:16px}.grid{grid-template-columns:1fr}article p{min-height:0}}</style>
<main><h1>What changes when learned noise offsets are reset?</h1>
<p class="scope"><strong>Diagnostic, not a model candidate.</strong> This compares raw G at iteration 100 of the released-source convolution-layout probing control (<code>source_flattened</code>). No new training, Fisher estimation or main adaptation occurred in this ablation. Nothing here establishes production quality or server generation latency.</p>
<p>All four fixed latents are shown without filtering, repairs or selection. Each row contains the original FFHQ baseline and all three ablation arms. Click any image for its native 1024 × 1024 original. These source-population latents can include children, women and non-clergy; they are not a proposed priest catalog.</p>
<p><strong>Zeroing an offset retains the original noise strength.</strong> It removes the learned additive change, not the pretrained coefficient or the fixed noise maps. The single-offset arm changes only <code>b1024.conv0</code>; the all-offset arm changes all 17 learned noise-strength offsets. All other checkpoint state is verified unchanged.</p>
<p class="meta">The unchanged raw arm reproduces the four archived checkpoint-100 PNGs byte for byte. Same latents; constant noise; truncation 1; FP32; unfused modulated convolution. The original FFHQ baseline is shown for context. Four fixed latents cannot establish generalization or unseen-latent diversity.</p>
''' + '\n'.join(rows) + '''<footer><a href="manifest.json">Hash-bound manifest</a> · <a href="README.md">Rebuild and verification</a> · <a href="../adam-native-probing/">Probing review</a> · <a href="decision.json">Diagnostic decision</a> · <a href="native-review-root.json">Native review 1</a> · <a href="native-review-agent.json">Native review 2</a><p class="meta">Completed diagnostic run; review is non-blocking. No visual acceptance is implied. No model runs are started by this page or builder.</p></footer></main></html>\n'''


def main():
    manifest = collect()
    contents = [('manifest.json', json.dumps(manifest, indent=2) + '\n'), ('index.html', render(manifest))]
    for name, data in contents:
        temporary = HERE / (name + '.partial')
        temporary.write_text(data)
        temporary.replace(HERE / name)
    print(json.dumps({'run_complete': True, 'ablation_images': 12, 'baseline_images': 4,
                      'manifest_sha256': artifact(HERE / 'manifest.json')['sha256']}))


if __name__ == '__main__':
    main()
