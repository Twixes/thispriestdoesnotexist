"""Tiny actual-vendor CPU fixtures; never loads FFHQ or runs a 1024 backward."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch

from trainer import Generator, grayscale, masked_l1, polygon_mask, quantized, freeze_student, frozen_state, state_digest, partition_masks, partition_loss

HERE = Path(__file__).resolve().parent


def equal_tree(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right), 'Checkpoint tensor mismatch'
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            equal_tree(first, second)
    else:
        assert left == right, (left, right)


def run_fixture(args, name, expect_success=True):
    result = subprocess.run([sys.executable, '-u', str(HERE / 'trainer.py'), *args],
                            capture_output=True, text=True, timeout=120)
    (HERE / f'{name}.log').write_text(result.stdout + result.stderr)
    if expect_success and result.returncode != 0:
        raise AssertionError(f'{name} failed; see saved log')
    if not expect_success and result.returncode == 0:
        raise AssertionError(f'{name} incorrectly succeeded')
    return result


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(418)
    root = HERE / 'tiny-fixtures'
    if root.exists():
        raise SystemExit('Refusing to overwrite prior CPU evidence')
    root.mkdir()
    kwargs = {'z_dim': 8, 'c_dim': 0, 'w_dim': 8, 'img_resolution': 64, 'img_channels': 3,
              'mapping_kwargs': {'num_layers': 2},
              'synthesis_kwargs': {'channel_base': 256, 'channel_max': 16, 'num_fp16_res': 0}}
    source = Generator(**kwargs).cpu().eval().requires_grad_(False)
    # Nonzero strengths ensure freezing both noise tensor and coefficient matters.
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith('noise_strength'):
                parameter.fill_(.15)
    bundle = root / 'bundle'
    bundle.mkdir()
    weights = bundle / 'generator.safetensors'
    save_file({name: tensor.contiguous() for name, tensor in source.state_dict().items()}, str(weights))
    (bundle / 'model.json').write_text(json.dumps({'schema_version': 1, 'init_kwargs': kwargs,
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'truncation_psi': 1,
        'review': {'approved': False}}))
    polygons = [[[0, .75], [1, .75], [1, 1], [0, 1]]]
    mask = polygon_mask(polygons, 64)
    collar_polygons = [[[.45,.8],[.55,.8],[.55,.95],[.45,.95]]]
    collar, rest, region_info = partition_masks(mask, collar_polygons, 64)
    pairs = []
    rng = torch.Generator().manual_seed(941)
    for index in range(3):
        z = torch.randn(1, 8, generator=rng)
        with torch.no_grad():
            w = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
            original = source.synthesis(w, noise_mode='const', force_fp32=True)
        # Synthetic tensor fixture, not a generated photographic asset.
        target = original * (1 - mask) + torch.full_like(original, -.6 + index * .1) * rest + torch.full_like(original, .8) * collar
        Image.fromarray(quantized(original)).save(root / f'source-{index}.png')
        Image.fromarray(quantized(target)).resize((96, 96), Image.Resampling.NEAREST).save(root / f'target-{index}.png')
        np.savez(root / f'latent-{index}.npz', z=z.numpy(), w=w.numpy())
        pairs.append({'id': str(index), 'split': 'validation' if index == 2 else 'train',
                      'source_path': f'source-{index}.png', 'target_path': f'target-{index}.png',
                      'latent_path': f'latent-{index}.npz', 'clothing_polygons': polygons,
                      **({'collar_polygons': collar_polygons} if index != 2 else {})})
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'version': 2, 'requires_trainer_feature': 'equal_area_collar_partition_v1', 'production_approved': False, 'source_bundle': 'bundle', 'pairs': pairs}, indent=2))

    # Region area and batch normalization: unequal masks, identical error => same loss.
    image = torch.ones(2, 1, 4, 4, requires_grad=True)
    masks = torch.zeros_like(image)
    masks[0, :, 0, 0] = 1
    masks[1] = 1
    loss = masked_l1(image, torch.zeros_like(image), masks)
    assert float(loss.detach()) == 1
    loss.backward()
    assert float(image.grad[0, 0, 0, 0]) == .5
    assert float(image.grad[1, 0, 0, 0]) == .5 / 16
    assert float(image.grad[0, 0, 1, 1]) == 0
    try:
        masked_l1(image, torch.zeros_like(image), torch.zeros_like(image))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty region accepted')
    # Independent area normalization gives each disjoint region half the loss
    # and half the summed pixel gradient despite radically different areas.
    pred = torch.ones(1, 1, 8, 8, requires_grad=True)
    tiny = torch.zeros_like(pred); tiny[0,0,0,0] = 1
    large = torch.ones_like(pred)-tiny
    combined, tab_loss, rest_loss = partition_loss(pred, torch.zeros_like(pred), tiny, large)
    assert float(combined.detach()) == 1
    combined.backward()
    assert float((pred.grad*tiny).sum()) == .5
    assert abs(float((pred.grad*large).sum())-.5) < 1e-7
    for invalid_polygon in [None, [[[0,0],[1,0],[1,.1],[0,.1]]], polygons]:
        try:
            partition_masks(mask, invalid_polygon, 64)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing, empty intersection, or full clothing collar accepted')
    # A trace may cross clothing, but only the strict intersection is optimized.
    crossing = [[[.4,.7],[.6,.7],[.6,.85],[.4,.85]]]
    clipped, remaining, info = partition_masks(mask, crossing,64)
    assert info['trace_outside_clothing_pixels'] > 0
    assert torch.equal(clipped+remaining,mask)
    assert not bool((clipped*(1-mask)).any())
    student = copy.deepcopy(source)
    trainable = freeze_student(student)
    assert all(name.startswith('synthesis.b64.') and not name.endswith('noise_strength') for name in trainable)
    initial_frozen = state_digest(frozen_state(student))
    initial_student = state_digest(student.state_dict())

    shared = ['--manifest', str(manifest), '--device', 'cpu', '--threads', '1', '--batch', '2',
              '--fresh-weight', '.5', '--preview-count', '2', '--checkpoint-every', '1']
    run_fixture(shared + ['--run', str(root / 'continuous'), '--total-steps', '5'], 'cpu-continuous')
    run_fixture(shared + ['--run', str(root / 'first'), '--total-steps', '2'], 'cpu-first')
    run_fixture(shared + ['--run', str(root / 'resumed'), '--total-steps', '5',
                         '--resume', str(root / 'first/resume.pt')], 'cpu-resumed')
    continuous = torch.load(root / 'continuous/resume.pt', weights_only=True)
    resumed = torch.load(root / 'resumed/resume.pt', weights_only=True)
    for key in ['student', 'optimizer', 'rng', 'step', 'paired_images_seen', 'fresh_images_seen',
                'source_digest', 'frozen_digest', 'fixed_z', 'options', 'provenance']:
        equal_tree(continuous[key], resumed[key])
    assert continuous['frozen_digest'] == initial_frozen
    assert continuous['source_digest'] == state_digest(source.state_dict())
    assert state_digest(continuous['student']) != initial_student
    assert all(torch.isfinite(tensor).all() for tensor in continuous['student'].values())
    records = [json.loads(line) for line in (root / 'continuous/metrics.jsonl').read_text().splitlines()]
    assert all('2' not in record['pair_ids'] for record in records)
    assert all(record['collar_l1'] > 0 and record['rest_clothing_l1'] > 0 for record in records)
    assert any(record['fresh_preservation_l1'] > 0 for record in records[1:])
    student.load_state_dict(continuous['student'])
    with np.load(root / 'latent-0.npz') as latent, torch.no_grad():
        output = student.synthesis(torch.from_numpy(latent['w']), noise_mode='const', force_fp32=True)
    expected = quantized(grayscale(output).repeat(1, 3, 1, 1))
    preview = np.array(Image.open(root / 'continuous/000005-pair-000-train.png'))
    assert np.array_equal(expected, preview), 'Preview is not the full generated output'
    for file in (root / 'continuous').glob('000005-*.png'):
        assert file.read_bytes() == (root / 'resumed' / file.name).read_bytes()
    bad = run_fixture(shared + ['--run', str(root / 'bad-options'), '--total-steps', '6',
                                 '--resume', str(root / 'first/resume.pt'), '--lr', '.002'], 'cpu-reject-options', False)
    assert 'matching options' in bad.stderr
    bad = run_fixture(shared + ['--run', str(root / 'bad-production'), '--production-approved'], 'cpu-reject-production', False)
    assert 'Production approval/export is unavailable' in bad.stderr
    # Corrupt source provenance, then prove exact CPU source checking catches it.
    saved = (root / 'source-0.png').read_bytes()
    corrupted = np.array(Image.open(root / 'source-0.png'))
    corrupted[0, 0, 0] ^= 127
    Image.fromarray(corrupted).save(root / 'source-0.png')
    try:
        bad = run_fixture(shared + ['--run', str(root / 'bad-source')], 'cpu-reject-source', False)
        assert 'source uint8 mismatch' in bad.stderr
    finally:
        (root / 'source-0.png').write_bytes(saved)
    # Resume rejects both the old objective and modified imported-helper provenance.
    tampered = torch.load(root / 'first/resume.pt', weights_only=True)
    tampered['format'] = 'paired-edit-research-v1'
    torch.save(tampered, root / 'broad-checkpoint.pt')
    bad = run_fixture(shared + ['--run', str(root/'bad-format'), '--total-steps','6', '--resume',str(root/'broad-checkpoint.pt')], 'cpu-reject-format', False)
    assert 'Only paired-regions' in bad.stderr
    tampered['format'] = 'paired-regions-research-v1'
    tampered['provenance']['source_files']['paired_edit_helpers'] = '0'*64
    torch.save(tampered, root / 'changed-code.pt')
    bad = run_fixture(shared + ['--run', str(root/'bad-code'), '--total-steps','6', '--resume',str(root/'changed-code.pt')], 'cpu-reject-code', False)
    assert 'matching options' in bad.stderr
    assert 'paired_edit_helpers' in continuous['provenance']['source_files']
    result = {'passed': True, 'device': 'cpu', 'threads': 1, 'resolution': 64, 'channels_max': 16,
              'heavy_ffhq_loaded': False, 'mps_used': False, 'continuous_updates': 5, 'resumed_updates': '2+3',
              'exact_optimizer_model_rng_resume': True, 'frozen_source_noise_and_blocks_unchanged': True,
              'actual_finite_student_update': True, 'validation_excluded_from_training': True,
              'mask_area_and_batch_normalization_gradients': True,
              'equal_partition_summed_gradients': True, 'empty_full_missing_tab_rejected': True,
              'intersection_preserves_original_clothing_protection': True,
              'old_objective_resume_rejected': True, 'changed_imported_helper_hash_rejected': True,
              'validation_needs_no_tab': True, 'empty_mask_rejected': True,
              'fresh_latent_preservation_exercised': True, 'previews_equal_full_generated_output': True,
              'resume_previews_byte_identical': True, 'source_mismatch_rejected': True,
              'changed_options_rejected': True, 'production_approval_rejected': True,
              'seconds': time.monotonic() - started,
              'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ['trainer.py', 'check_cpu.py']}}
    (HERE / 'cpu-evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
