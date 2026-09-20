"""Two actual 256px updates exercising trainer checkpoint save/load, no long run.

Only preview generation is replaced by a no-op (separately tested elsewhere).
Training, checkpoint serialization, model/optimizer loading, CDC and R1 run through
research/train.py. Hooks observe and assert, without changing tensors or RNG.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train

OUT = ROOT / 'experiments/cdc'
CREATE = ROOT / 'runs/cdc-resume-roundtrip-create'
RESUME = ROOT / 'runs/cdc-resume-roundtrip-resumed'


def assert_equal(left, right, path='state'):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu()), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f'{path}.{i}')
    else:
        assert left == right, (path, left, right)


def phase(name):
    torch.set_num_threads(2)
    evidence = {'phase': name, 'device': 'mps', 'resolution': 256, 'preview_grids_skipped': True,
                'trainer_sha256': hashlib.sha256((ROOT/'train.py').read_bytes()).hexdigest()}
    expected = torch.load(CREATE/'resume.pt', map_location='cpu', weights_only=False) if name == 'resume' else None
    original_adam_load = torch.optim.Adam.load_state_dict
    original_adam_step = torch.optim.Adam.step
    original_randn = torch.randn
    original_cdc = train.cdc_loss
    optimizers = []
    source_ref = []
    cdc_observations = []
    loads = []
    rng_observations = []
    if expected is not None:
        expected_generator = torch.Generator(device='cpu')
        expected_generator.set_state(expected['cdc_rng'])
        expected_z = original_randn(4, 512, generator=expected_generator)

    def checked_load(optimizer, state):
        result = original_adam_load(optimizer, state)
        label = ['G_opt', 'D_opt'][len(loads)]
        assert_equal(state, expected[label], label+'.checkpoint_input')
        assert_equal(optimizer.state_dict(), expected[label], label+'.restored')
        loads.append({'optimizer': label, 'state_entries': len(state['state']), 'exact_restoration': True})
        return result

    def checked_step(optimizer, *args, **kwargs):
        assert all(torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params'] if p.grad is not None)
        result = original_adam_step(optimizer, *args, **kwargs)
        assert all(torch.isfinite(p).all() for group in optimizer.param_groups for p in group['params'])
        optimizers.append(optimizer)
        return result

    def checked_randn(*args, **kwargs):
        generator = kwargs.get('generator')
        if generator is not None and expected is not None:
            assert torch.equal(generator.get_state(), expected['cdc_rng']), 'CDC RNG state was not restored before next latent draw'
        value = original_randn(*args, **kwargs)
        if generator is not None and expected is not None:
            assert torch.equal(value.cpu(), expected_z), 'Next CDC latent batch differs from checkpoint continuation'
            rng_observations.append({'restored_state_exact': True, 'next_latents_exact': True,
                                     'next_latents_sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()})
        return value

    def checked_cdc(source, target, latents, **kwargs):
        source_hash = train.state_hash(source)
        target_hash = train.state_hash(target)
        # Hash from independently loaded base raw G in preceding bounded smoke.
        assert source_hash == '5a5e5e22461f7c1a94f69bc813230a8ac5d5705b7c0b4da30c812feb51d4a9d4'
        if expected is not None:
            assert target_hash != source_hash, 'Resumed adapted target incorrectly replaced frozen source'
        source_ref.append((source, source_hash))
        loss, details = original_cdc(source, target, latents, **kwargs)
        cdc_observations.append({'loss': float(loss.detach()), 'source_sha256': source_hash,
                                 'target_sha256': target_hash, 'details': details})
        return loss, details

    train.save_comparison_grids = lambda *args, **kwargs: None
    torch.optim.Adam.load_state_dict = checked_load
    torch.optim.Adam.step = checked_step
    torch.randn = checked_randn
    train.cdc_loss = checked_cdc
    argv = ['train.py', '--device', 'mps', '--threads', '2', '--batch', '1', '--freeze-d-layers', '4',
            '--cdc-weight', '1000', '--cdc-batch', '4', '--data', str(ROOT/'data/hot110-256'),
            '--run', CREATE.name if name == 'create' else RESUME.name]
    if name == 'create':
        argv += ['--steps', '1', '--snapshot-every', '1']
    else:
        argv += ['--resume', str(CREATE/'resume.pt'), '--smoke-test']
    sys.argv = argv
    started = time.monotonic()
    train.main()
    assert len(optimizers) == 2, 'Expected exactly one G and one D optimizer update'
    assert all(train.state_hash(source) == before for source, before in source_ref)
    assert all(p.grad is None and not p.requires_grad for source, _ in source_ref for p in source.parameters())
    evidence.update({'seconds': time.monotonic()-started, 'finite_optimizer_updates': 2,
                     'source_unchanged': True, 'source_has_no_gradients': True,
                     'optimizer_load_checks': loads, 'rng_checks': rng_observations,
                     'cdc_observations': cdc_observations,
                     'mps_current_bytes': torch.mps.current_allocated_memory(),
                     'mps_driver_bytes': torch.mps.driver_allocated_memory()})
    if expected is not None:
        assert len(loads) == 2 and len(rng_observations) == 1
        initial_config = json.loads((CREATE/'config.json').read_text())
        resumed_config = json.loads((RESUME/'config.json').read_text())
        assert resumed_config['cdc'] == expected['cdc_config'] == initial_config['cdc']
        for field in ['mapping_lr', 'synthesis_lr', 'd_lr', 'freeze_d_layers', 'batch', 'base_sha256', 'dataset_sha256']:
            assert initial_config[field] == resumed_config[field], field
        assert resumed_config['resume_step'] == 1 and resumed_config['resume_images_seen'] == 1
        evidence['config_preserved'] = True
        evidence['resume_step_and_images_seen_restored'] = True
        evidence['resumed_smoke'] = json.loads((RESUME/'smoke-test.json').read_text())
    (OUT/f'resume-roundtrip-{name}.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['create', 'resume'])
    args = parser.parse_args()
    if args.phase:
        phase(args.phase)
    else:
        if (CREATE/'resume.pt').exists() or (RESUME/'smoke-test.json').exists():
            raise SystemExit('Evidence already exists; choose new run paths before repeating')
        for stage in ['create', 'resume']:
            with (OUT/f'resume-roundtrip-{stage}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', __file__, '--phase', stage], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Full CDC checkpoint round trip passed; evidence in research/experiments/cdc/resume-roundtrip-*.json')
