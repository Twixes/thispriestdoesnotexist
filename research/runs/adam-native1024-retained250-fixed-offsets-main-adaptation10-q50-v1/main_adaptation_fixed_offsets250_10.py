"""Prepared retained250 EMA-route native10 main-adaptation smoke; never auto-launched.

Requires fixed-policy FI1000 at the specifically reviewed retained250 EMA and explicit
quantile. Resets to original G/D, fresh factors/Adam/RNG/path target. No CLI
resume or long training mode. Tiny integration tests exercise the actual engine.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = ROOT / 'research'
GDIR = RESEARCH / 'runs/inference-cpu/ffhq1024/baseline-bundle'
DDIR = RESEARCH / 'models/ffhq1024-discriminator'
DATA = RESEARCH / 'data/flux-priest-domain-v1'
G_HASH = 'f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d'
D_HASH = 'd91ebf17ce8ef94ba50db60d5452583085a4c4723eb7b294daea92510558f1a5'
DATA_HASH = '6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b'
EVAL_Z = RESEARCH / 'runs/adam-native1024-probing500-v1/eval-z.npz'
EVAL_HASH = 'c4eb3bee33c6f9a0502c62b620a1a984aa431d1b43eadcaab1bbc53c8ebe804e'
ITERATIONS = 10
SEED = 2026092271
PARENT_RUNNER_SHA = '9ff54a546593aebeffc0320f897ba66865c13674b5b83be3095cf1471eb73b6b'
RETAINED_SOURCE_REL = 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1'
RETAINED250_PINS = {'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/protocol.json': '0be7e94ef47513d9eb733e6d41621dcb5eb5db5da3c1b3d625ceb857784174ad', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/checkpoint-250.pt': '864bc321a032cce4fa8b7d72afdb31cc6083444a140680d084529627c1cff2a8', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/checkpoint-250.json': '9fb262ab9c30153ed4161f1b87804328acb3aa793eb60078eb097a4cb5077270', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/step-250/manifest.json': '661e4cb3cdae19ba0638083afd1b44f9511f5356a0aa4409767751fe58c1942d', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/modulation-inventory.json': '8c9117c5adf06e1c987d6cc7723970837262af02a3bf2cf543b82886d9c2fa72', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/runtime.json': '3d9d9a9347f0f70a1b7ff7d7fab228bf36c8b444277c4075ef194d57231373d7', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/termination.json': '985e679527a2f77aee88a07f2ae92ba76d6d7ebe8ee9492dfd00631c6e0c29e1', 'research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/supervisor.json': 'fab4e669cfd3fbacc14c63d1c0085793aff29a289cbca3cbf3e7dd42130a1a61', 'research/reviews/adam-fixed-offsets500/native-review-250-root.json': 'b91734fd80e87dc326eb7b9b0fa5ed46baeb8a5bf3467f2f27afcec0bc1b04e5', 'research/reviews/adam-fixed-offsets500/native-review-250-agent.json': 'ffb9bbd72503b8097721acf3eeaed1e467ad352379e507d63d673e2f7537279e'}
RETAINED250_REVIEWS = {'research/reviews/adam-fixed-offsets500/native-review-250-root.json': 'b91734fd80e87dc326eb7b9b0fa5ed46baeb8a5bf3467f2f27afcec0bc1b04e5', 'research/reviews/adam-fixed-offsets500/native-review-250-agent.json': 'ffb9bbd72503b8097721acf3eeaed1e467ad352379e507d63d673e2f7537279e'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def under_root(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def expected_steps(count):
    return {'g': count + (count + 3)//4, 'd': count + (count + 15)//16}


def state_digest(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            digest.update(f'tensor:{x.dtype}:{list(x.shape)}'.encode())
            digest.update(x.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            digest.update(f'{type(item).__name__}:{item!r}'.encode())
    visit(value)
    return digest.hexdigest()


def validate_retained250_source(source_run, source, checkpoint):
    """Narrow exception for this exact reviewed checkpoint; never generic failure acceptance.

    Caller still executes validate_input's full source/checkpoint authentication.
    This additionally binds stop semantics, both reviews and their actual images.
    """
    source_run = source_run.resolve()
    assert str(source_run.relative_to(ROOT)) == RETAINED_SOURCE_REL, 'Unrecognized retained source'
    for relative, expected in RETAINED250_PINS.items():
        assert sha(under_root(relative)) == expected, f'Retained250 evidence changed: {relative}'
    assert source['iterations'] == 500 and 250 in source['checkpoint_steps']
    assert checkpoint['complete'] is True and checkpoint['step'] == 250
    assert checkpoint['checkpoint'] == 'checkpoint-250.pt'
    assert checkpoint['checkpoint_sha256'] == RETAINED250_PINS[RETAINED_SOURCE_REL+'/checkpoint-250.pt']
    assert checkpoint['protocol_sha256'] == RETAINED250_PINS[RETAINED_SOURCE_REL+'/protocol.json']
    assert checkpoint['snapshot_manifest_sha256'] == RETAINED250_PINS[RETAINED_SOURCE_REL+'/step-250/manifest.json']
    assert checkpoint['model_optimizer_rng_path_state_restored_exactly'] is True
    assert checkpoint['optimizer_steps'] == {'g': 313, 'd': 266}
    termination = read(source_run/'termination.json'); supervisor = read(source_run/'supervisor.json')
    assert supervisor['complete'] is False and 'KeyboardInterrupt' in supervisor['failure']
    assert supervisor['protocol_sha256'] == checkpoint['protocol_sha256']
    assert termination['status'] == 'stopped_after_visual_review'
    assert termination['last_completed_step'] == 264 and termination['last_complete_checkpoint'] == 250
    assert termination['planned_iterations'] == 500 and termination['optimizer_steps'] == {'g':330,'d':281}
    assert termination['supervisor_and_worker_absent'] is True
    assert termination['production_approved'] is False and termination['server_latency_proven'] is False
    assert termination['review_files'] == RETAINED250_REVIEWS
    covered = set(); viewed_retained = set()
    for relative, expected in RETAINED250_REVIEWS.items():
        assert RETAINED250_PINS[relative] == expected
        review = read(under_root(relative))
        assert review['step'] == 250
        covered.update(review['indices'])
        if review['reviewer'] == 'root':
            assert review['indices'] == [0,1]
            assert review['photographic_regression'] is True
            assert review['adult_priest_domain_achieved'] is False and review['production_approved'] is False
        else:
            assert review['indices'] == [2,3]
            assert review['snapshot_manifest_sha256'] == checkpoint['snapshot_manifest_sha256']
            assert review['overall']['major_raw_photographic_regression_vs100'] is True
            assert review['overall']['ema_close_to_source'] is True and review['overall']['quality_accepted'] is False
        for image in review['viewed_images']:
            assert sha(under_root(image['path'])) == image['sha256'], 'Reviewed image changed'
            if image['path'].startswith(RETAINED_SOURCE_REL+'/step-250/'):
                viewed_retained.add(image['path'])
    assert covered == {0,1,2,3}
    expected_images = {f'{RETAINED_SOURCE_REL}/step-250/{arm}-{i:03}.png' for arm in ('raw','ema') for i in range(4)}
    assert viewed_retained == expected_images, 'All four raw and EMA images must have been reviewed'
    return {'source_run_complete':False,'source_step':250,'stopped_after_step':264,'reviewed_ema_diagnostic_only':True}


def validated_selection(importance_run, quantile):
    """Authenticate terminal FI and source lineage before returning real masks."""
    import torch
    from importance import EmpiricalSquaredGradients
    from fixed_offset_policy import POLICY
    from selection_fixed_offsets import select_rows_fixed_offsets
    from estimate_importance_fixed_offsets import validate_input
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)) or not math.isfinite(quantile) or not 0 <= quantile <= 100:
        raise ValueError('An explicit finite quantile in [0,100] is required')
    run = importance_run.resolve(); run.relative_to(RESEARCH / 'runs')
    protocol = read(run / 'protocol.json'); result = read(run / 'result.json')
    supervisor = read(run / 'supervisor.json'); marker = read(run / 'importance-1000.json')
    protocol_sha = sha(run / 'protocol.json')
    assert protocol['name'] == 'adam-output-rank1-fixed-offsets-ema-importance'
    assert protocol['pairs'] == 1000 and protocol['diagnostic_only'] is False
    assert protocol['fixed_offset_policy'] == POLICY and protocol['layout'] == 'output_rank1'
    assert protocol['state_keys'] == ['Gema', 'Dema'] and protocol['resolution'] == 1024
    assert protocol['source_step'] == 250, 'Retained250 variant requires EMA importance from exactly source step250'
    assert result['complete'] is True and result['pairs'] == 1000 and result['diagnostic_only'] is False
    assert result['fixed_offset_policy'] == POLICY and result['fixed_offsets_zero_and_originals_unchanged'] is True
    assert result['ema_state_unchanged'] is True
    assert supervisor['complete'] is True and supervisor['failure'] is None
    assert marker['complete'] is True and marker['pairs'] == 1000 and marker['state_roundtrip_exact'] is True
    for record in (result, supervisor, marker): assert record['protocol_sha256'] == protocol_sha
    assert marker['path'] == result['importance_path'] == 'importance-1000.pt'
    importance_path = run / marker['path']
    assert sha(importance_path) == marker['sha256'] == result['importance_sha256']
    assert sha(run / 'sample-plan.npz') == marker['sample_plan_sha256'] == result['sample_plan_sha256']
    for relative, expected in protocol['pins'].items(): assert sha(under_root(relative)) == expected, relative
    source_run = under_root(protocol['source_run']); source_run.relative_to(RESEARCH / 'runs')
    source, checkpoint = validate_input(source_run, protocol['source_step'])
    validate_retained250_source(source_run, source, checkpoint)
    assert under_root(protocol['checkpoint']) == source_run/'checkpoint-250.pt'
    assert sha(source_run / 'protocol.json') == protocol['source_protocol_sha256']
    assert checkpoint['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert source['g_weights_sha256'] == G_HASH and source['d_weights_sha256'] == D_HASH
    assert source['dataset_manifest_sha256'] == DATA_HASH and source['training'] == protocol['training']
    inventory = read(source_run / 'modulation-inventory.json')
    payload = torch.load(importance_path, map_location='cpu', weights_only=True)
    assert payload['pairs'] == 1000 and payload['protocol_sha256'] == protocol_sha
    assert payload['sample_plan_sha256'] == sha(run / 'sample-plan.npz')
    selections = {}
    for key in ('G', 'D'):
        acc = EmpiricalSquaredGradients.from_state_dict(payload[key])
        assert acc.sample_count == 1000 and acc.missing == 'error'
        selections[key] = select_rows_fixed_offsets(acc.mean_squared_gradients(), inventory[key], quantile=quantile)
        assert all(not pool['all_equal'] and not pool['empty_high'] for pool in selections[key]['pools'].values()), 'Degenerate FI pool requires investigation'
    return selections, protocol


class AdaptationEngine:
    """Actual masked optimizer/loss/checkpoint mechanics; accepts tiny test models."""
    def __init__(self, G, D, selections, *, seed):
        import torch
        from adaptation_masks_fixed_offsets import FixedOffsetsAdaptationMasks
        from fixed_offset_policy import capture_fixed_reference
        self.G, self.D = G, D
        self.selections = copy.deepcopy(selections)
        self.fixed = capture_fixed_reference(G)
        self.cg = FixedOffsetsAdaptationMasks(G, selections['G'])
        self.cd = FixedOffsetsAdaptationMasks(D, selections['D'])
        self.Gema = copy.deepcopy(G).eval().requires_grad_(False)
        self.go = torch.optim.Adam(self.cg.parameters(), lr=.002*4/5, betas=(0.0, .99**(4/5)))
        self.do = torch.optim.Adam(self.cd.parameters(), lr=.002*16/17, betas=(0.0, .99**(16/17)))
        self.cg.bind_optimizer(self.go); self.cd.bind_optimizer(self.do)
        self.rng = torch.Generator().manual_seed(seed)
        self.path_mean = torch.zeros(())
        self.iterations = 0
        self.counts = {'g': 0, 'd': 0}
        self.check()

    def check(self):
        from fixed_offset_policy import verify_fixed_offsets
        self.cg.verify_invariants(self.go); self.cd.verify_invariants(self.do)
        verify_fixed_offsets(self.Gema, self.fixed)
        assert self.counts == expected_steps(self.iterations)

    def generate(self):
        import torch
        z = torch.randn(1, self.G.z_dim, generator=self.rng)
        ws = self.G.mapping(z, None, skip_w_avg_update=True)
        if torch.rand((), generator=self.rng) < .9:
            cut = int(torch.randint(1, self.G.num_ws, (), generator=self.rng))
            ws2 = self.G.mapping(torch.randn(1, self.G.z_dim, generator=self.rng), None, skip_w_avg_update=True)
            ws = torch.cat([ws[:, :cut], ws2[:, cut:]], dim=1)
        return self.G.synthesis(ws, noise_mode='random', force_fp32=True, fused_modconv=False), ws

    def apply(self, component):
        import torch
        control, optimizer = (self.cg, self.go) if component == 'g' else (self.cd, self.do)
        grads = [p.grad for p in control.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
        assert any(bool(torch.count_nonzero(g)) for g in grads), 'No active gradients'
        control.step(optimizer)
        self.counts[component] += 1

    def update(self, component, loss):
        import torch
        assert bool(torch.isfinite(loss))
        optimizer = self.go if component == 'g' else self.do
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); self.apply(component)
        return float(loss.detach())

    def iteration(self, real_images):
        import torch
        from torch.nn import functional as F
        i = self.iterations
        self.G.zero_grad(set_to_none=True); self.D.zero_grad(set_to_none=True)
        self.cg.set_enabled(False); self.cd.set_enabled(True)
        index = int(torch.randint(len(real_images), (), generator=self.rng))
        real = real_images[index][None]
        flip = bool(torch.rand((), generator=self.rng) < .5)
        if flip: real = real.flip(-1)
        record = {'iteration': i, 'step': i+1, 'image_index': index, 'horizontal_flip': flip}
        with torch.no_grad(): fake, _ = self.generate()
        self.do.zero_grad(set_to_none=True)
        lf = F.softplus(self.D(fake, None, force_fp32=True)).mean(); lf.backward()
        lr = F.softplus(-self.D(real, None, force_fp32=True)).mean(); lr.backward()
        assert bool(torch.isfinite(lf)) and bool(torch.isfinite(lr))
        self.apply('d'); record['d_loss'] = float((lf + lr).detach())
        del fake, lf, lr
        if i % 16 == 0:
            real_r1 = real.detach().requires_grad_(True)
            logits = self.D(real_r1, None, force_fp32=True)
            gradient = torch.autograd.grad(logits.sum(), real_r1, create_graph=True)[0]
            penalty = gradient.square().flatten(1).sum(1).mean()
            record['r1_loss'] = self.update('d', penalty*80 + logits.sum()*0)
            del real_r1, logits, gradient, penalty
        self.D.zero_grad(set_to_none=True); self.G.zero_grad(set_to_none=True)
        self.cd.set_enabled(False); self.cg.set_enabled(True)
        fake, _ = self.generate()
        record['g_loss'] = self.update('g', F.softplus(-self.D(fake, None, force_fp32=True)).mean())
        del fake
        if i % 4 == 0:
            fake, ws = self.generate()
            noise = torch.randn(fake.shape, generator=self.rng) / math.sqrt(fake.shape[2]*fake.shape[3])
            gradient = torch.autograd.grad((fake*noise).sum(), ws, create_graph=True)[0]
            lengths = gradient.square().sum(2).mean(1).sqrt()
            target = self.path_mean + .01*(lengths.mean() - self.path_mean)
            penalty = (lengths - target).square().mean()
            record['path_loss'] = self.update('g', 8*penalty + fake[0, 0, 0, 0]*0)
            self.path_mean = target.detach()
            del fake, ws, noise, gradient, lengths, target, penalty
        with torch.no_grad():
            for pe, p in zip(self.Gema.parameters(), self.G.parameters()):
                pe.copy_(p.lerp(pe, .5**(32/10000)))
        self.iterations += 1
        self.check()
        record.update(optimizer_steps=dict(self.counts), path_mean=float(self.path_mean))
        return record

    def state(self):
        import torch
        self.check()
        return {'G': self.G.state_dict(), 'D': self.D.state_dict(), 'Gema': self.Gema.state_dict(),
                'g_optimizer': self.go.state_dict(), 'd_optimizer': self.do.state_dict(),
                'torch_rng': torch.get_rng_state(), 'sampling_rng': self.rng.get_state(),
                'path_mean': self.path_mean, 'iterations': self.iterations, 'optimizer_steps': dict(self.counts),
                'selections': self.selections, 'controller_enabled': [self.cg.enabled, self.cd.enabled],
                'optimizer_parameter_names': {'g': [n for n, _ in self.cg.named_parameters()], 'd': [n for n, _ in self.cd.named_parameters()]}}

    def restore(self, payload):
        """Use only on a controller rooted in authenticated original references."""
        import torch
        assert payload['selections'] == self.selections, 'Mask/policy changed'
        assert payload['optimizer_parameter_names'] == self.state()['optimizer_parameter_names'], 'Optimizer order changed'
        assert payload['optimizer_steps'] == expected_steps(payload['iterations']), 'Bad cumulative counts'
        assert len(payload['controller_enabled']) == 2 and all(type(v) is bool for v in payload['controller_enabled'])
        for name in ('G', 'D', 'Gema'):
            getattr(self, name).load_state_dict(payload[name], strict=True)
        self.go.load_state_dict(payload['g_optimizer']); self.do.load_state_dict(payload['d_optimizer'])
        self.G.zero_grad(set_to_none=True); self.D.zero_grad(set_to_none=True)
        self.cg.set_enabled(payload['controller_enabled'][0]); self.cd.set_enabled(payload['controller_enabled'][1])
        self.iterations = payload['iterations']; self.counts = dict(payload['optimizer_steps'])
        self.path_mean = payload['path_mean'].clone()
        torch.set_rng_state(payload['torch_rng']); self.rng.set_state(payload['sampling_rng'])
        self.check()


def prepare(importance_run, quantile, out):
    import psutil
    from fixed_offset_policy import POLICY
    assert not out.exists(), 'Use a fresh output directory'
    assert sha(HERE / 'main_adaptation_fixed_offsets10.py') == PARENT_RUNNER_SHA, 'Frozen parent runner changed'
    selections, fi = validated_selection(importance_run, quantile)
    assert sha(GDIR / 'generator.safetensors') == G_HASH
    assert sha(DDIR / 'D.safetensors') == D_HASH
    assert sha(DATA / 'manifest.json') == DATA_HASH and sha(EVAL_Z) == EVAL_HASH
    assert psutil.virtual_memory().available / psutil.virtual_memory().total >= .35
    pins = dict(fi['pins'])
    files = [HERE / n for n in ('main_adaptation_fixed_offsets250_10.py', 'main_adaptation_fixed_offsets10.py', 'adaptation_masks_fixed_offsets.py', 'selection_fixed_offsets.py', 'selection.py', 'fixed_offset_policy.py', 'importance.py', 'modulation.py')]
    files += [GDIR / 'model.json', DDIR / 'metadata.json', EVAL_Z]
    files += [importance_run / n for n in ('protocol.json', 'result.json', 'supervisor.json', 'importance-1000.json', 'importance-1000.pt', 'sample-plan.npz')]
    files += [under_root(relative) for relative in RETAINED250_PINS]
    for path in files: pins[str(path.resolve().relative_to(ROOT))] = sha(path)
    protocol = {'name': 'adam-native1024-fixed-offsets-retained250-main-adaptation10', 'iterations': ITERATIONS,
        'conv_layout': 'output_rank1', 'fixed_offset_policy': POLICY, 'seed': SEED, 'sampling_seed': SEED+2,
        'importance_run': str(importance_run.resolve().relative_to(ROOT)), 'importance_pairs': 1000,
        'importance_source_run': RETAINED_SOURCE_REL, 'importance_source_step': 250,
        'retained250_lineage': {'source_protocol_planned_iterations': 500, 'source_completed_iterations': 264, 'last_complete_checkpoint': 250, 'overall_source_run_complete': False, 'source_status': 'stopped_after_visual_review', 'pins': dict(RETAINED250_PINS), 'reviews': dict(RETAINED250_REVIEWS)},
        'route_rationale': 'Reviewed coherent/source-like EMA250 may supply diagnostic FI despite rejected raw250; main resets original G/D. This is not a completed500 probe or quality approval.',
        'numerical_disjoint_stability_review_required_before_launch': True,
        'quantile': quantile, 'quantile_quality_validated': False, 'selections': selections,
        'initialization': 'Original source G_ema and D; fresh factors, Adam, RNG and path target; FI supplies masks only',
        'g_weights_sha256': G_HASH, 'd_weights_sha256': D_HASH, 'dataset_manifest_sha256': DATA_HASH,
        'training': fi['training'], 'pins': pins, 'resolution': 1024, 'device': 'cpu', 'threads': 1, 'batch': 1,
        'g_lr': .002*4/5, 'd_lr': .002*16/17, 'g_beta2': .99**(4/5), 'd_beta2': .99**(16/17),
        'r1_gamma': 10, 'r1_every': 16, 'path_weight': 2, 'path_every': 4, 'path_decay': .01,
        'style_mixing_probability': .9, 'ema_decay': .5**(32/10000), 'ema_components': ['G'],
        'augmentation': 'Real horizontal flips only', 'preview_steps': [0, 10], 'preview_count_per_state': 4,
        'guards': {'start_available_fraction': .35, 'runtime_available_fraction': .20, 'rss_gib': 12, 'swap_growth_mib': 512, 'seconds': 1200},
        'deviations': ['Output-rank1 instead of released source flattened layout', 'Native NVIDIA1024 CPU FP32 batch1', 'Original G_ema reset prior', '34 G noise/activation-bias offsets fixed at zero', 'Exactly10 main-adaptation smoke iterations; no quality/convergence claim', 'FI from reviewed retained EMA250 instead of completed500 probing; raw250 rejected and source stopped at264'],
        'expected_optimizer_steps': expected_steps(10), 'production_approved': False, 'server_latency_proven': False}
    out.mkdir(parents=True); (out / '.gitignore').write_text('*.partial\n')
    write(out / 'protocol.json', protocol)
    for key, value in selections.items(): write(out / f'selection-{key}.json', value)
    for path in files:
        if path.parent == HERE: shutil.copy2(path, out / path.name)
    return protocol


def worker(out):
    import numpy as np
    import platform
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from modulation import install_modulation, fold_modulation
    sys.path.insert(0, str(RESEARCH / 'vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator, Discriminator
    protocol = read(out / 'protocol.json')
    for relative, expected in protocol['pins'].items(): assert sha(under_root(relative)) == expected, relative
    assert sha(GDIR / 'generator.safetensors') == G_HASH and sha(DDIR / 'D.safetensors') == D_HASH
    selections, _ = validated_selection(under_root(protocol['importance_run']), protocol['quantile'])
    assert selections == protocol['selections']
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
    write(out / 'runtime.json', {'torch': torch.__version__, 'python': sys.version, 'platform': platform.platform(), 'threads': 1})
    G = Generator(**read(GDIR / 'model.json')['init_kwargs']).cpu().eval().requires_grad_(False)
    D = Discriminator(**read(DDIR / 'metadata.json')['init_kwargs']).cpu().eval().requires_grad_(False)
    G.load_state_dict(load_file(str(GDIR / 'generator.safetensors')), strict=True)
    D.load_state_dict(load_file(str(DDIR / 'D.safetensors')), strict=True)
    with np.load(EVAL_Z, allow_pickle=False) as fixture: z = torch.from_numpy(fixture['z'].copy())
    assert z.shape == (4, 512)
    np.savez(out / 'eval-z.npz', z=z.numpy())
    def render(model, latent):
        return model(latent[None], None, truncation_psi=1, noise_mode='const', force_fp32=True, fused_modconv=False)
    def save_image(value, path):
        data = ((value[0].detach().permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
        temporary = Path(str(path)+'.partial'); Image.fromarray(data).save(temporary, format='PNG'); temporary.replace(path)
    with torch.no_grad():
        baseline = [render(G, latent) for latent in z]
        d_base = D(baseline[0], None, force_fp32=True)
    inventory = {'G': install_modulation(G, component='G', conv_layout='output_rank1'), 'D': install_modulation(D, component='D', conv_layout='output_rank1')}
    write(out / 'modulation-inventory.json', inventory)
    engine = AdaptationEngine(G, D, selections, seed=SEED+2)
    assert len(engine.fixed) == 34
    with torch.no_grad():
        assert all(torch.equal(render(G, latent), before) for latent, before in zip(z, baseline))
        assert torch.equal(D(baseline[0], None, force_fp32=True), d_base)
    write(out / 'initial-equality.json', {'generator_exact': True, 'generator_cases': 4, 'discriminator_exact': True, 'discriminator_cases': 1, 'fresh_optimizer_states_empty': not engine.go.state and not engine.do.state})
    del baseline, d_base
    write(out / 'trainable-inventory.json', {'fixed_offsets': sorted(engine.fixed), 'components': {key: [{'name': n, 'shape': list(p.shape), 'elements': p.numel()} for n,p in control.named_parameters()] for key,control in [('G',engine.cg),('D',engine.cd)]}})
    def preview(step):
        engine.check(); old_rng = torch.get_rng_state(); old_sampling = engine.rng.get_state()
        directory = out / f'step-{step:03}'; directory.mkdir()
        with torch.no_grad():
            for label, model in [('raw', G), ('ema', engine.Gema)]:
                for i, latent in enumerate(z): save_image(render(model, latent), directory/f'{label}-{i:03}.png')
        assert torch.equal(old_rng,torch.get_rng_state()) and torch.equal(old_sampling,engine.rng.get_state())
        engine.check()
        write(directory/'manifest.json', {'complete': True, 'step': step, 'rng_unchanged': True, 'protocol_sha256': sha(out/'protocol.json'), 'images': [{'path': p.name, 'sha256': sha(p)} for p in sorted(directory.glob('*.png'))]})
    real_images=[]
    for entry in protocol['training']:
        path = under_root(entry['path']); assert '/train/' in str(path) and sha(path) == entry['sha256']
        with Image.open(path) as im:
            assert im.size == (1024,1024)
            real_images.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1)
    assert len(real_images)==20
    np.savez(out/'initial-training-rng.npz', torch_rng=torch.get_rng_state().numpy(), sampling_rng=engine.rng.get_state().numpy())
    preview(0); started=time.monotonic()
    for _ in range(ITERATIONS):
        tick=time.monotonic(); record=engine.iteration(real_images); record['seconds']=time.monotonic()-tick
        record['real_id']=protocol['training'][record['image_index']]['id']
        with (out/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
    preview(10)
    payload={'engine':engine.state(),'protocol_sha256':sha(out/'protocol.json'),'runtime_sha256':sha(out/'runtime.json'),'modulation_inventory_sha256':sha(out/'modulation-inventory.json')}
    expected=state_digest(payload); temporary=out/'checkpoint-010.pt.partial'; torch.save(payload,temporary); temporary.replace(out/'checkpoint-010.pt')
    restored=torch.load(out/'checkpoint-010.pt',weights_only=True,map_location='cpu',mmap=True)
    assert state_digest(restored)==expected
    # Exercise restore with a changed permissible adapter and RNG, retaining original controller references.
    with torch.no_grad(): engine.cg.parameters()[0].add_(1)
    torch.rand(());torch.rand((),generator=engine.rng)
    engine.restore(restored['engine'])
    assert state_digest({**restored,'engine':engine.state()})==expected
    write(out/'checkpoint-010.json',{'complete':True,'step':10,'checkpoint_sha256':sha(out/'checkpoint-010.pt'),'state_sha256':expected,'protocol_sha256':sha(out/'protocol.json'),'model_optimizer_rng_path_state_restored_exactly':True})
    del payload,restored
    errors={}
    for label, model in [('raw',G),('ema',engine.Gema)]:
        folded=fold_modulation(model,inplace=False).eval().requires_grad_(False)
        with torch.no_grad():
            errors[label]=[float((render(model,latent)-render(folded,latent)).abs().max()) for latent in z]
        assert errors[label]==[0.0]*4
        temp=out/f'folded-{label}.safetensors.partial';save_file({k:v.detach().contiguous() for k,v in folded.state_dict().items()},str(temp));temp.replace(out/f'folded-{label}.safetensors')
        del folded
    engine.check()
    write(out/'result.json',{'complete':True,'iterations':10,'optimizer_steps':engine.counts,'protocol_sha256':sha(out/'protocol.json'),'seconds_training_previews_export':time.monotonic()-started,'fixed_offsets_and_mask_invariants':True,'folded_native_max_errors':errors,'checkpoint_state_restored_exactly':True,'main_adaptation_run':True,'mechanics_smoke_only':True,'production_approved':False,'server_latency_proven':False})


def supervise(importance_run, quantile, out):
    import psutil
    protocol=prepare(importance_run,quantile,out)
    started=time.monotonic();swap=psutil.swap_memory().used;proc=None;failure=None;peak=0;minimum=1
    try:
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'main_adaptation_fixed_offsets250_10.py'),'--worker','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                mem=psutil.virtual_memory()
                try:
                    current=psutil.Process(proc.pid);rss=current.memory_info().rss+sum(p.memory_info().rss for p in current.children(recursive=True))
                except psutil.NoSuchProcess: continue
                peak=max(peak,rss);minimum=min(minimum,mem.available/mem.total)
                if rss>12*2**30: failure='RSS guard'
                elif mem.available/mem.total<.20: failure='available memory guard'
                elif psutil.swap_memory().used-swap>512*2**20: failure='swap growth guard'
                elif time.monotonic()-started>protocol['guards']['seconds']: failure='wall time guard'
                if failure: os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=proc.wait()
        if code and failure is None: failure=f'worker exit {code}'
        if failure is None:
            result=read(out/'result.json')
            assert result['complete'] is True and result['iterations']==10 and result['optimizer_steps']==expected_steps(10)
            assert result['protocol_sha256']==sha(out/'protocol.json')
    except BaseException:
        failure=traceback.format_exc()
        if proc is not None and proc.poll() is None: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        write(out/'supervisor.json',{'complete':failure is None,'failure':failure,'seconds':time.monotonic()-started,'peak_rss_gib':peak/2**30,'min_available_fraction':minimum,'protocol_sha256':sha(out/'protocol.json')})
    if failure: raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--importance-run',type=Path);parser.add_argument('--quantile',type=float)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker(args.output.resolve())
    else:
        if args.importance_run is None or args.quantile is None: parser.error('--importance-run and --quantile are required')
        supervise(args.importance_run.resolve(),args.quantile,args.output.resolve())
