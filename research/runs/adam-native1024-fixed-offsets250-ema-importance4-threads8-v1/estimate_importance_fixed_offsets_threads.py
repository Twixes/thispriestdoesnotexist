"""Prepared CPU-thread comparison variant of fixed-policy EMA importance; no optimization.

Consumes only a completed output_rank1 probing checkpoint. Four pairs are a
compatibility diagnostic; 1000 pairs retain the source's nominal example count.
Neither mode claims reliable rankings, image quality, or a deployable model.
"""
import argparse
import hashlib
import json
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
MODULATION_SHA = 'b032f7965f397a5ebdc93fb1081adaeace651d31a614347123ab83fc46ce456a'
from fixed_offset_policy import capture_fixed_reference, verify_fixed_offsets, importance_named_parameters, POLICY

IMPORTANCE_SHA = '1d1d253e8659863ed567f22201423389f71df36a572657896aadcda8d1ab9bbb'
GIB = 2**30
SEED = 2026092231
BASE_ESTIMATOR_SHA = '2c80a760a562aad5fb8fa33848729a6ae0acad4683e0fa5123928956cbc03ec5'
THREAD_CHOICES = (1, 2, 4, 8)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def make_sample_plan(torch, pairs):
    """The diagnostic is exactly the first four pairs of the fixed 1000-pair plan."""
    assert pairs in (4, 1000)
    rng = torch.Generator().manual_seed(SEED)
    z = torch.randn(1000, 512, generator=rng)
    indices = torch.cat([torch.randperm(20, generator=rng) for _ in range(50)])
    flips = torch.rand(1000, generator=rng) < .5
    seeds = torch.randint(0, 2**63 - 1, (1000,), generator=rng)
    return z[:pairs], indices[:pairs], flips[:pairs], seeds[:pairs]


def validate_input(run, step):
    run = run.resolve()
    run.relative_to(RESEARCH / 'runs')
    protocol_path = run / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol.get('conv_layout') == 'output_rank1', 'Reject source-layout factor scores'
    assert protocol.get('fixed_offset_policy') == {'offset_count': 34, 'noise_count': 17, 'synthesis_bias_count': 17, 'offset_value': 0, 'optimizer_excluded': True, 'raw_and_ema_fixed': True, 'original_strengths_and_biases_unchanged': True}, 'Require fixed-offset probing provenance'
    assert protocol['resolution'] == 1024 and protocol['batch'] == 1
    assert protocol['iterations'] == 500 and step in protocol['checkpoint_steps']
    marker = run / f'checkpoint-{step:03}.json'
    record = json.loads(marker.read_text())
    assert record['complete'] is True and record['step'] == step
    assert record['model_optimizer_rng_path_state_restored_exactly'] is True
    assert record['protocol_sha256'] == sha(protocol_path)
    assert record['snapshot_manifest_sha256'] == sha(run / f'step-{step:03}/manifest.json')
    assert record['checkpoint'] == f'checkpoint-{step:03}.pt'
    checkpoint = run / record['checkpoint']
    assert sha(checkpoint) == record['checkpoint_sha256']
    inventory_path = run / 'modulation-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    assert all(inventory[k]['conv_layout'] == 'output_rank1' for k in ('G', 'D'))
    assert sha(HERE / 'modulation.py') == MODULATION_SHA
    assert sha(HERE / 'importance.py') == IMPORTANCE_SHA
    for name, digest in protocol['pins'].items():
        path = (ROOT / name).resolve(); path.relative_to(ROOT)
        assert sha(path) == digest, name
    assert len(protocol['training']) == 20
    for entry in protocol['training']:
        path = (ROOT / entry['path']).resolve(); path.relative_to(RESEARCH / 'data')
        assert '/train/' in str(path) and sha(path) == entry['sha256']
    return protocol, record


def prepare(run, step, pairs, out, threads=1):
    import psutil
    assert not out.exists(), 'Use a fresh output directory'
    assert pairs in (4, 1000)
    assert type(threads) is int and threads in THREAD_CHOICES, 'Threads must be 1, 2, 4 or 8'
    assert sha(HERE / 'estimate_importance_fixed_offsets.py') == BASE_ESTIMATOR_SHA, 'Frozen parent estimator changed'
    source, record = validate_input(run, step)
    assert psutil.virtual_memory().available / psutil.virtual_memory().total >= .35
    pins = dict(source['pins'])
    for path in (HERE / 'estimate_importance_fixed_offsets_threads.py', HERE / 'estimate_importance_fixed_offsets.py', HERE / 'importance.py', HERE / 'fixed_offset_policy.py',
                 run / 'protocol.json', run / f'checkpoint-{step:03}.json',
                 run / 'modulation-inventory.json', run / 'runtime.json'):
        pins[str(path.relative_to(ROOT))] = sha(path)
    protocol = {
        'name': 'adam-output-rank1-fixed-offsets-ema-importance', 'fixed_offset_policy': POLICY, 'pairs': pairs, 'seed': SEED,
        'diagnostic_only': pairs == 4, 'source_run': str(run.relative_to(ROOT)),
        'source_step': step, 'checkpoint': str((run / record['checkpoint']).relative_to(ROOT)),
        'checkpoint_sha256': record['checkpoint_sha256'], 'state_keys': ['Gema', 'Dema'],
        'source_protocol_sha256': sha(run / 'protocol.json'), 'pins': pins,
        'layout': 'output_rank1', 'training': source['training'],
        'resolution': 1024, 'device': 'cpu', 'threads': threads, 'interop_threads': 1,
        'thread_variant_parent_sha256': BASE_ESTIMATOR_SHA,
        'sample_plan': 'Seeded Gaussian z, deterministic shuffled cycles of twenty training images, random flips and independent synthesis-noise seeds; no style mixing.',
        'objectives': {'G': 'softplus(-Dema(Gema(z)))', 'D': 'softplus(-Dema(real)) + softplus(Dema(same_fake))'},
        'normalization': 'Mean of per-pair elementwise gradient squares; real/fake D terms combined before differentiation.',
        'parameter_scope': 'All u/v and permitted b modulation coordinates, excluding the 34 fixed synthesis noise/activation-bias offsets; no ordinary base-weight or D epilogue gradients are stored. Their partial derivatives are unused by factor-score selection.',
        'accumulator_dtype': 'torch.float64',
        'deviations': ['CPU FP32 native1024 batch1', 'Float64 accumulation instead of source FP32',
                       'Deterministic balanced real-image cycles instead of continuing shuffled loader state',
                       '34 G offsets excluded from measurement and gradients, matching the fixed-offset probing policy (not upstream AdAM)',
                       'Only modulation gradients requested; base weights kept frozen without changing requested partial derivatives'],
        'guards': {'start_available_fraction': .35, 'runtime_available_fraction': .20,
                   'rss_gib': 12, 'swap_growth_mib': 512, 'seconds': 1200 if pairs == 4 else 14400},
        'production_approved': False, 'ranking_stability_proven': False, 'server_latency_proven': False,
    }
    out.mkdir(parents=True)
    (out / '.gitignore').write_text('*.partial\n')
    write(out / 'protocol.json', protocol)
    for name in ('estimate_importance_fixed_offsets_threads.py', 'estimate_importance_fixed_offsets.py', 'importance.py', 'modulation.py', 'fixed_offset_policy.py'):
        shutil.copy2(HERE / name, out / name)
    return protocol


def supervise(run, step, pairs, out, threads=1):
    import psutil
    protocol = prepare(run, step, pairs, out, threads=threads)
    start = time.monotonic(); swap = psutil.swap_memory().used
    proc = None; failure = None; peak = 0; min_available = 1.0
    try:
        with (out / 'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'estimate_importance_fixed_offsets_threads.py'),
                '--worker', '--output', str(out)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                mem = psutil.virtual_memory()
                try:
                    p = psutil.Process(proc.pid)
                    rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                except psutil.NoSuchProcess:
                    continue
                peak = max(peak, rss); min_available = min(min_available, mem.available / mem.total)
                if rss > 12 * GIB: failure = 'RSS guard'
                elif mem.available / mem.total < .20: failure = 'available memory guard'
                elif psutil.swap_memory().used - swap > 512 * 2**20: failure = 'swap growth guard'
                elif time.monotonic() - start > protocol['guards']['seconds']: failure = 'wall time guard'
                if failure:
                    os.killpg(proc.pid, signal.SIGKILL); break
                time.sleep(.1)
            code = proc.wait()
        if code != 0 and failure is None: failure = f'worker exit {code}'
        if failure is None:
            result = json.loads((out / 'result.json').read_text())
            assert result['complete'] and result['pairs'] == pairs
            assert result['protocol_sha256'] == sha(out / 'protocol.json')
    except BaseException:
        failure = traceback.format_exc()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        raise
    finally:
        write(out / 'supervisor.json', {'complete': failure is None, 'failure': failure,
            'seconds': time.monotonic() - start, 'peak_rss_gib': peak / GIB,
            'min_available_fraction': min_available, 'protocol_sha256': sha(out / 'protocol.json')})
    if failure: raise RuntimeError(failure)


def pair_gradients(G, D, z, real, noise_seed, named):
    """One real/fake pair, with the same fake used for both source objectives."""
    import torch
    from torch.nn import functional as F
    expected = {'G': importance_named_parameters(G), 'D': importance_named_parameters(D)}
    for key in expected:
        assert len(named[key]) == len(expected[key])
        assert all(n == en and p is ep for (n, p), (en, ep) in zip(named[key], expected[key])), 'FI parameter scope mismatch'
    fixed_reference = capture_fixed_reference(G)
    verify_fixed_offsets(G, fixed_reference)
    torch.manual_seed(noise_seed)
    for _, p in named['G']: p.requires_grad_(True)
    for _, p in named['D']: p.requires_grad_(False)
    ws = G.mapping(z, None, skip_w_avg_update=True)
    fake = G.synthesis(ws, noise_mode='random', force_fp32=True, fused_modconv=False)
    g_loss = F.softplus(-D(fake, None, force_fp32=True)).mean()
    gg = torch.autograd.grad(g_loss, [p for _, p in named['G']], allow_unused=False)
    fake = fake.detach()
    for _, p in named['G']: p.requires_grad_(False)
    for _, p in named['D']: p.requires_grad_(True)
    d_loss = F.softplus(-D(real, None, force_fp32=True)).mean() + F.softplus(D(fake, None, force_fp32=True)).mean()
    dg = torch.autograd.grad(d_loss, [p for _, p in named['D']], allow_unused=False)
    assert torch.isfinite(g_loss) and torch.isfinite(d_loss)
    assert all(torch.isfinite(x).all() for x in (*gg, *dg))
    verify_fixed_offsets(G, fixed_reference)
    return fake, g_loss.detach(), d_loss.detach(), gg, dg


def worker(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, digest in protocol['pins'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / protocol['checkpoint']) == protocol['checkpoint_sha256']
    import platform
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    sys.path.insert(0, str(RESEARCH / 'vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator, Discriminator
    from modulation import install_modulation, modulation_named_parameters
    from importance import EmpiricalSquaredGradients
    assert type(protocol['threads']) is int and protocol['threads'] in THREAD_CHOICES
    assert protocol['interop_threads'] == 1
    torch.set_num_threads(protocol['threads']); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
    assert torch.get_num_threads() == protocol['threads'] and torch.get_num_interop_threads() == 1
    write(out / 'runtime.json', {'torch': torch.__version__, 'python': sys.version,
        'platform': platform.platform(), 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()})
    checkpoint = torch.load(ROOT / protocol['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    assert checkpoint['iterations'] == protocol['source_step']
    assert checkpoint['protocol_sha256'] == protocol['source_protocol_sha256']
    run = ROOT / protocol['source_run']
    assert checkpoint['runtime_sha256'] == sha(run / 'runtime.json')
    assert checkpoint['modulation_inventory_sha256'] == sha(run / 'modulation-inventory.json')
    gmeta = json.loads((GDIR / 'model.json').read_text())
    dmeta = json.loads((DDIR / 'metadata.json').read_text())
    G = Generator(**gmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    D = Discriminator(**dmeta['init_kwargs']).cpu().eval().requires_grad_(False)
    for key, model in [('G', G), ('D', D)]:
        install_modulation(model, component=key, conv_layout='output_rank1')
        model.load_state_dict(checkpoint[key + 'ema'], strict=True)
        model.requires_grad_(False)
    del checkpoint
    fixed_reference = capture_fixed_reference(G)
    assert len(fixed_reference) == 34
    verify_fixed_offsets(G, fixed_reference)
    named = {'G': importance_named_parameters(G), 'D': importance_named_parameters(D)}
    write(out / 'fixed-offset-policy.json', {'policy': POLICY, 'excluded_names': sorted(fixed_reference),
        'count': len(fixed_reference), 'measured_names': {k: [n for n, _ in v] for k, v in named.items()}})
    acc = {key: EmpiricalSquaredGradients(items) for key, items in named.items()}
    def state_hash(model):
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            x = tensor.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.numpy().tobytes())
        return h.hexdigest()
    before = {'G': state_hash(G), 'D': state_hash(D)}
    z, indices, flips, seeds = make_sample_plan(torch, protocol['pairs'])
    np.savez(out / 'sample-plan.npz', z=z.numpy(), image_index=indices.numpy(), flip=flips.numpy(), noise_seed=seeds.numpy())
    write(out / 'sample-plan.json', {'pairs': protocol['pairs'], 'npz_sha256': sha(out / 'sample-plan.npz'),
        'training_ids': [e['id'] for e in protocol['training']], 'style_mixing': False})
    real_images = []
    for entry in protocol['training']:
        path = ROOT / entry['path']; assert sha(path) == entry['sha256']
        with Image.open(path) as image:
            assert image.size == (1024, 1024)
            real_images.append(torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).float() / 127.5 - 1)
    def save_accumulators(count):
        assert acc['G'].sample_count == acc['D'].sample_count == count
        value = {'G': acc['G'].state_dict(), 'D': acc['D'].state_dict(),
            'pairs': count, 'protocol_sha256': sha(out / 'protocol.json'), 'sample_plan_sha256': sha(out / 'sample-plan.npz')}
        path = out / f'importance-{count:04}.pt'; temp = Path(str(path) + '.partial')
        torch.save(value, temp); temp.replace(path)
        # Independent serialization read, then validate schema and every tensor.
        restored = torch.load(path, weights_only=True, map_location='cpu')
        for key in ('G', 'D'):
            check = EmpiricalSquaredGradients.from_state_dict(restored[key])
            assert check.sample_count == count
            for kind in ('gradient_sums', 'squared_gradient_sums'):
                for name, tensor in value[key][kind].items():
                    assert torch.equal(tensor, restored[key][kind][name])
        write(out / f'importance-{count:04}.json', {'complete': True, 'pairs': count,
            'path': path.name, 'sha256': sha(path), 'protocol_sha256': sha(out / 'protocol.json'),
            'sample_plan_sha256': sha(out / 'sample-plan.npz'), 'state_roundtrip_exact': True})
        return path
    start = time.monotonic()
    for i in range(protocol['pairs']):
        tick = time.monotonic()
        real = real_images[int(indices[i])][None]
        if bool(flips[i]): real = real.flip(-1)
        fake, g_loss, d_loss, gg, dg = pair_gradients(G, D, z[i:i+1], real, int(seeds[i]), named)
        verify_fixed_offsets(G, fixed_reference)
        acc['G'].add_example({n: grad for (n, _), grad in zip(named['G'], gg)})
        acc['D'].add_example({n: grad for (n, _), grad in zip(named['D'], dg)})
        if i < 4:
            a = ((fake[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
            temp = out / f'sample-{i:03}.png.partial'; Image.fromarray(a).save(temp, format='PNG')
            temp.replace(out / f'sample-{i:03}.png')
        record = {'pairs': i + 1, 'g_loss': float(g_loss.detach()), 'd_loss': float(d_loss.detach()), 'seconds': time.monotonic() - tick}
        with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
        del fake, real, g_loss, d_loss, gg, dg
        if i + 1 in (4, 100, 250, 500, 750, 1000): final_path = save_accumulators(i + 1)
    verify_fixed_offsets(G, fixed_reference)
    after = {'G': state_hash(G), 'D': state_hash(D)}
    assert before == after, 'Importance measurement changed EMA parameters or buffers'
    write(out / 'result.json', {'complete': True, 'pairs': protocol['pairs'],
        'seconds': time.monotonic() - start, 'fixed_offsets_zero_and_originals_unchanged': True, 'fixed_offset_policy': POLICY, 'ema_state_unchanged': True, 'ema_state_sha256': after,
        'importance_path': final_path.name, 'importance_sha256': sha(final_path),
        'protocol_sha256': sha(out / 'protocol.json'), 'sample_plan_sha256': sha(out / 'sample-plan.npz'),
        'diagnostic_only': protocol['diagnostic_only'], 'ranking_stability_proven': False,
        'production_approved': False, 'server_latency_proven': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--step', type=int, choices=(100, 250, 500), default=500)
    parser.add_argument('--pairs', type=int, choices=(4, 1000), default=4)
    parser.add_argument('--threads', type=int, choices=THREAD_CHOICES, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if args.worker:
        try: worker(out)
        except BaseException:
            write(out / 'failure.json', {'complete': False, 'traceback': traceback.format_exc()}); raise
    else:
        assert args.run is not None, '--run is required for supervised evaluation'
        supervise(args.run.resolve(), args.step, args.pairs, out, threads=args.threads)
