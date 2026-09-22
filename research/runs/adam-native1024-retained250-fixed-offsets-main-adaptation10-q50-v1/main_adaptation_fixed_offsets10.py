"""Prepared supervised native10 main-adaptation smoke; never auto-launched.

Requires a completed fixed-policy 1000-pair EMA importance run and explicit
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
    assert protocol['source_step'] == 500, 'First main-adaptation smoke requires end-of-probe EMA importance'
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
    assert read(source_run / 'supervisor.json')['complete'] is True
    assert read(source_run / 'result.json')['complete'] is True
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
    selections, fi = validated_selection(importance_run, quantile)
    assert sha(GDIR / 'generator.safetensors') == G_HASH
    assert sha(DDIR / 'D.safetensors') == D_HASH
    assert sha(DATA / 'manifest.json') == DATA_HASH and sha(EVAL_Z) == EVAL_HASH
    assert psutil.virtual_memory().available / psutil.virtual_memory().total >= .35
    pins = dict(fi['pins'])
    files = [HERE / n for n in ('main_adaptation_fixed_offsets10.py', 'adaptation_masks_fixed_offsets.py', 'selection_fixed_offsets.py', 'selection.py', 'fixed_offset_policy.py', 'importance.py', 'modulation.py')]
    files += [GDIR / 'model.json', DDIR / 'metadata.json', EVAL_Z]
    files += [importance_run / n for n in ('protocol.json', 'result.json', 'supervisor.json', 'importance-1000.json', 'importance-1000.pt', 'sample-plan.npz')]
    for path in files: pins[str(path.resolve().relative_to(ROOT))] = sha(path)
    protocol = {'name': 'adam-native1024-fixed-offsets-main-adaptation10', 'iterations': ITERATIONS,
        'conv_layout': 'output_rank1', 'fixed_offset_policy': POLICY, 'seed': SEED, 'sampling_seed': SEED+2,
        'importance_run': str(importance_run.resolve().relative_to(ROOT)), 'importance_pairs': 1000,
        'quantile': quantile, 'quantile_quality_validated': False, 'selections': selections,
        'initialization': 'Original source G_ema and D; fresh factors, Adam, RNG and path target; FI supplies masks only',
        'g_weights_sha256': G_HASH, 'd_weights_sha256': D_HASH, 'dataset_manifest_sha256': DATA_HASH,
        'training': fi['training'], 'pins': pins, 'resolution': 1024, 'device': 'cpu', 'threads': 1, 'batch': 1,
        'g_lr': .002*4/5, 'd_lr': .002*16/17, 'g_beta2': .99**(4/5), 'd_beta2': .99**(16/17),
        'r1_gamma': 10, 'r1_every': 16, 'path_weight': 2, 'path_every': 4, 'path_decay': .01,
        'style_mixing_probability': .9, 'ema_decay': .5**(32/10000), 'ema_components': ['G'],
        'augmentation': 'Real horizontal flips only', 'preview_steps': [0, 10], 'preview_count_per_state': 4,
        'guards': {'start_available_fraction': .35, 'runtime_available_fraction': .20, 'rss_gib': 12, 'swap_growth_mib': 512, 'seconds': 1200},
        'deviations': ['Output-rank1 instead of released source flattened layout', 'Native NVIDIA1024 CPU FP32 batch1', 'Original G_ema reset prior', '34 G noise/activation-bias offsets fixed at zero', 'Exactly10 main-adaptation smoke iterations; no quality/convergence claim'],
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
            proc=subprocess.Popen([sys.executable,str(HERE/'main_adaptation_fixed_offsets10.py'),'--worker','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
