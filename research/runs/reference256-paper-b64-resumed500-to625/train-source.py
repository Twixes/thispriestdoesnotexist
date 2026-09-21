"""Single-device outer loop around unmodified NVIDIA StyleGAN2-ADA components.

Phase schedule, accumulation gains, lazy Adam compensation, EMA and ADA follow
vendored training/training_loop.py; losses/augmentation/data sampler are imported.
See trainer-notes.md for provenance and explicit portability deviations.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'vendor/stylegan2-ada-pytorch'
sys.path.insert(0, str(VENDOR))
import legacy
from training.loss import StyleGAN2Loss
from training.augment import AugmentPipe
from training.dataset import ImageFolderDataset
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from smoke import FP32Call, BGC


class PortableInfiniteSampler(misc.InfiniteSampler):
    """Keep NVIDIA's iterator; adapt its removed Sampler(data_source) constructor."""
    def __init__(self, dataset, seed):
        torch.utils.data.Sampler.__init__(self)
        self.dataset = dataset
        self.rank = 0
        self.num_replicas = 1
        self.shuffle = True
        self.seed = seed
        self.window_size = .5


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        torch.save(to_cpu(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_generator_snapshot(ema_state, step, images_seen, config, recipe, run):
    if step <= 0:
        raise ValueError('Generator export snapshots require a positive training step')
    # JSON round trip strips non-primitive subclasses (e.g. TorchVersion) and
    # excludes NumPy/RNG objects, keeping the existing weights_only exporter safe.
    payload = {'G_ema':ema_state,'step':int(step),'images_seen':int(images_seen),
               'config':json.loads(json.dumps(config)),'recipe':json.loads(json.dumps(recipe))}
    path = Path(run)/f'generator-{step:06}.pt'
    atomic_save(payload,path)
    return path


def rng_state(device):
    return {'torch': torch.get_rng_state(), 'python': random.getstate(), 'numpy': np.random.get_state(),
            'mps': torch.mps.get_rng_state() if device == 'mps' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if device == 'mps':
        if state['mps'] is None:
            raise ValueError('MPS exact resume requires a saved MPS RNG state')
        torch.mps.set_rng_state(state['mps'])


def freeze_discriminator(source, layers):
    if layers == 0:
        return source, []
    kwargs = copy.deepcopy(source.init_kwargs)
    kwargs['block_kwargs'] = dict(kwargs.get('block_kwargs', {}), freeze_layers=layers)
    with torch.random.fork_rng(devices=[]):
        result = type(source)(*source.init_args, **kwargs)
    result.load_state_dict(source.state_dict(), strict=True)
    frozen = sorted(set(dict(source.named_parameters())) & set(dict(result.named_buffers())))
    return result, frozen


def snapshot(g, ema, fixed, run, step, device):
    state = rng_state(device)
    try:
        for name, model, psi in [('raw', g, 1.0), ('untruncated', ema, 1.0), ('ema', ema, .7)]:
            modes = {module: module.training for module in model.modules()}
            try:
                model.eval()
                with torch.no_grad():
                    images = torch.cat([model(z[None].to(device), None, truncation_psi=psi,
                                              noise_mode='const', force_fp32=True).cpu() for z in fixed])
                arrays = ((images+1)*127.5).clamp(0,255).byte().permute(0,2,3,1).numpy()
                grid = np.concatenate([np.concatenate(arrays[i:i+4], axis=1) for i in range(0,len(arrays),4)], axis=0)
                Image.fromarray(grid).save(run/f'samples-{step:06}-{name}.png')
            finally:
                for module, mode in modes.items():
                    module.training = mode
    finally:
        restore_rng(state, device)


def stats_state(collectors):
    # Pinned NVIDIA internal counters are needed to preserve the pending ADA
    # window across checkpoints that are not aligned to ada_interval.
    return {'counters': training_stats._counters, 'cumulative': training_stats._cumulative,
            'sync_called': training_stats._sync_called,
            'collectors': {key: {'cumulative': value._cumulative, 'moments': value._moments}
                           for key, value in collectors.items()}}


def restore_stats(state, collectors):
    training_stats._counters = state['counters']
    training_stats._cumulative = state['cumulative']
    training_stats._sync_called = state['sync_called']
    for key, value in collectors.items():
        value._cumulative = state['collectors'][key]['cumulative']
        value._moments = state['collectors'][key]['moments']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','mps'], default='mps')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--data', type=Path, default=ROOT/'alignment/collar-only/eyes42')
    parser.add_argument('--base', type=Path, default=ROOT/'models/ffhq256.pkl')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=6000, help='Total updates including prior completed updates')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--pl-batch-shrink', type=int, default=2)
    parser.add_argument('--freeze-d-layers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.0025)
    parser.add_argument('--r1-gamma', type=float, default=1)
    parser.add_argument('--ema-kimg', type=float, default=20)
    parser.add_argument('--augment-p', type=float, default=0)
    parser.add_argument('--ada-target', type=float, default=.6)
    parser.add_argument('--ada-kimg', type=float, default=100)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--no-snapshots', action='store_true', help='Bounded plumbing tests only')
    parser.add_argument('--verify-updates', action='store_true', help='Check each gradient/weight and actual parameter update')
    args = parser.parse_args()
    if args.batch < 1 or args.microbatch < 1 or args.batch % args.microbatch:
        parser.error('batch must be a positive multiple of microbatch')
    if args.pl_batch_shrink < 1 or args.microbatch < args.pl_batch_shrink:
        parser.error('microbatch must be at least pl-batch-shrink')
    if args.steps < 1 or args.freeze_d_layers < 0 or args.checkpoint_every < 1 or args.snapshot_every < 1:
        parser.error('invalid step, interval, or freeze setting')
    if args.lr <= 0 or args.ema_kimg <= 0 or args.ada_kimg <= 0 or args.r1_gamma < 0 or args.augment_p < 0:
        parser.error('invalid optimizer/regularization/augmentation setting')
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run/'resume.pt').exists() and args.resume is None:
        parser.error('run already has a checkpoint; pass --resume explicitly or choose a new run')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    conv2d_gradfix.enabled = True; grid_sample_gradfix.enabled = True
    # Always use CPU reporting counters, including on CPU, so checkpoint keys and
    # pending ADA windows have one exact representation independent of accelerator.
    upstream_report = training_stats.report
    def report_cpu(name, value):
        upstream_report(name, value.detach().cpu() if torch.is_tensor(value) else value)
        return value
    training_stats.report = report_cpu
    source_hashes = {name: hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()
                     for name in ['training/loss.py','training/augment.py','training/training_loop.py',
                                  'training/dataset.py','torch_utils/misc.py','torch_utils/training_stats.py']}
    dataset_files = sorted(args.data.rglob('*.png'))
    if not dataset_files:
        parser.error('dataset directory contains no PNG images')
    data_digest = hashlib.sha256()
    for path in dataset_files:
        data_digest.update(str(path.relative_to(args.data)).encode())
        data_digest.update(path.read_bytes())
    with args.base.open('rb') as handle:
        nets = legacy.load_network_pkl(handle)
    g = nets['G'].to(args.device).train().requires_grad_(False)
    d, frozen_names = freeze_discriminator(nets['D'], args.freeze_d_layers)
    d = d.to(args.device).train().requires_grad_(False)
    ema = nets['G_ema'].to(args.device).eval().requires_grad_(False)
    del nets
    data = ImageFolderDataset(path=str(args.data), resolution=g.img_resolution, use_labels=False, xflip=args.mirror)
    if data.num_channels != g.img_channels or g.c_dim != 0 or d.c_dim != 0:
        raise ValueError('This prototype requires an unconditional RGB-compatible source and dataset')
    sampler = iter(PortableInfiniteSampler(dataset=data, seed=args.seed))
    augment = AugmentPipe(**BGC).to(args.device).train().requires_grad_(False)
    augment.p.copy_(torch.tensor(args.augment_p))
    loss = StyleGAN2Loss(device=torch.device(args.device), G_mapping=g.mapping, G_synthesis=FP32Call(g.synthesis),
                        D=FP32Call(d), augment_pipe=augment, r1_gamma=args.r1_gamma,
                        pl_batch_shrink=args.pl_batch_shrink, pl_weight=2, pl_decay=.01, style_mixing_prob=.9)
    phases = []
    optimizers = {}
    for name, module, interval in [('G',g,4),('D',d,16)]:
        ratio = interval/(interval+1)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr*ratio, betas=(0.0,.99**ratio), eps=1e-8)
        optimizers[name] = optimizer
        phases += [(name+'main',module,optimizer,1),(name+'reg',module,optimizer,interval)]
    collectors = {'ada': training_stats.Collector(regex='Loss/signs/real'),
                  'metrics': training_stats.Collector(regex='Loss/.*')}
    recipe = {key: value for key,value in vars(args).items() if key not in
              ['run','resume','steps','checkpoint_every','snapshot_every','no_snapshots','verify_updates','base','data','threads']}
    recipe.update({'base_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
                   'dataset_sha256': data_digest.hexdigest(), 'dataset_count':len(data),
                   'source_hashes':source_hashes, 'resolution':g.img_resolution,
                   'g_reg_interval':4,'d_reg_interval':16,'ada_interval':4,'pl_weight':2,'pl_decay':.01,
                   'style_mixing_probability':.9,'augmentation':BGC,'ema_rampup':None,
                   'accumulation_gain':'phase interval per microbatch, without round normalization'})
    config = {'recipe':recipe,'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
              'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'frozen_d_tensor_names':frozen_names,'torch_version':torch.__version__,
              'architecture':{'G':str(g.init_kwargs),'D':str(d.init_kwargs)},
              'sampler':'upstream InfiniteSampler; exact seed + consumed count replay',
              'deviations':['single device','FP32 network calls','detached CPU statistics','CPU-staged synchronous image reads','Sampler constructor compatibility shim; unchanged iterator',
                            'batch32/microbatch4 default instead of paper256 batch64','loaded source architecture',
                            'full durable training state checkpoint rather than network-only transfer pickle']}
    batch_idx = 0; images_seen = 0
    fixed = torch.from_numpy(np.random.RandomState(20260920).randn(16,g.z_dim).astype(np.float32))
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint['recipe'] != recipe:
            differences = [key for key in set(checkpoint['recipe'])|set(recipe) if checkpoint['recipe'].get(key)!=recipe.get(key)]
            raise ValueError('Resume recipe mismatch: '+', '.join(sorted(differences)))
        g.load_state_dict(checkpoint['G']); d.load_state_dict(checkpoint['D']); ema.load_state_dict(checkpoint['G_ema'])
        optimizers['G'].load_state_dict(checkpoint['G_opt']); optimizers['D'].load_state_dict(checkpoint['D_opt'])
        loss.pl_mean.copy_(checkpoint['pl_mean']); augment.load_state_dict(checkpoint['augment'])
        batch_idx = checkpoint['batch_idx']; images_seen = checkpoint['images_seen']; fixed = checkpoint['fixed_z']
        if checkpoint['sampler']['consumed'] != images_seen or checkpoint['sampler']['seed'] != args.seed:
            raise ValueError('Sampler checkpoint count/seed mismatch')
        sampler = itertools.islice(sampler,images_seen,None)
        restore_stats(checkpoint['training_stats'],collectors)
        restore_rng(checkpoint['rng'],args.device)
        del checkpoint
    if args.steps <= batch_idx:
        raise ValueError('Requested total steps must exceed checkpoint step')
    (args.run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    frozen_before = {name:dict(d.named_buffers())[name].detach().cpu().clone() for name in frozen_names}
    if not args.no_snapshots:
        snapshot(g,ema,fixed,args.run,batch_idx,args.device)
    started = time.monotonic()
    while batch_idx < args.steps:
        examples = [data[next(sampler)] for _ in range(args.batch)]
        real_images = torch.from_numpy(np.stack([item[0] for item in examples]))
        real_labels = torch.from_numpy(np.stack([item[1] for item in examples]))
        # Match the reference's draws for all four phases, including unused reg phases.
        all_z = torch.randn(len(phases)*args.batch,g.z_dim,device=args.device).split(args.batch)
        all_c = torch.from_numpy(np.stack([data.get_label(np.random.randint(len(data)))
                                          for _ in range(len(phases)*args.batch)])).to(args.device).split(args.batch)
        phase_records = []
        for (name,module,optimizer,interval), phase_z, phase_c in zip(phases,all_z,all_c):
            if batch_idx % interval != 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            module.requires_grad_(True)
            before = next(module.parameters()).detach().cpu().clone() if args.verify_updates else None
            phase_start = time.monotonic()
            for start in range(0,args.batch,args.microbatch):
                end = start+args.microbatch
                real = real_images[start:end].to(args.device).float()/127.5-1
                real_c = real_labels[start:end].to(args.device)
                # NVIDIA does NOT divide this gain by accumulation rounds.
                loss.accumulate_gradients(name,real,real_c,phase_z[start:end],phase_c[start:end],
                                          sync=end==args.batch,gain=interval)
            module.requires_grad_(False)
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if args.verify_updates:
                assert gradients and all(torch.isfinite(value).all() for value in gradients), name+' nonfinite gradients'
            # Exact reference guard; verify mode asserts before sanitizing so
            # validation cannot hide bad gradients.
            for grad in gradients:
                misc.nan_to_num(grad,nan=0,posinf=1e5,neginf=-1e5,out=grad)
            optimizer.step()
            if args.verify_updates:
                assert all(torch.isfinite(value).all() for value in module.parameters()), name+' nonfinite weights'
                assert not torch.equal(before,next(module.parameters()).detach().cpu()), name+' parameter did not update'
            phase_records.append({'phase':name,'gain_per_microbatch':interval,'rounds':args.batch//args.microbatch,
                                  'seconds':time.monotonic()-phase_start})
        with torch.no_grad():
            beta = .5**(args.batch/(args.ema_kimg*1000))
            for target,source in zip(ema.parameters(),g.parameters()):
                target.copy_(source.lerp(target,beta))
            for target,source in zip(ema.buffers(),g.buffers()):
                target.copy_(source)
        batch_idx += 1; images_seen += args.batch
        if batch_idx % 4 == 0:
            collectors['ada'].update()
            adjust = np.sign(collectors['ada']['Loss/signs/real']-args.ada_target)*(args.batch*4)/(args.ada_kimg*1000)
            augment.p.copy_((augment.p+adjust).max(misc.constant(0,device=args.device)))
        collectors['metrics'].update()
        record = {'step':batch_idx,'images_seen':images_seen,'kimg':images_seen/1000,
                  'seconds':time.monotonic()-started,'augment_p':float(augment.p),'pl_mean':float(loss.pl_mean),
                  'phases':phase_records,'statistics':{key:value.mean for key,value in collectors['metrics'].as_dict().items()}}
        with (args.run/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not args.no_snapshots and (batch_idx%args.snapshot_every==0 or batch_idx==args.steps):
            snapshot(g,ema,fixed,args.run,batch_idx,args.device)
        if batch_idx%args.checkpoint_every==0 or batch_idx==args.steps:
            assert all(torch.equal(value,dict(d.named_buffers())[name].detach().cpu()) for name,value in frozen_before.items())
            state = {'format_version':1,'recipe':recipe,'config':config,'G':g.state_dict(),'D':d.state_dict(),
                     'G_ema':ema.state_dict(),'G_opt':optimizers['G'].state_dict(),'D_opt':optimizers['D'].state_dict(),
                     'pl_mean':loss.pl_mean,'augment':augment.state_dict(),'batch_idx':batch_idx,'images_seen':images_seen,
                     'rng':rng_state(args.device),'sampler':{'seed':args.seed,'consumed':images_seen},
                     'training_stats':stats_state(collectors),'fixed_z':fixed}
            atomic_save(state,args.run/'resume.pt')
            save_generator_snapshot(ema.state_dict(),batch_idx,images_seen,config,recipe,args.run)
    (args.run/'completed.json').write_text(json.dumps({'step':batch_idx,'images_seen':images_seen,
                'finite_update_checks':args.verify_updates,'frozen_d_unchanged':True},indent=2)+'\n')


if __name__ == '__main__':
    main()
