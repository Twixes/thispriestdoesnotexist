"""Portable StyleGAN2 transfer learning with DiffAugment and ADA probability control.

Network implementations remain NVIDIA's unmodified reference. This trainer uses
standard PyTorch ops on MPS/CPU, logistic GAN losses, lazy R1, EMA, and style mixing.
It is an experiment, not an assertion of acceptable model quality.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'vendor/stylegan2-ada-pytorch'), str(ROOT / 'vendor/diffaugment')]
import legacy
from DiffAugment_pytorch import DiffAugment


def cpu_state(module):
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def transfer_discriminator(source, freeze_layers):
    """Use NVIDIA's buffer-based FreezeD so training toggles cannot unfreeze it."""
    if freeze_layers == 0:
        return source, []
    kwargs = copy.deepcopy(source.init_kwargs)
    kwargs['block_kwargs'] = dict(kwargs.get('block_kwargs', {}), freeze_layers=freeze_layers)
    # Initialization is overwritten below; do not change the experiment's RNG.
    with torch.random.fork_rng(devices=[]):
        discriminator = type(source)(*source.init_args, **kwargs)
    discriminator.load_state_dict(source.state_dict(), strict=True)
    frozen_names = sorted(set(dict(source.named_parameters())) & set(dict(discriminator.named_buffers())))
    return discriminator, frozen_names


def save_grid(g, codes, path, psi=.7):
    # Raw G is normally training: eval prevents mapping.w_avg updates. Preserve
    # RNG explicitly so snapshots cannot alter the subsequent optimization path.
    training_modes={module:module.training for module in g.modules()}
    uses_mps=next(g.parameters()).device.type=='mps'
    mps_rng=torch.mps.get_rng_state() if uses_mps else None
    try:
        g.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            images = torch.cat([g(z[None], None, truncation_psi=psi, noise_mode='const', force_fp32=True).cpu() for z in codes])
    finally:
        for module,training in training_modes.items(): module.training=training
        if uses_mps: torch.mps.set_rng_state(mps_rng)
    images = ((images + 1) * 127.5).clamp(0,255).byte().permute(0,2,3,1).numpy()
    grid = np.concatenate([np.concatenate(images[i:i+4], axis=1) for i in range(0,len(images),4)], axis=0)
    Image.fromarray(grid).save(path)


def save_comparison_grids(g, ema, codes, run, step):
    save_grid(ema,codes,run/f'samples-{step:06}.png')
    save_grid(g,codes,run/f'samples-{step:06}-raw.png',psi=1.0)
    save_grid(ema,codes,run/f'samples-{step:06}-untruncated.png',psi=1.0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device', default='mps')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--smoke-test', action='store_true', help='One real update including R1, without saving large checkpoints')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--snapshot-every', type=int, default=250)
    p.add_argument('--run', default='pilot256')
    p.add_argument('--resume', type=Path)
    p.add_argument('--base', type=Path, default=ROOT/'models/ffhq256.pkl')
    p.add_argument('--data', type=Path, default=ROOT/'data/priests256')
    p.add_argument('--mapping-lr', type=float, default=.0005)
    p.add_argument('--synthesis-lr', type=float, default=.001)
    p.add_argument('--d-lr', type=float, default=.001)
    p.add_argument('--ema-kimg', type=float, default=.5)
    p.add_argument('--ada-kimg', type=float, default=100)
    p.add_argument('--augment-p', type=float, default=.5)
    p.add_argument('--r1-gamma', type=float, default=2)
    p.add_argument('--freeze-d-layers', type=int, default=0,
                   help='Freeze input-side discriminator layers using NVIDIA FreezeD buffers')

    args=p.parse_args()
    if args.freeze_d_layers < 0:
        p.error('--freeze-d-layers must be non-negative')
    run=ROOT/'runs'/args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260920)
    np.random.seed(20260920)
    random.seed(20260920)
    with args.base.open('rb') as f:
        nets=legacy.load_network_pkl(f)
    g=nets['G'].to(args.device).train().requires_grad_(True)
    d,frozen_d_names=transfer_discriminator(nets['D'],args.freeze_d_layers)
    d=d.to(args.device).train().requires_grad_(True)
    ema=copy.deepcopy(nets['G_ema']).to(args.device).eval().requires_grad_(False)
    # The upstream model can request FP16 at high resolutions. Metal and CPU use FP32.
    g_opt=torch.optim.Adam([{'params': g.mapping.parameters(), 'lr': args.mapping_lr},
                           {'params':g.synthesis.parameters(), 'lr':args.synthesis_lr}],betas=(0.0,.99))
    d_opt=torch.optim.Adam(d.parameters(), lr=args.d_lr, betas=(0.0,.99))
    fixed=torch.randn(16, g.z_dim, device=args.device)
    data=torch.stack([torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1)
                      for path in sorted(args.data.glob('*.png'))]).to(args.device).float()/127.5-1
    aug_p=args.augment_p
    start_step=0
    prior_images_seen=0
    if args.resume:
        checkpoint=torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_freeze=checkpoint.get('freeze_d_layers',0)
        if saved_freeze != args.freeze_d_layers:
            raise ValueError(f'Resume FreezeD mismatch: checkpoint freezes {saved_freeze} layers, '
                             f'but --freeze-d-layers={args.freeze_d_layers}; use matching settings '
                             'to preserve discriminator optimizer state, or start a fresh run')
        g.load_state_dict(checkpoint['G']);d.load_state_dict(checkpoint['D']);ema.load_state_dict(checkpoint['G_ema'])
        g_opt.load_state_dict(checkpoint['G_opt']); d_opt.load_state_dict(checkpoint['D_opt'])
        aug_p=checkpoint['aug_p'];start_step=checkpoint['step']
        if 'images_seen' in checkpoint:
            prior_images_seen=checkpoint['images_seen']
        else:
            old_config=json.loads((args.resume.parent/'config.json').read_text())
            prior_images_seen=start_step*old_config['batch']
        torch.set_rng_state(checkpoint['torch_rng'])
        if args.device=='mps':torch.mps.set_rng_state(checkpoint['device_rng'])
        if 'python_rng' in checkpoint:
            random.setstate(checkpoint['python_rng'])
        else:
            # Exact replay of this trainer's two style-mixing decisions per step
            # for pilot checkpoints made before Python RNG state was saved.
            for _ in range(2*start_step):
                if random.random()<.9: random.randint(1,g.num_ws-1)
        if 'numpy_rng' in checkpoint:
            state=checkpoint['numpy_rng']
            np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
    # Optimizer load restores old hyperparameters; explicit CLI parameters govern
    # the continuation and must be applied AFTER loading its momentum state.
    g_opt.param_groups[0]['lr']=args.mapping_lr
    g_opt.param_groups[1]['lr']=args.synthesis_lr
    d_opt.param_groups[0]['lr']=args.d_lr
    d_opt.param_groups[0]['betas']=(0.0,.99)

    if args.smoke_test:
        frozen_before={name:dict(d.named_buffers())[name].detach().cpu().clone() for name in frozen_d_names}
        trainable_name,trainable_tensor=next(iter(d.named_parameters()))
        trainable_before=trainable_tensor.detach().cpu().clone()
        optimizer_ids={id(value) for group in d_opt.param_groups for value in group['params']}
        assert optimizer_ids=={id(value) for value in d.parameters()}, 'D optimizer parameter mismatch'
        assert not optimizer_ids & {id(value) for value in d.buffers()}, 'Frozen buffer entered optimizer'

    if args.smoke_test: args.steps=start_step+1
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config['base_sha256']=hashlib.sha256(args.base.read_bytes()).hexdigest()
    config['trainer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['frozen_d_tensor_names']=frozen_d_names
    config['dataset_sha256']=hashlib.sha256(''.join(hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.data.glob('*.png'))).encode()).hexdigest()
    config.update({'dataset_count':len(data),'resume_step':start_step,'resume_images_seen':prior_images_seen, 'seed':20260920,'r1_gamma':args.r1_gamma,'r1_interval':16,
                   'g_lr':args.synthesis_lr,'mapping_lr':args.mapping_lr,'d_lr':args.d_lr,'ema_kimg':args.ema_kimg,'ada_target':.6,'ada_kimg':args.ada_kimg,
                   'augment':'DiffAugment color,translation,cutout plus horizontal flip',
                   'path_length_regularization':False,'style_mixing_probability':.9})
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    save_comparison_grids(g,ema,fixed,run,start_step)
    def augment(x):
        flip=torch.rand(x.shape[0],1,1,1,device=args.device)<.5
        x=torch.where(flip,x.flip(3),x)
        augmented=DiffAugment(x,policy='color,translation,cutout')
        return torch.where(torch.rand(x.shape[0],1,1,1,device=args.device)<aug_p,augmented,x)
    def generate():
        ws=g.mapping(torch.randn(args.batch,g.z_dim,device=args.device),None)
        if random.random()<.9:
            cutoff=random.randint(1,g.num_ws-1)
            ws2=g.mapping(torch.randn(args.batch,g.z_dim,device=args.device),None,skip_w_avg_update=True)
            ws=torch.cat((ws[:,:cutoff],ws2[:,cutoff:]),dim=1)
        return g.synthesis(ws,force_fp32=True)
    started=time.monotonic();signs=[]
    for step in range(start_step+1,args.steps+1):
        # Generator adversarial update.
        d.requires_grad_(False);g.requires_grad_(True)
        g_opt.zero_grad(set_to_none=True)
        fake=generate()
        loss_g=F.softplus(-d(augment(fake),None,force_fp32=True)).mean()
        loss_g.backward();g_opt.step()
        # Discriminator logistic loss and lazy R1 regularization.
        g.requires_grad_(False);d.requires_grad_(True)
        real=data[torch.randint(len(data),(args.batch,),device=args.device)]
        regularize=step%16==0 or args.smoke_test
        real=real.detach().requires_grad_(regularize)
        d_opt.zero_grad(set_to_none=True)
        with torch.no_grad(): fake=generate()
        fake_score=d(augment(fake),None,force_fp32=True)
        real_score=d(augment(real),None,force_fp32=True)
        loss_d=F.softplus(fake_score).mean()+F.softplus(-real_score).mean()
        r1=torch.zeros((),device=args.device)
        if regularize:
            grad=torch.autograd.grad(real_score.sum(),real,create_graph=True)[0]
            r1=grad.square().sum((1,2,3)).mean()
            loss_d=loss_d+r1*(args.r1_gamma/2)*16
        loss_d.backward();d_opt.step()
        if not torch.isfinite(loss_g+loss_d):raise RuntimeError(f'Non-finite loss at step {step}')
        signs.append(float(real_score.detach().sign().mean()))
        if step%4==0:
            aug_p=max(0.0,min(.85,aug_p+np.sign(np.mean(signs)-.6)*args.batch*4/(args.ada_kimg*1000)))
            signs=[]
        with torch.no_grad():
            beta=.5**(args.batch/(args.ema_kimg*1000))
            for e,v in zip(ema.parameters(),g.parameters()): e.lerp_(v,1-beta)
            for e,v in zip(ema.buffers(),g.buffers()): e.copy_(v)
        if step%10==0 or args.smoke_test:
            record={'step':step,'kimg':(prior_images_seen+(step-start_step)*args.batch)/1000,'seconds':time.monotonic()-started,
                    'g_loss':float(loss_g.detach()),'d_loss':float(loss_d.detach()),'r1':float(r1.detach()),'augment_p':float(aug_p),
                    'real_sign_mean':float(real_score.detach().sign().mean()),'real_logit_mean':float(real_score.detach().mean()),'fake_logit_mean':float(fake_score.detach().mean())}
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if not args.smoke_test and (step%args.snapshot_every==0 or step==args.steps):
            save_comparison_grids(g,ema,fixed,run,step)
            torch.save({'G_ema':cpu_state(ema),'step':step,'config':config},run/f'generator-{step:06}.pt')
            checkpoint={'G':cpu_state(g),'D':cpu_state(d),'G_ema':cpu_state(ema),
                        'G_opt':g_opt.state_dict(),'D_opt':d_opt.state_dict(),'step':step,'aug_p':aug_p,
                        'freeze_d_layers':args.freeze_d_layers,
                        'images_seen':prior_images_seen+(step-start_step)*args.batch,
                        'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                        'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),
                        'device_rng':torch.mps.get_rng_state() if args.device=='mps' else torch.get_rng_state()}
            torch.save(checkpoint,run/'resume.tmp.pt')
            (run/'resume.tmp.pt').replace(run/'resume.pt')
    if args.smoke_test:
        assert all(torch.isfinite(value).all() for value in g.parameters()), 'Non-finite generator weights'
        assert all(torch.isfinite(value).all() for value in d.parameters()), 'Non-finite discriminator weights'
        frozen_after={name:dict(d.named_buffers())[name].detach().cpu() for name in frozen_d_names}
        assert all(torch.equal(value,frozen_after[name]) for name,value in frozen_before.items()), 'Frozen D tensor changed'
        assert not torch.equal(trainable_before,dict(d.named_parameters())[trainable_name].detach().cpu()), 'Trainable D weight did not update'
        evidence={'step':step,'resumed':bool(args.resume),'freeze_d_layers':args.freeze_d_layers,
                  'frozen_tensor_count':len(frozen_d_names),'frozen_tensors_bitwise_unchanged':True,
                  'trainable_tensor_updated':trainable_name,'optimizer_excludes_buffers':True,
                  'r1_finite':bool(torch.isfinite(r1)),'all_weights_finite':True,
                  'frozen_sha256':{name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in frozen_before.items()}}
        (run/'smoke-test.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('Smoke test: optimizer update and R1 succeeded; weights finite, trainable D updated, frozen D unchanged',flush=True)
    print('Training completed',flush=True)

if __name__=='__main__':main()
