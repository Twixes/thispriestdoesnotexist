"""Portable StyleGAN2 transfer learning with DiffAugment and ADA probability control.

Network implementations remain NVIDIA's unmodified reference. This trainer uses
standard PyTorch ops on MPS/CPU, logistic GAN losses, lazy R1, EMA, and style mixing.
It is an experiment, not an assertion of acceptable model quality.
"""
import argparse
import copy
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


def save_grid(g, codes, path):
    with torch.no_grad():
        images = torch.cat([g(z[None], None, truncation_psi=.7, noise_mode='const', force_fp32=True).cpu() for z in codes])
    images = ((images + 1) * 127.5).clamp(0,255).byte().permute(0,2,3,1).numpy()
    grid = np.concatenate([np.concatenate(images[i:i+4], axis=1) for i in range(0,len(images),4)], axis=0)
    Image.fromarray(grid).save(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device', default='mps')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--snapshot-every', type=int, default=250)
    p.add_argument('--run', default='pilot256')
    p.add_argument('--resume', type=Path)
    args=p.parse_args()
    run=ROOT/'runs'/args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(20260920)
    np.random.seed(20260920)
    random.seed(20260920)
    with (ROOT/'models/ffhq256.pkl').open('rb') as f:
        nets=legacy.load_network_pkl(f)
    g=nets['G'].to(args.device).train().requires_grad_(True)
    d=nets['D'].to(args.device).train().requires_grad_(True)
    ema=copy.deepcopy(nets['G_ema']).to(args.device).eval().requires_grad_(False)
    # The upstream model can request FP16 at high resolutions. Metal and CPU use FP32.
    g_opt=torch.optim.Adam([{'params': g.mapping.parameters(), 'lr': .0001},
                           {'params':g.synthesis.parameters(), 'lr':.001}],betas=(0.0,.99))
    d_opt=torch.optim.Adam(d.parameters(), lr=.001*16/17, betas=(0.0,.99**(16/17)))
    fixed=torch.randn(16, g.z_dim, device=args.device)
    data=torch.stack([torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1)
                      for path in sorted((ROOT/'data/priests256').glob('*.png'))]).to(args.device).float()/127.5-1
    aug_p=.5
    start_step=0
    if args.resume:
        checkpoint=torch.load(args.resume, map_location='cpu', weights_only=False)
        g.load_state_dict(checkpoint['G']);d.load_state_dict(checkpoint['D']);ema.load_state_dict(checkpoint['G_ema'])
        g_opt.load_state_dict(checkpoint['G_opt']); d_opt.load_state_dict(checkpoint['D_opt'])
        aug_p=checkpoint['aug_p'];start_step=checkpoint['step']
        torch.set_rng_state(checkpoint['torch_rng'])
        if args.device=='mps':torch.mps.set_rng_state(checkpoint['device_rng'])
    config=vars(args).copy();config['resume']=str(args.resume) if args.resume else None
    config.update({'dataset_count':len(data), 'seed':20260920,'r1_gamma':2.0,'r1_interval':16,
                   'g_lr':.001,'mapping_lr':.0001,'d_lr':.001,'ema_kimg':1,'ada_target':.6,'ada_kimg':100,
                   'augment':'DiffAugment color,translation,cutout plus horizontal flip',
                   'path_length_regularization':False,'style_mixing_probability':.9})
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    save_grid(ema,fixed,run/f'samples-{start_step:06}.png')
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
        regularize=step%16==0
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
            loss_d=loss_d+r1*16 # gamma / 2 * interval
        loss_d.backward();d_opt.step()
        if not torch.isfinite(loss_g+loss_d):raise RuntimeError(f'Non-finite loss at step {step}')
        signs.append(float(real_score.detach().sign().mean()))
        if step%4==0:
            aug_p=max(0.0,min(.85,aug_p+np.sign(np.mean(signs)-.6)*args.batch*4/100000))
            signs=[]
        with torch.no_grad():
            beta=.5**(args.batch/1000)
            for e,v in zip(ema.parameters(),g.parameters()): e.lerp_(v,1-beta)
            for e,v in zip(ema.buffers(),g.buffers()): e.copy_(v)
        if step%10==0:
            record={'step':step,'kimg':step*args.batch/1000,'seconds':time.monotonic()-started,
                    'g_loss':float(loss_g),'d_loss':float(loss_d),'r1':float(r1),'augment_p':float(aug_p)}
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%args.snapshot_every==0 or step==args.steps:
            save_grid(ema,fixed,run/f'samples-{step:06}.png')
            torch.save({'G_ema':cpu_state(ema),'step':step,'config':config},run/f'generator-{step:06}.pt')
            checkpoint={'G':cpu_state(g),'D':cpu_state(d),'G_ema':cpu_state(ema),
                        'G_opt':g_opt.state_dict(),'D_opt':d_opt.state_dict(),'step':step,'aug_p':aug_p,
                        'torch_rng':torch.get_rng_state(),
                        'device_rng':torch.mps.get_rng_state() if args.device=='mps' else torch.get_rng_state()}
            torch.save(checkpoint,run/'resume.tmp.pt')
            (run/'resume.tmp.pt').replace(run/'resume.pt')
    print('Training completed',flush=True)

if __name__=='__main__':main()
