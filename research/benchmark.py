"""Measure the actual pretrained StyleGAN2 on CPU and Apple Metal; no training claims."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/stylegan2-ada-pytorch'))
import legacy


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps'], required=True)
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260920)
    model_path = ROOT / 'models/ffhq256.pkl'
    with model_path.open('rb') as f:
        nets = legacy.load_network_pkl(f)
    g = nets['G_ema'].to(args.device).eval().requires_grad_(False)
    d = nets['D'].to(args.device).eval().requires_grad_(False)
    print(f'Loaded {g.img_resolution}px model on {args.device}', flush=True)
    report = {'device': args.device, 'torch': torch.__version__, 'platform': platform.platform(),
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'inference_seconds': []}
    for i in range(4):
        sync(args.device)
        start = time.perf_counter()
        with torch.no_grad():
            x = g(torch.randn(1, g.z_dim, device=args.device), None, truncation_psi=.7, noise_mode='const', force_fp32=True)
        sync(args.device)
        elapsed = time.perf_counter() - start
        report['inference_seconds'].append(elapsed)
        print(f'Inference {i}: {elapsed:.3f}s', flush=True)
        if i == 3:
            image = ((x[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(ROOT / f'runs/baseline-{args.device}.png')
    if args.training:
        g.train().requires_grad_(True)
        d.train().requires_grad_(True)
        report['training_step_seconds'] = []
        opt_g = torch.optim.Adam(g.parameters(), lr=.002, betas=(0.0, .99))
        opt_d = torch.optim.Adam(d.parameters(), lr=.002, betas=(0.0, .99))
        real = torch.from_numpy(np.asarray(Image.open(ROOT.parent / 'public/portraits/51.webp').convert('RGB').resize((256,256))).copy()).permute(2,0,1).unsqueeze(0).to(args.device).float() / 127.5 - 1
        for i in range(3):
            sync(args.device)
            start = time.perf_counter()
            d.requires_grad_(False)
            fake = g(torch.randn(1, g.z_dim, device=args.device), None, force_fp32=True)
            loss_g = torch.nn.functional.softplus(-d(fake, None, force_fp32=True)).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            d.requires_grad_(True)
            loss_d = torch.nn.functional.softplus(d(fake.detach(), None, force_fp32=True)).mean() + torch.nn.functional.softplus(-d(real, None, force_fp32=True)).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            sync(args.device)
            elapsed = time.perf_counter() - start
            report['training_step_seconds'].append(elapsed)
            print(f'Train step {i}: {elapsed:.3f}s', flush=True)
        # R1 needs second derivatives, a frequent portability failure on non-CUDA backends.
        real.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits = d(real, None, force_fp32=True)
        grad = torch.autograd.grad(logits.sum(), real, create_graph=True)[0]
        penalty = grad.square().sum((1,2,3)).mean()
        penalty.backward()
        sync(args.device)
        report['r1_seconds'] = time.perf_counter() - start
        report['r1_finite'] = bool(torch.isfinite(penalty))
        print('R1 passed', flush=True)
    (ROOT / f'runs/benchmark-{args.device}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
