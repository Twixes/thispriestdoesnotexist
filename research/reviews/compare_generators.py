"""Bounded CPU-only comparison of instantaneous and EMA transfer generators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/stylegan2-ada-pytorch'))
import legacy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    # One open file descriptor stays on the same inode across trainer's atomic
    # replacement. Read once, hash once, deserialize the same immutable bytes.
    import io
    snapshot_bytes=args.resume.read_bytes()
    # This is our own trainer's trusted resume checkpoint, which includes NumPy
    # augmentation scalars and optimizer states. This diagnostic is offline only.
    snapshot=torch.load(io.BytesIO(snapshot_bytes),map_location='cpu',weights_only=False)
    source_hash=hashlib.sha256(snapshot_bytes).hexdigest()
    del snapshot_bytes
    with (ROOT/'models/ffhq256.pkl').open('rb') as source:
        nets=legacy.load_network_pkl(source)
    seed=77142026
    codes=torch.from_numpy(np.random.default_rng(seed).standard_normal((8,512), dtype=np.float32))
    results={}
    grids=[]
    for name,state in [('base-G',nets['G'].state_dict()),('base-EMA',nets['G_ema'].state_dict()),
                       ('trained-G',snapshot['G']),('trained-EMA',snapshot['G_ema'])]:
        generator=nets['G'].cpu().eval().requires_grad_(False)
        generator.load_state_dict(state)
        start=time.monotonic()
        with torch.inference_mode():
            values=torch.cat([generator(code[None],None,truncation_psi=.7,noise_mode='const',force_fp32=True) for code in codes])
        images=((values+1)*127.5).clamp(0,255).byte().permute(0,2,3,1).numpy()
        grid=np.concatenate([np.concatenate(images[i:i+4],axis=1) for i in range(0,8,4)],axis=0)
        Image.fromarray(grid).save(args.output/f'{name}.png')
        grids.append(grid)
        results[name]={'seconds':time.monotonic()-start,'finite':bool(torch.isfinite(values).all()),
                       'pixel_mean':float(values.mean()),'pixel_std':float(values.std())}
    Image.fromarray(np.concatenate(grids,axis=0)).save(args.output/'comparison.png')
    deltas=[]
    for key,value in snapshot['G'].items():
        if value.is_floating_point():
            difference=(value-snapshot['G_ema'][key]).float()
            deltas.append({'tensor':key,'rms':float(difference.square().mean().sqrt()),
                           'relative_rms':float(difference.square().mean().sqrt()/(value.float().square().mean().sqrt()+1e-12))})
    deltas.sort(key=lambda row:row['relative_rms'],reverse=True)
    report={'step':snapshot['step'],'resume_sha256':source_hash,'seed':seed,'threads':2,
            'grid_order':['base-G','base-EMA','trained-G','trained-EMA'],
            'results':results,'G_vs_EMA_largest_relative_rms':deltas[:20]}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
