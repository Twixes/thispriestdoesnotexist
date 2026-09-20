"""CPU-only check of fused versus unfused trained StyleGAN modulation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/stylegan2-ada-pytorch'))
import legacy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with (ROOT/'models/ffhq256.pkl').open('rb') as source:
        model=legacy.load_network_pkl(source)['G_ema'].cpu().eval().requires_grad_(False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['G_ema'])
    seed=88142026
    codes=torch.from_numpy(np.random.default_rng(seed).standard_normal((4,512),dtype=np.float32))
    with torch.inference_mode():
        styles=model.mapping(codes,None,truncation_psi=.7)
        fused=torch.cat([model.synthesis(ws[None],noise_mode='const',force_fp32=True,fused_modconv=True) for ws in styles])
        unfused=torch.cat([model.synthesis(ws[None],noise_mode='const',force_fp32=True,fused_modconv=False) for ws in styles])
    difference=(fused-unfused).abs()
    grids=[]
    for name,values in [('fused',fused),('unfused',unfused)]:
        images=((values+1)*127.5).clamp(0,255).byte().permute(0,2,3,1).numpy()
        grid=np.concatenate(images,axis=1)
        Image.fromarray(grid).save(args.output/f'{name}.png')
        grids.append(grid)
    Image.fromarray(np.concatenate(grids,axis=0)).save(args.output/'comparison.png')
    report={'step':checkpoint['step'],'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'device':'cpu','threads':2,'seed':seed,'same_styles':True,'max_absolute_difference':float(difference.max()),
            'mean_absolute_difference':float(difference.mean()),
            'max_difference_in_8bit_units':float(difference.max()*127.5),
            'finite':bool(torch.isfinite(fused).all() and torch.isfinite(unfused).all()),
            'allclose_at_1e_minus_4':bool(torch.allclose(fused,unfused,atol=1e-4,rtol=1e-4)),
            'limitation':'CPU forward equivalence only; does not prove MPS backward equivalence.'}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
