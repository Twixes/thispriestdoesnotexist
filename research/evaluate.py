"""Unseen seed grids and LPIPS nearest-neighbor diagnostics; never a beauty classifier.

Visual review is still required. Perceptual distance cannot prove different identity,
quality, priest attire, age, or attractiveness. It helps expose duplicates/collapse.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lpips
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/stylegan2-ada-pytorch'))
import legacy


def image_tensor(path):
    image=Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1


def pil(tensor):
    array=((tensor.detach().cpu().permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
    return Image.fromarray(array)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--seed-start',type=int,default=10000)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--data',type=Path,default=ROOT/'data/priests256')
    args=p.parse_args()
    args.checkpoint=args.checkpoint.resolve()
    args.data=args.data.resolve()
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(ROOT/'models/metrics'))
    out=args.checkpoint.parent/f'eval-{args.checkpoint.stem}'
    out.mkdir(parents=True,exist_ok=True)
    with (ROOT/'models/ffhq256.pkl').open('rb') as f:g=legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    g.load_state_dict(checkpoint['G_ema'])
    targets=sorted(args.data.glob('*.png'))
    reals=torch.stack([image_tensor(path) for path in targets])
    model=lpips.LPIPS(net='squeeze').eval().requires_grad_(False)
    generated=[];records=[]
    started=time.monotonic()
    with torch.inference_mode():
        for seed in range(args.seed_start,args.seed_start+args.count):
            rng=torch.Generator().manual_seed(seed)
            z=torch.randn(1,g.z_dim,generator=rng)
            x=g(z,None,truncation_psi=.7,noise_mode='const',force_fp32=True).cpu()
            generated.append(x[0])
            image=pil(x[0]);image.save(out/f'seed-{seed}.png')
            # Compare both orientations because training uses random horizontal flips.
            distances=[]
            for start in range(0,len(reals),8):
                batch=reals[start:start+8]
                distances.append(torch.minimum(model(x.expand(len(batch),-1,-1,-1),batch).flatten(),
                                                 model(x.expand(len(batch),-1,-1,-1),batch.flip(3)).flatten()))
            distances=torch.cat(distances)
            nearest=int(distances.argmin())
            records.append({'seed':seed,'nearest_training_file':str(targets[nearest].relative_to(ROOT.parent)),
                            'lpips':float(distances[nearest]),'decoded_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest()})
            if len(records)%8==0:print(f'Evaluated {len(records)}/{args.count}',flush=True)
        all_images=torch.stack(generated)
        pair_distance=model(all_images,all_images.roll(1,0)).flatten().tolist()
    grid=Image.new('RGB',(8*256,math_rows(args.count,8)*280),'#111111')
    draw=ImageDraw.Draw(grid)
    nearest_grid=Image.new('RGB',(4*512,math_rows(args.count,4)*280),'#111111')
    near_draw=ImageDraw.Draw(nearest_grid)
    for index,(tensor,record) in enumerate(zip(generated,records)):
        gx=(index%8)*256;gy=(index//8)*280
        grid.paste(pil(tensor),(gx,gy));draw.text((gx+5,gy+259),str(record['seed']),fill='white')
        nx=(index%4)*512;ny=(index//4)*280
        nearest_grid.paste(pil(tensor),(nx,ny))
        nearest_grid.paste(Image.open(ROOT.parent/record['nearest_training_file']).convert('RGB').resize((256,256)),(nx+256,ny))
        near_draw.text((nx+5,ny+259),f"seed {record['seed']} / nearest {Path(record['nearest_training_file']).stem} / LPIPS {record['lpips']:.3f}",fill='white')
    grid.save(out/'unseen-seeds.png');nearest_grid.save(out/'nearest-training-images.png')
    report={'checkpoint':str(args.checkpoint.relative_to(ROOT.parent)), 'sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'count':args.count,'training_count':len(reals),'unique_decoded_images':len({r['decoded_sha256'] for r in records}),
            'nearest_lpips_min':min(r['lpips'] for r in records),'nearest_lpips_median':float(np.median([r['lpips'] for r in records])),
            'generated_pair_lpips_median':float(np.median(pair_distance)),'generated_pair_lpips_min':min(pair_distance),
            'seconds':time.monotonic()-started,'metric':'LPIPS v0.1 SqueezeNet; orientation invariant nearest training comparison',
            'limitations':'Diagnostics only. These scores do not prove new identity or visual acceptability. Human visual inspection required.',
            'records':records,'generated_pair_distances':pair_distance}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','generated_pair_distances']},indent=2))


def math_rows(count,columns):return (count+columns-1)//columns

if __name__=='__main__':main()
