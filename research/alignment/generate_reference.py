"""Generate a small deterministic unadapted-FFHQ geometry reference on CPU."""
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

torch.set_num_threads(2)
output=ROOT/'alignment/reference'
output.mkdir(exist_ok=True)
model_path=ROOT/'models/ffhq256.pkl'
with model_path.open('rb') as source:
    model=legacy.load_network_pkl(source)['G_ema'].cpu().eval().requires_grad_(False)
seed=98142026
codes=torch.from_numpy(np.random.default_rng(seed).standard_normal((64,512),dtype=np.float32))
entries=[]
with torch.inference_mode():
    for index,code in enumerate(codes):
        values=model(code[None],None,truncation_psi=.7,noise_mode='const',force_fp32=True)
        pixels=((values[0].permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
        target=output/f'{index:03}.png'
        Image.fromarray(pixels).save(target)
        entries.append({'path':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(output/'manifest.json').write_text(json.dumps({'seed':seed,'count':64,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
    'purpose':'Unadapted FFHQ geometry comparison only, not priest training data or production outputs','entries':entries},indent=2)+'\n')
print('Generated 64 CPU reference samples')
