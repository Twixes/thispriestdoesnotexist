"""Actual 256px preview check: model state, flags and RNG must not change."""
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

import trainer
from check_trainer_resume import equal

OUT = Path(__file__).resolve().parent/'prototype-tests/snapshots'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(19)
with (trainer.ROOT/'models/ffhq256.pkl').open('rb') as handle:
    nets = trainer.legacy.load_network_pkl(handle)
g = nets['G'].train().requires_grad_(False)
ema = nets['G_ema'].eval().requires_grad_(False)
del nets


def state_hash(module):
    digest = hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


before = [state_hash(g),state_hash(ema)]
modes = [[module.training for module in model.modules()] for model in [g,ema]]
fixed = torch.randn(4,g.z_dim)
rng_before = trainer.rng_state('cpu')
trainer.snapshot(g,ema,fixed,OUT,0,'cpu')
assert before == [state_hash(g),state_hash(ema)]
assert modes == [[module.training for module in model.modules()] for model in [g,ema]]
equal(rng_before,trainer.rng_state('cpu'),'rng',{'tensors':0,'elements':0})
files = sorted(OUT.glob('samples-*.png'))
assert len(files)==3
for path in files:
    with Image.open(path) as image:
        assert image.size == (1024,256) and image.mode=='RGB'
result = {'passed':True,'device':'cpu','resolution':256,'model_states_unchanged':True,
          'all_training_flags_unchanged':True,'all_rng_states_unchanged':True,
          'grid_sizes':[1024,256],'grid_files':[path.name for path in files],
          'before_and_after_state_hashes':before,
          'trainer_sha256':hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest()}
(OUT/'snapshot-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
