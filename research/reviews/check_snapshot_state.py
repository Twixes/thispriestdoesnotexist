"""Bounded CPU snapshot side-effect regression with the actual pretrained G."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import legacy, save_grid

torch.set_num_threads(2)
torch.manual_seed(404)
with (ROOT / 'models/ffhq256.pkl').open('rb') as source:
    model = legacy.load_network_pkl(source)['G'].cpu().train()
codes = torch.randn(4, model.z_dim)

def state_hash(net):
    digest = hashlib.sha256()
    for name, value in net.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()

before = state_hash(model)
buffers_before = {name: value.clone() for name, value in model.named_buffers()}
rng_before = torch.get_rng_state().clone()
modes_before = [module.training for module in model.modules()]
output = ROOT / 'runs/frozen-d-smoke'
save_grid(model, codes, output / 'snapshot-state-check.png', psi=1.0)
assert before == state_hash(model), 'Snapshot changed model state'
assert torch.equal(rng_before, torch.get_rng_state()), 'Snapshot changed CPU RNG'
assert modes_before == [module.training for module in model.modules()], 'Training modes changed'
assert all(torch.equal(value, dict(model.named_buffers())[name])
           for name, value in buffers_before.items()), 'Snapshot changed a buffer'
report = {'device': 'cpu', 'model': 'ffhq256.pkl raw G in training mode',
          'psi': 1.0, 'samples': len(codes), 'model_state_sha256': before,
          'parameters_and_buffers_unchanged': True, 'cpu_rng_unchanged': True,
          'all_module_training_modes_restored': True,
          'buffer_count_checked': len(buffers_before)}
(output / 'snapshot-state-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
