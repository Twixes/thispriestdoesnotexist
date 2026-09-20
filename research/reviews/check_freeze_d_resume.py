"""Small CPU regression check of NVIDIA FreezeD state/optimizer round trips."""
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import transfer_discriminator
from training.networks import Discriminator

torch.set_num_threads(2)
torch.manual_seed(73)
source = Discriminator(c_dim=0, img_resolution=8, img_channels=3,
                       channel_base=128, channel_max=16)
rng_before = torch.get_rng_state().clone()
model, frozen = transfer_discriminator(source, 4)
assert torch.equal(rng_before, torch.get_rng_state()), 'Reconstruction consumed RNG'
assert frozen, 'No frozen tensors discovered'
before = {name: dict(model.named_buffers())[name].clone() for name in frozen}
optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(0.0, .99))

def update(net, opt):
    net.requires_grad_(False)
    net.requires_grad_(True)
    real = torch.randn(2, 3, 8, 8, requires_grad=True)
    score = net(real, None, force_fp32=True)
    grad = torch.autograd.grad(score.sum(), real, create_graph=True)[0]
    loss = F.softplus(-score).mean() + grad.square().sum((1, 2, 3)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

update(model, optimizer)
stream = io.BytesIO()
torch.save({'D': model.state_dict(), 'D_opt': optimizer.state_dict(),
            'freeze_d_layers': 4}, stream)
stream.seek(0)
checkpoint = torch.load(stream, weights_only=True)
restored, restored_frozen = transfer_discriminator(source, checkpoint['freeze_d_layers'])
restored.load_state_dict(checkpoint['D'], strict=True)
restored_opt = torch.optim.Adam(restored.parameters(), lr=.001, betas=(0.0, .99))
restored_opt.load_state_dict(checkpoint['D_opt'])
assert restored_frozen == frozen
assert len(restored_opt.state) == len(list(restored.parameters())), 'Missing restored Adam state'
for name, value in model.state_dict().items():
    assert torch.equal(value, restored.state_dict()[name]), 'Model round-trip changed state'
update(restored, restored_opt)
for name, value in before.items():
    assert torch.equal(value, dict(restored.named_buffers())[name]), name
    assert not dict(restored.named_buffers())[name].requires_grad, name
assert not set(frozen) & set(dict(restored.named_parameters()))
report = {'device': 'cpu', 'resolution': 8, 'freeze_d_layers': 4,
          'frozen_tensors': frozen, 'updates_with_r1': 2,
          'checkpoint_round_trip': True, 'adam_state_restored': True,
          'requires_grad_toggles_keep_buffers_frozen': True,
          'frozen_tensors_bitwise_unchanged': True, 'initialization_rng_preserved': True}
output = ROOT / 'runs/frozen-d-smoke/resume-regression.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
