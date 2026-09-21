"""Small random-network/plan checks only; never load a pretrained model."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimate_importance import make_sample_plan, pair_gradients, validate_input, RESEARCH
from test_modulation import small_models
from modulation import install_modulation, modulation_named_parameters


class EstimatorTests(unittest.TestCase):
    def test_plan_prefix_balanced_cycles_and_global_rng_unchanged(self):
        before = torch.get_rng_state().clone()
        short = make_sample_plan(torch, 4)
        full = make_sample_plan(torch, 1000)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(short, full):
            self.assertTrue(torch.equal(a, b[:4]))
        self.assertEqual(tuple(full[0].shape), (1000, 512))
        self.assertTrue(torch.isfinite(full[0]).all())
        for cycle in full[1].reshape(50, 20):
            self.assertTrue(torch.equal(torch.sort(cycle).values, torch.arange(20)))
        for a, b in zip(full, make_sample_plan(torch, 1000)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(full[2].dtype, torch.bool)
        self.assertEqual(torch.unique(full[3]).numel(), 1000)

    def test_phased_factor_gradients_match_joint_graph_all_parameters_enabled(self):
        torch.manual_seed(71)
        G, D = small_models(32)
        install_modulation(G, component='G', conv_layout='output_rank1')
        install_modulation(D, component='D', conv_layout='output_rank1')
        named = {'G': modulation_named_parameters(G), 'D': modulation_named_parameters(D)}
        z = torch.randn(1, 8); real = torch.randn(1, 3, 32, 32)
        state = {key: {n: t.detach().clone() for n, t in model.state_dict().items()}
                 for key, model in [('G', G), ('D', D)]}
        G.requires_grad_(True); D.requires_grad_(True)
        torch.manual_seed(953)
        ws = G.mapping(z, None, skip_w_avg_update=True)
        fake = G.synthesis(ws, noise_mode='random', force_fp32=True, fused_modconv=False)
        fake_pred = D(fake, None, force_fp32=True)
        gl = F.softplus(-fake_pred).mean()
        dl = F.softplus(-D(real, None, force_fp32=True)).mean() + F.softplus(fake_pred).mean()
        expected_g = torch.autograd.grad(gl, [p for _, p in named['G']], retain_graph=True)
        expected_d = torch.autograd.grad(dl, [p for _, p in named['D']])
        G.requires_grad_(False); D.requires_grad_(False)
        got_fake, got_gl, got_dl, got_g, got_d = pair_gradients(G, D, z, real, 953, named)
        self.assertTrue(torch.equal(fake.detach(), got_fake))
        self.assertTrue(torch.equal(gl.detach(), got_gl))
        self.assertTrue(torch.equal(dl.detach(), got_dl))
        for expected, got in [(expected_g, got_g), (expected_d, got_d)]:
            self.assertEqual(len(expected), len(got))
            for a, b in zip(expected, got):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-6)
        self.assertTrue(any(torch.count_nonzero(t) for t in got_g))
        self.assertTrue(any(torch.count_nonzero(t) for t in got_d))
        for key, model in [('G', G), ('D', D)]:
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(state[key][name], tensor), name)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_existing_source_layout_is_rejected_before_checkpoint_access(self):
        with self.assertRaisesRegex(AssertionError, 'Reject source-layout'):
            validate_input(RESEARCH / 'runs/adam-native1024-probing500-v1', 500)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
