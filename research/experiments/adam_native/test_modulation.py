"""Small random NVIDIA networks and scalar mechanics only; no pretrained loads."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/vendor/stylegan2-ada-pytorch"))
from training.networks import Generator, Discriminator
from research.experiments.adam_native.modulation import (
    AdditiveOffset, RankOneWeight, describe_modulation, fold_modulation,
    frozen_parameters, install_modulation, modulation_named_parameters,
    probing_named_parameters, probing_parameters, set_probing_grad,
)


def small_models(resolution=32):
    # 1024 is instantiated only for enumeration; two/four-channel weights and
    # no native forward/backward, no pretrained source or teacher files loaded.
    channel_base = max(128, resolution * 2)
    g = Generator(z_dim=8, c_dim=0, w_dim=8, img_resolution=resolution, img_channels=3,
        mapping_kwargs={"num_layers": 2},
        synthesis_kwargs={"channel_base": channel_base, "channel_max": 8}).eval()
    d = Discriminator(c_dim=0, img_resolution=resolution, img_channels=3,
        channel_base=channel_base, channel_max=8,
        epilogue_kwargs={"mbstd_group_size": 2}).eval()
    return g, d


def render(g, z):
    return g(z, None, noise_mode="const", force_fp32=True, fused_modconv=False)


class ModulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

    def setUp(self):
        torch.manual_seed(51)

    def test_rank_equations_identity_and_initial_factor_gradients(self):
        weight = torch.randn(3, 2, 3, 3)
        adapter = RankOneWeight(weight, layout="source_flattened", generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(adapter(weight), weight))
        adapter(weight).square().sum().backward()
        for parameter in adapter.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with torch.no_grad():
            adapter.u_vector.copy_(torch.linspace(-.2, .1, 18))
            adapter.v_vector.copy_(torch.tensor([.3, -.1, .2]))
        expected = weight * (1 + torch.outer(adapter.u_vector, adapter.v_vector).reshape_as(weight))
        self.assertTrue(torch.equal(adapter(weight), expected))
        output_rank = RankOneWeight(weight, layout="output_rank1", generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            output_rank.u_vector.copy_(adapter.u_vector)
            output_rank.v_vector.copy_(adapter.v_vector)
        expected_output_rank = weight * (1 + torch.outer(adapter.v_vector, adapter.u_vector).reshape_as(weight))
        self.assertTrue(torch.equal(output_rank(weight), expected_output_rank))
        self.assertFalse(torch.equal(expected, expected_output_rank))
        fc_weight = torch.randn(3, 7)
        fc = RankOneWeight(fc_weight, layout="source_flattened", generator=torch.Generator().manual_seed(3))
        self.assertEqual(fc.layout, "output_rank1")

    def test_small_native_g_d_identity_and_complete_probing_scope(self):
        g, d = small_models()
        z = torch.randn(2, 8)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            before_g, before_d = render(g, z), d(x, None, force_fp32=True)
        rng = torch.random.get_rng_state().clone()
        gm = install_modulation(g, component="G", seed=41)
        dm = install_modulation(d, component="D", seed=42)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        with torch.no_grad():
            self.assertTrue(torch.equal(before_g, render(g, z)))
            self.assertTrue(torch.equal(before_d, d(x, None, force_fp32=True)))
        paths = {(r["module"], r["tensor"]) for r in gm["targets"]}
        for expected in [("mapping.fc0", "weight"), ("mapping.fc1", "bias"),
                         ("synthesis.b4.conv1", "weight"), ("synthesis.b4.conv1", "noise_strength"),
                         ("synthesis.b4.conv1.affine", "weight"), ("synthesis.b32.torgb", "bias"),
                         ("synthesis.b32.torgb.affine", "bias")]:
            self.assertIn(expected, paths)
        self.assertFalse(g.synthesis.b4.const.requires_grad)
        self.assertEqual(dm["ordinary_trainable_names"],
            ["b4.conv.weight", "b4.conv.bias", "b4.fc.weight", "b4.fc.bias", "b4.out.weight", "b4.out.bias"])
        self.assertTrue(all(not name.startswith("b4.") for name, _ in modulation_named_parameters(d)))
        self.assertEqual(len(probing_parameters(g)), 79)
        self.assertEqual(len(probing_parameters(d)), 33)
        for name, parameter in frozen_parameters(g) + frozen_parameters(d):
            self.assertFalse(parameter.requires_grad, name)

    def test_optimizer_updates_only_probing_and_preserves_frozen_bases(self):
        g, d = small_models()
        install_modulation(g, component="G")
        install_modulation(d, component="D", seed=52)
        frozen_g = {name: p.detach().clone() for name, p in frozen_parameters(g)}
        frozen_d = {name: p.detach().clone() for name, p in frozen_parameters(d)}
        g_opt = torch.optim.Adam(probing_parameters(g), lr=.002)
        d_opt = torch.optim.Adam(probing_parameters(d), lr=.002)
        original_g_trainable = {name: p.detach().clone() for name, p in probing_named_parameters(g)}
        original_epilogue = d.b4.out.weight.detach().clone()
        z = torch.randn(2, 8)
        real = torch.randn(2, 3, 32, 32)
        set_probing_grad(d, False)
        g_opt.zero_grad(set_to_none=True)
        torch.nn.functional.softplus(-d(render(g, z), None, force_fp32=True)).mean().backward()
        nonzero = 0
        for name, p in probing_named_parameters(g):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            nonzero += int(bool(p.grad.abs().sum() > 0))
        self.assertGreater(nonzero, 0)
        g_opt.step()
        set_probing_grad(g, False)
        set_probing_grad(d, True)
        d_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = render(g, z)
        loss = (torch.nn.functional.softplus(d(fake, None, force_fp32=True)).mean()
                + torch.nn.functional.softplus(-d(real, None, force_fp32=True)).mean())
        loss.backward()
        for name, p in probing_named_parameters(d):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        d_opt.step()
        for model, original in ((g, frozen_g), (d, frozen_d)):
            for name, p in frozen_parameters(model):
                self.assertIsNone(p.grad, name)
                self.assertTrue(torch.equal(p, original[name]), name)
        self.assertTrue(any(not torch.equal(p, original_g_trainable[name]) for name, p in probing_named_parameters(g)))
        self.assertFalse(torch.equal(original_epilogue, d.b4.out.weight))

    def test_fold_and_state_dict_roundtrip_preserve_nonzero_modulations(self):
        g, d = small_models()
        plain_g, plain_d = copy.deepcopy(g), copy.deepcopy(d)
        for model, role in ((g, "G"), (d, "D")):
            install_modulation(model, component=role)
            with torch.no_grad():
                for _, p in modulation_named_parameters(model):
                    p.add_(torch.randn_like(p) * .025)
        z = torch.randn(2, 8)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected_g, expected_d = render(g, z), d(x, None, force_fp32=True)
        # Switching grad flags must not disable learned offsets (upstream bug).
        set_probing_grad(g, False)
        set_probing_grad(d, False)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected_g, render(g, z)))
            self.assertTrue(torch.equal(expected_d, d(x, None, force_fp32=True)))
        folded_g, folded_d = fold_modulation(g), fold_modulation(d)
        self.assertTrue(any(parametrize.is_parametrized(m) for m in g.modules()))
        self.assertTrue(any(parametrize.is_parametrized(m) for m in d.modules()))
        self.assertFalse(any(parametrize.is_parametrized(m) for m in folded_g.modules()))
        self.assertFalse(any(parametrize.is_parametrized(m) for m in folded_d.modules()))
        plain_g.load_state_dict(folded_g.state_dict(), strict=True)
        plain_d.load_state_dict(folded_d.state_dict(), strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected_g, render(g, z)))
            self.assertTrue(torch.equal(expected_d, d(x, None, force_fp32=True)))
            self.assertTrue(torch.equal(expected_g, render(folded_g, z)))
            self.assertTrue(torch.equal(expected_g, render(plain_g, z)))
            self.assertTrue(torch.equal(expected_d, folded_d(x, None, force_fp32=True)))
            self.assertTrue(torch.equal(expected_d, plain_d(x, None, force_fp32=True)))

    def test_native1024_dynamic_enumeration_without_native_compute(self):
        g, d = small_models(1024)
        gm, dm = install_modulation(g, component="G"), install_modulation(d, component="D")
        synthesis = [r for r in gm["targets"] if r["tensor"] == "weight"
                     and r["module"].startswith("synthesis.") and r["module"].split(".")[-1] in ("conv0", "conv1")]
        rgb = [r for r in gm["targets"] if r["tensor"] == "weight" and r["module"].endswith(".torgb")]
        d_weights = [r for r in dm["targets"] if r["tensor"] == "weight"]
        self.assertEqual(len(synthesis), 17)
        self.assertEqual(len(rgb), 9)
        self.assertEqual(len(d_weights), 25)
        self.assertIn("synthesis.b1024.conv1", [r["module"] for r in synthesis])
        self.assertIn("b1024.fromrgb", [r["module"] for r in d_weights])
        # Mapping has2 layers in this tiny fixture instead of native default8.
        self.assertEqual(len(probing_parameters(g)), 197 - 6 * 3)
        self.assertEqual(len(probing_parameters(d)), 73)

    def test_small_native_second_derivatives_and_adapter_state_reload(self):
        g, d = small_models()
        restored_g, restored_d = copy.deepcopy(g), copy.deepcopy(d)
        install_modulation(g, component="G")
        install_modulation(d, component="D")
        image = torch.randn(2, 3, 32, 32, requires_grad=True)
        score = d(image, None, force_fp32=True)
        image_grad = torch.autograd.grad(score.sum(), image, create_graph=True)[0]
        r1 = image_grad.square().sum((1, 2, 3)).mean()
        r1.backward()
        active_d = [p.grad for p in probing_parameters(d) if p.grad is not None]
        self.assertTrue(active_d and all(torch.isfinite(v).all() for v in active_d))
        self.assertGreater(sum(float(v.abs().sum()) for v in active_d), 0)
        g.zero_grad(set_to_none=True)
        ws = g.mapping(torch.randn(2, 8), None, skip_w_avg_update=True)
        output = g.synthesis(ws, noise_mode="const", force_fp32=True, fused_modconv=False)
        projection = (output * torch.randn_like(output) / 32).sum()
        path_grad = torch.autograd.grad(projection, ws, create_graph=True)[0]
        penalty = path_grad.square().sum(2).mean(1).sqrt().square().mean()
        penalty.backward()
        active_g = [p.grad for p in probing_parameters(g) if p.grad is not None]
        self.assertTrue(active_g and all(torch.isfinite(v).all() for v in active_g))
        self.assertGreater(sum(float(v.abs().sum()) for v in active_g), 0)
        with torch.no_grad():
            for model in (g, d):
                for _, p in modulation_named_parameters(model):
                    p.add_(torch.randn_like(p) * .02)
        install_modulation(restored_g, component="G", seed=456)
        install_modulation(restored_d, component="D", seed=457)
        restored_g.load_state_dict(g.state_dict(), strict=True)
        restored_d.load_state_dict(d.state_dict(), strict=True)
        z = torch.randn(2, 8)
        with torch.no_grad():
            self.assertTrue(torch.equal(render(g, z), render(restored_g, z)))
            self.assertTrue(torch.equal(d(image, None, force_fp32=True), restored_d(image, None, force_fp32=True)))

    def test_reject_unsupported_or_repeated_install_before_mutation(self):
        g, _ = small_models()
        before = {name: p.requires_grad for name, p in g.named_parameters()}
        with self.assertRaises(ValueError):
            install_modulation(g, component="G", init_scale=.3)
        self.assertFalse(any(parametrize.is_parametrized(m) for m in g.modules()))
        self.assertEqual(before, {name: p.requires_grad for name, p in g.named_parameters()})
        install_modulation(g, component="G")
        with self.assertRaises(ValueError):
            install_modulation(g, component="G")


if __name__ == "__main__":
    unittest.main(verbosity=2)
