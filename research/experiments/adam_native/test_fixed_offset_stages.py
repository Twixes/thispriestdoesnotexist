"""Small random native32 networks; no source checkpoints, native1024, or training run."""
import unittest
import copy
import torch
from torch.nn import functional as F
from test_adaptation_masks import models, render
from fixed_offset_policy import (capture_fixed_reference, verify_fixed_offsets,
    fixed_offset_names, importance_named_parameters, is_fixed_offset)
from estimate_importance import pair_gradients as old_pair_gradients
from estimate_importance_fixed_offsets import pair_gradients
from adaptation_masks_fixed_offsets import FixedOffsetsAdaptationMasks
from modulation import modulation_named_parameters, describe_modulation, fold_modulation
from selection import select_rows
from selection_fixed_offsets import select_rows_fixed_offsets


def selected(model):
    means = {n: torch.linspace(.01, .9, p.numel(), dtype=torch.float64).reshape(p.shape)
             for n, p in importance_named_parameters(model)}
    return select_rows_fixed_offsets(means, describe_modulation(model), quantile=50)


class FixedStageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(281)

    def test_fi_omission_retains_same_requested_gradients_and_selection(self):
        G, D = models(); G.requires_grad_(False); D.requires_grad_(False)
        before = {n: t.clone() for n, t in G.state_dict().items()}
        refs = capture_fixed_reference(G)
        self.assertEqual(len(refs), 14)  # seven tiny32 synthesis convs, two families.
        all_named = {'G': modulation_named_parameters(G), 'D': modulation_named_parameters(D)}
        keep = {'G': importance_named_parameters(G), 'D': importance_named_parameters(D)}
        z = torch.randn(1, 8); real = torch.randn(1, 3, 32, 32)
        old = old_pair_gradients(G, D, z, real, 719, all_named)
        G.requires_grad_(False); D.requires_grad_(False)
        got = pair_gradients(G, D, z, real, 719, keep)
        for index in range(3): self.assertTrue(torch.equal(old[index], got[index]))
        for key, index in [('G', 3), ('D', 4)]:
            expected = dict(zip([n for n, _ in all_named[key]], old[index]))
            for (name, _), grad in zip(keep[key], got[index]):
                torch.testing.assert_close(grad, expected[name], rtol=1e-6, atol=1e-10)
            full_means = {n: x.double().square() for n, x in expected.items()}
            reduced = {n: x for n, x in full_means.items() if not is_fixed_offset(n)}
            a = select_rows(full_means, describe_modulation(G if key == 'G' else D), quantile=50)
            b = select_rows_fixed_offsets(reduced, describe_modulation(G if key == 'G' else D), quantile=50)
            self.assertEqual(a['masks'], b['masks']); self.assertEqual(a['pools'], b['pools'])
        self.assertTrue(verify_fixed_offsets(G, refs))
        self.assertTrue(all(torch.equal(t, before[n]) for n, t in G.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in G.parameters()))
        with self.assertRaises(AssertionError):
            pair_gradients(G, D, z, real, 719, all_named)

    def test_actual_adversarial_path_updates_preserve_both_families_and_fold(self):
        g, d = models(); g.requires_grad_(False); d.requires_grad_(False)
        # Nonzero original strengths/biases must survive; fixing offsets is not
        # a reset of the pretrained synthesis tensors.
        with torch.no_grad():
            for name in fixed_offset_names(g):
                g.get_parameter(name.replace('.0.b_vector','.original')).fill_(.07)
        ema = copy.deepcopy(g).requires_grad_(False)
        c = FixedOffsetsAdaptationMasks(g, selected(g))
        dc = FixedOffsetsAdaptationMasks(d, selected(d))
        opt = torch.optim.Adam(c.parameters(), lr=.002, betas=(0.0,.99))
        dop = torch.optim.Adam(dc.parameters(), lr=.002, betas=(0.0,.99))
        c.bind_optimizer(opt); dc.bind_optimizer(dop)
        refs = capture_fixed_reference(g)
        originals = {n: p.clone() for n, p in c.named_parameters()}
        for _ in range(2):
            c.set_enabled(False); dc.set_enabled(True); dop.zero_grad(set_to_none=True)
            with torch.no_grad(): fake = render(g, torch.randn(2,8))
            real = torch.randn(2,3,32,32)
            (F.softplus(d(fake,None,force_fp32=True)).mean()+F.softplus(-d(real,None,force_fp32=True)).mean()).backward()
            dc.step(dop)
            dop.zero_grad(set_to_none=True); real.requires_grad_(True)
            dr, = torch.autograd.grad(d(real,None,force_fp32=True).sum(),real,create_graph=True)
            dr.square().mean().backward(); dc.step(dop)
            dc.set_enabled(False); c.set_enabled(True); opt.zero_grad(set_to_none=True)
            F.softplus(-d(render(g,torch.randn(2,8)),None,force_fp32=True)).mean().backward(); c.step(opt)
            opt.zero_grad(set_to_none=True)
            ws = g.mapping(torch.randn(2,8),None,skip_w_avg_update=True)
            im = g.synthesis(ws,noise_mode='const',force_fp32=True,fused_modconv=False)
            gp, = torch.autograd.grad((im*torch.randn_like(im)).sum(),ws,create_graph=True)
            gp.square().mean().backward(); c.step(opt)
            self.assertTrue(verify_fixed_offsets(g,refs,opt))
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), g.parameters()):
                    ep.copy_(p.lerp(ep, .5 ** (32/10000)))
            self.assertTrue(verify_fixed_offsets(ema,refs))
        for name in ('mapping.fc0.parametrizations.bias.0.b_vector',
                     'synthesis.b8.conv0.affine.parametrizations.bias.0.b_vector',
                     'synthesis.b32.torgb.parametrizations.bias.0.b_vector'):
            self.assertIn(name, dict(c.named_parameters()))
            self.assertFalse(torch.equal(originals[name], g.get_parameter(name)))
        self.assertTrue(any(n.endswith('.original') and not torch.equal(p, originals[n]) for n,p in c.named_parameters()))
        with torch.no_grad():
            z=torch.randn(1,8); before=render(g,z); folded=fold_modulation(g)
            self.assertTrue(torch.equal(before,render(folded,z)))
        self.assertTrue(c.verify_invariants(opt))
        with torch.no_grad(): g.get_parameter(fixed_offset_names(g)[0]).fill_(.1)
        with self.assertRaisesRegex(AssertionError,'fixed zero'): c.verify_invariants(opt)

    def test_reject_nonzero_offsets_and_old_selection_without_mutation(self):
        g,_=models(); g.requires_grad_(False)
        s=selected(g); offset=g.get_parameter(fixed_offset_names(g)[0])
        with torch.no_grad(): offset.fill_(.1)
        before={n:p.clone() for n,p in g.named_parameters()}
        with self.assertRaisesRegex(ValueError,'fixed offset must be zero'): FixedOffsetsAdaptationMasks(g,s)
        self.assertTrue(all(torch.equal(before[n],p) for n,p in g.named_parameters()))
        with torch.no_grad(): offset.zero_()
        del s['fixed_offset_policy']
        with self.assertRaisesRegex(ValueError,'explicitly fixed-offset'): FixedOffsetsAdaptationMasks(g,s)
        means = {n: torch.ones_like(p,dtype=torch.float64) for n,p in modulation_named_parameters(g)}
        with self.assertRaisesRegex(ValueError,'must omit protected'): select_rows_fixed_offsets(means,describe_modulation(g),quantile=50)


if __name__=='__main__':
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
