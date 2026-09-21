"""Tiny random native32px networks, scalar objectives and real Adam updates."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/vendor/stylegan2-ada-pytorch'))
from training.networks import Generator, Discriminator
from adaptation_masks import AdaptationMasks, adaptation_inventory
from modulation import install_modulation, fold_modulation, describe_modulation
from selection import select_rows


def models():
    g = Generator(z_dim=8, c_dim=0, w_dim=8, img_resolution=32, img_channels=3,
        mapping_kwargs={'num_layers': 2}, synthesis_kwargs={'channel_base': 128, 'channel_max': 8}).eval()
    d = Discriminator(c_dim=0, img_resolution=32, img_channels=3,
        channel_base=128, channel_max=8, epilogue_kwargs={'mbstd_group_size': 2}).eval()
    install_modulation(g, component='G', conv_layout='output_rank1')
    install_modulation(d, component='D', conv_layout='output_rank1')
    return g, d


def selection(model, uniform=None):
    result = adaptation_inventory(model)
    result['masks'] = {name: [bool(i % 2) != name.endswith('.affine') if uniform is None else uniform
        for i in range(record['weight_shape'][0])] for name, record in result['modules'].items()}
    return result


def render(g, z):
    return g(z, None, noise_mode='const', force_fp32=True, fused_modconv=False)


class AdaptationMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def setUp(self):
        torch.manual_seed(92)

    def test_inventory_exclusions_exact_initial_output_and_support(self):
        g, d = models(); zg = torch.randn(1, 8); xd = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            before_g = render(g, zg); before_d = d(xd, None, force_fp32=True)
        for model in (g, d):
            s = selection(model); names = tuple(dict(model.named_parameters()))
            count = sum(p.numel() for p in model.parameters())
            control = AdaptationMasks(model, s)
            self.assertEqual(tuple(dict(model.named_parameters())), names)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertTrue(control.verify_invariants())
            for name, rec in s['modules'].items():
                high = torch.tensor(s['masks'][name]); low = ~high
                v = model.get_parameter(rec['v_parameter'])
                self.assertEqual(int(torch.count_nonzero(v[low])), 0)
                self.assertTrue(model.get_parameter(rec['u_parameter']).requires_grad)
            if s['component'] == 'G':
                self.assertEqual(len(s['modules']), 12)  # six convs, six independently masked affines.
                for name in ('mapping.fc0', 'synthesis.b4.conv1', 'synthesis.b32.torgb'):
                    self.assertNotIn(name, s['modules'])
                    self.assertFalse(model.get_parameter(name+'.parametrizations.weight.original').requires_grad)
                    self.assertTrue(model.get_parameter(name+'.parametrizations.weight.0.u_vector').requires_grad)
                for name in ('synthesis.b8.conv0', 'synthesis.b16.conv1'):
                    self.assertFalse(model.get_parameter(name+'.parametrizations.bias.original').requires_grad)
                    self.assertFalse(model.get_parameter(name+'.parametrizations.noise_strength.original').requires_grad)
                    self.assertTrue(model.get_parameter(name+'.parametrizations.bias.0.b_vector').requires_grad)
                    self.assertTrue(model.get_parameter(name+'.parametrizations.noise_strength.0.b_vector').requires_grad)
            else:
                self.assertEqual(len(s['modules']), 9)
                self.assertTrue(all(p.requires_grad for n,p in model.named_parameters() if n.startswith('b4.')))
                self.assertFalse(model.get_parameter('b32.fromrgb.parametrizations.weight.original').requires_grad)
        with torch.no_grad():
            self.assertTrue(torch.equal(before_g, render(g, zg)))
            self.assertTrue(torch.equal(before_d, d(xd, None, force_fp32=True)))
        layer = g.synthesis.b8.conv0
        with torch.no_grad():
            layer.parametrizations.weight[0].u_vector.fill_(.2)
            reference = layer.weight.clone()
            layer.parametrizations.weight[0].v_vector[1] += .3
            changed = (reference != layer.weight).flatten(1).any(1)
        self.assertEqual(changed.nonzero().flatten().tolist(), [1])

    def test_multiple_actual_adversarial_and_r1_path_adam_steps(self):
        g,d = models(); cg = AdaptationMasks(g, selection(g)); cd = AdaptationMasks(d, selection(d))
        go = torch.optim.Adam(cg.parameters(), lr=.002, betas=(0.0,.99), amsgrad=True)
        do = torch.optim.Adam(cd.parameters(), lr=.002, betas=(0.0,.99), amsgrad=True)
        cg.bind_optimizer(go); cd.bind_optimizer(do)
        originals = {n:p.detach().clone() for n,p in cg.named_parameters()}
        epilogue = d.b4.out.weight.detach().clone()
        for _ in range(2):
            cg.set_enabled(False); cd.set_enabled(True); do.zero_grad(set_to_none=True)
            with torch.no_grad(): fake = render(g, torch.randn(2,8))
            real = torch.randn(2,3,32,32)
            loss = F.softplus(d(fake,None,force_fp32=True)).mean() + F.softplus(-d(real,None,force_fp32=True)).mean()
            loss.backward(); cd.step(do)
            do.zero_grad(set_to_none=True); real.requires_grad_(True)
            logits = d(real,None,force_fp32=True)
            grad, = torch.autograd.grad(logits.sum(), real, create_graph=True)
            grad.square().sum().backward(); cd.step(do)
            cd.set_enabled(False); cg.set_enabled(True); go.zero_grad(set_to_none=True)
            F.softplus(-d(render(g,torch.randn(2,8)),None,force_fp32=True)).mean().backward(); cg.step(go)
            go.zero_grad(set_to_none=True)
            ws = g.mapping(torch.randn(2,8),None,skip_w_avg_update=True)
            image = g.synthesis(ws,noise_mode='const',force_fp32=True,fused_modconv=False)
            pl, = torch.autograd.grad((image * torch.randn_like(image)).sum(), ws, create_graph=True)
            pl.square().mean().backward(); cg.step(go)
        self.assertTrue(cg.verify_invariants(go)); self.assertTrue(cd.verify_invariants(do))
        self.assertFalse(torch.equal(epilogue,d.b4.out.weight))
        changed_low = [n for n,p in cg.named_parameters() if n.endswith('.original') and not torch.equal(p,originals[n])]
        self.assertTrue(changed_low)
        # Fold must retain offsets and must not damage the original parametrized model.
        with torch.no_grad():
            z = torch.randn(1,8); before = render(g,z)
            folded = fold_modulation(g)
            self.assertTrue(torch.equal(before,render(folded,z)))
            self.assertTrue(torch.equal(before,render(g,z)))
        self.assertTrue(cg.verify_invariants(go))

    def test_optimizer_state_corruption_and_bypassing_masks_detected(self):
        g,_ = models(); c = AdaptationMasks(g,selection(g)); opt = torch.optim.Adam(c.parameters(),lr=.01)
        c.bind_optimizer(opt)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            sum(p.sum() + .01*p.square().sum() for p in c.parameters()).backward()
            c.step(opt)
        name = 'synthesis.b8.conv0.parametrizations.weight.original'; p = g.get_parameter(name)
        high = torch.tensor(c.selection['masks']['synthesis.b8.conv0'])
        opt.state[p]['exp_avg'][high] = 1
        with self.assertRaisesRegex(AssertionError,'protected Adam'):
            c.mask_gradients(opt)
        opt.state[p]['exp_avg'][high] = 0
        old_step=opt.state[p]['step'].clone()
        opt.state[p]['step'].fill_(float('nan'))
        with self.assertRaisesRegex(AssertionError,'step count'):
            c.verify_invariants(opt)
        opt.state[p]['step'].copy_(old_step)
        with torch.no_grad(): p[high] += 1
        with self.assertRaisesRegex(AssertionError,'protected parameter'):
            c.verify_invariants(opt)

    def test_preflight_rejection_is_nonmutating_and_no_source_flattened(self):
        for kind in ('missing', 'dimension', 'mapping', 'component'):
            g,_ = models(); s=selection(g); before={n:p.detach().clone() for n,p in g.named_parameters()}
            flags={n:p.requires_grad for n,p in g.named_parameters()}; path=next(iter(s['modules']))
            if kind=='missing': del s['masks'][path]
            if kind=='dimension': s['masks'][path]=[True]
            if kind=='mapping': s['modules'][path]['v_parameter']='wrong'
            if kind=='component': s['component']='D'
            with self.assertRaises(ValueError): AdaptationMasks(g,s)
            self.assertTrue(all(torch.equal(p,before[n]) for n,p in g.named_parameters()))
            self.assertEqual(flags,{n:p.requires_grad for n,p in g.named_parameters()})
        g,_=models(); g.__dict__['_adam_native_metadata']['conv_layout']='source_flattened'
        with self.assertRaises(ValueError): adaptation_inventory(g)

    def test_all_high_and_all_low_ties_have_explicit_parameter_policy(self):
        for high in (True,False):
            g,_ = models(); s=selection(g,uniform=high); c=AdaptationMasks(g,s)
            opt=torch.optim.Adam(c.parameters(),lr=.002); c.bind_optimizer(opt)
            for name,rec in s['modules'].items():
                self.assertEqual(g.get_parameter(rec['weight_parameter']).requires_grad,not high)
                self.assertEqual(g.get_parameter(rec['v_parameter']).requires_grad,high)
                self.assertTrue(g.get_parameter(rec['u_parameter']).requires_grad)
            sum(p.sum() for p in c.parameters()).backward();c.step(opt)
            self.assertTrue(c.verify_invariants(opt))

    def test_selector_contract_works_for_actual_native_inventory(self):
        g,d = models()
        for model in (g,d):
            inventory=describe_modulation(model)
            scores={n: torch.linspace(0.1,1,p.numel(),dtype=torch.float64).reshape(p.shape)
                    for n,p in model.named_parameters() if n.endswith(('u_vector','v_vector','b_vector'))}
            selected=select_rows(scores,inventory,quantile=50)
            control=AdaptationMasks(model,selected)
            optimizer=torch.optim.Adam(control.parameters(),lr=.001)
            control.bind_optimizer(optimizer)
            sum(p.square().sum()+p.sum() for p in control.parameters()).backward()
            control.step(optimizer)
            self.assertTrue(control.verify_invariants(optimizer))

    def test_optimizer_guard_and_requires_grad_misuse(self):
        g,_=models();c=AdaptationMasks(g,selection(g))
        for opt in (torch.optim.AdamW(c.parameters()),torch.optim.Adam(c.parameters(),weight_decay=.1)):
            with self.assertRaises(ValueError):c.bind_optimizer(opt)
        stale=torch.optim.Adam(c.parameters());sum(p.sum() for p in c.parameters()).backward();stale.step()
        with self.assertRaises(ValueError):c.bind_optimizer(stale)
        g,_=models();c=AdaptationMasks(g,selection(g));opt=torch.optim.Adam(c.parameters());c.bind_optimizer(opt)
        g.requires_grad_(True)
        with self.assertRaisesRegex(AssertionError,'requires_grad'):c.verify_invariants(opt)
        c.set_enabled(False)
        with self.assertRaises(ValueError):c.step(opt)
        c.set_enabled(True)
        with self.assertRaises(ValueError):AdaptationMasks(g,selection(g))


if __name__=='__main__':
    unittest.main(verbosity=2)
