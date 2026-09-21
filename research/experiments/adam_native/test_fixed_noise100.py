"""Preparation checks only; tiny random32px networks, no pretrained/native loads."""
import ast
import copy
from pathlib import Path
import sys
import unittest

import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'research/vendor/stylegan2-ada-pytorch'))
from training.networks import Generator
from modulation import install_modulation, probing_named_parameters
from probe_output_rank1_fixed_noise100 import (
    assert_fixed_noise, capture_noise_reference, set_stability_grad, stability_named_parameters,
    expected_optimizer_steps, PREVIEW_STEPS, CHECKPOINT_STEPS, ITERATIONS,
)


def model():
    g=Generator(z_dim=8,c_dim=0,w_dim=8,img_resolution=32,img_channels=3,
        mapping_kwargs={'num_layers':2},synthesis_kwargs={'channel_base':128,'channel_max':8}).eval()
    with torch.no_grad():
        for n,p in g.named_parameters():
            if n.endswith('.noise_strength'):p.fill_(.0125)
    install_modulation(g,component='G',conv_layout='output_rank1')
    set_stability_grad(g,True)
    return g


class FixedNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_noise_excluded_and_frozen_across_toggles_updates_and_ema(self):
        torch.manual_seed(70);g=model();ref=capture_noise_reference(g,7)
        ema=copy.deepcopy(g).requires_grad_(False)
        regular=dict(probing_named_parameters(g));actual=dict(stability_named_parameters(g))
        self.assertEqual(set(regular)-set(actual),set(ref))
        optimizer=torch.optim.Adam(list(actual.values()),lr=.002,betas=(0.0,.99))
        for _ in range(3):
            for enabled in (False,True,False,True):
                set_stability_grad(g,enabled);assert_fixed_noise(g,ref,optimizer)
                self.assertTrue(all(p.requires_grad==enabled for p in actual.values()))
            optimizer.zero_grad(set_to_none=True)
            image=g(torch.randn(1,8),None,noise_mode='random',force_fp32=True,fused_modconv=False)
            (image.square().mean()+image.mean()).backward()
            assert_fixed_noise(g,ref,optimizer)
            self.assertTrue(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in actual.values()))
            optimizer.step();assert_fixed_noise(g,ref,optimizer)
            with torch.no_grad():
                for pe,p in zip(ema.parameters(),g.parameters()):pe.copy_(p.lerp(pe,.9))
            assert_fixed_noise(ema,ref)
            self.assertTrue(all(g.get_parameter(n) not in optimizer.state for n in ref))
        # Actual original .0125 values survive (not merely initially zero defaults).
        self.assertTrue(all(torch.equal(g.get_parameter(n.replace('.0.b_vector','.original')),v) for n,v in ref.items()))

    def test_invariants_detect_offset_optimizer_toggle_and_original_corruption(self):
        g=model();ref=capture_noise_reference(g,7);name=next(iter(ref));offset=g.get_parameter(name)
        opt=torch.optim.Adam([p for _,p in stability_named_parameters(g)]+[offset])
        with self.assertRaisesRegex(AssertionError,'entered optimizer'):assert_fixed_noise(g,ref,opt)
        offset.requires_grad_(True)
        with self.assertRaisesRegex(AssertionError,'gradients enabled'):assert_fixed_noise(g,ref)
        set_stability_grad(g,True)
        with torch.no_grad():offset.fill_(.1)
        with self.assertRaisesRegex(AssertionError,'no longer zero'):assert_fixed_noise(g,ref)
        with torch.no_grad():
            offset.zero_();g.get_parameter(name.replace('.0.b_vector','.original')).add_(1)
        with self.assertRaisesRegex(AssertionError,'strength changed'):assert_fixed_noise(g,ref)

    def test_static_horizon_losses_rng_and_no_resume(self):
        self.assertEqual(ITERATIONS,100);self.assertEqual(CHECKPOINT_STEPS,(100,))
        self.assertEqual(PREVIEW_STEPS,(0,10,25,50,100))
        self.assertEqual(expected_optimizer_steps(100),{'g':125,'d':107})
        def nested(filename,name):
            tree=ast.parse((HERE/filename).read_text())
            worker=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
            return ast.dump(next(x for x in worker.body if isinstance(x,ast.FunctionDef) and x.name==name))
        for name in ('generate','update'):
            self.assertEqual(nested('probe_output_rank1.py',name),nested('probe_output_rank1_fixed_noise100.py',name))
        original=(HERE/'probe_output_rank1.py').read_text()
        variant=(HERE/'probe_output_rank1_fixed_noise100.py').read_text()
        start='# Separate backwards lower peak memory';end="        assert frozen(G)==frozen_g and frozen(D)==frozen_d, 'frozen source changed'"
        loss_block=lambda s:s[s.index(start):s.index(end)].replace('set_stability_grad(G,True)','set_probing_grad(G,True)')
        self.assertEqual(loss_block(original),loss_block(variant))
        self.assertNotIn("'--resume'",variant)


if __name__=='__main__':unittest.main(verbosity=2)
