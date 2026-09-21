"""Tiny native32 tests of the actual prepared main-adaptation engine; no source weights."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/vendor/stylegan2-ada-pytorch'))
from training.networks import Generator,Discriminator
from modulation import install_modulation,describe_modulation,fold_modulation
from fixed_offset_policy import importance_named_parameters
from selection_fixed_offsets import select_rows_fixed_offsets
from main_adaptation_fixed_offsets10 import AdaptationEngine,state_digest,expected_steps,validated_selection


def make_engine(seed=821):
    torch.manual_seed(seed)
    G=Generator(z_dim=8,c_dim=0,w_dim=8,img_resolution=32,img_channels=3,
                mapping_kwargs={'num_layers':2},synthesis_kwargs={'channel_base':128,'channel_max':8}).eval().requires_grad_(False)
    D=Discriminator(c_dim=0,img_resolution=32,img_channels=3,channel_base=128,channel_max=8,
                    epilogue_kwargs={'mbstd_group_size':None}).eval().requires_grad_(False)
    # Nonzero protected original noise/bias values expose accidental zeroing of source tensors.
    with torch.no_grad():
        for name,p in G.named_parameters():
            if name.endswith('noise_strength'):p.fill_(.03)
    z=torch.randn(1,8)
    def render(model):return model(z,None,noise_mode='const',force_fp32=True,fused_modconv=False)
    with torch.no_grad():baseline=render(G);d_base=D(baseline,None,force_fp32=True)
    selections={}
    for key,model in [('G',G),('D',D)]:
        install_modulation(model,component=key,conv_layout='output_rank1')
        named=importance_named_parameters(model)
        scores={name:torch.linspace(.1,1,p.numel(),dtype=torch.float64).reshape(p.shape) for name,p in named}
        selections[key]=select_rows_fixed_offsets(scores,describe_modulation(model),quantile=50)
    engine=AdaptationEngine(G,D,selections,seed=seed+2)
    with torch.no_grad():
        assert torch.equal(render(G),baseline)
        assert torch.equal(D(baseline,None,force_fp32=True),d_base)
    return engine,z


class MainAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_actual_ten_iteration_schedule_original_updates_and_exact_fold(self):
        engine,z=make_engine();real=[torch.randn(3,32,32) for _ in range(3)]
        before={n:p.detach().clone() for n,p in engine.cg.named_parameters()}
        records=[engine.iteration(real) for _ in range(10)]
        self.assertEqual(engine.counts,{'g':13,'d':11})
        self.assertEqual([r['iteration'] for r in records if 'r1_loss' in r],[0])
        self.assertEqual([r['iteration'] for r in records if 'path_loss' in r],[0,4,8])
        changed=[n for n,p in engine.cg.named_parameters() if n.endswith('.original') and not torch.equal(before[n],p)]
        self.assertTrue(changed)
        engine.check()
        with torch.no_grad():
            for model in (engine.G,engine.Gema):
                folded=fold_modulation(model,inplace=False)
                options=dict(noise_mode='const',force_fp32=True,fused_modconv=False)
                self.assertTrue(torch.equal(model(z,None,**options),folded(z,None,**options)))
        self.assertTrue(all(pool['high_count'] and pool['low_count'] for s in engine.selections.values() for pool in s['pools'].values()))

    def test_exact_restored_next_update_with_both_regularizers(self):
        engine,_=make_engine();real=[torch.randn(3,32,32) for _ in range(2)]
        for _ in range(16):engine.iteration(real)
        # The next update includes both R1 and path regularization and exercises optimizer momentum.
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.pt';torch.save(engine.state(),path)
            saved=torch.load(path,weights_only=True)
        record=engine.iteration(real);expected=state_digest(engine.state())
        self.assertIn('r1_loss',record);self.assertIn('path_loss',record)
        restored,_=make_engine()
        restored.restore(copy.deepcopy(saved))
        self.assertEqual(state_digest(restored.state()),state_digest(saved))
        restored.iteration(real)
        self.assertEqual(state_digest(restored.state()),expected)
        # Zero momentum is not an equivalent continuation.
        wrong,_=make_engine();wrong.restore(copy.deepcopy(saved))
        for optimizer in (wrong.go,wrong.do):
            for value in optimizer.state.values():
                value['exp_avg'].zero_();value['exp_avg_sq'].zero_()
        wrong.iteration(real)
        self.assertNotEqual(state_digest(wrong.state()),expected)

    def test_restore_rejects_changed_masks_before_mutation_and_detects_corruption(self):
        engine,_=make_engine();real=[torch.randn(3,32,32)];engine.iteration(real)
        saved=copy.deepcopy(engine.state());reference=state_digest(saved)
        bad=copy.deepcopy(saved);path=next(iter(bad['selections']['G']['masks']))
        bad['selections']['G']['masks'][path][0]=not bad['selections']['G']['masks'][path][0]
        with self.assertRaisesRegex(AssertionError,'Mask/policy'):engine.restore(bad)
        self.assertEqual(state_digest(engine.state()),reference)
        bad=copy.deepcopy(saved);bad['optimizer_steps']['g']+=1
        with self.assertRaisesRegex(AssertionError,'cumulative'):engine.restore(bad)
        self.assertEqual(state_digest(engine.state()),reference)
        name=next(n for n in engine.cg._allowed_rows if n.endswith('.original') and bool((~engine.cg._allowed_rows[n]).any()))
        parameter=engine.G.get_parameter(name);protected=~engine.cg._allowed_rows[name]
        engine.go.state[parameter]['exp_avg'][protected]=1
        with self.assertRaisesRegex(AssertionError,'protected Adam'):engine.check()

    def test_importance_gate_rejects_absent_input_and_implicit_quantile(self):
        with self.assertRaises(ValueError):validated_selection(ROOT/'research/runs/absent',None)
        with self.assertRaises(FileNotFoundError):validated_selection(ROOT/'research/runs/absent',50)
        self.assertEqual(expected_steps(100),{'g':125,'d':107})


if __name__=='__main__':unittest.main(verbosity=2)
