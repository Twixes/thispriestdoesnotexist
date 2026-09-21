"""Tiny CPU tests: no real weights, image generation, or live trainer import."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import unittest

import torch
from torch import nn

from losses import FrozenDPrefix, collar_boundary_loss, feature_reconstruction_loss, region_mean

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parents[1]/'vendor/stylegan2-ada-pytorch'
sys.path.insert(0, str(VENDOR))
from training.networks import Discriminator

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class IdentityFeatures(nn.Module):
    def forward(self, x):
        return [x]


class ForbiddenEpilogue(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError('Minibatch-statistics epilogue must never be called')


class LossTests(unittest.TestCase):
    def fixtures(self, size=16, batch=1):
        rng = torch.Generator().manual_seed(27)
        y = torch.randn(batch,3,size,size,generator=rng).requires_grad_()
        target = torch.randn(batch,3,size,size,generator=rng).requires_grad_()
        source = torch.randn(batch,3,size,size,generator=rng).requires_grad_()
        mask = torch.zeros(batch,1,size,size); mask[:,:,size//2:,:] = 1
        collar = torch.zeros_like(mask); collar[:,:,3*size//4:,size//3:2*size//3] = 1
        return y,target,source,mask.requires_grad_(),collar.requires_grad_()

    def assert_gradients(self, y,t,s,m,c):
        self.assertTrue(bool(torch.isfinite(y.grad).all()))
        self.assertGreater(float((y.grad*m.detach()).abs().sum()),0)
        self.assertEqual(float((y.grad*(1-m.detach())).abs().sum()),0)
        self.assertIsNone(t.grad); self.assertIsNone(s.grad)
        self.assertIsNone(m.grad); self.assertIsNone(c.grad)

    def test_real_vendor_prefix_gradients_and_preservation(self):
        torch.manual_seed(8)
        d = Discriminator(c_dim=0,img_resolution=64,img_channels=3,channel_base=256,
                          channel_max=16,num_fp16_res=0)
        d.b4 = ForbiddenEpilogue()
        before = {k:v.clone() for k,v in d.state_dict().items()}
        prefix = FrozenDPrefix(d,input_resolution=64,tap_blocks=(32,16))
        y,t,s,m,c = self.fixtures(64)
        originals = [x.detach().clone() for x in [y,t,s,m,c]]
        loss,metrics = feature_reconstruction_loss(prefix,y,t,s,m,c)
        self.assertGreater(float(loss.detach()),0)
        self.assertEqual([a['shape'][2:] for a in metrics['layers']],[[16,16],[8,8]])
        loss.backward()
        self.assert_gradients(y,t,s,m,c)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in d.parameters()))
        for k,v in d.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
        for x,b in zip([y,t,s,m,c], originals): self.assertTrue(torch.equal(x.detach(),b))
        self.assertEqual(len(prefix._forward_hooks),0)
        self.assertEqual([*prefix.blocks],['64','32','16'])

    def test_unequal_area_normalization_and_feature_partition(self):
        y = torch.ones(1,1,8,8,requires_grad=True)
        z = torch.zeros_like(y)
        m = torch.zeros_like(y); m[:,:,4:,:] = 1
        c = torch.zeros_like(y); c[:,:,4,0] = 1
        loss,_ = feature_reconstruction_loss(IdentityFeatures().eval(),y,z,z,m,c)
        self.assertEqual(float(loss.detach()),1)
        loss.backward()
        self.assertAlmostEqual(float((y.grad*c).sum()),.5,places=6)
        self.assertAlmostEqual(float((y.grad*(m-c)).sum()),.5,places=6)
        self.assertEqual(float((y.grad*(1-m)).sum()),0)
        values = torch.ones(2,3,8,8,requires_grad=True)
        masks = torch.ones(2,1,8,8); masks[0]=0; masks[0,0,0,0]=.125
        losses = region_mean(values,masks)
        self.assertTrue(torch.equal(losses,torch.ones(2)))
        losses.mean().backward()
        self.assertAlmostEqual(float(values.grad[0].sum()),.5,places=6)
        self.assertAlmostEqual(float(values.grad[1].sum()),.5,places=6)

    def test_boundary_gradients_detach_and_image_preservation(self):
        y,t,s,m,c = self.fixtures()
        saved = [x.detach().clone() for x in [y,t,s,m,c]]
        loss,metrics = collar_boundary_loss(y,t,m,c,radius=2)
        self.assertGreater(metrics['valid_edges_per_image'][0],0)
        loss.backward(); self.assert_gradients(y,t,s,m,c)
        for x,b in zip([y,t,s,m,c],saved): self.assertTrue(torch.equal(x.detach(),b))

    def test_boundary_zero_for_matching_differences(self):
        y,t,s,m,c = self.fixtures()
        # Constant intensity offsets have identical first differences.
        loss,_ = collar_boundary_loss(t.detach()+.25,t,m,c,radius=2)
        self.assertLess(float(loss.detach()),1e-6)

    def test_invalid_masks_and_nonfinite_inputs(self):
        y,t,s,m,c = self.fixtures()
        for bad in [torch.zeros_like(c),m.detach().clone(),1-m.detach()]:
            with self.assertRaises(ValueError): feature_reconstruction_loss(IdentityFeatures().eval(),y,t,s,m,bad)
            with self.assertRaises(ValueError): collar_boundary_loss(y,t,m,bad)
        for bad in [c.detach()*.5,torch.full_like(c,float('nan'))]:
            with self.assertRaises(ValueError): collar_boundary_loss(y,t,m,bad)
        with self.assertRaises(ValueError): feature_reconstruction_loss(IdentityFeatures().eval(),y*float('nan'),t,s,m,c)
        with self.assertRaises(ValueError): region_mean(y,torch.zeros_like(m))

    def test_boundary_empty_edges_rejected(self):
        y = torch.zeros(1,1,8,8)
        m = torch.zeros_like(y); m[:,:,1,1]=1; m[:,:,6,6]=1
        c = torch.zeros_like(y); c[:,:,1,1]=1
        with self.assertRaises(ValueError): collar_boundary_loss(y,y,m,c,radius=1)

    def test_prefix_rejects_unfrozen_or_train_mode_and_bad_taps(self):
        d=Discriminator(c_dim=0,img_resolution=64,img_channels=3,channel_base=64,channel_max=4)
        with self.assertRaises(ValueError): FrozenDPrefix(d,input_resolution=1024)
        with self.assertRaises(ValueError): FrozenDPrefix(d,input_resolution=64,tap_blocks=(4,))
        prefix=FrozenDPrefix(d,input_resolution=64,tap_blocks=(32,16))
        prefix.train()
        with self.assertRaises(ValueError): prefix(torch.zeros(1,3,64,64))
        prefix.eval(); next(prefix.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError): prefix(torch.zeros(1,3,64,64))


if __name__ == '__main__':
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LossTests))
    files=[HERE/'losses.py',Path(__file__)]
    for directory in ['training','torch_utils','dnnlib']:
        files += [p for p in sorted((VENDOR/directory).rglob('*'))
                  if p.is_file() and p.suffix in ['.py','.cpp','.cu','.h','.hpp']]
    evidence={'passed':result.wasSuccessful(),'tests':result.testsRun,'seconds':time.perf_counter()-started,
              'device':'cpu','threads':1,'interop_threads':1,'real_weights_loaded':False,
              'largest_test_resolution':64,'max_discriminator_channels':16,'mps_used':False,
              'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'torch_version':str(torch.__version__),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (HERE/'test-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
