"""Tiny CPU fixtures only; never opens or unpickles pretrained weights."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extract_d',HERE/'extract_discriminator.py')
extract=importlib.util.module_from_spec(spec);spec.loader.exec_module(extract)


class GuardTests(unittest.TestCase):
    def test_import_and_help_do_not_import_torch_or_legacy(self):
        code=f"import runpy,sys; runpy.run_path({str(HERE/'extract_discriminator.py')!r},run_name='audit_import'); assert 'torch' not in sys.modules and 'legacy' not in sys.modules and 'pickle' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_no_execute_flag_cannot_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'must-not-exist'
            result=subprocess.run([sys.executable,str(HERE/'extract_discriminator.py'),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists())
            self.assertIn('No real pickle',result.stderr)

    def test_pressure_guard_fail_closed(self):
        self.assertEqual(extract.parse_free_percentage('System-wide memory free percentage: 25%'),25)
        for text in ['', 'System-wide memory free percentage: 101%']:
            with self.assertRaises(ValueError):extract.parse_free_percentage(text)
        with mock.patch.object(extract.sys,'platform','darwin'), mock.patch.object(extract.subprocess,'run',return_value=mock.Mock(stdout='System-wide memory free percentage: 24%')):
            with self.assertRaises(RuntimeError):extract.memory_guard()

    def test_constructor_rejects_unreviewed_shape_and_payload(self):
        valid={'c_dim':0,'img_resolution':1024,'img_channels':3}
        self.assertEqual(extract.validate_kwargs([],valid)[1],valid)
        for updates in [{'img_resolution':256},{'c_dim':1},{'channel_base':16384},{'block_kwargs':{'freeze_layers':4}},{'unknown':1}]:
            with self.assertRaises(ValueError):extract.validate_kwargs([],dict(valid,**updates))
        with self.assertRaises(ValueError):extract.json_value(float('nan'))
        with self.assertRaises(ValueError):extract.json_value(object())

    def test_pinned_vendor_and_source_identity_constant(self):
        pins=extract.verify_vendor()
        self.assertEqual(pins['training/networks.py'],'df49fc78a0364015a09b592bb43546af83a6cdaad753d89a2a1a69bd6e171c24')
        self.assertEqual(extract.SOURCE_SHA256,'a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c')


class TinyStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        cls.torch=torch
        sys.path.insert(0,str(extract.VENDOR))
        from training.networks import Discriminator
        cls.Constructor=Discriminator

    def test_exact_reconstruction_prefix_and_safetensors(self):
        from safetensors.torch import load_file,save_file
        t=self.torch
        source=self.Constructor(c_dim=0,img_resolution=16,img_channels=3,channel_base=64,channel_max=16).eval().requires_grad_(False)
        d,args,kwargs=extract.reconstruct(source,self.Constructor,t,strict_ffhq=False)
        original=extract.state_inventory(d,t)
        with mock.patch.object(d,'forward',side_effect=AssertionError('Full D forward must not execute')),mock.patch.object(d.b4,'forward',side_effect=AssertionError('Epilogue must not execute')):
            result=extract.check_prefix(d,t,resolution=16,blocks=(16,8),expected=((1,8,8,8),(1,16,4,4)))
        self.assertTrue(result['frozen_state_unchanged']);self.assertEqual(original,extract.state_inventory(d,t))
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in d.parameters()))
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'D.safetensors';save_file({k:v.contiguous() for k,v in d.state_dict().items()},str(p))
            restored=self.Constructor(*args,**kwargs);restored.load_state_dict(load_file(str(p)),strict=True)
            self.assertEqual(extract.state_inventory(restored,t),original)
        with t.no_grad():next(d.parameters()).flatten()[0]=float('nan')
        with self.assertRaises(ValueError):extract.state_inventory(d,t)


if __name__=='__main__':unittest.main(verbosity=2)
