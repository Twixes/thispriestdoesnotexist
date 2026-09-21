"""Stdlib-only refusal/pinning fixtures. Never import Torch or create models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

spec=importlib.util.spec_from_file_location('restoration_diagnostic',Path(__file__).with_name('diagnostic.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class Refusals(unittest.TestCase):
    def test_incomplete_nada_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name,value in [('supervisor-result.json',{'complete':False,'worker_exit_code':1}),('worker-result.json',{}),('checkpoint-500-review-ready.json',{})]:
                (p/name).write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'supervisor'): d.nada_completion(p,'a'*64)
    def test_nada_missing_supplied_digest(self):
        with self.assertRaisesRegex(RuntimeError,'independently reviewed'): d.nada_completion(Path('/absent'),None)
    def test_network_refused(self):
        with self.assertRaisesRegex(RuntimeError,'Network is disabled'): d.forbid_network('example.com')
    def test_memory_boundary(self):
        with patch.object(d.subprocess,'check_output',return_value='System-wide memory free percentage: 34%'):
            with self.assertRaisesRegex(RuntimeError,'35%'):d.memory_guard()
        with patch.object(d.subprocess,'check_output',return_value='System-wide memory free percentage: 35%'):
            self.assertEqual(d.memory_guard()['free_percent'],35)
    def test_active_training_refused(self):
        row=f'999999 2 {d.ROOT}/research/.venv/bin/python {d.ROOT}/research/experiments/example/runner.py --execute'
        with patch.object(d.subprocess,'check_output',return_value=row):
            with self.assertRaisesRegex(RuntimeError,'Other model work'):d.process_guard()
    def test_source_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'file').write_text('changed')
            with patch.object(d,'ROOT',p),patch.object(d,'read',return_value={'python_version':sys.version,'packages':{},'files':{'file':'a'*64}}):
                with self.assertRaisesRegex(RuntimeError,'Pinned file changed'):d.pins_check()
    def test_no_heavy_imports(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)
    def test_resource_constants(self):
        self.assertEqual(d.LIMIT,6*1024**3);self.assertEqual(d.SECONDS,600)

if __name__=='__main__':unittest.main(verbosity=2)
