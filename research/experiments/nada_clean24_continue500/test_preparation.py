"""Small data/source/policy checks; no torch/model imports or training."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = HERE.with_name('nada_clean24')
spec = importlib.util.spec_from_file_location('continuation_runner', HERE/'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Preparation(unittest.TestCase):
    def test_pins_and_lazy_import(self):
        pins = runner.verify_pins()
        for name in ['runner.py', 'worker.py', 'inputs.json', 'latents.npz']:
            self.assertIn(str((HERE/name).relative_to(ROOT)), pins['files_sha256'])
        self.assertNotIn('torch', sys.modules)

    def test_exact_recipe_inputs_and_disjoint_fresh_latents(self):
        old=json.loads((ORIGINAL/'inputs.json').read_text())
        new=json.loads((HERE/'inputs.json').read_text())
        self.assertEqual(old['targets24'], new['targets24'])
        self.assertEqual(old['source32'], new['source32'])
        with np.load(ORIGINAL/'latents.npz',allow_pickle=False) as a, np.load(HERE/'latents.npz',allow_pickle=False) as b:
            for name in ['source_z','source_w','eval_z']:
                self.assertTrue(np.array_equal(a[name],b[name]))
            self.assertTrue(np.array_equal(a['train_z'],b['previous_train_z']))
            fresh=np.random.Generator(np.random.PCG64(202609210800)).standard_normal((490,512)).astype(np.float32)
            self.assertTrue(np.array_equal(fresh,b['train_z']))
            self.assertEqual(len({row.tobytes() for name in ['source_z','previous_train_z','train_z','eval_z'] for row in b[name]}),540)
            self.assertEqual({k:list(b[k].shape) for k in b.files},new['array_shapes'])
            self.assertTrue(all(b[k].dtype==np.float32 and np.isfinite(b[k]).all() for k in b.files))
        self.assertEqual(runner.sha(HERE/'latents.npz'),new['latents_npz_sha256'])

    def test_math_and_rng_helpers_identical_to_successful_smoke(self):
        def functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        a,b=functions(ORIGINAL/'worker.py'),functions(HERE/'worker.py')
        for name in ['unit','encode','generate','rng','restore','nested_hash','frozen_checks']:
            self.assertEqual(a[name],b[name],name)

    def test_competing_processes_refused_and_owned_worker_excluded(self):
        for command in ['reference625_validation/validate.py --check-static',
                        'nada_clean24_overlap/supervise.py --execute',
                        'nada_clean24_continue500/runner.py --execute']:
            text=f'999998 python /repo/research/experiments/{command}\n'
            with patch.object(runner.subprocess,'check_output',return_value=text):
                with self.assertRaisesRegex(RuntimeError,'another known'):
                    runner.concurrency_guard()
                runner.concurrency_guard(owned_pid=999998)

    def test_34percent_refused(self):
        with patch.object(runner.subprocess,'check_output',return_value='System-wide memory free percentage: 34%'):
            with self.assertRaisesRegex(RuntimeError,'35%'):
                runner.memory_guard()

    def test_sources_compile_without_importing_models(self):
        for name in ['runner.py','worker.py']:
            compile((HERE/name).read_text(),str(HERE/name),'exec')
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)
