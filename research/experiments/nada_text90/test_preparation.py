"""Data/math checks only. Never start a child or import model libraries."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.with_name('nada_clean24_continue500')
spec=importlib.util.spec_from_file_location('text_probe_guards',HERE/'runner.py')
guards=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guards)


class Preparation(unittest.TestCase):
    def test_actual_input_pins_and_no_model_import(self):
        pins=guards.verify_pins()
        for name in ['runner.py','worker.py','inputs.json','latents.npz']:
            self.assertIn(str((HERE/name).relative_to(ROOT)),pins['files_sha256'])
        self.assertNotIn('torch',sys.modules)
        self.assertEqual(guards.MAX_SECONDS,1200)
        self.assertEqual(guards.MAX_RSS,12*1024**3)

    def test_matched90_latents_and_same_targets(self):
        old=json.loads((BASE/'inputs.json').read_text())
        new=json.loads((HERE/'inputs.json').read_text())
        self.assertEqual(old['targets24'],new['targets24'])
        self.assertEqual(old['source32'],new['source32'])
        self.assertEqual(old['continuation']['checkpoint_sha256'],new['continuation']['checkpoint_sha256'])
        with np.load(BASE/'latents.npz',allow_pickle=False) as a,np.load(HERE/'latents.npz',allow_pickle=False) as b:
            self.assertTrue(np.array_equal(a['train_z'][:90],b['train_z']))
            for name in ['source_z','source_w','previous_train_z','eval_z']:
                self.assertTrue(np.array_equal(a[name],b[name]))
            self.assertEqual(len({row.tobytes() for name in ['source_z','previous_train_z','train_z','eval_z'] for row in b[name]}),140)
            self.assertTrue(all(b[k].dtype==np.float32 and np.isfinite(b[k]).all() for k in b.files))
            self.assertEqual({k:list(b[k].shape) for k in b.files},new['array_shapes'])
        self.assertEqual(guards.sha(HERE/'latents.npz'),new['latents_npz_sha256'])

    def test_image_math_gradients_and_resume_helpers_unchanged(self):
        def funcs(path):
            return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
        before,after=funcs(BASE/'worker.py'),funcs(HERE/'worker.py')
        for name in ['unit','encode','generate','rng','restore','nested_hash','frozen_checks','snapshot','save_checkpoint']:
            self.assertEqual(ast.dump(before[name]),ast.dump(after[name]),name)
        # Every update operation is identical; only checkpoint schedule changes.
        def update_loop(path):
            tree=ast.parse(path.read_text())
            loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(index, z)')
            return ast.dump(ast.Module(body=loop.body[:-1],type_ignores=[]))
        self.assertEqual(update_loop(BASE/'worker.py'),update_loop(HERE/'worker.py'))

    def test_fixed_prompts_and_sources_compile(self):
        spec=json.loads((HERE/'inputs.json').read_text())['text_guidance']
        self.assertEqual(spec['source_class'],'a man wearing an ordinary shirt')
        self.assertEqual(spec['target_class'],'a man wearing a black clerical shirt with a white Roman collar')
        self.assertEqual(len(spec['templates']),4)
        self.assertTrue(all('photo' in t and t.count('{}')==1 for t in spec['templates']))
        for name in ['runner.py','worker.py']:
            compile((HERE/name).read_text(),str(HERE/name),'exec')
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)
