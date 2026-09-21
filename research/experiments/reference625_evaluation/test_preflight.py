"""Small provenance fixtures only: no model construction or checkpoint loading."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('candidate625', HERE / 'evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original = self.root / 'research/runs/reference256-paper-b64'
        self.run = self.root / 'research/runs/reference256-paper-b64-resumed500-to625'
        self.previous = self.original / 'unseen32-000500'
        self.original.mkdir(parents=True)
        self.run.mkdir()
        self.previous.mkdir()
        baseline = json.loads(evaluation.ORIGINAL_CONFIG.read_text())
        self.write(self.original / 'config.json', baseline)
        self.config = copy.deepcopy(baseline)
        self.config['arguments'].update(run=str(self.run),
            resume=str(self.original / 'resume-000500.pt'), steps=625,
            data=str(self.root / 'research/alignment/collar-only/eyes42'),
            base=str(self.root / 'research/models/ffhq256.pkl'))
        self.write(self.run / 'config.json', self.config)
        for name in ('resume.pt', 'resume-000625.pt'):
            (self.run / name).write_bytes(b'FIXTURE ONLY: never torch-loaded')
        for name in ('train-source.py', 'adapters-source.py'):
            (self.original / name).write_text(name)
            (self.run / name).write_text(name)
        launcher = self.root / 'research/experiments/reference_resume500_625/launch.py'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('fixture launcher')
        (self.run / 'launch-source.py').write_bytes(launcher.read_bytes())
        (self.previous / 'latents.npz').write_bytes(b'fixture latents, not numpy-loaded')
        self.write(self.previous / 'evaluation.json', {'complete': True,
            'latents': {'sha256': evaluation.sha(self.previous / 'latents.npz')}})
        self.write(self.previous / 'supervisor-result.json', {'complete': True, 'worker_exit_code': 0})
        self.write(self.run / 'completed.json', {'step': 625, 'images_seen': 40000})
        self.launch = {'complete': True, 'child_exit_code': 0, 'start_step': 500, 'stop_step': 625,
            'source_sha256': evaluation.sha(launcher), 'outputs_sha256': {
                name: evaluation.sha(self.run / name) for name in ('resume.pt', 'config.json')}}
        self.write(self.run / 'launch.json', self.launch)
        self.patches = patch.multiple(evaluation, ROOT=self.root,
            CHECKPOINT=self.run / 'resume-000625.pt', CONFIG=self.run / 'config.json',
            ORIGINAL_CONFIG=self.original / 'config.json', PREVIOUS=self.previous,
            LATENTS=self.previous / 'latents.npz')
        self.patches.start()
        self.addCleanup(self.patches.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def verify(self):
        return evaluation.verify_future(evaluation.sha(self.run / 'resume-000625.pt'),
                                        evaluation.sha(self.run / 'config.json'))

    def test_complete_matching_fixture_passes_without_torch(self):
        self.assertEqual(self.verify()['checkpoint_sha256'], evaluation.sha(self.run / 'resume.pt'))
        self.assertNotIn('torch', sys.modules)

    def test_incomplete_training_is_rejected(self):
        self.launch['complete'] = False
        self.write(self.run / 'launch.json', self.launch)
        with self.assertRaisesRegex(RuntimeError, 'not successfully complete'):
            self.verify()

    def test_changed_dataset_recipe_is_rejected_even_with_new_config_hash(self):
        self.config['recipe']['dataset_count'] = 268
        self.config['recipe']['dataset_sha256'] = '2d50e93823001b08ae30eb4e0e1086050b4c8f6808cf791e57e60e176a44ebeb'
        self.write(self.run / 'config.json', self.config)
        with self.assertRaisesRegex(RuntimeError, 'config changed: recipe'):
            self.verify()

    def test_supplied_wrong_checkpoint_hash_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            evaluation.verify_future('0' * 64, evaluation.sha(self.run / 'config.json'))

    def test_changed_immutable_copy_is_rejected_against_completion(self):
        (self.run / 'resume-000625.pt').write_bytes(b'different fixture')
        with self.assertRaisesRegex(RuntimeError, 'differ from completed launch'):
            self.verify()

    def test_wrong_counter_is_rejected(self):
        self.write(self.run / 'completed.json', {'step': 624, 'images_seen': 39936})
        with self.assertRaisesRegex(RuntimeError, 'not625/40000'):
            self.verify()


class PythonVersionTests(unittest.TestCase):
    def test_pinned_python_version_mismatch_is_rejected_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            pins = Path(directory) / 'pins.json'
            pins.write_text(json.dumps({'python_version': 'deliberately different Python',
                                        'files': {}, 'versions': {}}))
            with patch.object(evaluation, 'PINS', pins):
                with self.assertRaisesRegex(RuntimeError, 'Pinned Python sys.version changed'):
                    evaluation.verify_pins()
        self.assertNotIn('torch', sys.modules)


class SavedLatentTests(unittest.TestCase):
    def test_actual_saved32_match_seeds_without_model_import(self):
        import numpy as np
        with np.load(evaluation.LATENTS, allow_pickle=False) as saved:
            z = saved['z']
            seeds = saved['seeds']
            fixed = saved['trainer_fixed_z']
        expected = np.stack([np.random.Generator(np.random.PCG64(seed)).standard_normal(512).astype(np.float32)
                             for seed in evaluation.SEEDS])
        self.assertTrue(np.array_equal(z, expected))
        self.assertTrue(np.array_equal(seeds, np.asarray(evaluation.SEEDS, dtype=np.uint64)))
        self.assertEqual(fixed.shape, (16, 512))
        self.assertEqual(len({row.tobytes() for row in z}), 32)
        self.assertFalse(any(np.array_equal(row, previous) for row in z for previous in fixed))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
