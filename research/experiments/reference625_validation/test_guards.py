"""Provenance/refusal fixtures only; no tensor checkpoint load or fake numeric pass."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validator625', HERE / 'validate.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads((validator.ORIGINAL / 'config.json').read_text())
        self.config = copy.deepcopy(self.baseline)
        self.config['arguments'].update(run=str(validator.RUN), steps=625,
            resume=str(validator.ORIGINAL / 'resume-000500.pt'),
            data=str(validator.ROOT / 'research/alignment/collar-only/eyes42'),
            base=str(validator.ROOT / 'research/models/ffhq256.pkl'))
        self.completed = {'step': 625, 'images_seen': 40000}
        self.launch = {'complete': True, 'child_exit_code': 0, 'start_step': 500, 'stop_step': 625,
                       'outputs_sha256': {name: 'fixture-not-validated' for name in validator.OUTPUT_NAMES}}

    def check(self):
        validator.check_completion(self.completed, self.launch, self.config, self.baseline)

    def test_expected_completion_metadata_accepted_without_model_import(self):
        self.check()
        self.assertNotIn('torch', sys.modules)

    def test_incomplete_launch_rejected(self):
        self.launch['complete'] = False
        with self.assertRaisesRegex(RuntimeError, 'terminal-success'):
            self.check()

    def test_wrong625_counter_rejected(self):
        self.completed['images_seen'] = 39936
        with self.assertRaisesRegex(RuntimeError, '625/40000'):
            self.check()

    def test_expanded_dataset_rejected(self):
        self.config['recipe']['dataset_count'] = 268
        with self.assertRaisesRegex(RuntimeError, 'config changed: recipe'):
            self.check()

    def test_missing_export_hash_rejected(self):
        del self.launch['outputs_sha256']['generator-000625.pt']
        with self.assertRaisesRegex(RuntimeError, 'output hash set'):
            self.check()


class ProcessTests(unittest.TestCase):
    def test_live_trainer_other_evaluator_and_unknown_research_runner_detected(self):
        processes = '\n'.join([
            '10 python own/reference625_validation/validate.py',
            '11 python own/reference625_validation/validate.py --_worker',
            '12 python renamed-process.py',
            '13 python research/experiments/reference625_evaluation/evaluate.py',
            '14 python research/experiments/new_experiment/runner.py',
            '15 python research/reviews/report.py'])
        found = validator.active_model_processes(processes, 12, {10, 11})
        self.assertEqual([entry['pid'] for entry in found], [12, 13, 14])


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.preserved = self.run / 'resume-000625.pt'
        self.copying = self.run / 'resume-000625.pt.copying'
        self.source = self.run / 'resume.pt'
        self.source.write_bytes(b'ordinary byte-copy fixture; not a torch checkpoint\n')
        self.patches = patch.multiple(validator, RUN=self.run, PRESERVED=self.preserved, COPYING=self.copying)
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_byte_copy_primitive_publishes_exactly_without_numeric_claim(self):
        expected = validator.sha(self.source)
        self.assertEqual(validator.capture_after_validation(expected), expected)
        self.assertEqual(self.preserved.read_bytes(), self.source.read_bytes())
        self.assertFalse(self.copying.exists())
        self.assertNotIn('torch', sys.modules)

    def test_existing_final_never_overwritten(self):
        self.preserved.write_bytes(b'preserve me')
        with self.assertRaisesRegex(RuntimeError, 'existing preserved'):
            validator.capture_after_validation(validator.sha(self.source))
        self.assertEqual(self.preserved.read_bytes(), b'preserve me')

    def test_existing_partial_never_overwritten(self):
        self.copying.write_bytes(b'inspect partial')
        with self.assertRaisesRegex(RuntimeError, 'partial copying'):
            validator.capture_after_validation(validator.sha(self.source))
        self.assertEqual(self.copying.read_bytes(), b'inspect partial')

    def test_wrong_hash_retains_partial_without_publishing(self):
        with self.assertRaisesRegex(RuntimeError, 'digest changed'):
            validator.capture_after_validation('0' * 64)
        self.assertFalse(self.preserved.exists())
        self.assertEqual(self.copying.read_bytes(), self.source.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
