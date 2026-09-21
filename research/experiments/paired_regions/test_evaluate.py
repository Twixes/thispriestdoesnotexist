"""Lightweight region-checkpoint guards; never imports torch or loads a model."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.experiments.paired_regions import evaluate


class RegionEvaluatorGuards(unittest.TestCase):
    def setUp(self):
        self.metadata = {'schema_version': 1, 'truncation_psi': 1, 'weights_sha256': 'weights',
                         'init_kwargs': {'img_resolution': 1024, 'z_dim': 512}}
        provenance = {'source_weights_sha256': 'weights', 'source_metadata_sha256': 'metadata'}
        self.checkpoint = {'format': 'paired-regions-research-v1', 'production_approved': False, 'step': 200,
                           'init_kwargs': copy.deepcopy(self.metadata['init_kwargs']),
                           'provenance': provenance, 'options': {'seed': 1}}
        self.configuration = {'total_steps': 200, 'provenance': copy.deepcopy(provenance), 'options': {'seed': 1}}
        self.samples = {'model': {'model_sha256': 'weights'}}

    def verify(self):
        evaluate.verify_metadata(self.metadata, self.checkpoint, self.configuration, self.samples, 'weights', 'metadata')

    def test_completed_regions_format_accepted_without_torch(self):
        self.verify()
        self.assertNotIn('torch', sys.modules)

    def test_broad_mask_or_production_checkpoint_rejected(self):
        self.checkpoint['format'] = 'paired-edit-research-v1'
        with self.assertRaisesRegex(ValueError, 'broad-mask'):
            self.verify()
        self.checkpoint['format'] = 'paired-regions-research-v1'
        self.checkpoint['production_approved'] = True
        with self.assertRaises(ValueError):
            self.verify()

    def test_intermediate_step_rejected(self):
        self.checkpoint['step'] = 100
        with self.assertRaisesRegex(ValueError, 'completed configured training'):
            self.verify()

    def test_architecture_source_and_config_mismatch_rejected(self):
        for target, key, value in [
            (self.checkpoint, 'init_kwargs', {'img_resolution': 256}),
            (self.metadata, 'weights_sha256', 'changed'),
            (self.configuration, 'options', {'seed': 2}),
            (self.samples, 'model', {'model_sha256': 'another-model'}),
        ]:
            with self.subTest(key=key):
                saved = copy.deepcopy(target[key]); target[key] = value
                with self.assertRaises(ValueError):
                    self.verify()
                target[key] = saved

    def test_complete_file_inventory_and_imported_helpers_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = ['research/experiments/paired_regions/trainer.py',
                     'research/experiments/paired_edit/trainer.py',
                     'research/vendor/stylegan2-ada-pytorch/training/networks.py',
                     'research/vendor/stylegan2-ada-pytorch/torch_utils/ops/filter.cu',
                     'research/vendor/stylegan2-ada-pytorch/dnnlib/util.py']
            for name in files:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            recorded = evaluate.current_source_hashes(root)
            evaluate.verify_code_provenance(recorded, evaluate.current_source_hashes(root))
            self.assertIn('paired_edit_helpers', recorded)
            self.assertIn('regions_trainer', recorded)
            self.assertIn('torch_utils/ops/filter.cu', recorded)
            helper = root / files[1]
            helper.write_text('changed helper')
            with self.assertRaisesRegex(ValueError, 'paired_edit_helpers'):
                evaluate.verify_code_provenance(recorded, evaluate.current_source_hashes(root))
            helper.write_text(files[1])
            added = root / 'research/vendor/stylegan2-ada-pytorch/training/new.py'; added.write_text('new')
            with self.assertRaisesRegex(ValueError, 'added='):
                evaluate.verify_code_provenance(recorded, evaluate.current_source_hashes(root))
            added.unlink()
            (root / files[-1]).unlink()
            with self.assertRaisesRegex(ValueError, 'missing='):
                evaluate.verify_code_provenance(recorded, evaluate.current_source_hashes(root))


if __name__ == '__main__':
    unittest.main()
