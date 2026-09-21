"""Lightweight provenance/statistics checks; no torch import or model inference."""
import copy
import unittest

import numpy as np

from evaluate import latent_digest, region_errors, verify_metadata


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.metadata = {'schema_version': 1, 'truncation_psi': 1, 'weights_sha256': 'weights',
                         'init_kwargs': {'img_resolution': 1024, 'z_dim': 512}}
        provenance = {'source_weights_sha256': 'weights', 'source_metadata_sha256': 'metadata'}
        self.checkpoint = {'format': 'paired-edit-research-v1', 'production_approved': False, 'step': 200,
                           'init_kwargs': copy.deepcopy(self.metadata['init_kwargs']), 'provenance': provenance,
                           'options': {'seed': 1}}
        self.configuration = {'total_steps': 200, 'provenance': copy.deepcopy(provenance), 'options': {'seed': 1}}
        self.samples = {'model': {'model_sha256': 'weights'}}

    def verify(self):
        return verify_metadata(self.metadata, self.checkpoint, self.configuration, self.samples, 'weights', 'metadata')

    def test_valid_completed_provenance(self):
        self.verify()

    def test_active_checkpoint_is_rejected(self):
        self.checkpoint['step'] = 100
        with self.assertRaisesRegex(ValueError, 'completed configured training'):
            self.verify()

    def test_mismatched_model_and_architecture_rejected(self):
        for field, value in [('weights_sha256', 'other'), ('init_kwargs', {'img_resolution': 256})]:
            with self.subTest(field=field):
                saved = copy.deepcopy(self.metadata)
                self.metadata[field] = value
                with self.assertRaises(ValueError):
                    self.verify()
                self.metadata = saved

    def test_source_samples_from_different_model_rejected(self):
        self.samples['model']['model_sha256'] = 'different-source'
        with self.assertRaisesRegex(ValueError, 'different source model'):
            self.verify()

    def test_checkpoint_provenance_or_configuration_mismatch_rejected(self):
        self.checkpoint['provenance']['source_metadata_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source provenance'):
            self.verify()
        self.checkpoint['provenance']['source_metadata_sha256'] = 'metadata'
        self.configuration['options']['seed'] = 2
        with self.assertRaisesRegex(ValueError, 'configuration/checkpoint'):
            self.verify()

    def test_production_artifact_rejected(self):
        self.checkpoint['production_approved'] = True
        with self.assertRaisesRegex(ValueError, 'unapproved'):
            self.verify()

    def test_horizontal_regions_and_unclipped_error(self):
        source = np.zeros((4, 2), dtype=np.float32)
        student = np.array([[.25, .25], [.25, .25], [.25, .25], [2, 2]], dtype=np.float32)
        result = region_errors(source, student)
        self.assertEqual(result['top75_mae_0_1'], .25)
        self.assertEqual(result['bottom25_mae_0_1'], 2)
        self.assertEqual(result['full_mae_0_1'], .6875)
        self.assertEqual(result['max_absolute_difference_0_1'], 2)
        with self.assertRaises(ValueError):
            region_errors(source, student[:2])

    def test_latent_content_digest_ignores_npz_container(self):
        z = np.arange(8, dtype=np.float32).reshape(1, 8)
        self.assertEqual(latent_digest(z), latent_digest(z.copy()))
        changed = z.copy(); changed[0, 0] += .01
        self.assertNotEqual(latent_digest(z), latent_digest(changed))


if __name__ == '__main__':
    unittest.main()
