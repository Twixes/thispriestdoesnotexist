"""Lightweight paired split/provenance guards; no torch/model compute."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.experiments.paired_regions import evaluate_pairs


class PairedEvaluationGuards(unittest.TestCase):
    def setUp(self):
        pairs=[{'id':f'train-{index}', 'split':'train'} for index in range(6)]
        pairs.append({'id':'028','split':'validation'})
        self.manifest={'version':2,'production_approved':False,
                       'requires_trainer_feature':'equal_area_collar_partition_v1','pairs':pairs}
        self.checkpoint={'step':600,'provenance':{'manifest_sha256':'exact','pairs':copy.deepcopy(pairs)},
                         'options':{'source_max_uint8_error':0,'w_atol':0}}

    def verify(self):
        evaluate_pairs.verify_paired_manifest(self.manifest,'exact',self.checkpoint)

    def test_exact_split_accepted_without_torch(self):
        self.verify()
        self.assertNotIn('torch',sys.modules)
        self.assertTrue(evaluate_pairs.split_label('train').startswith('TRAIN'))
        self.assertTrue(evaluate_pairs.split_label('validation').startswith('HOLDOUT'))

    def test_intermediate_step_and_changed_manifest_rejected(self):
        self.checkpoint['step']=500
        with self.assertRaisesRegex(ValueError,'completed step'):
            self.verify()
        self.checkpoint['step']=600
        with self.assertRaisesRegex(ValueError,'checksum'):
            evaluate_pairs.verify_paired_manifest(self.manifest,'changed',self.checkpoint)

    def test_holdout_cannot_be_reclassified_as_train_or_different_identity(self):
        self.manifest['pairs'][-1]['split']='train'
        with self.assertRaisesRegex(ValueError,'six training'):
            self.verify()
        self.manifest['pairs'][-1]['split']='validation'
        self.manifest['pairs'][-1]['id']='other'
        with self.assertRaisesRegex(ValueError,'holdout'):
            self.verify()

    def test_duplicate_unsafe_and_checkpoint_split_mismatch_rejected(self):
        for value in ['train-1','../unsafe']:
            self.manifest['pairs'][0]['id']=value
            with self.assertRaisesRegex(ValueError,'Unsafe or repeated'):
                self.verify()
        self.manifest['pairs'][0]['id']='train-0'
        self.checkpoint['provenance']['pairs'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'split/order'):
            self.verify()

    def test_zero_tolerance_and_exact_reconstructed_provenance_required(self):
        for key in ['source_max_uint8_error','w_atol']:
            self.checkpoint['options'][key]=1
            with self.assertRaisesRegex(ValueError,'zero-tolerance'):
                self.verify()
            self.checkpoint['options'][key]=0
        recorded=[{'files':{'source_path':{'sha256':'original'}},'partition':{'collar_pixels':123}}]
        evaluate_pairs.verify_pair_provenance(copy.deepcopy(recorded),recorded)
        changed=copy.deepcopy(recorded);changed[0]['partition']['collar_pixels']=124
        with self.assertRaisesRegex(ValueError,'provenance'):
            evaluate_pairs.verify_pair_provenance(changed,recorded)


if __name__=='__main__':
    unittest.main()
