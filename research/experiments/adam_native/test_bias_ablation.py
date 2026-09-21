"""Tiny offset-support tests only; no native networks or model checkpoints loaded."""
import unittest

import torch

from bias_ablation import ARMS, FAMILY_BY_ARM, apply_offset_arm, offset_families, reset_names_for_arm


class TinyOffsets:
    def __init__(self, values):
        self.values = {name: tensor.clone() for name, tensor in values.items()}

    def get_parameter(self, name):
        return self.values[name]


class BiasAblationSupport(unittest.TestCase):
    def setUp(self):
        self.names = {
            'noise': 'synthesis.b1024.conv0.parametrizations.noise_strength.0.b_vector',
            'torgb_bias': 'synthesis.b1024.torgb.parametrizations.bias.0.b_vector',
            'synthesis_bias': 'synthesis.b1024.conv0.parametrizations.bias.0.b_vector',
            'style_affine_bias': 'synthesis.b1024.torgb.affine.parametrizations.bias.0.b_vector',
            'mapping_bias': 'mapping.fc0.parametrizations.bias.0.b_vector',
        }
        self.offsets = {name: torch.tensor([i + 1., -i - 1.]) for i, name in enumerate(self.names.values())}
        self.originals = {
            name.replace('.0.b_vector', '.original'): torch.tensor([4., 5.]) for name in self.names.values()
        }
        self.originals['synthesis.b1024.conv0.parametrizations.weight.0.u_vector'] = torch.tensor([6., 7.])
        self.originals['synthesis.b1024.conv0.parametrizations.weight.0.v_vector'] = torch.tensor([8., 9.])
        self.all_values = {**self.offsets, **self.originals}

    def test_family_partition_and_weight_source_exclusions(self):
        groups = offset_families(self.all_values)
        self.assertEqual(groups, {family: [name] for family, name in self.names.items()})
        conv_affine = 'synthesis.b8.conv1.affine.parametrizations.bias.0.b_vector'
        self.assertEqual(offset_families([conv_affine])['style_affine_bias'], [conv_affine])

    def test_each_arm_changes_exact_support_and_restores_previous_arm(self):
        model = TinyOffsets(self.all_values)
        groups = offset_families(self.all_values)
        for arm in ARMS:
            resets = reset_names_for_arm(groups, arm)
            expected = {self.names['noise']}
            if arm != ARMS[0]:
                expected.add(self.names[FAMILY_BY_ARM[arm]])
            self.assertEqual(set(resets), expected)
            apply_offset_arm(model, self.offsets, resets)
            changed = {name for name, tensor in model.values.items() if not torch.equal(tensor, self.all_values[name])}
            self.assertEqual(changed, expected)
            for name in expected:
                self.assertEqual(torch.count_nonzero(model.values[name]).item(), 0)
        apply_offset_arm(model, self.offsets, [])
        self.assertTrue(all(torch.equal(t, self.all_values[n]) for n, t in model.values.items()))

    def test_unknown_offset_or_arm_fails_closed(self):
        with self.assertRaises(ValueError):
            offset_families(['unexpected.parametrizations.bias.0.b_vector'])
        with self.assertRaises(ValueError):
            reset_names_for_arm(offset_families(self.all_values), 'unknown')
        with self.assertRaises(ValueError):
            apply_offset_arm(TinyOffsets(self.all_values), self.offsets, [next(iter(self.originals))])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
