"""Analytical metadata/vector fixtures only; no model construction or loading."""
import copy
import json
import unittest

import torch

from selection import build_score_inventory, select_rows


def inventory_fixture(component, resolution=16):
    inventory = {'component': component, 'conv_layout': 'output_rank1', 'targets': [],
                 'original_parameter_names': [], 'ordinary_trainable_names': []}
    def add(module, tensor, shape):
        inventory['original_parameter_names'].append(module + '.' + tensor)
        inventory['targets'].append({'module': module, 'tensor': tensor, 'shape': shape,
            'layout': 'output_rank1' if tensor == 'weight' else None,
            'mechanism': 'multiplicative_rank_factors' if tensor == 'weight' else 'additive_offset',
            'base_parameter': module + '.parametrizations.' + tensor + '.original'})
    def weight(module, shape, bias=True, noise=False):
        add(module, 'weight', shape)
        if bias: add(module, 'bias', [shape[0]])
        if noise: add(module, 'noise_strength', [])
    if component == 'G':
        for i in range(2): weight(f'mapping.fc{i}', [2, 2])
        inventory['original_parameter_names'].append('synthesis.b4.const')
        current = 4
        while current <= resolution:
            for layer in (('conv1', 'torgb') if current == 4 else ('conv0', 'conv1', 'torgb')):
                module = f'synthesis.b{current}.{layer}'
                weight(module, [3 if layer == 'torgb' else 2, 2, 1 if layer == 'torgb' else 3,
                                1 if layer == 'torgb' else 3], noise=layer != 'torgb')
                weight(module + '.affine', [2, 2])
            current *= 2
    else:
        current = 8
        while current <= resolution:
            for layer in ('conv0', 'conv1', 'skip'):
                weight(f'b{current}.{layer}', [2, 2, 1 if layer == 'skip' else 3, 1 if layer == 'skip' else 3],
                       bias=layer != 'skip')
            current *= 2
        weight(f'b{resolution}.fromrgb', [2, 3, 1, 1])
        for layer in ('conv', 'fc', 'out'):
            for tensor in ('weight', 'bias'):
                name = f'b4.{layer}.{tensor}'
                inventory['original_parameter_names'].append(name)
                inventory['ordinary_trainable_names'].append(name)
    return inventory


def means_fixture(inventory):
    descriptor = build_score_inventory(inventory)
    return {name: torch.zeros(descriptor['score_shapes'][name], dtype=torch.float64)
            for name in descriptor['required_score_names']}


class SelectionTests(unittest.TestCase):
    def test_dynamic_counts_and_policies(self):
        for resolution, g_count, d_count in ((8, 4, 3), (32, 12, 9), (1024, 32, 24)):
            with self.subTest(resolution=resolution):
                g = build_score_inventory(inventory_fixture('G', resolution))
                d = build_score_inventory(inventory_fixture('D', resolution))
                self.assertEqual(len(g['modules']), g_count)
                self.assertEqual(len(d['modules']), d_count)
                self.assertEqual(sum(x['pool'] == 'g_conv' for x in g['modules'].values()), g_count // 2)
                self.assertEqual(g['excluded_modules']['mapping.fc0']['policy'], 'modulation_only')
                self.assertEqual(g['excluded_modules']['synthesis.b4.const']['policy'], 'frozen_original')
                self.assertEqual(d['excluded_modules']['b4.fc']['policy'], 'full_finetune')
                self.assertEqual(g['parameter_policies']['synthesis.b8.conv0.parametrizations.bias.original'], 'frozen_original')
                self.assertEqual(g['parameter_policies']['synthesis.b8.conv0.parametrizations.bias.0.b_vector'], 'trainable_offset')
                self.assertIsNone(g['modules']['synthesis.b8.conv0']['b_parameter'])
                self.assertIsNone(d['modules']['b8.skip']['b_parameter'])

    def test_three_source_pool_formulas_and_linear_threshold(self):
        for component in ('G', 'D'):
            inventory = inventory_fixture(component, 8)
            descriptor = build_score_inventory(inventory)
            means = means_fixture(inventory)
            for record in descriptor['modules'].values():
                means[record['u_parameter']].fill_(2)
                means[record['v_parameter']] = torch.tensor([0., 4.], dtype=torch.float64)
                if record['b_parameter']:
                    means[record['b_parameter']] = torch.tensor([4., 10.], dtype=torch.float64)
            result = select_rows(means, inventory, quantile=50)
            self.assertEqual(set(result['pools']), {'g_conv', 'g_affine'} if component == 'G' else {'d_conv'})
            for module, record in result['modules'].items():
                self.assertEqual(record['scores'], [3., 8.] if record['b_parameter'] else [2., 6.])
                self.assertEqual(result['masks'][module], [False, True])
            for pool in result['pools'].values():
                self.assertEqual(pool['threshold'], {'g_conv': 4, 'g_affine': 5.5, 'd_conv': 4.5}[result['modules'][pool['modules'][0]]['pool']])
                self.assertEqual(pool['method'], 'linear')
                self.assertEqual(pool['ties_at_threshold'], 0)
            json.dumps(result, allow_nan=False)

    def test_pooling_is_global_per_family_and_ties_are_low(self):
        inventory = inventory_fixture('G', 8); means = means_fixture(inventory)
        descriptor = build_score_inventory(inventory)
        records = [r for r in descriptor['modules'].values() if r['pool'] == 'g_conv']
        means[records[0]['v_parameter']] = torch.tensor([0., 1.], dtype=torch.float64)
        means[records[1]['v_parameter']] = torch.tensor([10., 11.], dtype=torch.float64)
        result = select_rows(means, inventory, quantile=50)
        conv = [m for m, r in result['modules'].items() if r['pool'] == 'g_conv']
        self.assertEqual(result['masks'][conv[0]], [False, False])
        self.assertEqual(result['masks'][conv[1]], [True, True])
        self.assertEqual(result['pools']['g_conv']['threshold'], 5.5)
        self.assertEqual(result['pools']['g_affine']['high_count'], 0)
        self.assertEqual(result['pools']['g_affine']['ties_at_threshold'], 4)
        self.assertTrue(result['pools']['g_affine']['empty_high'])
        self.assertTrue(result['pools']['g_affine']['all_equal'])

    def test_missing_duplicate_nonfinite_wrong_shape_and_unknown_scores(self):
        inventory = inventory_fixture('D', 8); means = means_fixture(inventory)
        key = next(iter(means))
        malformed = [dict(means), list(means.items()) + [(key, means[key])],
                     {**means, 'unknown': torch.zeros(1)}]
        del malformed[0][key]
        for bad in (torch.zeros(99), torch.full_like(means[key], float('nan')),
                    torch.full_like(means[key], float('inf')), -torch.ones_like(means[key]),
                    torch.ones(means[key].shape, dtype=torch.int64)):
            malformed.append({**means, key: bad})
        for values in malformed:
            with self.assertRaises(ValueError): select_rows(values, inventory, quantile=50)

    def test_inventory_fails_closed(self):
        original = inventory_fixture('G', 16)
        bad = copy.deepcopy(original); bad['conv_layout'] = 'source_flattened'
        cases = [bad]
        bad = copy.deepcopy(original); bad['targets'].append(copy.deepcopy(bad['targets'][0])); cases.append(bad)
        bad = copy.deepcopy(original); bad['original_parameter_names'].append(bad['original_parameter_names'][0]); cases.append(bad)
        bad = copy.deepcopy(original); bad['targets'][0]['shape'] = [2, -1]; cases.append(bad)
        bad = copy.deepcopy(original); bad['targets'][0]['base_parameter'] = 'wrong'; cases.append(bad)
        bad = copy.deepcopy(original); bad['targets'] = [t for t in bad['targets'] if t['module'] != 'synthesis.b8.conv0']; cases.append(bad)
        bad = copy.deepcopy(original)
        bad['targets'] = [t for t in bad['targets'] if not t['module'].startswith('synthesis.b8.')]
        bad['original_parameter_names'] = [n for n in bad['original_parameter_names'] if not n.startswith('synthesis.b8.')]
        cases.append(bad)
        for inventory in cases:
            with self.assertRaises(ValueError): build_score_inventory(inventory)
        d = inventory_fixture('D'); d['ordinary_trainable_names'].pop()
        with self.assertRaises(ValueError): build_score_inventory(d)

    def test_explicit_quantile_and_unused_scores_are_not_pooled(self):
        inventory = inventory_fixture('G', 8); means = means_fixture(inventory)
        before = {n: t.clone() for n, t in means.items()}
        descriptor = build_score_inventory(inventory)
        unused = next(n for n in descriptor['score_shapes'] if n not in means)
        means[unused] = torch.full(descriptor['score_shapes'][unused], 1e20, dtype=torch.float64)
        result = select_rows(means, inventory, quantile=75)
        self.assertEqual(result['provided_unused_score_names'], [unused])
        self.assertTrue(all(p['threshold'] == 0 for p in result['pools'].values()))
        for name, tensor in before.items(): self.assertTrue(torch.equal(tensor, means[name]))
        with self.assertRaises(TypeError): select_rows(means, inventory)
        for quantile in (None, True, -1, 101, float('nan'), float('inf')):
            with self.assertRaises(ValueError): select_rows(means, inventory, quantile=quantile)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
