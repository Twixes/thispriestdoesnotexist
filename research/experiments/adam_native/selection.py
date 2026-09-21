"""Architecture-aware AdAM score selection, without loading or changing a model.

Supports the explicitly inventoried NVIDIA unconditional skip-G/resnet-D family.
Quantile is required from the caller; no priest-specific threshold is selected.
Only output_rank1 inventories can produce output-kernel masks. G convolution,
G affine and D convolution pools preserve the pinned source formulas. NumPy
percentile method='linear'; high is strictly greater, ties belong to low.

build_score_inventory describes required normalized factor names. select_rows
accepts these names, optionally accompanied by other declared modulation scores.
Unknown names and malformed supplied scores fail; excluded scores never enter
a pool. Values must already be means of per-example squared gradients.
"""
from collections.abc import Mapping
import math
import re

import numpy as np
import torch

from importance import source_uvb_score


def _unique_strings(values, label):
    if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'{label}: require unique string list')
    return set(values)


def _role(module, component):
    if component == 'G':
        if re.fullmatch(r'mapping\.fc\d+', module):
            return 'mapping', None
        match = re.fullmatch(r'synthesis\.b(\d+)\.(conv[01]|torgb)(\.affine)?', module)
        if not match:
            raise ValueError('Unsupported generator module: ' + module)
        resolution = int(match[1])
        if resolution < 4 or resolution & (resolution - 1) or (resolution == 4 and match[2] == 'conv0'):
            raise ValueError('Invalid generator resolution/module: ' + module)
        if resolution == 4 or match[2] == 'torgb':
            return 'excluded_synthesis', resolution
        return ('g_affine' if match[3] else 'g_conv'), resolution
    match = re.fullmatch(r'b(\d+)\.(conv[01]|skip|fromrgb)', module)
    if not match:
        raise ValueError('Unsupported discriminator module: ' + module)
    resolution = int(match[1])
    if resolution < 8 or resolution & (resolution - 1):
        raise ValueError('Invalid discriminator resolution/module: ' + module)
    return ('fromrgb' if match[2] == 'fromrgb' else 'd_conv'), resolution


def build_score_inventory(inventory):
    """Return primitive mask/parameter policies from a complete explicit inventory.

    Input is describe_modulation()/install_modulation() metadata for one model.
    Every original parameter must have an explicit policy. No model is loaded.
    """
    if inventory.get('component') not in ('G', 'D') or inventory.get('conv_layout') != 'output_rank1':
        raise ValueError('Require component G/D and output_rank1 metadata')
    component = inventory['component']
    originals = _unique_strings(inventory.get('original_parameter_names'), 'original parameters')
    ordinary = _unique_strings(inventory.get('ordinary_trainable_names'), 'ordinary trainable parameters')
    targets = inventory.get('targets')
    if not isinstance(targets, list) or not targets:
        raise ValueError('Missing target inventory')
    grouped = {}
    shapes = {}
    policies = {}
    for target in targets:
        module, tensor = target.get('module'), target.get('tensor')
        if not isinstance(module, str) or tensor not in ('weight', 'bias', 'noise_strength'):
            raise ValueError('Malformed modulation target')
        role, resolution = _role(module, component)
        shape = target.get('shape')
        if (not isinstance(shape, list) or any(type(x) is not int or x <= 0 for x in shape)
                or (not shape and tensor != 'noise_strength')):
            raise ValueError('Invalid target shape: ' + module + '.' + tensor)
        key = module + '.' + tensor
        base = module + '.parametrizations.' + tensor + '.original'
        if key not in originals or target.get('base_parameter') != base:
            raise ValueError('Target original-parameter mapping mismatch: ' + key)
        group = grouped.setdefault(module, {'role': role, 'resolution': resolution, 'targets': {}})
        if tensor in group['targets']:
            raise ValueError('Duplicate target: ' + key)
        group['targets'][tensor] = target
        policies[base] = 'frozen_original'
        prefix = module + '.parametrizations.' + tensor + '.0.'
        if tensor == 'weight':
            if len(shape) not in (2, 4) or target.get('layout') != 'output_rank1' or target.get('mechanism') != 'multiplicative_rank_factors':
                raise ValueError('Invalid output-row weight target: ' + key)
            shapes[prefix + 'u_vector'] = [math.prod(shape[1:])]
            shapes[prefix + 'v_vector'] = [shape[0]]
            policies[prefix + 'u_vector'] = 'trainable_shared_factor'
            policies[prefix + 'v_vector'] = 'trainable_modulation'
        else:
            if target.get('mechanism') != 'additive_offset' or target.get('layout') is not None:
                raise ValueError('Invalid offset target: ' + key)
            shapes[prefix + 'b_vector'] = shape
            policies[prefix + 'b_vector'] = 'trainable_offset'

    weights = {name for name, group in grouped.items() if 'weight' in group['targets']}
    if len(weights) != len(grouped):
        raise ValueError('Offset module has no inventoried weight')
    resolutions = sorted({g['resolution'] for g in grouped.values() if g['resolution'] is not None})
    minimum = 4 if component == 'G' else 8
    if not resolutions or resolutions != [2**i for i in range(int(math.log2(minimum)), int(math.log2(max(resolutions))) + 1)]:
        raise ValueError('Noncontiguous resolution inventory')
    expected_modules = set()
    if component == 'G':
        mapping = sorted(int(name.rsplit('fc', 1)[1]) for name in weights if name.startswith('mapping.'))
        if not mapping or mapping != list(range(len(mapping))):
            raise ValueError('Missing/noncontiguous mapping layers')
        expected_modules.update(f'mapping.fc{i}' for i in mapping)
        for resolution in resolutions:
            for layer in (('conv1', 'torgb') if resolution == 4 else ('conv0', 'conv1', 'torgb')):
                module = f'synthesis.b{resolution}.{layer}'
                expected_modules.update((module, module + '.affine'))
        extra_originals = {'synthesis.b4.const'}
        if ordinary:
            raise ValueError('G has unexpected ordinary trainable parameters')
        policies['synthesis.b4.const'] = 'frozen_original'
    else:
        for resolution in resolutions:
            expected_modules.update(f'b{resolution}.{layer}' for layer in ('conv0', 'conv1', 'skip'))
        expected_modules.add(f'b{max(resolutions)}.fromrgb')
        extra_originals = {f'b4.{layer}.{tensor}' for layer in ('conv', 'fc', 'out') for tensor in ('weight', 'bias')}
        if ordinary != extra_originals:
            raise ValueError('D epilogue ordinary trainable coverage mismatch')
        policies.update({name: 'full_finetune' for name in ordinary})
    if weights != expected_modules:
        raise ValueError('Architecture module coverage mismatch')
    target_originals = {module + '.' + tensor for module, group in grouped.items() for tensor in group['targets']}
    if originals != target_originals | extra_originals:
        raise ValueError('Original parameter coverage mismatch')

    selected, excluded = {}, {}
    required = set()
    for module, group in sorted(grouped.items()):
        target = group['targets']['weight']
        shape, role = target['shape'], group['role']
        is_fc = module.startswith('mapping.') or module.endswith('.affine')
        if len(shape) != (2 if is_fc else 4):
            raise ValueError('FC/convolution shape mismatch: ' + module)
        bias = group['targets'].get('bias')
        if bias is not None and bias['shape'] != [shape[0]]:
            raise ValueError('Bias/output shape mismatch: ' + module)
        if module.endswith('.skip'):
            if bias is not None or set(group['targets']) != {'weight'}:
                raise ValueError('Resnet skip must be bias-free')
        elif bias is None:
            raise ValueError('Expected module bias: ' + module)
        noise = group['targets'].get('noise_strength')
        if noise is not None and (component != 'G' or is_fc or module.endswith('.torgb') or noise['shape'] != []):
            raise ValueError('Unexpected noise-strength offset: ' + module)
        if role not in ('g_conv', 'g_affine', 'd_conv'):
            excluded[module] = {'policy': 'modulation_only', 'reason': role}
            continue
        use_bias = role == 'g_affine' or (role == 'd_conv' and bias is not None)
        prefix = module + '.parametrizations.weight.'
        record = {'pool': role, 'weight_shape': shape[:], 'weight_parameter': prefix + 'original',
                  'u_parameter': prefix + '0.u_vector', 'v_parameter': prefix + '0.v_vector',
                  'bias_parameter': module + '.parametrizations.bias.original' if use_bias else None,
                  'b_parameter': module + '.parametrizations.bias.0.b_vector' if use_bias else None}
        selected[module] = record
        required.update((record['u_parameter'], record['v_parameter']))
        policies[record['weight_parameter']] = 'high_frozen_low_finetune_rows'
        policies[record['v_parameter']] = 'high_trainable_low_zero'
        if use_bias:
            required.add(record['b_parameter'])
            policies[record['bias_parameter']] = 'high_frozen_low_finetune_rows'
            policies[record['b_parameter']] = 'high_trainable_low_zero'
    if not selected:
        raise ValueError('Architecture has no selectable residual modules')
    if component == 'G':
        excluded['synthesis.b4.const'] = {'policy': 'frozen_original', 'reason': 'learned_constant'}
    else:
        excluded.update({f'b4.{layer}': {'policy': 'full_finetune', 'reason': 'discriminator_epilogue'} for layer in ('conv', 'fc', 'out')})
    return {'version': 1, 'component': component, 'conv_layout': 'output_rank1',
            'architecture': 'skip_generator' if component == 'G' else 'resnet_discriminator',
            'resolutions': resolutions, 'modules': selected, 'excluded_modules': excluded,
            'parameter_policies': policies, 'score_shapes': shapes, 'required_score_names': sorted(required),
            'coverage': {'selected_modules': len(selected), 'inventoried_weight_modules': len(weights),
                         'original_parameters': len(originals), 'required_score_parameters': len(required)}}


def select_rows(named_means, inventory, *, quantile):
    """Return JSON-compatible high masks and complete source-pool diagnostics.

    named_means is a mapping or unique(name, tensor) sequence. Inputs are CPU
    floating, sample-normalized nonnegative second moments. No gradients or
    sample normalization are computed here. No input is mutated.
    """
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)) or not math.isfinite(quantile) or not 0 <= quantile <= 100:
        raise ValueError('Caller must supply finite quantile in [0,100]')
    result = build_score_inventory(inventory)
    items = list(named_means.items() if isinstance(named_means, Mapping) else named_means)
    names = [name for name, _ in items]
    if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('Duplicate or invalid score names')
    values = dict(items)
    if not set(result['required_score_names']) <= set(values) or not set(values) <= set(result['score_shapes']):
        raise ValueError('Missing required or unknown score names')
    for name, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.layout != torch.strided
                or not value.is_floating_point() or list(value.shape) != result['score_shapes'][name]
                or not bool(torch.isfinite(value).all()) or bool((value < 0).any())):
            raise ValueError('Invalid normalized score tensor: ' + name)
    scores = {}
    grouped = {}
    for module, record in result['modules'].items():
        score = source_uvb_score(values[record['u_parameter']], values[record['v_parameter']],
                                 None if record['b_parameter'] is None else values[record['b_parameter']])
        array = score.to(torch.float64).numpy().copy()
        scores[module] = array
        record['scores'] = array.tolist()
        grouped.setdefault(record['pool'], []).append(module)
    pools, masks = {}, {}
    for pool, modules in sorted(grouped.items()):
        vector = np.concatenate([scores[module] for module in modules])
        threshold = float(np.percentile(vector, quantile, method='linear'))
        high = vector > threshold
        pools[pool] = {'quantile': float(quantile), 'method': 'linear', 'comparison': 'strict_gt',
                       'threshold': threshold, 'modules': modules, 'coordinates': int(vector.size),
                       'high_count': int(high.sum()), 'low_count': int((~high).sum()),
                       'ties_at_threshold': int((vector == threshold).sum()),
                       'minimum': float(vector.min()), 'maximum': float(vector.max()),
                       'all_equal': bool((vector == vector[0]).all()), 'empty_high': not bool(high.any())}
        for module in modules:
            masks[module] = (scores[module] > threshold).tolist()
    result.update(quantile=float(quantile), masks=masks, pools=pools,
                  numpy_version=np.__version__, score_semantics='output_rank1_per_output_row_source_formula',
                  normalized_input_assumed=True, sample_normalization_performed=False,
                  provided_unused_score_names=sorted(set(values) - set(result['required_score_names'])))
    result['coverage']['selected_coordinates'] = sum(len(mask) for mask in masks.values())
    return result
