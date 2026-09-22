"""Explicit fixed-noise/activation-bias policy shared by new research stages.

Never zero or repair a nonzero offset. Reject incompatible state. Only synthesis
conv noise and activation-bias offsets are protected; affine/ToRGB/mapping remain
permitted. Exact original tensors and zero offsets are checked independently.
"""
import re

POLICY = 'fixed_synthesis_noise_and_activation_bias_offsets_v1'
_PATTERN = re.compile(r'synthesis\.b\d+\.conv[01]\.parametrizations\.(noise_strength|bias)\.0\.b_vector')


def is_fixed_offset(name):
    return bool(_PATTERN.fullmatch(name))


def fixed_offset_names(model):
    from modulation import describe_modulation
    inventory = describe_modulation(model)
    if inventory['conv_layout'] != 'output_rank1':
        raise ValueError('Fixed-offset policy requires output_rank1')
    if inventory['component'] == 'D':
        return []
    blocks = model.synthesis.block_resolutions
    expected = {f'synthesis.b{r}.{c}.parametrizations.{t}.0.b_vector'
                for r in blocks for c in (('conv1',) if r == 4 else ('conv0', 'conv1'))
                for t in ('noise_strength', 'bias')}
    actual = {n for n, _ in model.named_parameters() if is_fixed_offset(n)}
    if actual != expected:
        raise ValueError('Incomplete synthesis noise/activation-bias offset inventory')
    return sorted(actual)


def capture_fixed_reference(model):
    """Check values before changing flags; originals must already be frozen."""
    import torch
    references = {}
    for name in fixed_offset_names(model):
        offset = model.get_parameter(name)
        original = model.get_parameter(name.replace('.0.b_vector', '.original'))
        if torch.count_nonzero(offset) or offset.grad is not None:
            raise ValueError(f'{name}: fixed offset must be zero with no gradient')
        if original.requires_grad or original.grad is not None:
            raise ValueError(f'{name}: original must already be frozen')
        references[name] = original.detach().clone()
    return references


def verify_fixed_offsets(model, references, optimizer=None):
    import torch
    if fixed_offset_names(model) != sorted(references):
        raise AssertionError('Fixed offset inventory changed')
    optimizer_ids = set() if optimizer is None else {id(p) for group in optimizer.param_groups for p in group['params']}
    for name, reference in references.items():
        offset = model.get_parameter(name)
        original = model.get_parameter(name.replace('.0.b_vector', '.original'))
        if (offset.requires_grad or offset.grad is not None or torch.count_nonzero(offset)
                or original.requires_grad or original.grad is not None):
            raise AssertionError(f'{name}: fixed zero/gradient policy violated')
        if not torch.equal(original, reference):
            raise AssertionError(f'{name}: original tensor changed')
        if id(offset) in optimizer_ids or id(original) in optimizer_ids:
            raise AssertionError(f'{name}: protected parameter entered optimizer')
    return True


def importance_named_parameters(model):
    from modulation import modulation_named_parameters
    fixed = set(fixed_offset_names(model))
    return [(n, p) for n, p in modulation_named_parameters(model) if n not in fixed]
