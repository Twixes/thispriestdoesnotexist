"""Same source score formulas/pools, with honest fixed-offset policy metadata."""
from collections.abc import Mapping
from fixed_offset_policy import POLICY, is_fixed_offset
from selection import select_rows


def select_rows_fixed_offsets(named_means, inventory, *, quantile):
    items = list(named_means.items() if isinstance(named_means, Mapping) else named_means)
    if any(is_fixed_offset(n) for n, _ in items):
        raise ValueError('Fixed-offset FI must omit protected offsets, not store measured or zero scores')
    result = select_rows(items, inventory, quantile=quantile)
    names = sorted(n for n in result['parameter_policies'] if is_fixed_offset(n))
    assert not set(names) & set(result['required_score_names'])
    for name in names:
        result['parameter_policies'][name] = 'fixed_zero_offset_excluded_from_optimizer'
        del result['score_shapes'][name]
    result['fixed_offset_policy'] = POLICY
    result['fixed_offset_names'] = names
    return result
