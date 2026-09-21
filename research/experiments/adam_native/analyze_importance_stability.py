"""Read-only FI1000 ranking diagnostics; no threshold recommendation or model load.

Compare nested prefix500/full1000 AND disjoint first500/last500. Subtract raw
sums before normalization, never subtract normalized means. Undefined rank
correlations/Jaccards are null with reasons, never silently called perfect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch
from importance import EmpiricalSquaredGradients
from selection import build_score_inventory
from selection_fixed_offsets import select_rows_fixed_offsets
from fixed_offset_policy import POLICY, is_fixed_offset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_accumulator(payload, count, inventories):
    require(set(payload) == {'G', 'D', 'pairs', 'protocol_sha256', 'sample_plan_sha256'}, 'Accumulator wrapper schema mismatch')
    require(type(payload['pairs']) is int and payload['pairs'] == count, 'Wrong successful pair count')
    for name in ('protocol_sha256', 'sample_plan_sha256'):
        require(isinstance(payload[name], str) and bool(re.fullmatch('[0-9a-f]{64}', payload[name])), 'Invalid lineage hash: ' + name)
    result = {}
    for key in ('G', 'D'):
        require(inventories[key]['component'] == key, 'Inventory component mismatch')
        acc = EmpiricalSquaredGradients.from_state_dict(payload[key])
        require(acc.sample_count == count, 'G/D successful counts must match wrapper')
        require(acc.missing == 'error' and acc.dtype == torch.float64, 'Require complete FP64 FI observations')
        shapes = build_score_inventory(inventories[key])['score_shapes']
        expected = {n: shape for n, shape in shapes.items() if not is_fixed_offset(n)}
        require(set(acc.gradient_sums) == set(expected), 'Require complete permitted factor coverage')
        for name, shape in expected.items():
            require(list(acc.gradient_sums[name].shape) == shape, 'Inventory/tensor shape mismatch: ' + name)
        result[key] = acc
    return result


def average_ranks(values):
    order = np.argsort(values, kind='stable')
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
    return ranks


def distribution(values):
    _, counts = np.unique(values, return_counts=True)
    return {'coordinates': len(values), 'distinct_scores': len(counts),
        'tied_groups': int((counts > 1).sum()), 'tied_coordinates': int(counts[counts > 1].sum()),
        'tied_pairs': int(sum(int(c) * (int(c) - 1) // 2 for c in counts)),
        'zero_scores': int((values == 0).sum()), 'all_zero': bool((values == 0).all()),
        'all_equal': len(counts) == 1, 'minimum': float(values.min()), 'maximum': float(values.max())}


def correlation(a, b):
    ar = average_ranks(a); br = average_ranks(b)
    ar -= ar.mean(); br -= br.mean()
    norm_a = np.linalg.norm(ar); norm_b = np.linalg.norm(br)
    if norm_a == 0 or norm_b == 0:
        return {'spearman_rho': None, 'undefined_reason': 'At least one score vector is constant (all ranks tied).'}
    value = float(np.dot(ar / norm_a, br / norm_b))
    return {'spearman_rho': float(np.clip(value, -1, 1)), 'undefined_reason': None}


def compare_masks(a, b):
    intersection = int((a & b).sum()); union = int((a | b).sum())
    same = int((a == b).sum())
    return {'coordinates': len(a), 'left_high': int(a.sum()), 'right_high': int(b.sum()),
        'both_high': intersection, 'both_low': int((~a & ~b).sum()),
        'left_only_high': int((a & ~b).sum()), 'right_only_high': int((~a & b).sum()),
        'agreement_count': same, 'agreement_fraction': same / len(a),
        'high_union': union, 'high_jaccard': intersection / union if union else None,
        'jaccard_undefined_reason': None if union else 'Both high sets are empty.'}


def analyze_accumulators(prefix500, full1000, inventories):
    """Validated in-memory API. Caller authenticates files, marker and inventory."""
    require(set(inventories) == {'G', 'D'}, 'Require exactly G/D inventories')
    prefix = validate_accumulator(prefix500, 500, inventories)
    full = validate_accumulator(full1000, 1000, inventories)
    for name in ('protocol_sha256', 'sample_plan_sha256'):
        require(prefix500[name] == full1000[name], 'Incompatible lineage: ' + name)
    means = {'first500': {}, 'full1000': {}, 'last500': {}}
    for key in ('G', 'D'):
        first = prefix[key]; complete = full[key]
        count = complete.sample_count - first.sample_count
        require(count == 500, 'Disjoint sample count mismatch')
        second = complete.state_dict(); second['sample_count'] = count
        for name in first.gradient_sums:
            second['observed_counts'][name] = complete.observed_counts[name] - first.observed_counts[name]
            for field in ('gradient_sums', 'squared_gradient_sums'):
                require(getattr(complete, field)[name].shape == getattr(first, field)[name].shape, 'Prefix tensor shape mismatch')
                second[field][name] = getattr(complete, field)[name] - getattr(first, field)[name]
        # Strict rejection: no clamping of negative differences or nonfinite sums.
        last = EmpiricalSquaredGradients.from_state_dict(second)
        means['first500'][key] = first.mean_squared_gradients()
        means['full1000'][key] = complete.mean_squared_gradients()
        means['last500'][key] = last.mean_squared_gradients()
    selected = {name: {q: {key: select_rows_fixed_offsets(values[key], inventories[key], quantile=q)
                    for key in ('G', 'D')} for q in (50, 75)} for name, values in means.items()}
    comparisons = {}
    for label, left, right in [('nested_prefix500_vs_full1000', 'first500', 'full1000'),
                               ('disjoint_first500_vs_last500', 'first500', 'last500')]:
        pools = {}
        for key in ('G', 'D'):
            ls = selected[left][50][key]; rs = selected[right][50][key]
            for pool, metadata in ls['pools'].items():
                modules = metadata['modules']
                require(modules == rs['pools'][pool]['modules'], 'Pool coordinate identity differs')
                a = np.concatenate([ls['modules'][m]['scores'] for m in modules])
                b = np.concatenate([rs['modules'][m]['scores'] for m in modules])
                quantiles = {}
                for q in (50, 75):
                    lq = selected[left][q][key]; rq = selected[right][q][key]
                    am = np.concatenate([lq['masks'][m] for m in modules]).astype(bool)
                    bm = np.concatenate([rq['masks'][m] for m in modules]).astype(bool)
                    quantiles[str(q)] = {'left_threshold': lq['pools'][pool]['threshold'],
                        'right_threshold': rq['pools'][pool]['threshold'],
                        'left_ties_at_threshold': lq['pools'][pool]['ties_at_threshold'],
                        'right_ties_at_threshold': rq['pools'][pool]['ties_at_threshold'],
                        **compare_masks(am, bm)}
                pools[pool] = {'component': key, 'coordinate_order': [{'module': m, 'rows': len(ls['masks'][m])} for m in modules],
                    'left': distribution(a), 'right': distribution(b), **correlation(a, b), 'quantiles': quantiles}
        comparisons[label] = {'left': left, 'right': right, 'overlapping_samples': left == 'first500' and right == 'full1000', 'pools': pools}
    return {'schema_version': 1, 'fixed_offset_policy': POLICY,
        'protocol_sha256': prefix500['protocol_sha256'], 'sample_plan_sha256': prefix500['sample_plan_sha256'],
        'sample_counts': {'first500': 500, 'last500': 500, 'full1000': 1000},
        'disjoint_derivation': '(full raw sum - prefix raw sum) / (full actual count - prefix actual count)',
        'rank_method': 'Spearman correlation of average ranks; exact-value ties',
        'threshold_method': 'Existing pooled percentile linear interpolation; strict greater-than high mask',
        'quantiles_descriptive_only': [50, 75], 'comparisons': comparisons,
        'automatic_threshold_choice': None, 'pass_criteria': None, 'ranking_stability_proven': False,
        'limitations': ['Nested prefixes share samples and can overstate agreement.',
            'Disjoint halves diagnose sampling sensitivity, not independence of shared training images or correctness of the importance heuristic.',
            'High all-low mask agreement can be uninformative; empty high unions have undefined Jaccard.',
            'No quality, domain-conversion, diversity, or latency conclusion follows from this analysis.']}


def load_verified_run(run):
    run = run.resolve(); run.relative_to(ROOT / 'research/runs')
    protocol_file = run / 'protocol.json'; protocol = json.loads(protocol_file.read_text())
    protocol_sha = sha(protocol_file); plan_sha = sha(run / 'sample-plan.npz')
    require(protocol['name'] == 'adam-output-rank1-fixed-offsets-ema-importance' and protocol['fixed_offset_policy'] == POLICY, 'Wrong FI protocol')
    require(protocol['layout'] == 'output_rank1' and protocol['resolution'] == 1024 and protocol['accumulator_dtype'] == 'torch.float64', 'Wrong model/statistics protocol')
    require(type(protocol['pairs']) is int and protocol['pairs'] == 1000 and protocol['diagnostic_only'] is False and protocol['state_keys'] == ['Gema', 'Dema'], 'Require complete EMA FI1000 protocol')
    for relative, digest in protocol['pins'].items():
        path = (ROOT / relative).resolve(); path.relative_to(ROOT)
        require(sha(path) == digest, 'Pinned source changed: ' + relative)
    checkpoint = (ROOT / protocol['checkpoint']).resolve(); checkpoint.relative_to(ROOT / 'research/runs')
    require(sha(checkpoint) == protocol['checkpoint_sha256'], 'Source checkpoint hash mismatch')
    source = (ROOT / protocol['source_run']).resolve(); source.relative_to(ROOT / 'research/runs')
    inventory_file = source / 'modulation-inventory.json'
    require(protocol['pins'].get(str(inventory_file.relative_to(ROOT))) == sha(inventory_file), 'Unbound inventory')
    inventories = json.loads(inventory_file.read_text())
    require(sha(source / 'protocol.json') == protocol['source_protocol_sha256'], 'Source protocol lineage mismatch')
    plan_record = json.loads((run / 'sample-plan.json').read_text())
    require(plan_record['pairs'] == 1000 and plan_record['npz_sha256'] == plan_sha, 'Sample plan metadata mismatch')
    result = json.loads((run / 'result.json').read_text()); supervisor = json.loads((run / 'supervisor.json').read_text())
    require(result['complete'] is True and supervisor['complete'] is True and supervisor['failure'] is None, 'FI run not successfully complete')
    require(result['protocol_sha256'] == supervisor['protocol_sha256'] == protocol_sha, 'Terminal protocol lineage mismatch')
    require(result['pairs'] == 1000 and result['sample_plan_sha256'] == plan_sha, 'Terminal count/plan mismatch')
    require(result['ema_state_unchanged'] is True and result['fixed_offsets_zero_and_originals_unchanged'] is True, 'EMA/fixed policy evidence missing')
    payloads = []
    hashes = {'protocol.json': protocol_sha, 'sample-plan.npz': plan_sha,
        str(inventory_file.relative_to(ROOT)): sha(inventory_file), 'result.json': sha(run / 'result.json'), 'supervisor.json': sha(run / 'supervisor.json')}
    for count in (500, 1000):
        name = f'importance-{count:04}.pt'; marker_file = run / f'importance-{count:04}.json'
        marker = json.loads(marker_file.read_text()); digest = sha(run / name)
        require(marker['complete'] is True and marker['pairs'] == count and marker['path'] == name, 'Accumulator marker mismatch')
        require(marker['state_roundtrip_exact'] is True and marker['sha256'] == digest, 'Accumulator authentication failed')
        require(marker['protocol_sha256'] == protocol_sha and marker['sample_plan_sha256'] == plan_sha, 'Marker lineage mismatch')
        value = torch.load(run / name, weights_only=True, map_location='cpu')
        require(value['protocol_sha256'] == protocol_sha and value['sample_plan_sha256'] == plan_sha, 'Payload lineage mismatch')
        payloads.append(value); hashes[name] = digest; hashes[marker_file.name] = sha(marker_file)
    require(result['importance_path'] == 'importance-1000.pt' and result['importance_sha256'] == hashes['importance-1000.pt'], 'Terminal accumulator mismatch')
    return *payloads, inventories, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh artifact directory under research/')
    args = parser.parse_args(); out = args.output.resolve(); out.relative_to(ROOT / 'research')
    require(not out.exists(), 'Use a fresh output directory')
    prefix, full, inventories, hashes = load_verified_run(args.run)
    report = analyze_accumulators(prefix, full, inventories)
    report['source_run'] = str(args.run.resolve().relative_to(ROOT)); report['input_sha256'] = hashes
    report['analyzer_sources'] = {name: sha(HERE / name) for name in ('analyze_importance_stability.py', 'importance.py', 'selection.py', 'selection_fixed_offsets.py', 'fixed_offset_policy.py')}
    out.mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps({'report': str(out / 'report.json'), 'sha256': sha(out / 'report.json'), 'threshold_chosen': False}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
