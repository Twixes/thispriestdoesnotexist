"""Check a trusted local research checkpoint on CPU, without training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = digest(args.resume)
    # These are our own trusted training artifacts, not arbitrary downloaded pickles.
    state = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
    assert state['step'] == args.step
    counts = {}
    for name in ('G', 'D', 'G_ema'):
        counts[name] = len(state[name])
        assert all(not value.is_floating_point() or bool(torch.isfinite(value).all())
                   for value in state[name].values()), name
    optimizer_counts = {}
    for name in ('G_opt', 'D_opt'):
        moments = state[name]['state']
        assert moments, name
        tensors = [value for entry in moments.values() for value in entry.values()
                   if torch.is_tensor(value)]
        assert all(not value.is_floating_point() or bool(torch.isfinite(value).all())
                   for value in tensors), name
        optimizer_counts[name] = len(tensors)
    generator = torch.load(args.generator, map_location='cpu', weights_only=True, mmap=True)
    assert generator['step'] == args.step
    assert state['G_ema'].keys() == generator['G_ema'].keys()
    assert all(torch.equal(value, generator['G_ema'][key])
               for key, value in state['G_ema'].items())
    assert before == digest(args.resume), 'Checkpoint changed during validation'
    result = {
        'step': args.step, 'images_seen': state['images_seen'],
        'resume_sha256': before, 'generator_sha256': digest(args.generator),
        'validator_sha256': digest(Path(__file__)),
        'finite_state_tensor_counts': counts, 'finite_optimizer_tensor_counts': optimizer_counts,
        'generator_ema_matches_resume': True,
        'required_state_keys_present': sorted(state.keys()), 'quality_approved': False,
        'scope': 'Trusted local CPU tensor and export consistency check; no optimizer update or active-process change',
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
