"""Lightweight provenance guard; never imports torch, loads models, or starts training."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[3]
TRAINER = 'research/experiments/reference_phases/trainer.py'
ADAPTER = 'research/experiments/reference_phases/smoke.py'
VENDOR = 'research/vendor/stylegan2-ada-pytorch'
SOURCE_SUFFIXES = {'.py', '.cpp', '.cu', '.h', '.hpp'}
PACKAGES = ('torch', 'numpy', 'pillow', 'scipy', 'pyspng', 'ninja', 'sympy')
NUMERIC_ENVIRONMENT = ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH',
    'PYTORCH_MPS_PREFER_METAL', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
    'PYTORCH_MPS_LOW_WATERMARK_RATIO', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONHASHSEED',
    'CUBLAS_WORKSPACE_CONFIG')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    if versions['torch'] is None:
        raise ValueError('PyTorch distribution is absent; use the training Python environment')
    return {
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'system': platform.system(), 'machine': platform.machine(),
        'os_release': platform.release(), 'macos_version': platform.mac_ver()[0],
        'packages': versions,
        'numeric_environment': {key: os.environ.get(key) for key in NUMERIC_ENVIRONMENT},
    }


def relative(root, path):
    # Refuse symlinks escaping the reviewed repository and runs outside it.
    return str(path.resolve().relative_to(root.resolve()))


def snapshot(root, run, runtime):
    root, run = Path(root).resolve(), Path(run).resolve()
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    archive_trainer, archive_adapter = run / 'train-source.py', run / 'adapters-source.py'
    trainer, adapter = root / TRAINER, root / ADAPTER
    trainer_hash = sha256(trainer)
    if trainer_hash != sha256(archive_trainer) or trainer_hash != config['trainer_sha256']:
        raise ValueError('Trainer differs from archived source or run config')
    if sha256(adapter) != sha256(archive_adapter):
        raise ValueError('FP32 adapter differs from archived source')
    if runtime['packages']['torch'] != config['torch_version']:
        raise ValueError('PyTorch version differs from run config')
    vendor = root / VENDOR
    files = [trainer, adapter, archive_trainer, archive_adapter, config_path, vendor / 'legacy.py']
    for folder in ('training', 'torch_utils', 'dnnlib'):
        folder_path = vendor / folder
        if not folder_path.is_dir():
            raise ValueError(f'Missing vendor source directory: {folder}')
        files.extend(p for p in folder_path.rglob('*') if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    for required in ('training/networks.py', 'torch_utils/ops/conv2d_gradfix.py',
                     'torch_utils/ops/grid_sample_gradfix.py'):
        if not (vendor / required).is_file():
            raise ValueError(f'Missing required vendor source: {required}')
    for name, expected in config['recipe']['source_hashes'].items():
        if sha256(vendor / name) != expected:
            raise ValueError(f'Vendor file differs from run recipe: {name}')
    return {'run': relative(root, run), 'environment': runtime,
            'files_sha256': {relative(root, p): sha256(p) for p in sorted(set(files))}}


def record(root, run, manifest, runtime):
    observed = snapshot(root, run, runtime)
    record = {'format_version': 1, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Current trusted baseline only; cannot retroactively certify unrecorded prior environments.',
        'guard_sha256': sha256(Path(__file__)), **observed}
    # An existing baseline must never be silently refreshed to make a mismatch pass.
    with Path(manifest).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record


def check(root, run, manifest, runtime):
    baseline = json.loads(Path(manifest).read_text())
    if baseline['format_version'] != 1:
        raise ValueError('Unsupported baseline format')
    if baseline['guard_sha256'] != sha256(Path(__file__)):
        raise ValueError('Guard code differs from recorded baseline; review the change explicitly')
    observed = snapshot(root, run, runtime)
    problems = []
    if observed['run'] != baseline['run']:
        problems.append('different run')
    if observed['environment'] != baseline['environment']:
        problems.append('runtime environment differs')
    actual, expected = observed['files_sha256'], baseline['files_sha256']
    if actual.keys() != expected.keys():
        problems.append('source file inventory differs')
    problems.extend(f'changed file: {name}' for name in sorted(actual.keys() & expected.keys())
                    if actual[name] != expected[name])
    if problems:
        raise ValueError('; '.join(problems))
    return {'status': 'pass', 'run': observed['run'], 'checked_files': len(actual),
            'torch_version': runtime['packages']['torch'],
            'baseline_recorded_at_utc': baseline['recorded_at_utc']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('record', 'check'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'record':
            result = record(REPO, args.run, args.manifest, environment())
            print(json.dumps({'status': 'recorded', 'run': result['run'],
                              'checked_files': len(result['files_sha256'])}))
        else:
            print(json.dumps(check(REPO, args.run, args.manifest, environment())))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Resume preflight refused: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
