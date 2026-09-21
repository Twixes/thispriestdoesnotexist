"""Explicit, bounded CPU-only unseen sampling of the preserved reference625 continuation using exactly the reference500 latents."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import shutil
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS = HERE / 'pins.json'
CHECKPOINT = ROOT / 'research/runs/reference256-paper-b64-resumed500-to625/resume-000625.pt'
ORIGINAL_CONFIG = ROOT / 'research/runs/reference256-paper-b64/config.json'
PREVIOUS = ROOT / 'research/runs/reference256-paper-b64/unseen32-000500'
LATENTS = PREVIOUS / 'latents.npz'
CONFIG = CHECKPOINT.parent / 'config.json'
VENDOR = ROOT / 'research/vendor/stylegan2-ada-pytorch'
MAX_SECONDS = 600
MAX_RSS = 6 * 1024**3
SEEDS = tuple(range(202609210000, 202609210032))
VARIANTS = (('raw-psi1', 'G', 1.0), ('ema-psi1', 'G_ema', 1.0), ('ema-psi07', 'G_ema', .7))


# Reuse only stateless stdlib helpers from the immutable500 evaluator.
# Its run_worker/supervise/verify_pins are intentionally NOT called or monkeypatched.
HELPER = ROOT / 'research/experiments/reference_unseen/evaluate.py'
HELPER_SHA256 = 'bbfb5636c1e593cee7d6120816559dc48a44119c4d18b1acc9b1a7dd28202496'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA256:
    raise RuntimeError('Original evaluator helper source changed')
_spec = importlib.util.spec_from_file_location('reference500_pure_helpers', HELPER)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
sha, write_new, memory_guard = _helpers.sha, _helpers.write_new, _helpers.memory_guard


def verify_pins():
    pins = json.loads(PINS.read_text())
    if sys.version != pins['python_version']:
        raise RuntimeError('Pinned Python sys.version changed')
    for path, expected in pins['files'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f'Pinned input/source changed: {path}')
    for package, expected in pins['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'Pinned environment changed: {package}')
    return pins


def verify_future(checkpoint_sha256, config_sha256):
    for value in (checkpoint_sha256, config_sha256):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise RuntimeError('Root-supplied lowercase SHA256 values are required')
    if sha(CHECKPOINT) != checkpoint_sha256 or sha(CONFIG) != config_sha256:
        raise RuntimeError('Supplied625 checkpoint/config hash mismatch')
    config = json.loads(CONFIG.read_text())
    baseline = json.loads(ORIGINAL_CONFIG.read_text())
    for key in ('recipe', 'architecture', 'trainer_sha256', 'frozen_d_tensor_names',
                'torch_version', 'sampler', 'deviations'):
        if config[key] != baseline[key]:
            raise RuntimeError(f'Continuation625 config changed: {key}')
    if config['recipe']['dataset_count'] != 220 or config['recipe']['dataset_sha256'] != '3a6e258a9b9ddae8d415b70c3be540ff39cf44e8a37d80ab71540058f1ee0553':
        raise RuntimeError('Only original110 plus mirror continuation is eligible')
    args = config['arguments']
    for key in set(baseline['arguments']) - {'run', 'resume', 'steps', 'data', 'base'}:
        if args[key] != baseline['arguments'][key]:
            raise RuntimeError(f'Continuation625 argument changed: {key}')
    for key, expected in [('run', CONFIG.parent),
                          ('resume', ORIGINAL_CONFIG.parent / 'resume-000500.pt'),
                          ('data', ROOT / 'research/alignment/collar-only/eyes42'),
                          ('base', ROOT / 'research/models/ffhq256.pkl')]:
        if (ROOT / args[key]).resolve() != expected.resolve():
            raise RuntimeError(f'Unexpected continuation argument path: {key}')
    if args['steps'] != 625:
        raise RuntimeError('Expected declared stop625')
    completed = json.loads((CONFIG.parent / 'completed.json').read_text())
    launch = json.loads((CONFIG.parent / 'launch.json').read_text())
    if completed['step'] != 625 or completed['images_seen'] != 40000:
        raise RuntimeError('Training completion record is not625/40000')
    if not launch['complete'] or launch['child_exit_code'] != 0 or launch['start_step'] != 500 or launch['stop_step'] != 625:
        raise RuntimeError('Continuation supervisor is not successfully complete')
    if launch['outputs_sha256']['resume.pt'] != checkpoint_sha256 or launch['outputs_sha256']['config.json'] != config_sha256:
        raise RuntimeError('Preserved625 copy/config differ from completed launch outputs')
    for name, expected in launch['outputs_sha256'].items():
        if sha(CONFIG.parent / name) != expected:
            raise RuntimeError(f'Completed625 artifact changed: {name}')
    for filename in ('train-source.py', 'adapters-source.py'):
        if sha(CONFIG.parent / filename) != sha(ORIGINAL_CONFIG.parent / filename):
            raise RuntimeError(f'Archived trainer source changed: {filename}')
    if sha(CONFIG.parent / 'launch-source.py') != launch['source_sha256']:
        raise RuntimeError('Archived continuation launcher changed')
    # This source is fixed in static pins, not authorized by the future file itself.
    if launch['source_sha256'] != sha(ROOT / 'research/experiments/reference_resume500_625/launch.py'):
        raise RuntimeError('Unexpected continuation launcher provenance')
    previous = json.loads((PREVIOUS / 'evaluation.json').read_text())
    previous_supervisor = json.loads((PREVIOUS / 'supervisor-result.json').read_text())
    if not previous['complete'] or not previous_supervisor['complete'] or previous_supervisor['worker_exit_code'] != 0:
        raise RuntimeError('Reference500 evaluation lacks successful completion')
    if previous['latents']['sha256'] != sha(LATENTS):
        raise RuntimeError('Saved500 latent provenance mismatch')
    return {'checkpoint_sha256': checkpoint_sha256, 'config_sha256': config_sha256,
            'completion_sha256': sha(CONFIG.parent / 'completed.json'),
            'training_launch_sha256': sha(CONFIG.parent / 'launch.json'),
            'latents_sha256': sha(LATENTS)}


def run_worker(output, deadline, nonce, checkpoint_sha256, config_sha256):
    launch = json.loads((output / 'launch.json').read_text())
    if launch['nonce'] != nonce or launch['deadline_unix'] != deadline or launch['script_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Worker must match its supervisor launch')
    if set(p.name for p in output.iterdir()) - {'launch.json', 'worker.log'}:
        raise RuntimeError('Refuse existing evaluation files')
    if time.time() >= deadline or deadline > time.time() + MAX_SECONDS:
        raise RuntimeError('Invalid or expired original deadline')
    pins = verify_pins()
    future = verify_future(checkpoint_sha256, config_sha256)
    if launch['future_inputs'] != future:
        raise RuntimeError('Future625 inputs differ from supervisor launch')
    pressure = memory_guard()
    write_new(output / 'worker-preflight.json', pressure)
    stop = threading.Event()

    def watchdog():
        while not stop.wait(.05):
            reason = None
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS:
                reason = '6GiB observed peak RSS exceeded'
            if time.time() >= deadline:
                reason = 'Absolute ten-minute deadline expired'
            if reason:
                try:
                    write_new(output / 'watchdog-failure.json', {'reason': reason, 'production_approved': False})
                finally:
                    os._exit(70)

    watch = threading.Thread(target=watchdog, daemon=True)
    watch.start()
    try:
        # No numerical/model library is imported before pins and memory guard.
        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(VENDOR))
        from training.networks import Generator
        origin = time.monotonic()
        config = json.loads(CONFIG.read_text())
        constructor = ast.literal_eval(config['architecture']['G'])
        if not isinstance(constructor, dict) or any(constructor[k] != v for k, v in
                {'z_dim': 512, 'c_dim': 0, 'w_dim': 512, 'img_resolution': 256, 'img_channels': 3}.items()):
            raise RuntimeError('Unexpected pinned Generator architecture')
        # Trusted own trainer output includes NumPy RNG state; no external pickle.
        state = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
        if state['format_version'] != 1 or state['batch_idx'] != 625 or state['images_seen'] != 40000:
            raise RuntimeError('Expected preserved reference625 checkpoint')
        if state['config'] != config or state['recipe'] != config['recipe'] or state['sampler']['consumed'] != 40000:
            raise RuntimeError('Checkpoint config/recipe/sampler mismatch')
        fixed = state['fixed_z']
        if fixed.shape != (16, 512) or fixed.dtype != torch.float32 or not bool(torch.isfinite(fixed).all()):
            raise RuntimeError('Unexpected trainer fixed_z')
        with np.load(LATENTS, allow_pickle=False) as saved:
            z_array = saved['z'].copy()
            saved_seeds = saved['seeds'].copy()
            saved_fixed = saved['trainer_fixed_z'].copy()
        expected_z = np.stack([np.random.Generator(np.random.PCG64(seed)).standard_normal(512).astype(np.float32) for seed in SEEDS])
        if z_array.shape != (32, 512) or z_array.dtype != np.float32 or not np.array_equal(z_array, expected_z):
            raise RuntimeError('Saved500 z differs from pinned seed algorithm')
        if saved_seeds.dtype != np.uint64 or not np.array_equal(saved_seeds, np.asarray(SEEDS, dtype=np.uint64)):
            raise RuntimeError('Saved500 seed order changed')
        if not np.array_equal(saved_fixed, fixed.numpy()):
            raise RuntimeError('Continuation trainer fixed latents changed')
        z = torch.from_numpy(z_array)
        if any(torch.equal(row, previous) for row in z for previous in fixed):
            raise RuntimeError('Fresh latent duplicates a trainer fixed preview latent')
        if len({row.tobytes() for row in z_array}) != 32:
            raise RuntimeError('Duplicate evaluation latent')
        with (output / 'latents.npz').open('xb') as target, LATENTS.open('rb') as source:
            shutil.copyfileobj(source, target)
        if sha(output / 'latents.npz') != sha(LATENTS):
            raise RuntimeError('Saved comparison latent copy differs')
        model = Generator(**constructor).cpu().eval().requires_grad_(False)
        entries = []
        state_name_loaded = None
        for name, state_name, psi in VARIANTS:
            if state_name_loaded != state_name:
                model.load_state_dict(state[state_name], strict=True)
                for tensor in model.state_dict().values():
                    if not bool(torch.isfinite(tensor).all()):
                        raise RuntimeError('Nonfinite model state')
                state_name_loaded = state_name
            rows = []
            for index, seed in enumerate(SEEDS):
                if time.time() >= deadline:
                    raise RuntimeError('Original ten-minute deadline expired')
                tick = time.monotonic()
                with torch.inference_mode():
                    generated = model(z[index:index+1], None, truncation_psi=psi,
                                      noise_mode='const', force_fp32=True)
                if tuple(generated.shape) != (1, 3, 256, 256) or not bool(torch.isfinite(generated).all()):
                    raise RuntimeError('Invalid generated tensor; no visual filtering or retry')
                # Exactly the trainer snapshot encoding, not face restoration.
                array = ((generated[0] + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                file = output / f'{name}-{index:03}.png'
                with file.open('xb') as stream:
                    Image.fromarray(array, 'RGB').save(stream, format='PNG')
                rows.append(array)
                entries.append({'variant': name, 'index': index, 'seed': seed, 'path': file.name,
                                'sha256': sha(file), 'seconds': time.monotonic()-tick,
                                'raw_min': float(generated.min()), 'raw_max': float(generated.max())})
                del generated
            sheet = np.concatenate([np.concatenate(rows[i:i+8], axis=1) for i in range(0, 32, 8)], axis=0)
            with (output / f'{name}-contact.png').open('xb') as stream:
                Image.fromarray(sheet, 'RGB').save(stream, format='PNG')
            # Eval/inference must not mutate parameters, noise or mapping average.
            actual = model.state_dict()
            if set(actual) != set(state[state_name]) or any(not torch.equal(actual[k], v) for k, v in state[state_name].items()):
                raise RuntimeError('Generator state changed during evaluation')
            del rows, sheet, actual
        if len(entries) != 96:
            raise RuntimeError('Incomplete96-image evaluation')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS or time.time() >= deadline:
            raise RuntimeError('Resource limit exceeded; no successful report')
        if verify_pins() != pins or sha(Path(__file__)) != launch['script_sha256'] or sha(PINS) != launch['pins_sha256']:
            raise RuntimeError('Provenance changed during evaluation')
        if verify_future(checkpoint_sha256, config_sha256) != future:
            raise RuntimeError('Future625 inputs changed during inference')
        write_new(output / 'evaluation.json', {
            'complete': True, 'production_approved': False, 'quality_approved': False,
            'checkpoint_step': 625, 'images_seen_in_training': 40000,
            'checkpoint_sha256': checkpoint_sha256, 'config_sha256': config_sha256,
            'source_sha256': launch['script_sha256'], 'pins_sha256': launch['pins_sha256'],
            'input_hashes_before_and_after_equal': True, 'provenance': pins, 'future_inputs': future,
            'constructor': constructor, 'device': 'cpu', 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'batch': 1,
            'wall_seconds': time.monotonic()-origin, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'memory_preflight': pressure, 'sampling': 'Exact same32 saved500 z/seeds, copied byte-for-byte; no new seeds, search, filtering or replacement',
            'seeds': list(SEEDS), 'distinct_from_saved_fixed16': True,
            'all_output_pixels_generated': True, 'postprocessing': 'trainer-matched clamp/uint8 PNG encoding only',
            'latents': {'path': 'latents.npz', 'sha256': sha(output / 'latents.npz')},
            'contact_sheets': {f'{name}-contact.png': sha(output / f'{name}-contact.png') for name, _, _ in VARIANTS},
            'entries': entries, 'source_model_baseline_omitted': True,
            'limits': ['These seeds were unseen at500; at625 they are a matched longitudinal comparison, not a new held-out draw or proof of never overlapping training random draws.',
                       'Raw and EMA compare checkpoint states and truncation, not independent samples.',
                       'No automatic quality, age, gender or attractiveness threshold is applied.',
                       '6GiB is an observed-RSS abort threshold sampled by watchdog/supervisor, not an OS hard allocation guarantee.']})
    finally:
        stop.set()
        watch.join(timeout=1)


def supervise(output, checkpoint_sha256, config_sha256):
    if output.parent != CONFIG.parent or output.exists():
        raise RuntimeError('Use a new direct child of the resumed500-to625 run; no overwrite/retry')
    started = time.time()
    deadline = started + MAX_SECONDS
    pressure = memory_guard()
    pins = verify_pins()
    future = verify_future(checkpoint_sha256, config_sha256)
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    launch = {'started_unix': started, 'deadline_unix': deadline, 'nonce': nonce,
              'script_sha256': sha(Path(__file__)), 'pins_sha256': sha(PINS),
              'checkpoint_sha256': checkpoint_sha256, 'config_sha256': config_sha256,
              'memory_preflight': pressure, 'future_inputs': future, 'rss_abort_bytes': MAX_RSS,
              'rss_limit_is_sampled_not_hard_allocation_cap': True, 'production_approved': False}
    write_new(output / 'launch.json', launch)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--execute', '--output', str(output),
               '--checkpoint-sha256', checkpoint_sha256, '--config-sha256', config_sha256,
               '--_worker', '--_deadline', str(deadline), '--_nonce', nonce]
    process = None
    failure = None
    peak_sampled_rss = 0
    try:
        with (output / 'worker.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.time() >= deadline:
                    raise RuntimeError('Absolute ten-minute deadline expired')
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True,
                                        text=True, timeout=min(2, max(.01, deadline-time.time())))
                if sample.returncode == 0 and sample.stdout.strip():
                    peak_sampled_rss = max(peak_sampled_rss, int(sample.stdout.strip()) * 1024)
                    if peak_sampled_rss > MAX_RSS:
                        raise RuntimeError('6GiB observed worker RSS exceeded')
                time.sleep(.1)
            if process.returncode != 0:
                raise RuntimeError(f'Worker failed with exit{process.returncode}; retained partial evidence, no retry')
            if time.time() >= deadline or not (output / 'evaluation.json').exists():
                raise RuntimeError('Missing completed evaluation or expired deadline')
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        write_new(output / 'supervisor-result.json', {'complete': failure is None,
                  'worker_exit_code': process.returncode if process is not None else None,
                  'error': failure, 'wall_seconds': time.time()-started,
                  'deadline_unix': deadline, 'peak_sampled_worker_rss_bytes': peak_sampled_rss,
                  'production_approved': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-static', action='store_true', help='Only pinned source/metadata checks; no model imports')
    mode.add_argument('--check-only', action='store_true', help='Verify supplied future625 hashes/completion using stdlib only')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--config-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--_nonce', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_static:
        pins = verify_pins()
        print(json.dumps({'static_files': len(pins['files']), 'torch_imported': 'torch' in sys.modules,
                          'model_loaded': False, 'future625_checked': False}))
        return
    if args.checkpoint_sha256 is None or args.config_sha256 is None:
        parser.error('Both independently reviewed625 SHA256 arguments are required')
    if args.check_only:
        verify_pins()
        print(json.dumps({'future_inputs': verify_future(args.checkpoint_sha256, args.config_sha256),
                          'torch_imported': 'torch' in sys.modules, 'model_loaded': False}))
        return
    if args.output is None:
        parser.error('Explicit new --output required for execution')
    output = args.output.resolve()
    if output.parent != CONFIG.parent:
        parser.error('Output must be a new direct child of the resumed500-to625 run')
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        if args._deadline is None or args._nonce is None:
            parser.error('Internal worker requires supervisor provenance')
        run_worker(output, args._deadline, args._nonce, args.checkpoint_sha256, args.config_sha256)
    else:
        if args._deadline is not None or args._nonce is not None:
            parser.error('Cannot override the ten-minute supervisor deadline')
        supervise(output, args.checkpoint_sha256, args.config_sha256)


if __name__ == '__main__':
    main()
