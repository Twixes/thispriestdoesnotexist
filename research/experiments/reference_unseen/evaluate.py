"""Explicit, bounded CPU-only unseen sampling of the preserved reference500 model."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
CHECKPOINT = ROOT / 'research/runs/reference256-paper-b64/resume-000500.pt'
CONFIG = CHECKPOINT.parent / 'config.json'
VENDOR = ROOT / 'research/vendor/stylegan2-ada-pytorch'
MAX_SECONDS = 600
MAX_RSS = 6 * 1024**3
SEEDS = tuple(range(202609210000, 202609210032))
VARIANTS = (('raw-psi1', 'G', 1.0), ('ema-psi1', 'G_ema', 1.0), ('ema-psi07', 'G_ema', .7))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def memory_guard():
    if sys.platform != 'darwin':
        raise RuntimeError('This RSS/memory-pressure guard is reviewed for macOS only')
    text = subprocess.check_output(['memory_pressure'], text=True, timeout=15)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if not match or int(match[1]) < 35:
        raise RuntimeError('Defer: at least35% free memory required before model imports')
    return {'free_percent': int(match[1]), 'raw': text}


def verify_pins():
    pins = json.loads(PINS.read_text())
    for path, expected in pins['files'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f'Pinned input/source changed: {path}')
    for package, expected in pins['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'Pinned environment changed: {package}')
    return pins


def run_worker(output, deadline, nonce):
    launch = json.loads((output / 'launch.json').read_text())
    if launch['nonce'] != nonce or launch['deadline_unix'] != deadline or launch['script_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Worker must match its supervisor launch')
    if set(p.name for p in output.iterdir()) - {'launch.json', 'worker.log'}:
        raise RuntimeError('Refuse existing evaluation files')
    if time.time() >= deadline or deadline > time.time() + MAX_SECONDS:
        raise RuntimeError('Invalid or expired original deadline')
    pins = verify_pins()
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
        if state['format_version'] != 1 or state['batch_idx'] != 500 or state['images_seen'] != 32000:
            raise RuntimeError('Expected preserved reference500 checkpoint')
        if state['config'] != config or state['recipe'] != config['recipe'] or state['sampler']['consumed'] != 32000:
            raise RuntimeError('Checkpoint config/recipe/sampler mismatch')
        fixed = state['fixed_z']
        if fixed.shape != (16, 512) or fixed.dtype != torch.float32 or not bool(torch.isfinite(fixed).all()):
            raise RuntimeError('Unexpected trainer fixed_z')
        z_array = np.stack([np.random.Generator(np.random.PCG64(seed)).standard_normal(512).astype(np.float32) for seed in SEEDS])
        z = torch.from_numpy(z_array)
        if any(torch.equal(row, previous) for row in z for previous in fixed):
            raise RuntimeError('Fresh latent duplicates a trainer fixed preview latent')
        if len({row.tobytes() for row in z_array}) != 32:
            raise RuntimeError('Duplicate evaluation latent')
        with (output / 'latents.npz').open('xb') as stream:
            np.savez(stream, seeds=np.asarray(SEEDS, dtype=np.uint64), z=z_array, trainer_fixed_z=fixed.numpy())
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
        write_new(output / 'evaluation.json', {
            'complete': True, 'production_approved': False, 'quality_approved': False,
            'checkpoint_step': 500, 'images_seen_in_training': 32000,
            'checkpoint_sha256': pins['files'][str(CHECKPOINT.relative_to(ROOT))],
            'source_sha256': launch['script_sha256'], 'pins_sha256': launch['pins_sha256'],
            'input_hashes_before_and_after_equal': True, 'provenance': pins,
            'constructor': constructor, 'device': 'cpu', 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'batch': 1,
            'wall_seconds': time.monotonic()-origin, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'memory_preflight': pressure, 'sampling': '32 predeclared independent PCG64 seeds; float64 normal draws cast to float32; no search/filter/replacement',
            'seeds': list(SEEDS), 'distinct_from_saved_fixed16': True,
            'all_output_pixels_generated': True, 'postprocessing': 'trainer-matched clamp/uint8 PNG encoding only',
            'latents': {'path': 'latents.npz', 'sha256': sha(output / 'latents.npz')},
            'contact_sheets': {f'{name}-contact.png': sha(output / f'{name}-contact.png') for name, _, _ in VARIANTS},
            'entries': entries, 'source_model_baseline_omitted': True,
            'limits': ['Fresh means not the saved fixed16; not a proof of never overlapping training random draws.',
                       'Raw and EMA compare checkpoint states and truncation, not independent samples.',
                       'No automatic quality, age, gender or attractiveness threshold is applied.',
                       '6GiB is an observed-RSS abort threshold sampled by watchdog/supervisor, not an OS hard allocation guarantee.']})
    finally:
        stop.set()
        watch.join(timeout=1)


def supervise(output):
    if not output.is_relative_to(ROOT / 'research/runs') or output.exists():
        raise RuntimeError('Use a new directory below research/runs; no overwrite/retry')
    started = time.time()
    deadline = started + MAX_SECONDS
    pressure = memory_guard()
    pins = verify_pins()
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    launch = {'started_unix': started, 'deadline_unix': deadline, 'nonce': nonce,
              'script_sha256': sha(Path(__file__)), 'pins_sha256': sha(PINS),
              'checkpoint_sha256': pins['files'][str(CHECKPOINT.relative_to(ROOT))],
              'memory_preflight': pressure, 'rss_abort_bytes': MAX_RSS,
              'rss_limit_is_sampled_not_hard_allocation_cap': True, 'production_approved': False}
    write_new(output / 'launch.json', launch)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__)), '--execute', '--output', str(output),
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
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--_nonce', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required; no model imports or loads performed')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'research/runs'):
        parser.error('Output must be below research/runs')
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        if args._deadline is None or args._nonce is None:
            parser.error('Internal worker requires supervisor provenance')
        run_worker(output, args._deadline, args._nonce)
    else:
        if args._deadline is not None or args._nonce is not None:
            parser.error('Cannot override the ten-minute supervisor deadline')
        supervise(output)


if __name__ == '__main__':
    main()
