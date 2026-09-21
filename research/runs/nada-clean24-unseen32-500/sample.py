"""Unfiltered fresh-latent CPU evaluation of the completed NADA500 checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'research/runs/nada-clean24-cpu-continue500'
OUT = ROOT / 'research/runs/nada-clean24-unseen32-500'
EXPERIMENT = ROOT / 'research/experiments/nada_clean24_continue500'
LIMIT_SECONDS = 300
LIMIT_RSS = 6 * 1024**3


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def runner():
    spec = importlib.util.spec_from_file_location('original_nada_supervision', EXPERIMENT/'runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight():
    guard = runner()
    pins = guard.verify_pins()
    assert sha(HERE/'latents.npz') == json.loads((HERE/'inputs.json').read_text())['latents_sha256']
    result = json.loads((RUN/'supervisor-result.json').read_text())
    worker = json.loads((RUN/'worker-result.json').read_text())
    assert result['complete'] and result['worker_exit_code'] == 0
    assert worker['complete'] and worker['total_step'] == 500 and worker['updates'] == 490
    record = json.loads((RUN/'checkpoint-500-review-ready.json').read_text())
    assert record['step'] == 500 and record['actual_student_adam_rng_reload']
    assert record['native_preview_reload_equal'] and record['source_clip_and_frozen_student_unchanged']
    assert record['sha256'] == sha(RUN/'checkpoint-500.pt')
    assert any(c['sha256'] == record['sha256'] and c['step'] == 500 for c in worker['checkpoints'])
    return pins, record


def generate():
    pins, record = preflight()
    # Same six narrowly borrowed roots as the successful NADA worker.
    import importlib.abc
    import importlib.machinery
    class ExistingPackageFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if path is None and fullname in pins['borrowed_package_roots']:
                return importlib.machinery.PathFinder.find_spec(fullname, [str(ROOT/pins['borrowed_site'])])
            return None
    sys.meta_path.append(ExistingPackageFinder())
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('nada_pinned_helpers', ROOT/'research/experiments/paired_edit/trainer.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    metadata = json.loads((ROOT/'research/runs/inference-cpu/ffhq1024/baseline-bundle/model.json').read_text())
    model = base.Generator(**metadata['init_kwargs']).eval().requires_grad_(False)
    checkpoint = torch.load(RUN/'checkpoint-500.pt', map_location='cpu', mmap=True, weights_only=True)
    assert checkpoint['step'] == 500 and checkpoint['format_version'] == 2
    assert all(bool(torch.isfinite(t).all()) for t in checkpoint['student'].values())
    model.load_state_dict(checkpoint['student'], strict=True)
    del checkpoint
    initial_state = base.state_digest(model.state_dict())
    with np.load(EXPERIMENT/'latents.npz', allow_pickle=False) as old:
        fixed = torch.from_numpy(old['eval_z'][0:1].copy())
        seen = {r.tobytes() for n in ['source_z','previous_train_z','train_z','eval_z'] for r in old[n]}
    with np.load(HERE/'latents.npz', allow_pickle=False) as fresh:
        z = fresh['z'].copy()
    assert z.shape == (32,512) and z.dtype == np.float32 and np.isfinite(z).all()
    assert len({r.tobytes() for r in z}) == 32 and not any(r.tobytes() in seen for r in z)
    def image(tensor):
        w = model.mapping(tensor, None, truncation_psi=1, skip_w_avg_update=True)
        return model.synthesis(w, noise_mode='const', force_fp32=True)
    rows = []
    with torch.no_grad():
        control = image(fixed)
        for kind, pixels in [('rgb',base.quantized(control)),('gray',base.quantized(base.grayscale(control).expand(-1,3,-1,-1)))]:
            path = OUT/f'control-000-{kind}.png'
            Image.fromarray(pixels).save(path)
            assert sha(path) == sha(RUN/f'preview-500/000-student-{kind}.png'), 'Exact fixed500 preview reproduction failed'
        for index, row in enumerate(z):
            start = time.monotonic()
            output = image(torch.from_numpy(row[None]))
            seconds = time.monotonic()-start
            assert bool(torch.isfinite(output).all())
            files = {}
            for kind, pixels in [('rgb',base.quantized(output)),('gray',base.quantized(base.grayscale(output).expand(-1,3,-1,-1)))]:
                path = OUT/f'{index:03}-{kind}.png'
                Image.fromarray(pixels).save(path)
                files[kind] = {'path':path.name,'sha256':sha(path)}
            rows.append({'index':index,'generation_seconds':seconds,'files':files})
    assert base.state_digest(model.state_dict()) == initial_state
    assert sha(RUN/'checkpoint-500.pt') == record['sha256']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= LIMIT_RSS, 'Worker peak RSS exceeded cap'
    assert time.time() <= json.loads((OUT/'launch.json').read_text())['deadline_unix'], 'Worker deadline expired'
    write(OUT/'evaluation.json', {'complete':True,'checkpoint_sha256':record['sha256'],
        'unfiltered':True,'source':'fresh Gaussian z, disjoint all540 original/training/evaluation rows',
        'sampling':'psi1, constant noise, no restoration or output selection',
        'exact_fixed500_control_png_matches':2,'generator_state_unchanged':True,
        'threads':1,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rows':rows,'quality_approved':False,'production_approved':False})


def supervise():
    assert not OUT.exists(), 'Refuse overwrite/retry'
    pins, record = preflight()
    guard = runner()
    guard.concurrency_guard()
    pressure = guard.memory_guard()
    OUT.mkdir()
    for name in ['sample.py','latents.npz','inputs.json']:
        shutil.copyfile(HERE/name, OUT/name)
    write(OUT/'launch.json', {'checkpoint_sha256':record['sha256'],'source_sha256':sha(HERE/'sample.py'),
        'memory':pressure,'deadline_unix':time.time()+LIMIT_SECONDS,'seconds_limit':LIMIT_SECONDS,'rss_sampled_limit':LIMIT_RSS,'cpu_threads':1})
    env = {**os.environ, **guard.NUMERIC_ENV}
    process = None
    error = None
    started = time.monotonic()
    peak = 0
    try:
        with (OUT/'worker.log').open('x') as log:
            process = subprocess.Popen([sys.executable,str(HERE/'sample.py'),'--worker'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if time.monotonic()-started > LIMIT_SECONDS:
                    raise RuntimeError('Evaluation deadline exceeded')
                result = subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True,timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    peak=max(peak,int(result.stdout.strip())*1024)
                    if peak>LIMIT_RSS: raise RuntimeError('Sampled memory cap exceeded')
                time.sleep(.1)
            assert process.returncode == 0 and (OUT/'evaluation.json').exists(), 'Incomplete worker'
            assert time.monotonic()-started <= LIMIT_SECONDS, 'Final evaluation deadline exceeded'
            assert json.loads((OUT/'evaluation.json').read_text())['peak_rss_bytes'] <= LIMIT_RSS, 'Final worker RSS exceeded'
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        write(OUT/'supervisor-result.json',{'complete':error is None,'error':error,
            'worker_exit_code':None if process is None else process.returncode,
            'seconds':time.monotonic()-started,'peak_sampled_rss_bytes':peak})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute',action='store_true')
    group.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: generate()
    else:
        signal.signal(signal.SIGTERM, lambda signum,frame: sys.exit(128+signum))
        supervise()
