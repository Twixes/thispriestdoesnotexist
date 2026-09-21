"""Pinned official SG3-T/SG2 matched-latent baseline with bounded CPU supervision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / 'research/vendor/stylegan3'
LIMIT = 12 * 1024**3
MODELS = {
 'stylegan3-t': ('research/models/stylegan3-t-ffhq1024/stylegan3-t-ffhq-1024x1024.pkl', 'efd9fa1f967a11b5390399a8ed512dc32e341c245fa2d4dafa6d94e96222085b'),
 'stylegan2': ('research/models/ffhq1024.pkl', 'a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def verify():
    source = json.loads((ROOT / 'research/models/stylegan3-t-ffhq1024/source-provenance.json').read_text())
    for name, expected in source['files'].items():
        if sha(VENDOR / name) != expected:
            raise RuntimeError('Source changed: ' + name)
    for name, (path, expected) in MODELS.items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Model changed: ' + name)
    return source


def worker(output):
    import gc
    import resource
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    launch = json.loads((output / 'launch.json').read_text())
    if os.getppid() != launch['supervisor_pid'] or sha(__file__) != launch['source_sha256']:
        raise RuntimeError('Worker ownership/source changed')
    verify()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(202609214000)
    sys.path.insert(0, str(VENDOR))
    import legacy
    if Path(legacy.__file__).resolve() != VENDOR / 'legacy.py':
        raise RuntimeError('Wrong isolated runtime')
    seeds = np.arange(202609214000, 202609214008, dtype=np.int64)
    z = np.stack([np.random.Generator(np.random.PCG64(int(s))).standard_normal(512).astype(np.float32) for s in seeds])
    warm = np.random.Generator(np.random.PCG64(202609214099)).standard_normal((1,512)).astype(np.float32)
    np.savez(output / 'latents.npz', z=z, seeds=seeds, warmup_z=warm)
    # Sampling is from fresh predeclared seeds, not chosen after viewing model outputs.
    def guard():
        if time.monotonic() > launch['deadline_monotonic']:
            raise RuntimeError('Deadline expired')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMIT:
            raise RuntimeError('Peak RSS exceeded 12 GiB')
        memory = psutil.virtual_memory()
        if memory.available / memory.total < .2:
            raise RuntimeError('System available memory fell below 20%')
    report = {'complete': False, 'device': 'cpu', 'threads': 1, 'torch': torch.__version__,
              'psi': .7, 'noise_mode': 'const', 'force_fp32': True,
              'native_resolution': 1024, 'samples_per_model': 8,
              'sampling': 'PCG64(seed).standard_normal(512), float64 then float32; same saved z for both models, not same identities',
              'quality_selection': False, 'restoration': False,
              'production_approved': False, 'latents_sha256': sha(output/'latents.npz'), 'models': {}}
    write(output/'result.json', report)
    for name, (path, expected) in MODELS.items():
        guard(); directory=output/name;directory.mkdir()
        started=time.monotonic()
        # Only the already verified official NVIDIA source checkpoint is unpickled.
        with (ROOT/path).open('rb') as stream:
            network=legacy.load_network_pkl(stream)
        model=network['G_ema'].cpu().eval().requires_grad_(False)
        del network;gc.collect()
        if model.img_resolution != 1024 or model.z_dim != 512 or model.c_dim != 0:
            raise RuntimeError('Unexpected base shape')
        model_report={'checkpoint_path':path,'checkpoint_sha256':expected,
                      'load_seconds':time.monotonic()-started, 'parameter_count':sum(p.numel() for p in model.parameters()),
                      'entries':[], 'complete':False}
        report['models'][name]=model_report;write(output/'result.json',report)
        with torch.inference_mode():
            for index, latent in [(-1,warm), *[(i,z[i:i+1]) for i in range(8)]]:
                guard();t=time.monotonic()
                image=model(torch.from_numpy(latent),None,truncation_psi=.7,noise_mode='const',force_fp32=True)
                forward=time.monotonic()-t
                if image.shape != (1,3,1024,1024) or not bool(torch.isfinite(image).all()):
                    raise RuntimeError('Invalid generated image')
                pixels=(image.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
                im=Image.fromarray(pixels,'RGB');prefix='warmup' if index==-1 else f'{index:03d}'
                encode=time.monotonic();im.save(directory/(prefix+'.png'));im.save(directory/(prefix+'.webp'),quality=90,method=6)
                entry={'index':index,'seed':202609214099 if index==-1 else int(seeds[index]),
                       'warmup':index==-1,'forward_seconds':forward,'conversion_and_encode_seconds':time.monotonic()-t-forward,
                       'encode_seconds':time.monotonic()-encode,
                       'png':prefix+'.png','png_sha256':sha(directory/(prefix+'.png')),
                       'webp':prefix+'.webp','webp_sha256':sha(directory/(prefix+'.webp')),
                       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                model_report['entries'].append(entry);write(directory/(prefix+'.json'),entry);write(output/'result.json',report)
                print(name,prefix,'forward_seconds',round(forward,4),'peak_rss',entry['peak_rss_bytes'],flush=True)
                del image,pixels,im;guard()
        model_report['complete']=True;write(output/'result.json',report)
        del model;gc.collect()
    verify();guard();report['complete']=True;write(output/'result.json',report)


def supervisor(output):
    import psutil
    output.mkdir(parents=True,exist_ok=False)
    source=verify();memory=psutil.virtual_memory()
    if memory.available/memory.total<.2:
        raise RuntimeError('Initial available memory below 20%')
    launch={'source_sha256':sha(__file__),'supervisor_pid':os.getpid(),
            'deadline_monotonic':time.monotonic()+600,'source_commit':source['commit'],
            'limits':{'rss_bytes':LIMIT,'minimum_system_available_fraction':.2,'seconds':600,'poll_seconds':.1},
            'initial_available_bytes':memory.available,'initial_total_bytes':memory.total,
            'models':MODELS,'supervision_note':'RSS sampled every100ms plus worker peak checks; process group killed on gate. Sampling may observe overshoot; not an OS virtual-address cap.'}
    write(output/'launch.json',launch);(output/'run-source.py').write_bytes(Path(__file__).read_bytes())
    env=os.environ.copy();env.update({key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']})
    started=time.monotonic();peak=0;reason=None;process=None
    try:
        with (output/'worker.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if time.monotonic()>launch['deadline_monotonic']:reason='ten-minute deadline'
                try:
                    parent=psutil.Process(process.pid);members=[parent,*parent.children(recursive=True)]
                    rss=sum(p.memory_info().rss for p in members if p.is_running());peak=max(peak,rss)
                    if rss>LIMIT:reason='process group RSS above12GiB'
                except psutil.NoSuchProcess:pass
                memory=psutil.virtual_memory()
                if memory.available/memory.total<.2:reason='system available memory below20%'
                if reason:break
                time.sleep(.1)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        result={'complete':process is not None and process.returncode==0 and reason is None,
                'worker_exit_code':None if process is None else process.returncode,'stop_reason':reason,
                'elapsed_seconds':time.monotonic()-started,'sampled_peak_process_group_rss_bytes':peak,
                'production_approved':False}
        write(output/'supervisor-result.json',result);print(json.dumps(result,indent=2))
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    (worker if args.worker else supervisor)(args.output.resolve())
