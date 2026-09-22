#!/usr/bin/env python3
"""Preregister 32 new latents or explicitly render a completed main100 checkpoint.

No native models are loaded by preregistration, preview building or fixture tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENTS = ROOT / 'research/experiments/adam_native'
sys.path.insert(0, str(EXPERIMENTS))
import main_adaptation_fixed_offsets250_10 as base
from fixed_offset_policy import POLICY
sha, read, write = base.sha, base.read, base.write
COUNT, SEED = 32, 2026092291
OLD_Z = base.EVAL_Z
REGISTRY = HERE / 'preregistration.json'
LATENTS = HERE / 'latents32.npz'
SOURCE_NAME = 'adam-native1024-retained250-main-adaptation10-to100'


def rel(path):
    return str(path.resolve().relative_to(ROOT))


def check_latents(z, old):
    import numpy as np
    assert z.shape == (COUNT, 512) and z.dtype == np.float32 and np.isfinite(z).all()
    assert old.shape == (4, 512) and old.dtype == np.float32
    rows = [row.tobytes() for row in z]
    assert len(set(rows)) == COUNT and not set(rows).intersection(row.tobytes() for row in old)
    return [hashlib.sha256(row).hexdigest() for row in rows]


def preregister():
    import numpy as np
    assert not REGISTRY.exists() and not LATENTS.exists(), 'Preregistration is immutable; use a new version instead'
    assert sha(OLD_Z) == base.EVAL_HASH
    with np.load(OLD_Z, allow_pickle=False) as fixture: old = fixture['z'].copy()
    z = np.random.default_rng(SEED).standard_normal((COUNT, 512)).astype(np.float32)
    rows = check_latents(z, old)
    np.savez(LATENTS, z=z)
    files = [HERE / name for name in ('render.py', 'build.py', 'template.html')]
    files += [EXPERIMENTS / name for name in ('main_adaptation_fixed_offsets250_10.py', 'continue_main_adaptation_fixed_offsets250_100.py', 'modulation.py', 'fixed_offset_policy.py', 'adaptation_masks_fixed_offsets.py')]
    files += [base.GDIR/'model.json', ROOT/'research/vendor/stylegan2-ada-pytorch/LICENSE.txt', ROOT/'research/vendor/stylegan2-ada-pytorch/docs/license.html']
    files += sorted((ROOT/'research/vendor/stylegan2-ada-pytorch').rglob('*.py'))
    write(REGISTRY, {'schema_version':1, 'count':COUNT, 'seed':SEED,
        'rng':'NumPy default_rng PCG64, standard_normal float64 then float32', 'numpy_version':np.__version__,
        'latents_path':rel(LATENTS), 'latents_sha256':sha(LATENTS), 'row_sha256':rows,
        'excluded_original_four_path':rel(OLD_Z), 'excluded_original_four_sha256':sha(OLD_Z),
        'selection':'All 32 draws retained in draw order, no rejection or quality selection',
        'scope':'Future completed retained250-route main-adaptation100 raw/EMA; 64 full native PNGs',
        'source_protocol_name':SOURCE_NAME, 'checkpoint_step':100, 'fixed_offset_policy':POLICY,
        'g_weights_path':rel(base.GDIR/'generator.safetensors'), 'g_weights_sha256':base.G_HASH,
        'sampling':'Gaussian z only; c=None; truncation_psi=1; constant synthesis noise; native1024 FP32 unfused forward',
        'license_notice':'Source NVIDIA code/weights retain upstream license terms; repository MIT license does not relicense them.',
        'pins':{rel(path):sha(path) for path in files}, 'native_render_executed':False,
        'automatic_launch':False, 'production_approved':False, 'server_latency_proven':False})


def registered_latents():
    import numpy as np
    registry = read(REGISTRY)
    assert registry['count'] == COUNT and registry['seed'] == SEED and registry['fixed_offset_policy'] == POLICY
    assert sha(LATENTS) == registry['latents_sha256'] and sha(OLD_Z) == registry['excluded_original_four_sha256'] == base.EVAL_HASH
    for relative, expected in registry['pins'].items(): assert sha(ROOT / relative) == expected, relative
    with np.load(LATENTS, allow_pickle=False) as f: z = f['z'].copy()
    with np.load(OLD_Z, allow_pickle=False) as f: old = f['z'].copy()
    assert check_latents(z, old) == registry['row_sha256']
    return registry, z, old


def check_envelope(payload, marker, protocol, protocol_sha, runtime_sha, inventory_sha, digest):
    """No state mutation; tested using declared synthetic envelopes."""
    assert digest(payload) == marker['state_sha256']
    assert payload['protocol_sha256'] == protocol_sha
    assert payload['runtime_sha256'] == runtime_sha and payload['modulation_inventory_sha256'] == inventory_sha
    assert payload['source_main_checkpoint_sha256'] == protocol['source_main_checkpoint_sha256']
    state = payload['engine']
    assert state['iterations'] == 100 and state['optimizer_steps'] == base.expected_steps(100)
    assert state['selections'] == protocol['selections']
    assert all(k in state for k in ('G', 'Gema', 'D', 'g_optimizer', 'd_optimizer', 'torch_rng', 'sampling_rng', 'path_mean', 'controller_enabled', 'optimizer_parameter_names'))
    return state


def validate_source(run, registry):
    run = run.resolve(); run.relative_to(ROOT / 'research/runs')
    protocol = read(run/'protocol.json'); ph = sha(run/'protocol.json')
    result = read(run/'result.json'); supervisor = read(run/'supervisor.json'); marker = read(run/'checkpoint-100.json')
    assert protocol['name'] == SOURCE_NAME and protocol['iterations'] == 100
    assert protocol['importance_source_step'] == 250 and protocol['importance_source_run'] == base.RETAINED_SOURCE_REL
    assert protocol['fixed_offset_policy'] == POLICY and protocol['conv_layout'] == 'output_rank1'
    assert protocol['g_weights_sha256'] == registry['g_weights_sha256'] == base.G_HASH
    assert (protocol['resolution'],protocol['device'],protocol['threads']) == (1024,'cpu',1)
    assert protocol['checkpoint_steps'] == [50,100]
    for name in ('main_adaptation_fixed_offsets250_10.py','continue_main_adaptation_fixed_offsets250_100.py','modulation.py','fixed_offset_policy.py'):
        key = rel(EXPERIMENTS/name)
        assert protocol['pins'][key] == registry['pins'][key]
    for relative, expected in protocol['pins'].items(): assert sha(ROOT/relative) == expected, relative
    assert result['complete'] and result['iterations'] == 100 and result['main_adaptation_run']
    assert result['fixed_offsets_and_mask_invariants'] and result['optimizer_steps'] == base.expected_steps(100)
    assert result['folded_native_max_errors'] == {'raw':[0.0]*4,'ema':[0.0]*4}
    assert supervisor['complete'] and supervisor['failure'] is None
    assert marker['complete'] and marker['step'] == 100 and marker['checkpoint'] == 'checkpoint-100.pt'
    assert marker['optimizer_steps'] == base.expected_steps(100) and marker['model_optimizer_rng_path_state_restored_exactly']
    for record in (result,supervisor,marker): assert record['protocol_sha256'] == ph
    assert sha(run/'checkpoint-100.pt') == marker['checkpoint_sha256']
    assert sha(run/'step-100/manifest.json') == marker['snapshot_manifest_sha256']
    snap = read(run/'step-100/manifest.json')
    assert snap['complete'] and snap['step'] == 100 and snap['rng_unchanged'] and snap['protocol_sha256'] == ph
    assert len(snap['images']) == 8 and {im['path'] for im in snap['images']} == {f'{a}-{i:03}.png' for a in ('raw','ema') for i in range(4)}
    for im in snap['images']: assert sha(run/'step-100'/im['path']) == im['sha256']
    assert sha(base.GDIR/'generator.safetensors') == base.G_HASH
    assert sha(run/'eval-z.npz') == base.EVAL_HASH
    return protocol, marker, snap


def image_bytes(value):
    import io
    from PIL import Image
    # Exactly the frozen runner's pixel conversion; no crop, enhancement or restoration.
    pixels = ((value[0].detach().permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
    out = io.BytesIO(); Image.fromarray(pixels).save(out, format='PNG'); return out.getvalue()


def forward(model, latent):
    return model(latent[None],None,truncation_psi=1,noise_mode='const',force_fp32=True,fused_modconv=False)


def worker(run, out):
    import torch
    from safetensors.torch import load_file
    from modulation import install_modulation
    from fixed_offset_policy import capture_fixed_reference, verify_fixed_offsets
    from adaptation_masks_fixed_offsets import FixedOffsetsAdaptationMasks
    sys.path.insert(0,str(ROOT/'research/vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator
    registry,z,old = registered_latents(); protocol,marker,snap = validate_source(run,registry)
    assert sha(out/'render-protocol.json') == read(out/'launch.json')['render_protocol_sha256']
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    runtime = read(run/'runtime.json')
    assert runtime['torch'] == torch.__version__ and runtime['python'] == sys.version
    payload = torch.load(run/'checkpoint-100.pt',weights_only=True,map_location='cpu',mmap=True)
    state = check_envelope(payload,marker,protocol,sha(run/'protocol.json'),sha(run/'runtime.json'),sha(run/'modulation-inventory.json'),base.state_digest)
    G = Generator(**read(base.GDIR/'model.json')['init_kwargs']).cpu().eval().requires_grad_(False)
    G.load_state_dict(load_file(str(base.GDIR/'generator.safetensors')),strict=True)
    inventory = install_modulation(G,component='G',conv_layout='output_rank1')
    assert inventory == read(run/'modulation-inventory.json')['G']
    references = capture_fixed_reference(G); assert len(references) == 34
    # Same masks retain authenticated original-row/buffer invariants; no optimizer is constructed.
    control = FixedOffsetsAdaptationMasks(G,protocol['selections']['G']);control.set_enabled(False)
    records=[]; reproduction=[]; old_hashes={im['path']:im['sha256'] for im in snap['images']}
    for arm,key in [('raw','G'),('ema','Gema')]:
        G.load_state_dict(state[key],strict=True);control.verify_invariants();verify_fixed_offsets(G,references)
        rng_before = torch.get_rng_state()
        with torch.no_grad():
            # Reproduce the known four first; do not rebaseline or write duplicate previews.
            for i,latent in enumerate(old):
                digest=hashlib.sha256(image_bytes(forward(G,torch.from_numpy(latent.copy())))).hexdigest()
                assert digest == old_hashes[f'{arm}-{i:03}.png'], 'Native checkpoint reproduction differs'
                reproduction.append({'arm':arm,'index':i,'sha256':digest})
            for i,latent in enumerate(z):
                data=image_bytes(forward(G,torch.from_numpy(latent.copy())))
                path=out/f'{arm}-{i:03}.png';tmp=Path(str(path)+'.partial');tmp.write_bytes(data);tmp.replace(path)
                records.append({'arm':arm,'index':i,'path':path.name,'sha256':hashlib.sha256(data).hexdigest(),'width':1024,'height':1024,'latent_sha256':registry['row_sha256'][i]})
                print(json.dumps({'completed_arm':arm,'index':i}),flush=True)
        assert torch.equal(rng_before,torch.get_rng_state())
        control.verify_invariants();verify_fixed_offsets(G,references)
    write(out/'manifest.json',{'complete':True,'count':COUNT,'image_count':64,'images':records,
        'preregistration_sha256':sha(REGISTRY),'render_protocol_sha256':sha(out/'render-protocol.json'),
        'checkpoint_sha256':marker['checkpoint_sha256'],'source_protocol_sha256':sha(run/'protocol.json'),
        'parent_reproduction_exact':True,'reproduction':reproduction,'rng_unchanged':True,'fixed_offset_and_mask_invariants':True,
        'unfiltered':True,'all_native_whole_images':True,'production_approved':False,'server_latency_proven':False})


def supervise(run,out):
    import psutil
    assert not out.exists(), 'Use a fresh archive directory; no overwrite or resume'
    registry,_,_ = registered_latents();protocol,marker,_ = validate_source(run,registry)
    assert psutil.virtual_memory().available/psutil.virtual_memory().total >= .35
    out.mkdir(parents=True);(out/'.gitignore').write_text('*.partial\n')
    write(out/'render-protocol.json',{'source_run':rel(run),'source_protocol_sha256':sha(run/'protocol.json'),
        'checkpoint_path':rel(run/'checkpoint-100.pt'),'checkpoint_sha256':marker['checkpoint_sha256'],
        'preregistration_path':rel(REGISTRY),'preregistration_sha256':sha(REGISTRY),'latents_sha256':sha(LATENTS),
        'source_weights_sha256':base.G_HASH,'license_notice':registry['license_notice'],'pins':registry['pins'],
        'fixed_offset_policy':POLICY,'count':COUNT,'native_resolution':1024,'arms':['raw','ema'],
        'sampling':registry['sampling'],'guards':{'rss_gib':6,'seconds':600,'available_fraction':.20,'swap_growth_mib':512},
        'automatic_restart':False,'production_approved':False,'server_latency_proven':False})
    write(out/'launch.json',{'render_protocol_sha256':sha(out/'render-protocol.json')})
    started=time.monotonic();swap=psutil.swap_memory().used;peak=0;failure=None;proc=None
    try:
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'render.py'),'--worker','--run',str(run),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                try:
                    p=psutil.Process(proc.pid);rss=p.memory_info().rss+sum(c.memory_info().rss for c in p.children(recursive=True))
                except psutil.NoSuchProcess:continue
                peak=max(peak,rss);m=psutil.virtual_memory()
                if rss>6*2**30:failure='RSS guard'
                elif m.available/m.total<.20:failure='Available memory guard'
                elif psutil.swap_memory().used-swap>512*2**20:failure='Swap growth guard'
                elif time.monotonic()-started>600:failure='Wall time guard'
                if failure:os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(.2)
            code=proc.wait()
        if code and failure is None:failure=f'worker exit {code}'
        if failure is None:assert read(out/'manifest.json')['complete']
    except BaseException as exc:
        failure=repr(exc)
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        write(out/'supervisor.json',{'complete':failure is None,'failure':failure,'peak_rss_gib':peak/2**30,'seconds':time.monotonic()-started,'render_protocol_sha256':sha(out/'render-protocol.json')})
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preregister',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.preregister:
        assert not a.worker and a.run is None and a.output is None;preregister()
    else:
        if a.run is None or a.output is None:p.error('--run and --output are required for an explicit render')
        if a.worker:worker(a.run.resolve(),a.output.resolve())
        else:supervise(a.run.resolve(),a.output.resolve())
