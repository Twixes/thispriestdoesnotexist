"""PREPARED G-only native bias-family ablation on a common noise-offset reset; no training/image postprocessing.

Loads checkpoint100 raw G, never EMA/D models, preserving source_flattened layout.
All four parent latents and all five arms are retained. A controlled inference
intervention, not a production proposal or a claim of causality before review.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT/'research/runs/adam-native1024-probing500-v1'
GMETA = ROOT/'research/runs/inference-cpu/ffhq1024/baseline-bundle/model.json'
MODULATION_SHA = 'b032f7965f397a5ebdc93fb1081adaeace651d31a614347123ab83fc46ce456a'
CHECKPOINT_SHA = 'd406240601ec748f95ad09c2b2799106d04f0510cd9383526bdc0618bd7416ba'
PARENT_PROTOCOL_SHA = '9fd791ccd1e01c2bfa93dfa41a3119f052f873b48d998eb4b52168f2b68ac9ca'
GIB = 2**30
CONTROL = ROOT/'research/runs/adam-native1024-noise-ablation100-v1'
CONTROL_PROTOCOL_SHA = '4541ef0050e08498ea47ddd38ade166afc2fe5a206f7f21e49fa24743ce6a987'
CONTROL_RESULT_SHA = 'd79ace00890cf07280acf359feb0c46ecda76e99e624665d406d373c23060eb4'
SOURCE_ABLATION_SHA = '6c031d25c297c09f69966cec3f8808f86824c85997df94ab932f244f485e44e8'
ARMS = ('reset_all_noise_offsets', 'noise_plus_torgb_bias', 'noise_plus_synthesis_bias',
        'noise_plus_style_affine_bias', 'noise_plus_mapping_bias')
FAMILY_BY_ARM = dict(zip(ARMS[1:], ('torgb_bias', 'synthesis_bias', 'style_affine_bias', 'mapping_bias')))


def offset_families(names):
    """Partition only additive offsets; exact paths prevent accidental family overlap."""
    patterns = {
        'noise': r'synthesis\.b[0-9]+\.conv[01]\.parametrizations\.noise_strength\.0\.b_vector',
        'torgb_bias': r'synthesis\.b[0-9]+\.torgb\.parametrizations\.bias\.0\.b_vector',
        'synthesis_bias': r'synthesis\.b[0-9]+\.conv[01]\.parametrizations\.bias\.0\.b_vector',
        'style_affine_bias': r'synthesis\.b[0-9]+\.(conv[01]|torgb)\.affine\.parametrizations\.bias\.0\.b_vector',
        'mapping_bias': r'mapping\.fc[0-9]+\.parametrizations\.bias\.0\.b_vector',
    }
    groups = {family: [] for family in patterns}
    for name in sorted(names):
        if not name.endswith('.b_vector'):
            continue
        matches = [family for family, pattern in patterns.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f'Unclassified or overlapping additive offset: {name}')
        groups[matches[0]].append(name)
    return groups


def reset_names_for_arm(families, arm):
    if arm not in ARMS:
        raise ValueError('Unknown intervention arm')
    return sorted(families['noise'] + ([] if arm == ARMS[0] else families[FAMILY_BY_ARM[arm]]))



def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    tmp = Path(str(path)+'.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(out):
    import psutil
    assert not out.exists(), 'Use a new output directory'
    # Execution is serialized after the control is terminal, including failures.
    terminal = json.loads((PARENT/'supervisor.json').read_text())
    assert 'complete' in terminal and 'failure' in terminal, 'Control must be terminal before ablation'
    memory = psutil.virtual_memory()
    assert memory.available/memory.total >= .35, 'Need35% available memory'
    marker = json.loads((PARENT/'checkpoint-100.json').read_text())
    assert marker['complete'] and marker['step'] == 100
    assert marker['checkpoint'] == 'checkpoint-100.pt'
    assert marker['checkpoint_sha256'] == CHECKPOINT_SHA == sha(PARENT/'checkpoint-100.pt')
    assert marker['protocol_sha256'] == PARENT_PROTOCOL_SHA == sha(PARENT/'protocol.json')
    assert marker['snapshot_manifest_sha256'] == sha(PARENT/'step-100/manifest.json')
    parent_protocol = json.loads((PARENT/'protocol.json').read_text())
    inventory = json.loads((PARENT/'modulation-inventory.json').read_text())['G']
    assert inventory['conv_layout'] == 'source_flattened'
    assert sha(HERE/'modulation.py') == sha(PARENT/'modulation.py') == MODULATION_SHA
    assert parent_protocol['pins']['research/experiments/adam_native/modulation.py'] == MODULATION_SHA
    assert sha(GMETA) == parent_protocol['pins'][str(GMETA.relative_to(ROOT))]
    snapshot = json.loads((PARENT/'step-100/manifest.json').read_text())
    assert snapshot['complete'] and snapshot['step'] == 100
    assert snapshot['protocol_sha256'] == PARENT_PROTOCOL_SHA
    assert snapshot['eval_z_sha256'] == sha(PARENT/'eval-z.npz')
    archived = {e['path']: e['sha256'] for e in snapshot['images'] if e['path'].startswith('raw-')}
    assert set(archived) == {f'raw-{i:03}.png' for i in range(4)}
    for p, h in archived.items():
        assert sha(PARENT/'step-100'/p) == h
    assert sha(HERE/'noise_ablation.py') == SOURCE_ABLATION_SHA
    assert sha(CONTROL/'protocol.json') == CONTROL_PROTOCOL_SHA
    assert sha(CONTROL/'result.json') == CONTROL_RESULT_SHA
    control_result = json.loads((CONTROL/'result.json').read_text())
    assert control_result['complete'] and control_result['original_state_restored_exact']
    assert control_result['unchanged_raw_parent_pngs_exact']
    assert control_result['protocol_sha256'] == CONTROL_PROTOCOL_SHA
    control_supervisor = json.loads((CONTROL/'supervisor.json').read_text())
    assert control_supervisor['complete'] and control_supervisor['failure'] is None
    control_arm = json.loads((CONTROL/'reset_all_noise_offsets/manifest.json').read_text())
    assert control_arm['complete'] and control_arm['all_other_state_exact']
    assert control_arm['protocol_sha256'] == CONTROL_PROTOCOL_SHA
    assert len(control_arm['declared_resets']) == 17
    control_images = {entry['index']: entry for entry in control_arm['images']}
    assert set(control_images) == set(range(4))
    for entry in control_images.values():
        assert entry['arm'] == 'reset_all_noise_offsets'
        assert entry['path'] == f"reset_all_noise_offsets/{entry['index']:03}.png"
        assert sha(CONTROL/entry['path']) == entry['sha256']
    sources = [HERE/'bias_ablation.py', HERE/'noise_ablation.py', HERE/'modulation.py', GMETA,
        CONTROL/'protocol.json', CONTROL/'result.json', CONTROL/'supervisor.json',
        CONTROL/'reset_all_noise_offsets/manifest.json',
        *[CONTROL/entry['path'] for entry in control_images.values()],
        PARENT/'checkpoint-100.json', PARENT/'protocol.json', PARENT/'modulation-inventory.json',
        PARENT/'runtime.json', PARENT/'supervisor.json', PARENT/'eval-z.npz', PARENT/'step-100/manifest.json']
    sources += sorted((ROOT/'research/vendor/stylegan2-ada-pytorch').rglob('*.py'))
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for p, h in pins.items():
        if p.startswith('research/vendor/'):
            assert parent_protocol['pins'][p] == h, 'Vendor differs from parent render'
    protocol = {'name':'checkpoint100-rawG-bias-family-ablation', 'checkpoint':str((PARENT/'checkpoint-100.pt').relative_to(ROOT)),
        'checkpoint_sha256':CHECKPOINT_SHA, 'checkpoint_state':'G', 'checkpoint_step':100,
        'parent_protocol_sha256':PARENT_PROTOCOL_SHA, 'parent_modulation_sha256':MODULATION_SHA,
        'pins':pins, 'resolution':1024, 'device':'cpu', 'threads':1, 'interop_threads':1,
        'conv_layout':'source_flattened', 'arms':list(ARMS), 'latent_count':4,
        'all_noise_offset_count':17, 'bias_family_counts':{'torgb_bias':9,'synthesis_bias':17,'style_affine_bias':26,'mapping_bias':8},
        'common_background':'all17 learned additive noise-strength offsets zeroed; original source strengths retained',
        'source_ablation_sha256':SOURCE_ABLATION_SHA, 'control_protocol_sha256':CONTROL_PROTOCOL_SHA,
        'control_png_sha256':{str(i):entry['sha256'] for i,entry in control_images.items()},
        'archived_raw_png_sha256':archived,
        'render':{'truncation_psi':1,'noise_mode':'const','force_fp32':True,'fused_modconv':False},
        'guards':{'start_available_fraction':.35,'runtime_available_fraction':.25,'rss_gib':4,
                  'swap_growth_mib':512,'seconds':600},
        'no_training':True, 'no_image_editing_or_filtering':True, 'production_approved':False,
        'interpretation':'Prepared controlled inference interventions; inspect all native results before conclusions'}
    out.mkdir(parents=True)
    (out/'.gitignore').write_text('*.partial\n')
    write(out/'protocol.json',protocol)
    shutil.copy2(HERE/'bias_ablation.py',out/'bias_ablation.py')
    shutil.copy2(HERE/'noise_ablation.py',out/'source-noise_ablation.py')
    shutil.copy2(HERE/'modulation.py',out/'modulation.py')
    return protocol


def supervise(out):
    import psutil
    protocol = prepare(out)
    start=time.monotonic(); swap=psutil.swap_memory().used
    proc=None; failure=None; peak=0; low=1.
    try:
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'bias_ablation.py'),'--worker','--output',str(out)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                memory=psutil.virtual_memory()
                try:
                    p=psutil.Process(proc.pid)
                    rss=p.memory_info().rss+sum(c.memory_info().rss for c in p.children(recursive=True))
                except psutil.NoSuchProcess:
                    continue
                peak=max(peak,rss);low=min(low,memory.available/memory.total)
                if rss>4*GIB:failure='RSS guard'
                elif memory.available/memory.total<.25:failure='available memory guard'
                elif psutil.swap_memory().used-swap>512*2**20:failure='swap growth guard'
                elif time.monotonic()-start>600:failure='wall time guard'
                if failure:
                    os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=proc.wait()
        if code != 0 and failure is None:failure=f'worker exit {code}'
        if failure is None:
            result=json.loads((out/'result.json').read_text())
            assert result['complete'] and len(result['images'])==20
            assert result['protocol_sha256']==sha(out/'protocol.json')
            for image in result['images']:
                assert sha(out/image['path'])==image['sha256']
    except BaseException:
        failure=traceback.format_exc()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        write(out/'supervisor.json',{'complete':failure is None,'failure':failure,'seconds':time.monotonic()-start,
            'peak_rss_gib':peak/GIB,'min_available_fraction':low,'protocol_sha256':sha(out/'protocol.json')})
    if failure:raise RuntimeError(failure)


def apply_offset_arm(model, original_offsets, reset_names):
    """Restore every additive offset to raw checkpoint values, then zero declared offsets only."""
    import torch
    if not set(reset_names) <= set(original_offsets):
        raise ValueError('Reset list includes undeclared parameter')
    with torch.no_grad():
        for name, original in original_offsets.items():
            model.get_parameter(name).copy_(original)
        for name in reset_names:
            model.get_parameter(name).zero_()


def worker(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['pins'].items():assert sha(ROOT/p)==h,p
    import gc
    import platform
    import numpy as np
    import torch
    from PIL import Image, __version__ as pillow_version
    sys.path.insert(0,str(ROOT/'research/vendor/stylegan2-ada-pytorch'))
    from training.networks import Generator
    from modulation import install_modulation
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(2026092209)
    write(out/'runtime.json',{'torch':torch.__version__,'numpy':np.__version__,'pillow':pillow_version,
        'python':sys.version,'platform':platform.platform(),'threads':torch.get_num_threads(),
        'interop':torch.get_num_interop_threads(),'grad_enabled_during_render':False,
        'checkpoint_load':'weights_only=True,map_location=cpu,mmap=True; only G copied into instantiated model; no D/EMA model'})
    parent_runtime=json.loads((PARENT/'runtime.json').read_text())
    assert torch.__version__==parent_runtime['torch'], 'Match parent Torch runtime'
    tick=time.monotonic()
    assert sha(PARENT/'checkpoint-100.pt')==CHECKPOINT_SHA, 'Checkpoint bytes changed before worker load'
    checkpoint=torch.load(PARENT/'checkpoint-100.pt',map_location='cpu',weights_only=True,mmap=True)
    assert checkpoint['iterations']==100 and checkpoint['protocol_sha256']==PARENT_PROTOCOL_SHA
    assert checkpoint['modulation_inventory_sha256']==sha(PARENT/'modulation-inventory.json')
    state=checkpoint['G'];del checkpoint;gc.collect()
    metadata=json.loads(GMETA.read_text())
    g=Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
    inventory=json.loads((PARENT/'modulation-inventory.json').read_text())['G']
    installed=install_modulation(g,component='G',conv_layout=inventory['conv_layout'],
        seed=inventory['initialization_seed'],init_scale=inventory['init_scale'])
    assert installed['targets']==inventory['targets']
    g.load_state_dict(state,strict=True);g.eval().requires_grad_(False)
    del state;gc.collect()
    load_seconds=time.monotonic()-tick
    def tensor_hash(t):
        t=t.detach().cpu().contiguous()
        h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        return h.hexdigest()
    def state_hashes():return {n:tensor_hash(t) for n,t in g.state_dict().items()}
    original_hashes=state_hashes()
    offsets={n:p.detach().clone() for n,p in g.named_parameters() if n.endswith('.b_vector')}
    families=offset_families(offsets)
    assert {name:len(values) for name,values in families.items()} == {'noise':17, **protocol['bias_family_counts']}
    assert sum(len(values) for values in families.values()) == len(offsets) == 77
    assert all(offsets[n].numel()==1 for n in families['noise'])
    assert all(g.get_parameter(n).numel()==3 for n in families['torgb_bias'])
    for name in offsets:
        original_name=name.replace('.0.b_vector','.original')
        assert original_name in original_hashes
    noise_values={n:{'offset':float(t),'original':float(g.get_parameter(n.replace('.0.b_vector','.original')))} for n,t in offsets.items() if n in families['noise']}
    write(out/'loaded-state.json',{'load_seconds':load_seconds,'state_sha256_by_tensor':original_hashes,
        'noise_offsets':noise_values,'offset_families':families,'checkpoint_sha256':CHECKPOINT_SHA,'layout':inventory['conv_layout']})
    with np.load(PARENT/'eval-z.npz',allow_pickle=False) as fixture:
        z=torch.from_numpy(fixture['z'].copy())
    assert z.shape==(4,512) and z.dtype==torch.float32 and torch.isfinite(z).all()
    np.savez(out/'eval-z.npz',z=z.numpy())
    images=[];arm_records=[]
    with torch.no_grad():
        for arm in ARMS:
            reset=reset_names_for_arm(families,arm)
            apply_offset_arm(g,offsets,[])
            assert state_hashes()==original_hashes, 'Previous arm not fully restored'
            apply_offset_arm(g,offsets,reset)
            hashes=state_hashes()
            changed=[n for n in hashes if hashes[n]!=original_hashes[n]]
            assert set(changed)=={n for n in reset if torch.count_nonzero(offsets[n])}, 'Intervention state support differs from declared nonzero offsets'
            assert all(not torch.count_nonzero(g.get_parameter(n)) for n in reset)
            folder=out/arm;folder.mkdir()
            rng=torch.get_rng_state().clone()
            arm_images=[]
            for i,latent in enumerate(z):
                tick=time.monotonic()
                pixels=g(latent[None],None,**protocol['render'])
                generate_seconds=time.monotonic()-tick
                assert pixels.shape==(1,3,1024,1024) and torch.isfinite(pixels).all()
                array=((pixels.cpu()[0].permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
                path=folder/f'{i:03}.png';tmp=Path(str(path)+'.partial')
                Image.fromarray(array).save(tmp,format='PNG');tmp.replace(path)
                record={'arm':arm,'index':i,'path':str(path.relative_to(out)),'sha256':sha(path),
                    'generation_seconds':generate_seconds,'latent_tensor_sha256':tensor_hash(latent)}
                if arm==ARMS[0]:
                    expected=protocol['control_png_sha256'][str(i)]
                    with Image.open(CONTROL/'reset_all_noise_offsets'/f'{i:03}.png') as old:
                        assert np.array_equal(array,np.array(old)), 'Common noise-reset control pixels differ'
                    assert record['sha256']==expected, 'Common noise-reset control PNG bytes differ'
                    record['noise_reset_control_png_exact']=True
                arm_images.append(record);images.append(record)
                del pixels,array
            assert torch.equal(rng,torch.get_rng_state()), 'Constant-noise render changed RNG'
            assert state_hashes()==hashes, 'Rendering changed model state'
            record={'arm':arm,'complete':True,'declared_resets':reset,'actual_changed_state_names':changed,
                'all_other_state_exact':True,'images':arm_images,'protocol_sha256':sha(out/'protocol.json')}
            write(folder/'manifest.json',record);arm_records.append(record)
    apply_offset_arm(g,offsets,[])
    assert state_hashes()==original_hashes
    write(out/'result.json',{'complete':True,'images':images,'arms':arm_records,'original_state_restored_exact':True,
        'all_noise_reset_control_pngs_exact':True,'protocol_sha256':sha(out/'protocol.json'),
        'production_approved':False,'visual_review_pending':True,'server_latency_proven':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:
        try:worker(out)
        except BaseException:
            write(out/'failure.json',{'complete':False,'traceback':traceback.format_exc()});raise
    else:supervise(out)
