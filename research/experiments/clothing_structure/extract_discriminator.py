"""Extract pinned trusted-local FFHQ1024 D; research only, explicit execution required.

No torch, legacy or pickle import occurs until the resource/provenance guards pass.
The source pickle can execute code: only its fixed path AND pinned bytes are allowed.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = ROOT / 'research/vendor/stylegan2-ada-pytorch'
SOURCE = ROOT / 'research/models/ffhq1024.pkl'
SOURCE_SHA256 = 'a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c'
CONSTRUCTOR = 'training.networks.Discriminator'
PINS = HERE / 'extract-vendor-pins.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_value(value):
    """Reject executable/non-JSON constructor payloads rather than stringify them."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: json_value(v) for k, v in value.items()}
    raise ValueError('Non-JSON or nonfinite constructor value')


def parse_free_percentage(text):
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if not match or not 0 <= int(match.group(1)) <= 100:
        raise ValueError('Missing/unrecognized macOS memory_pressure percentage')
    return int(match.group(1))


def memory_guard():
    if sys.platform != 'darwin':
        raise RuntimeError('This extractor requires the reviewed macOS memory guard')
    raw = subprocess.run(['memory_pressure'], capture_output=True, text=True,
                         check=True, timeout=15).stdout
    if parse_free_percentage(raw) < 25:
        raise RuntimeError('Defer extraction: less than25% system memory free')
    return raw


def validate_kwargs(init_args, init_kwargs, strict_ffhq=True):
    args, kwargs = json_value(init_args), json_value(init_kwargs)
    if args:
        raise ValueError('Expected fully named discriminator constructor; do not guess positional mapping')
    allowed = {'c_dim','img_resolution','img_channels','architecture','channel_base','channel_max',
               'num_fp16_res','conv_clamp','cmap_dim','block_kwargs','mapping_kwargs','epilogue_kwargs'}
    if set(kwargs) - allowed:
        raise ValueError('Unexpected discriminator constructor fields')
    if strict_ffhq:
        expected = {'c_dim':0,'img_resolution':1024,'img_channels':3}
        for key, value in expected.items():
            if kwargs.get(key) != value:
                raise ValueError(f'Expected {key}={value}')
        for key, default in [('architecture','resnet'),('channel_base',32768),('channel_max',512)]:
            if kwargs.get(key, default) != default:
                raise ValueError(f'Unexpected FFHQ {key}')
        if kwargs.get('block_kwargs', {}).get('freeze_layers', 0) != 0:
            raise ValueError('Expected unfine-tuned, unfrozen original D construction')
    return args, kwargs


def verify_vendor():
    pins = json.loads(PINS.read_text())['files']
    if not {'legacy.py','training/networks.py','LICENSE.txt'} <= pins.keys():
        raise ValueError('Incomplete vendor pin manifest')
    for name, expected in pins.items():
        path = (VENDOR / name).resolve()
        if not path.is_relative_to(VENDOR.resolve()) or digest(path) != expected:
            raise ValueError(f'Vendor checksum mismatch: {name}')
    return pins


def state_inventory(model, torch):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    rows = {}
    for name, tensor in model.state_dict().items():
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
            raise ValueError(f'Expected CPU FP32 source tensor: {name}')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Nonfinite source tensor: {name}')
        rows[name] = {'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                      'numel':tensor.numel(),'bytes':tensor.numel()*tensor.element_size(),
                      'kind':'parameter' if name in parameters else 'buffer' if name in buffers else 'other',
                      'sha256':hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()}
    return {'tensors':rows,'tensor_count':len(rows),'parameter_count':sum(p.numel() for p in parameters.values()),
            'buffer_count':sum(b.numel() for b in buffers.values()),'state_bytes':sum(x['bytes'] for x in rows.values())}


def reconstruct(source, constructor, torch, strict_ffhq=True):
    args, kwargs = validate_kwargs(source.init_args, source.init_kwargs, strict_ffhq)
    model = constructor(*args, **kwargs).cpu().eval().requires_grad_(False)
    model.load_state_dict(source.state_dict(), strict=True)
    before, after = source.state_dict(), model.state_dict()
    if list(before) != list(after) or any(not torch.equal(before[k], after[k]) for k in before):
        raise ValueError('Strict reconstructed discriminator state differs from source')
    if list(dict(source.named_parameters())) != list(dict(model.named_parameters())) or list(dict(source.named_buffers())) != list(dict(model.named_buffers())):
        raise ValueError('Parameter/buffer registration differs after reconstruction')
    return model, args, kwargs


def check_prefix(model, torch, resolution=1024, blocks=(1024,512,256), expected=((1,64,512,512),(1,128,256,256),(1,256,128,128))):
    """One batch-one no-grad ramp through explicit blocks, never D.forward/b4."""
    before = state_inventory(model, torch)
    input_image = torch.linspace(-1,1,resolution*resolution,dtype=torch.float32).reshape(1,1,resolution,resolution).expand(1,3,-1,-1).contiguous()
    outputs=[]; x=None; img=input_image
    started=time.monotonic()
    with torch.no_grad():
        for index, resolution_key in enumerate(blocks):
            x,img=model._modules[f'b{resolution_key}'](x,img,force_fp32=True)
            if tuple(x.shape)!=expected[index] or not bool(torch.isfinite(x).all()) or x.requires_grad:
                raise ValueError('Unexpected prefix shape, nonfinite output, or retained graph')
            outputs.append({'block':f'b{resolution_key}','shape':list(x.shape),'dtype':str(x.dtype),
                            'selected_tap':index>0,'finite':True})
    if state_inventory(model,torch)!=before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen D changed or acquired gradients during prefix check')
    return {'input':'deterministic RGB-repeated horizontal/vertical ramp in[-1,1], batch1',
            'outputs':outputs,'wall_seconds':time.monotonic()-started,'frozen_state_unchanged':True,
            'D_forward_or_epilogue_called':False,'no_grad':True}


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def start_watchdog(output, stop, max_rss_bytes, max_seconds):
    """Process abort watchdog; sampled RSS ceiling, not an OS allocation hard limit."""
    started=time.monotonic()
    def watch():
        while not stop.wait(.25):
            peak=rss_bytes();elapsed=time.monotonic()-started
            if peak>max_rss_bytes or elapsed>max_seconds:
                failure={'complete':False,'reason':'resource watchdog abort','peak_rss_bytes':peak,'elapsed_seconds':elapsed}
                (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                os._exit(70)
    thread=threading.Thread(target=watch,daemon=True);thread.start();return thread


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicitly permit pinned local pickle extraction')
    parser.add_argument('--output',type=Path,required=True,help='NEW directory beneath research/')
    args=parser.parse_args()
    if not args.execute:
        parser.error('No real pickle is loaded without explicit --execute')
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/'research').resolve()) or output.exists():
        parser.error('Output must be a new research subdirectory')
    started=time.monotonic();before=memory_guard();pins=verify_vendor()
    # Hash streaming occurs before imports or unpickling; only this fixed local file.
    if SOURCE.is_symlink() or digest(SOURCE)!=SOURCE_SHA256:
        raise ValueError('Pinned official trusted-local FFHQ1024 pickle mismatch')
    source_provenance=ROOT/'research/models/ffhq1024.provenance.json'
    provenance=json.loads(source_provenance.read_text())
    if provenance['sha256']!=SOURCE_SHA256 or provenance['source_url']!='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl':
        raise ValueError('Official source provenance mismatch')
    before_import=memory_guard()
    output.mkdir(parents=True)
    (output/'memory-before.txt').write_text(before+'\nRECHECK BEFORE IMPORT:\n'+before_import)
    stop=threading.Event();watchdog=start_watchdog(output,stop,8*1024**3,300)
    try:
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20260921)
        from safetensors.torch import load_file, save_file
        sys.path.insert(0,str(VENDOR))
        import legacy
        from training.networks import Discriminator
        if Path(legacy.__file__).resolve()!=VENDOR/'legacy.py':
            raise ValueError('Unexpected legacy loader import location')
        # Trusted-pickle boundary: hash-verified official local bytes only.
        load_started=time.monotonic()
        with SOURCE.open('rb') as stream:
            nets=legacy.load_network_pkl(stream)
        source=nets['D'].cpu().eval().requires_grad_(False)
        del nets
        gc.collect()
        if (source.c_dim,source.img_resolution,source.img_channels)!=(0,1024,3):
            raise ValueError('Source D is not unconditional RGB1024')
        inventory=state_inventory(source,torch)
        reconstructed,init_args,init_kwargs=reconstruct(source,Discriminator,torch)
        load_seconds=time.monotonic()-load_started
        del source
        gc.collect()
        prefix=check_prefix(reconstructed,torch)
        state={k:v.detach().cpu().contiguous() for k,v in reconstructed.state_dict().items()}
        temporary=output/'D.safetensors.tmp';target=output/'D.safetensors'
        save_file(state,str(temporary),metadata={'purpose':'research-only frozen FFHQ1024 discriminator','source_sha256':SOURCE_SHA256})
        reloaded=load_file(str(temporary),device='cpu')
        if set(reloaded)!=set(state) or any(not torch.equal(state[k],reloaded[k]) for k in state):
            raise ValueError('Safetensors roundtrip state differs')
        # Full source reconstruction already checked; serialized values checked independently.
        del reloaded,state
        os.replace(temporary,target)
        shutil.copy2(VENDOR/'LICENSE.txt',output/'LICENSE.txt')
        metadata={'schema_version':1,'complete':True,'research_only':True,'production_approved':False,
                  'constructor':CONSTRUCTOR,'init_args':init_args,'init_kwargs':init_kwargs,
                  'weights_file':'D.safetensors','weights_sha256':digest(target),
                  'source_pickle':str(SOURCE.relative_to(ROOT)),'source_pickle_sha256':SOURCE_SHA256,
                  'source_provenance':provenance,'source_provenance_sha256':digest(source_provenance),
                  'vendor_pins':pins,'vendor_pins_sha256':digest(PINS),'script_sha256':digest(Path(__file__)),
                  'state_inventory':inventory,'strict_source_reconstruction_equal':True,'safetensors_roundtrip_equal':True,
                  'prefix_validation':prefix,'device':'cpu','threads':1,'interop_threads':1,'torch_version':torch.__version__,
                  'platform':platform.platform(),'load_reconstruct_seconds':load_seconds,'total_seconds':time.monotonic()-started,
                  'peak_rss_bytes':rss_bytes(),'watchdog':{'max_rss_bytes':8*1024**3,'max_seconds':300,'poll_seconds':.25,'hard_allocation_limit':False},
                  'license':{'name':'NVIDIA Source Code License; research/evaluation only, not MIT','file':'LICENSE.txt','url':provenance['license_url']},
                  'no_generator_forward':True,'no_training_or_optimizer':True,'epilogue_never_executed':True}
        (output/'metadata.json.tmp').write_text(json.dumps(metadata,indent=2)+'\n')
        os.replace(output/'metadata.json.tmp',output/'metadata.json')
        print(json.dumps({'complete':True,'output':str(output),'peak_rss_bytes':rss_bytes(),'total_seconds':metadata['total_seconds'],'weights_sha256':metadata['weights_sha256']}))
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'complete':False,'error':str(exc),'peak_rss_bytes':rss_bytes()},indent=2)+'\n')
        raise
    finally:
        stop.set();watchdog.join(timeout=1)


if __name__=='__main__':
    main()
