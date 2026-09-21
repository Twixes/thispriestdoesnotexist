"""Unfiltered reference625 development sampling; CPU only, no restoration/selection."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HELPER=ROOT/'research/experiments/reference625_evaluation/evaluate.py'
HELPER_SHA='fe33cb613315a1b1061c86b88ead72558ec79533b559827f2b28caf18c9fb584'
CHECKPOINT_SHA='c04482a462abf4ece7034cdc09be19dc857ee201b11427f6526c49ae378bf6dc'
CONFIG_SHA='b330c97f252c2b0f20d041923900676c806cfdac6b47241e062f50cabef09997'
PREVIOUS=ROOT/'research/runs/reference256-paper-b64-resumed500-to625/matched32-000625'
LIMIT=6*1024**3
SECONDS=600
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise RuntimeError(message)
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def helper():
    require(sha(HELPER)==HELPER_SHA,'Pinned reference helper changed')
    spec=importlib.util.spec_from_file_location('reference625_sampling_helpers',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def preflight():
    h=helper();pins=h.verify_pins();future=h.verify_future(CHECKPOINT_SHA,CONFIG_SHA)
    memory=subprocess.check_output(['memory_pressure'],text=True)
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',memory)
    require(match and int(match[1])>=35,'Require >=35% free system memory')
    old=read(PREVIOUS/'evaluation.json');sup=read(PREVIOUS/'supervisor-result.json')
    require(old['complete'] and sup['complete'] and sup['worker_exit_code']==0,'Control evaluation incomplete')
    require(old['checkpoint_sha256']==CHECKPOINT_SHA and old['config_sha256']==CONFIG_SHA,'Control model mismatch')
    control=next(x for x in old['entries'] if x['variant']=='raw-psi1' and x['index']==0)
    require(sha(PREVIOUS/control['path'])==control['sha256'],'Control image changed')
    return h,{'memory':{'free_percent':int(match[1]),'raw':memory},'pins':pins,'future':future,'control':control,
             'control_evaluation_sha256':sha(PREVIOUS/'evaluation.json')}

def guard(deadline):
    require(time.time()<deadline,'Ten-minute deadline expired')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMIT,'Worker peak RSS limit')

def worker(args):
    launch=read(args.output/'launch.json')
    require(os.getppid()==launch['supervisor_pid'] and sha(Path(__file__))==launch['source_sha256'],'Worker ownership/source mismatch')
    deadline=launch['deadline_unix'];guard(deadline)
    h,verified=preflight()
    require(verified['pins']==launch['verified']['pins'] and verified['future']==launch['verified']['future'],'Worker provenance mismatch')
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(h.VENDOR))
    from training.networks import Generator
    started=time.monotonic()
    config=read(h.CONFIG);constructor=ast.literal_eval(config['architecture']['G'])
    require(all(constructor[k]==v for k,v in {'z_dim':512,'c_dim':0,'w_dim':512,'img_resolution':256,'img_channels':3}.items()),'Unexpected generator')
    # Hash-pinned own trusted checkpoint contains NumPy RNG state, as in the original evaluator.
    state=torch.load(h.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    require(state['format_version']==1 and state['batch_idx']==625 and state['images_seen']==40000,'Wrong checkpoint step')
    require(state['config']==config and state['recipe']==config['recipe'] and state['sampler']['consumed']==40000,'Checkpoint configuration mismatch')
    seeds=np.arange(args.seed_base,args.seed_base+args.count,dtype=np.uint64)
    z=np.stack([np.random.Generator(np.random.PCG64(int(seed))).standard_normal(512).astype(np.float32) for seed in seeds])
    require(len({row.tobytes() for row in z})==args.count,'Duplicate new latent')
    previous_rows={row.tobytes() for row in state['fixed_z'].numpy()}
    previous_seeds=set();inventory=[]
    for item in launch['prior_npz']:
        path=ROOT/item['path'];require(sha(path)==item['sha256'],'Prior latent archive changed')
        count=0;keys=[]
        with np.load(path,allow_pickle=False) as saved:
            for key in saved.files:
                if key=='seeds':
                    values=saved[key]
                    if values.dtype.kind in 'iu':previous_seeds.update(int(v) for v in values.reshape(-1))
                if key=='z' or key.endswith('_z') or key.startswith('z_'):
                    a=saved[key]
                    if a.dtype==np.float32 and a.ndim>=1 and a.shape[-1]==512:
                        rows=a.reshape(-1,512);previous_rows.update(row.tobytes() for row in rows);count+=len(rows);keys.append(key)
        inventory.append({**item,'z_rows':count,'keys':keys})
    require(not(set(map(int,seeds)) & previous_seeds),'Previously recorded seed reused')
    require(not any(row.tobytes() in previous_rows for row in z),'Previously recorded latent reused')
    write(args.output/'latent-disjointness.json',{'archives':inventory,'unique_prior_z_rows':len(previous_rows),'prior_numeric_seeds':len(previous_seeds),'new_count':args.count,'all_new_distinct':True,'limitation':'Checks saved float32 z arrays and numeric seed arrays, not every unsaved historical training draw.'})
    np.savez(args.output/'latents.npz',z=z,seeds=seeds)
    model=Generator(**constructor).cpu().eval().requires_grad_(False)
    model.load_state_dict(state['G'],strict=True)
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),'Nonfinite model state')
    source_state=state['G'];del state
    def render(row,path):
        guard(deadline);tick=time.monotonic()
        with torch.inference_mode():out=model(torch.from_numpy(row[None]),None,truncation_psi=1,noise_mode='const',force_fp32=True)
        require(tuple(out.shape)==(1,3,256,256) and bool(torch.isfinite(out).all()),'Invalid generated pixels')
        pixels=((out[0]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy()
        with path.open('xb') as f:Image.fromarray(pixels,'RGB').save(f,format='PNG')
        return {'path':path.name,'sha256':sha(path),'seconds':time.monotonic()-tick,'raw_min':float(out.min()),'raw_max':float(out.max())}
    with np.load(h.LATENTS,allow_pickle=False) as saved:control_z=saved['z'][0].copy()
    control=render(control_z,args.output/'control-raw-psi1-000.png')
    require(control['sha256']==verified['control']['sha256'],'Control PNG must match exactly before new generation')
    write(args.output/'control.json',{**control,'matches_original_bytes':True,'original':str((PREVIOUS/verified['control']['path']).relative_to(ROOT))})
    print(json.dumps({'control_exact':True,'sha256':control['sha256']}),flush=True)
    entries=[]
    for index,row in enumerate(z):
        record={'index':index,'seed':int(seeds[index]),**render(row,args.output/f'{index:03}.png')}
        entries.append(record);print(json.dumps(record),flush=True)
    for page,start in enumerate(range(0,args.count,32)):
        subset=entries[start:start+32];sheet=Image.new('RGB',(8*256,4*278),'black');draw=ImageDraw.Draw(sheet)
        for j,entry in enumerate(subset):
            with Image.open(args.output/entry['path']) as im:sheet.paste(im,((j%8)*256,(j//8)*278))
            draw.text(((j%8)*256+5,(j//8)*278+257),f"{entry['index']:03}",fill='white')
        sheet.save(args.output/f'contact-{page:02}.png')
    require(all(torch.equal(v,source_state[k]) for k,v in model.state_dict().items()),'Generator state changed')
    require(sha(h.CHECKPOINT)==CHECKPOINT_SHA and sha(h.CONFIG)==CONFIG_SHA and sha(Path(__file__))==launch['source_sha256'],'Source changed during run')
    require(h.verify_pins()==verified['pins'],'Pinned inputs changed')
    guard(deadline)
    write(args.output/'evaluation.json',{'complete':True,'development_only':True,'quality_approved':False,'production_approved':False,
      'resolution':256,'count':args.count,'seed_base':args.seed_base,'checkpoint_sha256':CHECKPOINT_SHA,'config_sha256':CONFIG_SHA,'constructor':constructor,
      'source_sha256':launch['source_sha256'],'checkpoint_step':625,'state':'G','psi':1,'noise_mode':'const','force_fp32':True,'device':'cpu','threads':1,'interop_threads':1,
      'sampling':'Independent PCG64(seed).standard_normal(512) float64 then cast float32; same as reference625 evaluator; unfiltered, no restoration or retry',
      'latents_sha256':sha(args.output/'latents.npz'),'exact_control':control,'entries':entries,
      'contacts':{p.name:sha(p) for p in sorted(args.output.glob('contact-*.png'))},
      'worker_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'generator_state_unchanged':True})

def supervise(args):
    require(not args.output.exists(),'Refuse existing output directory')
    started=time.time();deadline=started+SECONDS
    h,verified=preflight()
    # Enumerate existing saved latent archives before creating this cohort.
    files=subprocess.check_output(['rg','--files','research'],cwd=ROOT,text=True).splitlines()
    inventory=[{'path':p,'sha256':sha(ROOT/p)} for p in sorted(files) if p.endswith('.npz')]
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),args.output/'sample-source.py');shutil.copyfile(h.CONFIG,args.output/'config.json')
    command=[sys.executable,str(Path(__file__)),'--output',str(args.output),'--count',str(args.count),'--seed-base',str(args.seed_base),'--worker']
    write(args.output/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':started,'deadline_unix':deadline,'rss_limit':LIMIT,'sampled_not_hard_limit':True,'source_sha256':sha(Path(__file__)),'command':command,'verified':verified,'prior_npz':inventory,'python':sys.version,'environment':ENV})
    process=None;error=None;peak=0;complete=False;reported_peak=None
    try:
        with (args.output/'worker.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                require(time.time()<deadline,'Supervisor deadline expired')
                rss=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True,timeout=2).stdout.strip()
                if rss:peak=max(peak,int(rss)*1024)
                require(peak<LIMIT,'Supervisor sampled RSS limit');time.sleep(.2)
    except BaseException as exc:error=repr(exc)
    finally:
        handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            if process is not None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                finally:process.wait()
            require(process is not None and process.returncode==0,'Worker did not exit0')
            result=read(args.output/'evaluation.json');reported_peak=result['peak_rss_bytes']
            require(result['complete'] and len(result['entries'])==args.count,'Incomplete cohort')
            require(reported_peak<LIMIT and peak<LIMIT and time.time()<deadline,'Final resource limit')
            complete=error is None
        except BaseException as exc:error=error or repr(exc)
        finally:
            try:write(args.output/'supervisor-result.json',{'complete':complete,'error':error,'worker_exit_code':None if process is None else process.returncode,'seconds':time.time()-started,'peak_sampled_rss_bytes':peak,'reported_peak_rss_bytes':reported_peak})
            finally:
                for sig,value in handlers.items():signal.signal(sig,value)
    require(complete,'Sampling failed: '+str(error))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--seed-base',type=int,default=202609211000);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve()
    require(args.output.parent==ROOT/'research/runs' and 1<=args.count<=128 and 0<=args.seed_base<2**63-128,'Invalid bounded sampling request')
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    (worker if args.worker else supervise)(args)
