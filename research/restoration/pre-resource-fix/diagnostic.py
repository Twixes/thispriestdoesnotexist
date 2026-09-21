"""Research-only GFPGAN diagnostic; imports ML libraries only in supervised worker."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
LIMIT = 6 * 1024**3
SECONDS = 600
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}
VENDOR = BASE / 'vendor/GFPGAN-7552a7791caad982045a7bbe5634bbf1cd5c8679'
REFERENCE = ROOT / 'research/runs/reference256-paper-b64-resumed500-to625/matched32-000625'
NADA = ROOT / 'research/runs/nada-clean24-cpu-continue500'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024*1024): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def require(ok, message):
    if not ok: raise RuntimeError(message)
def write(path, data):
    with Path(path).open('x') as f: json.dump(data, f, indent=2); f.write('\n')

def memory_guard():
    raw = subprocess.check_output(['memory_pressure'], text=True)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', raw)
    require(match is not None and int(match[1]) >= 35, 'Need fresh >=35% free memory')
    return {'free_percent':int(match[1]), 'raw':raw}

def process_guard():
    rows = subprocess.check_output(['ps','-axo','pid=,ppid=,command='], text=True)
    bad=[]
    for row in rows.splitlines():
        fields=row.strip().split(None,2)
        if len(fields)!=3: continue
        pid,ppid,cmd=fields
        if int(pid) in (os.getpid(),os.getppid()) or int(ppid)==os.getpid(): continue
        if 'python' in cmd.lower() and str(ROOT/'research') in cmd:
            if re.search(r'(runner|worker|train|evaluate|benchmark|validate|diagnostic)\S*\.py',cmd):bad.append(row)
    require(not bad, 'Other model work is active: '+repr(bad))
    return {'conflicting_processes':bad}

def pins_check():
    pins=read(BASE/'pins.json')
    require(sys.version==pins['python_version'], 'Python version changed')
    for name,version in pins['packages'].items():
        require(importlib.metadata.version(name)==version, 'Package version changed: '+name)
    for rel,digest in pins['files'].items(): require(sha(ROOT/rel)==digest,'Pinned file changed: '+rel)
    return pins

def nada_completion(folder, supplied_sha):
    require(bool(re.fullmatch('[0-9a-f]{64}',supplied_sha or '')), 'Supply independently reviewed NADA500 checkpoint SHA')
    sup=read(folder/'supervisor-result.json');worker=read(folder/'worker-result.json');record=read(folder/'checkpoint-500-review-ready.json')
    require(sup['complete'] is True and sup['worker_exit_code']==0, 'NADA supervisor not successful')
    require(worker['complete'] is True and worker['total_step']==500 and worker['restore_only'] is False, 'NADA500 training incomplete')
    require(record['step']==500 and record['actual_student_adam_rng_reload'] is True and record['native_preview_reload_equal'] is True,'NADA checkpoint lacks reload evidence')
    require(record['sha256']==supplied_sha and sha(folder/'checkpoint-500.pt')==supplied_sha,'NADA checkpoint SHA mismatch')
    rows=sorted([x for x in record['previews'] if x['kind']=='student-gray'],key=lambda x:x['latent_index'])
    require([x['latent_index'] for x in rows]==list(range(8)), 'Require every fixed NADA latent, exactly0..7')
    for i,x in enumerate(rows):
        require(x['step']==500 and x['path']==f'preview-500/{i:03}-student-gray.png','Unexpected NADA preview')
        require(sha(folder/x['path'])==x['sha256'],'NADA preview hash mismatch')
    return rows

def cohort_inputs(cohort, supplied_sha):
    if cohort=='reference625':
        e=read(REFERENCE/'evaluation.json');s=read(REFERENCE/'supervisor-result.json')
        require(e['complete'] and e['checkpoint_step']==625 and e['images_seen_in_training']==40000 and s['complete'] and s['worker_exit_code']==0,'Reference625 incomplete')
        rows=sorted([x for x in e['entries'] if x['variant']=='raw-psi1'],key=lambda x:x['index'])
        require([x['index'] for x in rows]==list(range(32)), 'Require all32 raw reference images')
        return [{'index':x['index'],'path':str((REFERENCE/x['path']).relative_to(ROOT)),'sha256':x['sha256'],'raw_challenge_control':x['index'] in (0,3,4,7,8,10,11,13,31)} for x in rows]
    rows=nada_completion(NADA,supplied_sha)
    return [{'index':x['latent_index'],'path':str((NADA/x['path']).relative_to(ROOT)),'sha256':x['sha256'],'raw_challenge_control':None} for x in rows]

def forbid_network(*args,**kwargs): raise RuntimeError('Network is disabled during restoration')

def worker(args):
    launch=read(args.output/'launch.json')
    require(os.getppid()==launch['supervisor_pid'] and args.deadline==launch['deadline_unix'],'Worker must belong to supervisor')
    pins_check();memory_guard();process_guard()
    inputs=read(args.output/'inputs.json')
    for item in inputs: require(sha(ROOT/item['path'])==item['sha256'],'Input changed before load')
    socket.socket.connect=forbid_network
    socket.create_connection=forbid_network
    socket.getaddrinfo=forbid_network
    sys.path.insert(0,str(VENDOR))
    # Network prohibition is in effect before importing helper code or constructing models.
    import random
    import resource
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from basicsr.utils import img2tensor,tensor2img
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    random.seed(1729);np.random.seed(1729);torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    model=GFPGANv1Clean(out_size=512,num_style_feat=512,channel_multiplier=2,decoder_load_path=None,fix_decoder=False,num_mlp=8,input_is_latent=True,different_w=True,narrow=1,sft_half=True)
    state=torch.load(BASE/'weights/GFPGANv1.4.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['params_ema'],strict=True);del state
    model.eval().requires_grad_(False)
    helper=FaceRestoreHelper(upscale_factor=4 if args.cohort=='reference625' else 1,face_size=512,crop_ratio=(1,1),det_model='retinaface_resnet50',save_ext='png',pad_blur=False,use_parse=True,device=torch.device('cpu'),model_rootpath=str(BASE/'weights'))
    load_seconds=time.monotonic()-start
    records=[]
    for item in inputs:
        require(time.time()<args.deadline,'Deadline reached')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMIT,'Peak RSS exceeded limit')
        folder=args.output/f"{item['index']:03}";folder.mkdir()
        shutil.copyfile(ROOT/item['path'],folder/'raw.png')
        record={**item,'error':None,'production_approved':False,'quality_approved':False}
        begin=time.monotonic()
        with Image.open(folder/'raw.png') as im:
            expected=256 if args.cohort=='reference625' else 1024
            require(im.size==(expected,expected),'Unexpected input dimensions')
            control=im.convert('L').convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS)
            control.save(folder/'lanczos-1024.png');control.save(folder/'lanczos-1024.webp',quality=90,method=6)
        try:
            helper.clean_all();helper.read_image(str(folder/'raw.png'))
            stage=time.monotonic()
            count=helper.get_face_landmarks_5(only_center_face=False,eye_dist_threshold=5)
            record['detected_faces']=int(count)
            record['all_detected_boxes']=np.asarray(helper.det_faces).tolist()
            # Diagnose one central face while preserving every other raw pixel outside paste mask.
            require(count>0,'No detected face; no restoration fallback permitted')
            boxes=np.asarray(helper.det_faces);centers=(boxes[:,:2]+boxes[:,2:4])/2
            chosen=int(np.argmin(np.sum((centers-expected/2)**2,axis=1)))
            helper.det_faces=[helper.det_faces[chosen]];helper.all_landmarks_5=[helper.all_landmarks_5[chosen]]
            helper.align_warp_face()
            require(len(helper.cropped_faces)==1,'Expected one central face')
            record['chosen_detection_index']=chosen;record['landmarks']=np.asarray(helper.all_landmarks_5).tolist()
            record['affine_matrices']=np.asarray(helper.affine_matrices).tolist()
            record['detect_align_seconds']=time.monotonic()-stage
            crop=helper.cropped_faces[0]
            require(cv2.imwrite(str(folder/'aligned-512.png'),crop),'Could not save crop')
            tensor=img2tensor(crop/255.,bgr2rgb=True,float32=True).unsqueeze(0)
            tensor=(tensor-.5)/.5
            stage=time.monotonic()
            with torch.inference_mode(): output=model(tensor,return_rgb=False,randomize_noise=False)[0]
            require(torch.isfinite(output).all().item(),'Nonfinite restored tensor')
            restored=tensor2img(output.squeeze(0),rgb2bgr=True,min_max=(-1,1))
            record['restoration_seconds']=time.monotonic()-stage
            require(cv2.imwrite(str(folder/'restored-512.png'),restored),'Could not save restored crop')
            stage=time.monotonic()
            helper.add_restored_face(restored.astype('uint8'))
            helper.get_inverse_affine(None)
            with torch.inference_mode(): full=helper.paste_faces_to_input_image(upsample_img=None)
            require(full.shape[:2]==(1024,1024),'Unexpected reconstructed dimensions')
            # Preserve color diagnostic, then use identical final monochrome and WebP settings.
            require(cv2.imwrite(str(folder/'restored-1024-rgb.png'),full),'Could not save composite')
            final=Image.fromarray(cv2.cvtColor(full,cv2.COLOR_BGR2RGB)).convert('L').convert('RGB')
            final.save(folder/'restored-1024.png');final.save(folder/'restored-1024.webp',quality=90,method=6)
            record['inverse_affine_matrices']=np.asarray(helper.inverse_affine_matrices).tolist()
            record['paste_encode_seconds']=time.monotonic()-stage
            del tensor,output,restored,full
        except Exception as exc:
            # Keep the raw/control and exact error. Never substitute raw pixels as successful restoration.
            record['error']=f'{type(exc).__name__}: {exc}'
        record['total_seconds']=time.monotonic()-begin
        record['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['outputs']={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        write(folder/'record.json',record);records.append(record)
        print(json.dumps({'index':item['index'],'error':record['error'],'seconds':record['total_seconds'],'peak_rss':record['peak_process_rss_bytes']}),flush=True)
    for item in inputs: require(sha(ROOT/item['path'])==item['sha256'],'Input changed during diagnostic')
    pins_check()
    write(args.output/'result.json',{'complete':True,'cohort':args.cohort,'attempted':len(records),'successful_restorations':sum(x['error'] is None for x in records),'errors':sum(x['error'] is not None for x in records),'model_and_aux_load_seconds':load_seconds,'worker_wall_after_import_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'input_and_source_hashes_unchanged':True,'records':records,'quality_approved':False,'production_approved':False})

def supervise(args):
    require(not args.output.exists(),'Refusing existing output directory')
    process_guard();mem=memory_guard();pins=pins_check();inputs=cohort_inputs(args.cohort,args.nada_checkpoint_sha256)
    for x in inputs:require(sha(ROOT/x['path'])==x['sha256'],'Input mismatch')
    args.output.mkdir(parents=True)
    write(args.output/'inputs.json',inputs)
    started=time.time();deadline=started+SECONDS
    write(args.output/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':started,'deadline_unix':deadline,'rss_abort_bytes':LIMIT,'sampled_not_os_hard_limit':True,'memory':mem,'cohort':args.cohort,'nada_checkpoint_sha256':args.nada_checkpoint_sha256,'pins_sha256':sha(BASE/'pins.json'),'source_sha256':sha(Path(__file__)),'thread_env':THREAD_ENV,'network_disabled':True,'background_upsampler':None,'decoder_randomize_noise':False,'fixed_seed':1729,'no_retry':True,'production_approved':False})
    env=dict(os.environ,**THREAD_ENV,PYTHONHASHSEED='1729',NUMBA_NUM_THREADS='1',MPLCONFIGDIR=str(args.output/'matplotlib-cache'))
    command=[sys.executable,str(Path(__file__)), '--cohort',args.cohort,'--output',str(args.output),'--_worker','--deadline',str(deadline)]
    process=None;failure=None;peak=0
    try:
        with (args.output/'worker.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                rss=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True).stdout.strip()
                if rss:peak=max(peak,int(rss)*1024)
                if peak>LIMIT or time.time()>deadline:
                    failure='RSS limit' if peak>LIMIT else '10 minute deadline';process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    break
                time.sleep(.25)
            if process.returncode!=0:failure=failure or f'worker exit {process.returncode}'
    finally:
        if process is not None and process.poll() is None:process.kill();process.wait()
        result=read(args.output/'result.json') if (args.output/'result.json').exists() else {}
        complete=failure is None and result.get('complete') is True
        write(args.output/'supervisor-result.json',{'complete':complete,'error':failure,'worker_exit_code':None if process is None else process.returncode,'wall_seconds':time.time()-started,'peak_sampled_worker_rss_bytes':peak,'quality_approved':False,'production_approved':False})
    require(complete,'Diagnostic did not complete: '+str(failure))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['reference625','nada500'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nada-checkpoint-sha256')
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--deadline',type=float,default=0,help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve()
    require(args.output.is_relative_to(BASE/'runs'),'Output must be a NEW restoration/runs child')
    (worker if args._worker else supervise)(args)
if __name__=='__main__':main()
