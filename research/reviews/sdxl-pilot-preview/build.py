"""Build every frozen SDXL development case, preserving missing/failed cells."""
import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

OUT=Path(__file__).resolve().parent
SOURCE=OUT
REPO=OUT.parents[2]
RUNS=REPO/'research/diffusion/runs'
PROTOCOL=REPO/'research/diffusion/sdxl-pilot-protocol.json'
ARMS=('base-plain','base-trigger','adapter-plain','adapter-trigger')
LABELS=('Pretrained base · plain','Pretrained base · trigger','Post-trained adapter · plain','Post-trained adapter · trigger')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def artifact(path):
    return {'url':os.path.relpath(path,OUT),'sha256':sha(path),'bytes':path.stat().st_size}


def require(test,message):
    if not test:raise ValueError(message)


def read_run(path,group,protocol_hash,steps):
    path=path.resolve()
    require(path.is_relative_to(REPO),'Run must be in this repository')
    if not path.exists():return {'path':path,'sources':{},'exists':False,'launch':None}
    sources={name:artifact(path/name) for name in ('launch.json','protocol.json','runtime.json','result.json','supervisor.json','worker-failure.json','decision.json') if (path/name).is_file()}
    if not (path/'launch.json').is_file():return {'path':path,'sources':sources,'exists':True,'launch':None}
    launch=json.loads((path/'launch.json').read_text())
    require(launch['protocol_sha256']==protocol_hash and sha(path/'protocol.json')==protocol_hash,'Run protocol differs: '+str(path))
    require(launch['arms']==group and launch['split']=='development' and launch['steps']==steps,'Expected matching development arm group and inference steps: '+str(path))
    if (path/'result.json').is_file():
        result=json.loads((path/'result.json').read_text())
        require(result['count']==48 and result['complete'] and len(result['records'])==48,'Unexpected completed run result')
    return {'path':path,'sources':sources,'exists':True,'launch':launch}


def cell(run,case,arm_index,protocol_hash,steps):
    arm=ARMS[arm_index]
    out={'arm':arm,'label':LABELS[arm_index],'native':None,'status':'Pending evaluation'}
    record_path=run['path']/case['case_id']/arm/'record.json'
    if not record_path.exists():
        if 'worker-failure.json' in run['sources']:
            out['status']='Run stopped before this image'
        return out
    require(run['launch'] is not None,'Record exists without launch provenance')
    record=json.loads(record_path.read_text())
    out['record']=artifact(record_path)
    require((record['case_id'],record['seed'],record['arm'])==(case['case_id'],case['seed'],arm),'Record case/seed/arm mismatch')
    prompt=('PR1EST_CAL. ' if arm.endswith('-trigger') else '')+case['prompt']
    require(record['prompt']==prompt and record['protocol_sha256']==protocol_hash,'Record prompt/protocol mismatch')
    require(record['resolution']==512 and record['steps']==steps,'Unexpected native resolution or step count')
    if record['status']!='complete':
        out['status']='Failed evaluation' if record['status']=='failed' else 'Evaluation in progress'
        return out
    require(record['model_provenance_sha256']==json.loads(PROTOCOL.read_text())['base']['provenance_sha256'],'Base provenance mismatch')
    require(record['runtime_sha256']==sha(run['path']/'runtime.json'),'Runtime evidence mismatch')
    if arm.startswith('adapter'):
        require(record['lora_sha256']==run['launch']['lora']['sha256'],'Adapter provenance mismatch')
    else:require(record['lora_sha256'] is None,'Base arm unexpectedly used an adapter')
    files={}
    for name,metadata in record['files'].items():
        path=record_path.parent/name
        require(path.resolve().is_relative_to(record_path.parent.resolve()),'Unsafe artifact path')
        actual=artifact(path)
        require(actual['sha256']==metadata['sha256'] and actual['bytes']==metadata['bytes'],'Output file hash mismatch: '+str(path))
        files[name]=actual
    native=record_path.parent/'native.png'
    with native.open('rb') as stream:header=stream.read(24)
    require(header[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',header[16:24])==(512,512),'Invalid native PNG dimensions')
    randomness=record['randomness']
    require(randomness['noise_file_sha256']==sha(record_path.parent.parent/'noise.npz'),'Noise artifact hash mismatch')
    require(randomness['generator_state_after_noise_sha256']==sha(record_path.parent.parent/'generator-state-after-noise.bin'),'Initial RNG state mismatch')
    out.update(status='complete',native=files['native.png'],files=files,prompt=prompt,
               randomness=randomness,generator_state_after_inference_sha256=record['generator_state_after_inference_sha256'],
               lora_sha256=record['lora_sha256'])
    return out


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run',type=Path)
    parser.add_argument('--checkpoint',action='append',default=[],metavar='UPDATES=RUN_PATH')
    parser.add_argument('--steps',type=int,choices=(1,4),default=1)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    require(OUT.is_relative_to(REPO/'research/reviews'),'Output must be a research review directory')
    require(not (args.steps==4 and OUT==SOURCE),'Four-step fallback needs a separate --output directory')
    OUT.mkdir(parents=True,exist_ok=True)
    if OUT!=SOURCE:
        for name in ('index.html','style.css','app.js'):
            shutil.copyfile(SOURCE/name,OUT/name)
    if args.base_run is None:
        args.base_run=RUNS/f'sdxl-pilot-base-development-{args.steps}step-v1'
    protocol=json.loads(PROTOCOL.read_text());protocol_hash=sha(PROTOCOL)
    require(len(protocol['development_cases'])==24,'Expected all24 development cases')
    mapping={}
    for spec in args.checkpoint:
        step,path=spec.split('=',1);step=int(step)
        require(step in (20,100,250) and step not in mapping,'Only unique20/100/250 checkpoints supported')
        mapping[step]=Path(path)
    if not mapping:
        mapping={20:RUNS/f'sdxl-pilot-step20-development-{args.steps}step-v1'}
        for step in (100,250):
            path=RUNS/f'sdxl-pilot-step{step}-development-{args.steps}step-v1'
            if path.exists():mapping[step]=path
    base=read_run(args.base_run,'base',protocol_hash,args.steps)
    checkpoints=[]
    for updates,path in sorted(mapping.items()):
        adapted=read_run(path,'adapter',protocol_hash,args.steps)
        cases=[]
        for case in protocol['development_cases']:
            cells=[cell(base if i<2 else adapted,case,i,protocol_hash,args.steps) for i in range(4)]
            complete=[c for c in cells if c['native']]
            if complete:
                first=complete[0]
                for other in complete[1:]:
                    for key in ('noise_values_sha256','generator_state_after_noise_sha256'):
                        require(other['randomness'][key]==first['randomness'][key],'Unmatched randomness between arms')
                    require(other['generator_state_after_inference_sha256']==first['generator_state_after_inference_sha256'],'Unmatched inference RNG stream')
            cases.append({'case_id':case['case_id'],'seed':case['seed'],'prompt':case['prompt'],'arms':cells})
        count=sum(c['native'] is not None for row in cases for c in row['arms'])
        checkpoints.append({'updates':updates,'completed_cells':count,'complete':count==96,'cases':cases,
            'base_sources':base['sources'],'adapter_sources':adapted['sources'],
            'base_run':os.path.relpath(base['path'],OUT),'adapter_run':os.path.relpath(adapted['path'],OUT)})
    data={'purpose':'Unfiltered non-blocking SDXL-Turbo post-training development comparison',
          'production_approved':False,'server_latency_proven':False,'manual_output_selection':False,
          'all24_cases_in_protocol_order':True,'grayscale_is_display_filter_only':True,
          'inference_steps':args.steps,'native_resolution':512,'protocol':artifact(PROTOCOL),
          'builder':artifact(Path(__file__)),'checkpoints':checkpoints}
    (OUT/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'data.js').write_text('window.PREVIEW_DATA = '+json.dumps(data).replace('</','<\\/')+';\n')
    print(json.dumps({'preview':str(OUT/'index.html'),'checkpoints':[{k:r[k] for k in ('updates','completed_cells','complete')} for r in checkpoints]},indent=2))

if __name__=='__main__':main()
