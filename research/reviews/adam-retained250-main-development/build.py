#!/usr/bin/env python3
"""Read only finalized main-adaptation snapshots; no model or checkpoint loading."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MAIN=ROOT/'research/runs/adam-native1024-retained250-fixed-offsets-main-adaptation10-q50-v1'
CONTINUATION=ROOT/'research/runs/adam-native1024-retained250-main10-to100-v1'
STEPS=[0,10,50,100]
POLICY='fixed_synthesis_noise_and_activation_bias_offsets_v1'
EVAL_SHA='c4eb3bee33c6f9a0502c62b620a1a984aa431d1b43eadcaab1bbc53c8ebe804e'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text())


def artifact(path):return {'path':str(path.relative_to(ROOT)),'url':os.path.relpath(path,HERE),'sha256':sha(path),'bytes':path.stat().st_size}


def image_record(path,digest):
    data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
    with Image.open(io.BytesIO(data)) as image:
        assert image.format=='PNG' and image.size==(1024,1024);image.verify()
    return {'status':'complete',**artifact(path),'width':1024,'height':1024}


def collect():
    sources={};runs={};snapshots=[]
    rows=[{'index':i,'images':{f'{arm}-{step}':{'status':'pending'} for step in STEPS for arm in ('raw','ema')}} for i in range(4)]
    def source(run,name):
        path=run/name;sources[str(path.relative_to(ROOT))]=artifact(path);return read(path)
    def inspect_run(run,is_continuation):
        if not (run/'protocol.json').exists():return {'status':'not_started','complete_steps':[]}
        protocol=source(run,'protocol.json');ph=sha(run/'protocol.json')
        assert protocol['conv_layout']=='output_rank1' and protocol['fixed_offset_policy']==POLICY
        assert protocol['quantile']==50.0 and protocol['resolution']==1024 and protocol['importance_source_step']==250
        assert protocol['importance_pairs']==1000 and protocol['preview_count_per_state']==4
        expected_name='adam-native1024-retained250-main-adaptation10-to100' if is_continuation else 'adam-native1024-fixed-offsets-retained250-main-adaptation10'
        assert protocol['name']==expected_name and protocol['iterations']==(100 if is_continuation else 10)
        assert protocol['preview_steps']==([10,50,100] if is_continuation else [0,10])
        assert protocol['production_approved'] is False and protocol['server_latency_proven'] is False
        names=['main_adaptation_fixed_offsets250_10.py','modulation.py','fixed_offset_policy.py','adaptation_masks_fixed_offsets.py']
        if is_continuation:names+=['continue_main_adaptation_fixed_offsets250_100.py']
        for name in names:
            path=run/name
            # Some continuation dependencies remain archived in its authenticated parent.
            if not path.exists():path=ROOT/'research/experiments/adam_native'/name
            record=artifact(path);assert record['sha256']==protocol['pins']['research/experiments/adam_native/'+name]
            sources[str(path.relative_to(ROOT))]=record
        if is_continuation:
            assert protocol['source_main_run']==str(MAIN.relative_to(ROOT))
            assert protocol['source_main_protocol_sha256']==sha(MAIN/'protocol.json')
            assert protocol['source_main_checkpoint_sha256']==read(MAIN/'checkpoint-010.json')['checkpoint_sha256']
            assert runs['main10']['status']=='complete'
        complete=[]
        for step in protocol['preview_steps']:
            folder=run/f'step-{step:03}';marker=folder/'manifest.json'
            if not marker.exists():continue
            if is_continuation and step in (50,100) and not (run/f'checkpoint-{step:03}.json').exists():continue
            snap=source(run,f'step-{step:03}/manifest.json')
            assert snap['complete'] and snap['step']==step and snap['rng_unchanged'] and snap['protocol_sha256']==ph
            assert sha(run/'eval-z.npz')==EVAL_SHA
            sources[str((run/'eval-z.npz').relative_to(ROOT))]=artifact(run/'eval-z.npz')
            if is_continuation:assert snap['eval_z_sha256']==EVAL_SHA
            expected={f'{arm}-{i:03}.png' for arm in ('raw','ema') for i in range(4)}
            assert len(snap['images'])==8 and {im['path'] for im in snap['images']}==expected
            images={im['path']:image_record(folder/im['path'],im['sha256']) for im in snap['images']}
            if is_continuation and step==10:
                assert snap['parent_pngs_exact']
                for row in rows:
                    for arm in ('raw','ema'):assert images[f"{arm}-{row['index']:03}.png"]['sha256']==row['images'][f'{arm}-10']['sha256']
            else:
                for row in rows:
                    for arm in ('raw','ema'):row['images'][f'{arm}-{step}']=images[f"{arm}-{row['index']:03}.png"]
            if step==0:
                eq=source(run,'initial-equality.json')
                assert eq['generator_exact'] and eq['generator_cases']==4
                for row in rows:assert row['images']['raw-0']['sha256']==row['images']['ema-0']['sha256']
            checkpoint=run/f'checkpoint-{step:03}.json'
            if checkpoint.exists():
                cp=source(run,checkpoint.name)
                assert cp['complete'] and cp['step']==step and cp['protocol_sha256']==ph and cp['model_optimizer_rng_path_state_restored_exactly']
                if is_continuation:assert cp['snapshot_manifest_sha256']==sha(marker)
            complete.append(step)
        result=source(run,'result.json') if (run/'result.json').exists() else None
        supervisor=source(run,'supervisor.json') if (run/'supervisor.json').exists() else None
        termination=source(run,'termination.json') if (run/'termination.json').exists() else None
        if result:
            assert result['complete'] and result['iterations']==protocol['iterations'] and result['protocol_sha256']==ph
            assert result['main_adaptation_run'] and result['fixed_offsets_and_mask_invariants']
            assert result['folded_native_max_errors']=={'raw':[0.0]*4,'ema':[0.0]*4}
            assert result['production_approved'] is False and result['server_latency_proven'] is False
            assert complete==protocol['preview_steps'] and (run/f"checkpoint-{protocol['iterations']:03}.json").exists()
        if supervisor:
            assert supervisor['protocol_sha256']==ph
            if supervisor['complete']:assert supervisor['failure'] is None and result is not None
        if termination:
            assert termination['status'] in ('stopped_after_visual_review','stopped_by_user','stopped_for_safety')
            assert termination['supervisor_and_worker_absent'] and not result
        status=('stopped' if termination else 'failed' if supervisor and not supervisor['complete'] else 'complete' if supervisor else 'awaiting_supervisor' if result else 'in_progress_or_awaiting_terminal_record')
        return {'status':status,'complete_steps':complete,'path':str(run.relative_to(ROOT)),'protocol_sha256':ph}
    runs['main10']=inspect_run(MAIN,False)
    runs['continuation100']=inspect_run(CONTINUATION,True)
    complete=[step for step in STEPS if all(row['images'][f'{arm}-{step}']['status']=='complete' for row in rows for arm in ('raw','ema'))]
    for step in STEPS:snapshots.append({'step':step,'status':'complete' if step in complete else 'pending'})
    return {'schema_version':1,'built_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':runs,
        'complete_steps':complete,'default_step':max(complete,default=0),'snapshots':snapshots,'rows':rows,
        'available_images':len(complete)*8,'expected_images':32,'sources':sources,
        'unfiltered':True,'scope':'Actual main-adaptation q50 development snapshots; four fixed latents, raw and EMA. Full main-adaptation updates begin from source weights; retained250 supplies importance masks only.',
        'production_approved':False,'quality_accepted':False,'server_latency_proven':False}


def main():
    manifest=collect();page=(HERE/'template.html').read_text().replace('__DATA__',json.dumps(manifest).replace('<','\\u003c'))
    for name,contents in [('manifest.json',json.dumps(manifest,indent=2)+'\n'),('index.html',page)]:
        temp=HERE/(name+'.partial');temp.write_text(contents);temp.replace(HERE/name)
    print(json.dumps({'complete_steps':manifest['complete_steps'],'images':manifest['available_images'],'manifest_sha256':sha(HERE/'manifest.json')}))


if __name__=='__main__':main()
