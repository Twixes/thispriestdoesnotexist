#!/usr/bin/env python3
"""Build a static unfiltered broad review; this module never imports torch."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIELDS=['apparent_adult','priest_clothing_and_collar','hat_present','gross_artifacts','photographic_coherence','repetition_or_nearest_training_image_concern','notes']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text())


def artifact(path):
    return {'path':str(path.resolve().relative_to(ROOT)),'url':os.path.relpath(path,HERE),'sha256':sha(path)}


def validate_coverage(manifest,registry):
    assert manifest['complete'] is True and manifest['count']==32 and manifest['image_count']==64
    assert manifest['unfiltered'] and manifest['all_native_whole_images']
    assert manifest['rng_unchanged'] and manifest['fixed_offset_and_mask_invariants'] and manifest['parent_reproduction_exact']
    expected={(arm,i) for arm in ('raw','ema') for i in range(32)}
    assert len(manifest['images'])==64 and {(im['arm'],im['index']) for im in manifest['images']}==expected
    for im in manifest['images']:
        assert im['path']==f"{im['arm']}-{im['index']:03}.png"
        assert im['width']==im['height']==1024 and im['latent_sha256']==registry['row_sha256'][im['index']]
    assert manifest['production_approved'] is False and manifest['server_latency_proven'] is False


def annotation_template(rows,source=None):
    return {'schema_version':1,'source_checkpoint_sha256':source,'review_status':'not_started',
        'instructions':'Review every raw and EMA image in fixed order. Unknown observations stay null. No attractiveness ratings for apparent minors. Record concrete photographic defects, hats, collar absence and repetition. A resemblance concern is not a proven duplicate or computed nearest neighbor. No automated quality approval.',
        'rows':[{'index':row['index'],'arm':arm,'image_sha256':row['images'][arm].get('sha256'),
                 'status':'not_reviewed',**{field:None for field in FIELDS}} for row in rows for arm in ('raw','ema')]}


def collect(run=None,annotations=None):
    registry=read(HERE/'preregistration.json')
    assert registry['count']==32 and len(registry['row_sha256'])==32 and len(set(registry['row_sha256']))==32
    assert sha(ROOT/registry['latents_path'])==registry['latents_sha256']
    rows=[{'index':i,'latent_sha256':registry['row_sha256'][i],'images':{arm:{'status':'pending'} for arm in ('raw','ema')}} for i in range(32)]
    sources={'preregistration':artifact(HERE/'preregistration.json'),'latents':artifact(ROOT/registry['latents_path'])}
    training_manifest=ROOT/'research/data/flux-priest-domain-v1/manifest.json'
    assert sha(training_manifest)=='6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b'
    training=[]
    for item in read(training_manifest)['entries']:
        if item['split']!='train':continue
        path=training_manifest.parent/item['destination'];record=artifact(path)
        assert record['sha256']==item['destination_sha256']
        training.append({'id':item['id'],**record})
    assert len(training)==20
    sources['training_manifest']=artifact(training_manifest)
    status='not_rendered';checkpoint=None
    if run is not None:
        run=run.resolve();run.relative_to(ROOT/'research/runs')
        protocol=read(run/'render-protocol.json');sources['render_protocol']=artifact(run/'render-protocol.json')
        assert protocol['preregistration_sha256']==sources['preregistration']['sha256']
        assert protocol['latents_sha256']==registry['latents_sha256']
        checkpoint=protocol['checkpoint_sha256'];status='render_in_progress_or_awaiting_terminal_record'
        marker=run/'manifest.json';supervisor=read(run/'supervisor.json') if (run/'supervisor.json').exists() else None
        if marker.exists():
            manifest=read(marker);validate_coverage(manifest,registry)
            assert manifest['preregistration_sha256']==sources['preregistration']['sha256']
            assert manifest['render_protocol_sha256']==sha(run/'render-protocol.json')
            assert manifest['checkpoint_sha256']==checkpoint
            sources['images_manifest']=artifact(marker)
            for im in manifest['images']:
                path=run/im['path'];data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==im['sha256']
                with Image.open(io.BytesIO(data)) as image:
                    assert image.format=='PNG' and image.size==(1024,1024);image.verify()
                rows[im['index']]['images'][im['arm']]={'status':'complete',**artifact(path),'width':1024,'height':1024}
            status='render_complete_supervisor_pending'
        if supervisor:
            sources['supervisor']=artifact(run/'supervisor.json')
            assert supervisor['render_protocol_sha256']==sha(run/'render-protocol.json')
            if supervisor['complete']:
                assert supervisor['failure'] is None and marker.exists();status='render_complete_review_pending'
            else:status='render_failed'
    template=annotation_template(rows,checkpoint)
    if annotations:
        reviewed=read(annotations)
        assert reviewed['source_checkpoint_sha256']==checkpoint and checkpoint is not None
        assert len(reviewed['rows'])==64
        expected={(entry['arm'],entry['index']):entry for entry in template['rows']}
        assert {(entry['arm'],entry['index']) for entry in reviewed['rows']}==set(expected)
        for entry in reviewed['rows']:
            original=expected[entry['arm'],entry['index']]
            assert original['image_sha256'] is not None and entry['image_sha256']==original['image_sha256']
            assert entry['status'] in ('not_reviewed','reviewed','uncertain') and all(field in entry for field in FIELDS)
        sources['annotations']=artifact(annotations)
    else:reviewed=template
    for item in reviewed['rows']:rows[item['index']]['images'][item['arm']]['review']=item
    return {'schema_version':1,'built_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':status,'checkpoint_sha256':checkpoint,'count':32,'available_images':sum(im['status']=='complete' for r in rows for im in r['images'].values()),
        'expected_images':64,'rows':rows,'sources':sources,'training_references':training,'license_notice':registry['license_notice'],
        'unfiltered':True,'production_approved':False,'server_latency_proven':False},template


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--annotations',type=Path);a=p.parse_args()
    manifest,template=collect(a.run,a.annotations)
    outputs={'manifest.json':json.dumps(manifest,indent=2)+'\n','index.html':(HERE/'template.html').read_text().replace('__DATA__',json.dumps(manifest).replace('<','\\u003c')),
             'annotation-template.json':json.dumps(template,indent=2)+'\n'}
    for name,value in outputs.items():
        temp=HERE/(name+'.partial');temp.write_text(value);temp.replace(HERE/name)
    print(json.dumps({'status':manifest['status'],'images':manifest['available_images'],'manifest_sha256':sha(HERE/'manifest.json')}))


if __name__=='__main__':main()
