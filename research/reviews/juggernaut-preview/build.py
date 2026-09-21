"""Build a static, hash-checked unfiltered comparison; no model execution."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'research/reviews/base-comparison-preview/manifest.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def checked_file(p, expected):
    assert sha(p) == expected, p
    return {'url': '../../' + str(p.relative_to(ROOT / 'research')), 'sha256': expected, 'bytes': p.stat().st_size}

original = json.loads(BASE.read_text())
models = []
for m in original['models']:
    model = {k: m[k] for k in ['run', 'title', 'native_resolution']}
    model['entries'] = []
    for entry in m['entries']:
        files = {name: checked_file((BASE.parent / value['url']).resolve(), value['sha256']) for name,value in entry['files'].items()}
        model['entries'].append({'index':entry['index'],'seed':entry['seed'],'prompt':entry['prompt'],'files':files})
    models.append(model)

runs = []
for name, title in [('juggernaut-hyper-tcd4-native1024-v2-tiled','Juggernaut · native 1024 · tiled VAE'),('juggernaut-hyper-tcd4-512-v1','Juggernaut · 4 steps · 512 sampling'),('juggernaut-hyper-tcd6-cfg1p5-native1024-v1-tiled','Juggernaut · 6 steps · CFG1.5 · tiled1024'),('realvis-v5-lightning-sde5-native1024-tiled-v1','RealVis V5 Lightning · SDE5 · tiled1024')]:
    p = ROOT / 'research/diffusion/runs' / name
    run = {'run':name, 'title':title, 'native_resolution':512 if name=='juggernaut-hyper-tcd4-512-v1' else 1024, 'entries':[], 'complete':False}
    if (p/'protocol.json').exists():
        protocol = json.loads((p/'protocol.json').read_text())
        run.update(protocol_sha256=sha(p/'protocol.json'), native_resolution=protocol['resolution'], vae_tiling=protocol['vae_tiling'], protocol_url=f'../../diffusion/runs/{name}/protocol.json')
        for case in protocol['cases']:
            if case['warmup']: continue
            folder=p/f"{case['index']:03d}"
            if not (folder/'record.json').exists(): continue
            record=json.loads((folder/'record.json').read_text())
            if record['status']!='complete': continue
            assert record['seed']==case['seed'] and record['prompt']==case['prompt']
            assert record['protocol_sha256']==run['protocol_sha256']
            assert record['steps']==protocol['steps'] and record['guidance_scale']==protocol['guidance_scale']
            if 'eta' in protocol:
                assert record['eta']==protocol['eta']
            files={n:checked_file(folder/n,h) for n,h in record['files'].items()}
            run['entries'].append({'index':case['index'],'seed':case['seed'],'prompt':case['prompt'],'record_sha256':sha(folder/'record.json'),'files':files})
        if (p/'result.json').exists():
            result=json.loads((p/'result.json').read_text())
            run.update(result_sha256=sha(p/'result.json'),complete=result['complete'] is True and len(run['entries'])==8)
    runs.append(run)
for run in runs:
    for e in run['entries']:
        for m in models:
            pair=next(x for x in m['entries'] if x['index']==e['index'])
            assert pair['seed']==e['seed'] and pair['prompt']==e['prompt'],(run['run'],m['run'],e['index'])
manifest={'purpose':'Non-blocking unfiltered pretrained model comparison','base_manifest_sha256':sha(BASE),'manual_selection':False,'warmups_excluded':True,'production_approved':False,'server_latency_proven':False,'post_trained_by_this_project':False,'comparison_note':'Identical prompts and numeric seeds; model families, scheduler and latent shapes differ, so these are not matched identities. 512 sampling of Juggernaut is a resolution tradeoff relative to its native1024 prior.','models':models,'juggernaut':[r for r in runs if r['run'].startswith('juggernaut-')],'references':runs}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'data.js').write_text('window.PREVIEW = '+json.dumps(manifest).replace('<','\\u003c')+';\n')
print(json.dumps({r['run']:len(r['entries']) for r in runs}))
