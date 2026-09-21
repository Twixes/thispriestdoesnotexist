"""Build an unfiltered static comparison from completed diffusion experiments."""
import hashlib
import html
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = [
    ('sd-turbo-1step-v1', 'SD Turbo · 1 step'),
    ('sdxl-turbo-1step-v1', 'SDXL Turbo · 1 step'),
    ('sdxl-turbo-4step-v1', 'SDXL Turbo · 4 steps'),
    ('sdxl-lightning-2step-1024-v1', 'SDXL Lightning · 2 steps'),
    ('sdxl-lightning-2step-512-v1', 'SDXL Lightning · 2 steps · 512px'),
    ('sdxl-turbo-1step-1024-v1', 'SDXL Turbo · 1 step · 1024px'),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(path):
    return os.path.relpath(path, OUT)

def main():
    models=[];pending=[]
    for dirname,title in RUNS:
        run=ROOT/'research/diffusion/runs'/dirname
        result_path=run/'result.json'
        if not result_path.exists():
            pending.append(dirname);continue
        result=json.loads(result_path.read_text())
        if result.get('complete') is not True:
            pending.append(dirname);continue
        rows=sorted((r for r in result['records'] if not r['warmup']),key=lambda r:r['index'])
        if [r['index'] for r in rows] != list(range(8)):
            raise ValueError(f'{dirname}: expected all eight original indices')
        entries=[]
        for row in rows:
            files={}
            for filename in ['native.png','display.webp']:
                path=run/f"{row['index']:03}"/filename
                actual=sha(path)
                if actual != row['files'][filename]:
                    raise ValueError(f'{dirname}: hash mismatch {path}')
                files[filename]={'url':relative(path),'sha256':actual}
            entries.append({'index':row['index'],'seed':row['seed'],'prompt':row['prompt'],
                            'generation_ms':row['generation_seconds']*1000,
                            'encoding_ms':row['encoding_seconds']*1000,
                            'combined_ms':row['generation_plus_encoding_seconds']*1000,
                            'files':files})
        times=[r['combined_ms'] for r in entries]
        models.append({'run':dirname,'title':title,'model':result['model'],'steps':result['steps'],
                       'native_resolution':result['native_resolution'],'device':result['device'],
                       'result_url':relative(result_path),'result_sha256':sha(result_path),
                       'post_trained':result.get('post_trained',False),'server_latency_proven':result.get('server_latency_proven',False),
                       'median_combined_ms':statistics.median(times),'max_combined_ms':max(times),
                       'median_generation_ms':statistics.median(r['generation_ms'] for r in entries),
                       'median_encoding_ms':statistics.median(r['encoding_ms'] for r in entries),
                       'entries':entries})
    if not models:
        raise ValueError('No completed comparison runs')
    for model in models[1:]:
        if [(r['seed'],r['prompt']) for r in model['entries']] != [(r['seed'],r['prompt']) for r in models[0]['entries']]:
            raise ValueError('Rows must preserve identical numerical seeds and prompts across models')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Non-blocking unfiltered pretrained-base comparison',
              'production_approved':False,'all_eight_outputs_per_completed_run_included':True,'manual_selection':False,
              'warmups_excluded_from_display_and_statistics':True,'pending_runs':pending,'models':models}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    headers=[]
    for model in models:
        headers.append(f'''<th scope="col"><strong>{html.escape(model['title'])}</strong><span>{model['native_resolution']} × {model['native_resolution']} native</span><span class="timing">{model['median_combined_ms']:.0f} ms median · {model['max_combined_ms']:.0f} ms max</span><small>Generation {model['median_generation_ms']:.0f} ms + WebP {model['median_encoding_ms']:.0f} ms<br>Local MPS medians, eight warm samples</small></th>''')
    body=[]
    for i in range(8):
        entry=models[0]['entries'][i]
        cells=[]
        for column,model in enumerate(models):
            row=model['entries'][i];alt=f"{model['title']}, sample {i:03}, seed {row['seed']}"
            cells.append(f'''<td><button class="portrait" data-model="{column}" data-row="{i}" aria-label="Open {html.escape(alt)}"><img src="{html.escape(row['files']['display.webp']['url'])}" width="{model['native_resolution']}" height="{model['native_resolution']}" alt="{html.escape(alt)}" loading="{'eager' if i==0 else 'lazy'}"></button><div class="caption">{row['combined_ms']:.0f} ms generation + WebP</div></td>''')
        body.append(f'''<tr><th scope="row"><span>{i:03}</span><small>Seed<br>{entry['seed']}</small></th>{''.join(cells)}</tr>''')
    pending_notice=f"<p class='pending'>Not yet completed: {html.escape(', '.join(pending))}. Rebuild to include completed runs.</p>" if pending else ''
    payload=json.dumps(models).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Priest model comparison</title>
<style>
:root{color-scheme:dark;font-family:Arial,Helvetica,sans-serif;background:#000;color:#ddd}*{box-sizing:border-box}body{margin:0}header{max-width:1280px;padding:28px 28px 16px}h1{font-size:22px;font-weight:500;margin:0 0 12px}p{font-size:14px;line-height:1.55;color:#aaa;margin:8px 0}strong{color:#eee}a{color:#bbb} .scroll{overflow:auto;padding:0 20px 30px}table{border-collapse:collapse;table-layout:fixed;width:100%;min-width:MINWIDTHpx}thead th{vertical-align:top;text-align:left;padding:14px 8px 20px;border-top:1px solid #252525;font-weight:400}thead th:first-child{width:95px}thead strong{display:block;font-size:16px;margin-bottom:8px}thead span{display:block;font-size:13px;color:#aaa;margin:5px 0}thead .timing{color:#ddd}small{display:block;color:#777;font-size:11px;line-height:1.6}tbody th{text-align:left;vertical-align:top;padding:12px 6px;font-weight:400;font-size:16px}tbody td{vertical-align:top;padding:0 8px 22px}.portrait{display:block;background:#080808;border:0;padding:0;cursor:zoom-in;width:100%;aspect-ratio:1}.portrait img{display:block;width:100%;height:100%;object-fit:contain}.caption{color:#777;font-size:11px;margin-top:7px}.pending{color:#b9a77c}dialog{border:0;padding:0;background:#000;color:#ccc;max-width:none;max-height:none;width:100vw;height:100dvh;margin:0}dialog::backdrop{background:#000}.modalbar{height:64px;display:flex;align-items:center;gap:12px;padding:12px 20px;border-bottom:1px solid #222}.modalbar strong{font-size:14px;font-weight:400;flex:1}.control{color:#ddd;background:#111;border:1px solid #444;padding:8px 12px;cursor:pointer;border-radius:3px}.modalimage{height:calc(100dvh - 108px);display:flex;align-items:center;justify-content:center;overflow:auto}.modalimage img{max-width:100%;max-height:100%;object-fit:contain}.modalcaption{height:44px;text-align:center;font-size:12px;color:#888;padding:12px}.help{margin:14px 28px 28px} @media(max-width:650px){header{padding:22px 16px 10px}.scroll{padding:0 8px 25px}.modalbar{gap:7px;padding:10px}.modalbar strong{font-size:12px}.control{padding:8px;font-size:12px}}
</style><header><h1>Priest model comparison</h1><p>All eight samples from each completed run, without hand-picking. Rows use the same numerical seed and prompt; different model resolutions mean they are <strong>not identical latent tensors</strong>.</p><p><strong>Pretrained bases, not post-trained priest models.</strong> Timings are generation + WebP on local Apple M5 Pro MPS, excluding load/warmup, queueing, network and HTTP overhead. They do not establish server latency or production readiness.</p><p>Compare faces, eyes, natural clothing, photographic quality and varied identities. Legitimate broad clerical collars are acceptable; the research prompt's precise collar shape is not a user requirement. Click any portrait for the native PNG or display WebP.</p>PENDING</header><div class="scroll"><table><thead><tr><th scope="col">Sample</th>HEADERS</tr></thead><tbody>ROWS</tbody></table></div><p class="help">In full-size view: ← / → switches model, ↑ / ↓ switches sample. Escape closes. <a href="manifest.json">Research manifest</a></p>
<dialog id="viewer" aria-label="Full-size model comparison"><div class="modalbar"><strong id="title"></strong><button class="control" id="format">Show display WebP</button><a class="control" id="original" target="_blank" rel="noopener">Open file</a><button class="control" id="close">Close</button></div><div class="modalimage"><img id="full" alt=""></div><div id="details" class="modalcaption"></div></dialog>
<script>
const models=PAYLOAD;const viewer=document.querySelector('#viewer'),full=document.querySelector('#full');let modelIndex=0,rowIndex=0,native=true;
function render(){const m=models[modelIndex],r=m.entries[rowIndex],name=native?'native.png':'display.webp';full.src=r.files[name].url;full.alt=`${m.title}, sample ${String(r.index).padStart(3,'0')}`;document.querySelector('#title').textContent=`${m.title} · ${m.native_resolution}px · sample ${String(r.index).padStart(3,'0')}`;document.querySelector('#format').textContent=native?'Show display WebP':'Show native PNG';document.querySelector('#original').href=full.src;document.querySelector('#details').textContent=`${native?'Native PNG':'Display WebP'} · seed ${r.seed} · ${Math.round(r.combined_ms)} ms local generation + WebP`;}
document.querySelectorAll('.portrait').forEach(button=>button.addEventListener('click',()=>{modelIndex=Number(button.dataset.model);rowIndex=Number(button.dataset.row);native=true;render();viewer.showModal()}));document.querySelector('#close').addEventListener('click',()=>viewer.close());document.querySelector('#format').addEventListener('click',()=>{native=!native;render()});document.addEventListener('keydown',event=>{if(!viewer.open)return;if(event.key==='ArrowRight')modelIndex=(modelIndex+1)%models.length;else if(event.key==='ArrowLeft')modelIndex=(modelIndex+models.length-1)%models.length;else if(event.key==='ArrowDown')rowIndex=(rowIndex+1)%8;else if(event.key==='ArrowUp')rowIndex=(rowIndex+7)%8;else return;event.preventDefault();render()});
</script></html>'''
    for key,value in [('MINWIDTH',str(95+len(models)*250)),('PENDING',pending_notice),('HEADERS',''.join(headers)),('ROWS',''.join(body)),('PAYLOAD',payload)]:
        page=page.replace(key,value)
    (OUT/'index.html').write_text(page)
    print(json.dumps({'output':str(OUT/'index.html'),'models':len(models),'portraits':len(models)*8,'pending':pending,'all_image_hashes_verified':True},indent=2))

if __name__=='__main__':main()
