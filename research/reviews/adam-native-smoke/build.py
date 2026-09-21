#!/usr/bin/env python3
"""Build the unfiltered native AdAM mechanics preview; never load a model."""
import hashlib
import html
import json
import os
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'research/runs/adam-native1024-probing-smoke10-v2'


def artifact(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'url': os.path.relpath(path, HERE),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    assert protocol['preview_count'] == 4 and protocol['preview_steps'] == [0, 5, 10]
    result = json.loads((RUN / 'result.json').read_text()) if (RUN / 'result.json').exists() else None
    sources = {name: artifact(RUN / name) for name in
               ['protocol.json', 'eval-z.npz', 'smoke.py', 'modulation.py', 'initial-equality.json']
               if (RUN / name).exists()}
    if result is not None:
        sources['result.json'] = artifact(RUN / 'result.json')
        assert result['protocol_sha256'] == sources['protocol.json']['sha256']
    rows = []
    for i in range(4):
        row = {'index': i, 'images': {}}
        paths = {'baseline': RUN / f'baseline-{i:03}.png'}
        for step in [0, 5, 10]:
            for arm in ['raw', 'ema']:
                paths[f'{arm}-{step}'] = RUN / f'step-{step:03}/{arm}-{i:03}.png'
        for key, path in paths.items():
            if not path.exists():
                row['images'][key] = {'status': 'pending'}
                continue
            with Image.open(path) as im:
                assert im.size == (1024, 1024) and im.format == 'PNG', path
                im.verify()
            row['images'][key] = {'status': 'complete', **artifact(path), 'width': 1024, 'height': 1024}
        rows.append(row)
    count = sum(im['status'] == 'complete' for row in rows for im in row['images'].values())
    if result and result['complete']:
        assert count == 28, 'A completed run must preserve all 28 native previews.'
    manifest = {
        'schema_version': 1, 'title': 'AdAM native mechanics smoke',
        'run': str(RUN.relative_to(ROOT)), 'run_complete': bool(result and result['complete']),
        'available_images': count, 'expected_images': 28, 'unfiltered': True,
        'preview_seed': protocol['preview_seed'], 'sampling': protocol['sampling'],
        'comparison': 'Same four Gaussian latent vectors in original order; source baseline and raw/EMA at iterations 0, 5, 10.',
        'scope': 'Ten importance-probing iterations only. No Fisher estimation or main adaptation. NOT a priest-generation candidate.',
        'production_approved': False, 'server_latency_proven': False,
        'sources': sources, 'builder': artifact(Path(__file__)), 'rows': rows,
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdAM native mechanics smoke</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101012;color:#eee}body{max-width:1600px;margin:auto;padding:28px}h1{font-size:clamp(25px,4vw,40px);margin-bottom:12px}p{max-width:1000px;line-height:1.55;color:#bcbcc5}.scope{color:#ffdc9b;border-left:3px solid;padding-left:16px}a{color:#b9caff}nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}nav a[aria-current=true]{color:white;font-weight:700}section{border-top:1px solid #34343b;padding:20px 0}h2{font-size:16px;font-weight:500}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figcaption{font-size:13px;margin-bottom:8px;color:#d8d8e0}img{display:block;width:100%;height:auto;background:black;aspect-ratio:1;object-fit:contain}.pending{display:grid;place-items:center;aspect-ratio:1;background:#222}small{display:block;color:#9696a0;margin-top:8px}footer{border-top:1px solid #34343b;margin-top:24px;padding-top:16px}a:focus-visible{outline:2px solid white;outline-offset:4px}@media(max-width:700px){body{padding:16px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.grid figure:first-child{grid-column:1/-1;max-width:512px;width:100%;justify-self:center}}
</style>
<h1>AdAM native mechanics smoke</h1>
<p class="scope"><strong>NOT a priest-generation candidate.</strong> Only 10 importance-probing iterations; no Fisher estimation or main adaptation. The original FFHQ model includes children, women and other non-priest subjects. Every sampled latent is shown without filtering, selection or repair.</p>
<p>Original NVIDIA StyleGAN2 FFHQ at native 1024 × 1024. Each row keeps the same Gaussian latent and constant synthesis noise across the source baseline, raw modulation weights and exponential moving average (EMA). Click any image for the untouched native PNG. This is a non-blocking mechanics review, with no production approval or server-speed claim.</p>
<p id="status"></p>
<nav aria-label="Snapshot iteration"><a href="#step-10" data-step="10">Iteration 10</a><a href="#step-5" data-step="5">Iteration 5</a><a href="#step-0" data-step="0">Iteration 0</a><a href="manifest.json">Hashes & manifest</a><a href="native-review.json">Raw iteration-10 review</a></nav>
<main id="rows"></main>
<footer><p>All 28 source PNGs are retained: four baselines and raw/EMA snapshots at iterations 0, 5 and 10. Preview RNG seed: 2026092210. Four latents cannot establish population quality or diversity. Ordinary display scaling is the only image transformation.</p></footer>
<script>
const data = __DATA__;
const host = document.querySelector('#rows');
function render(){
 const match = location.hash.match(/^#step-(0|5|10)$/); const step = match ? Number(match[1]) : 10;
 document.querySelectorAll('[data-step]').forEach(a=>a.setAttribute('aria-current',String(Number(a.dataset.step)===step)));
 document.querySelector('#status').textContent=`${data.run_complete?'Run complete':'Run pending'} · ${data.available_images} / ${data.expected_images} native images present`;
 host.replaceChildren();
 for(const row of data.rows){
  const section=document.createElement('section');const title=document.createElement('h2');title.textContent=`Latent ${String(row.index).padStart(3,'0')} · unfiltered`;section.append(title);
  const grid=document.createElement('div');grid.className='grid';
  for(const [key,label] of [['baseline','Original FFHQ baseline'],[`raw-${step}`,`Raw · iteration ${step}`],[`ema-${step}`,`EMA · iteration ${step}`]]){
   const fig=document.createElement('figure');const caption=document.createElement('figcaption');caption.textContent=label;fig.append(caption);const item=row.images[key];
   if(item.status==='complete'){const a=document.createElement('a');a.href=item.url;a.target='_blank';a.rel='noopener';a.title='Open original 1024 × 1024 PNG';const im=document.createElement('img');im.src=item.url;im.alt=`Latent ${row.index}, ${label}`;im.loading='lazy';a.append(im);fig.append(a);const small=document.createElement('small');small.textContent=`SHA-256 ${item.sha256.slice(0,16)}…`;fig.append(small)}
   else{const p=document.createElement('div');p.className='pending';p.textContent='Snapshot pending';fig.append(p)}
   grid.append(fig);
  }section.append(grid);host.append(section);
 }
}window.addEventListener('hashchange',render);render();
</script></html>'''
    page = page.replace('__DATA__', json.dumps(manifest).replace('<', '\\u003c'))
    (HERE / 'index.html').write_text(page)
    print(json.dumps({'run_complete': manifest['run_complete'], 'images': count,
                      'manifest_sha256': artifact(HERE / 'manifest.json')['sha256']}))


if __name__ == '__main__':
    main()
