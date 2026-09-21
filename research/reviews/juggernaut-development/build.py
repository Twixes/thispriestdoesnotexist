"""Build the fixed24-case native base/adapter review without selecting images."""
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
RUNS = RESEARCH / 'diffusion/runs'
CASE_FILE = RESEARCH / 'diffusion/juggernaut-development-cases.json'
ARMS = [('base','Unchanged base','juggernaut-development-base-v3-staged'),
        ('step20','20 training updates','juggernaut-development-step20-v1-cached')]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_arm(folder):
    result_path = RUNS / folder / 'result.json'
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    assert result['complete'] and result['count'] == 24
    protocol = json.loads((result_path.parent / 'protocol.json').read_text())
    assert protocol['case_manifest_sha256'] == sha(CASE_FILE)
    assert protocol['cached_text'] and protocol['steps'] == 6 and protocol['guidance_scale'] == 1.5
    assert protocol['prompt_prefix'] == 'PR1EST_CAL. ' and protocol['vae_tiling']['enabled']
    records = [r for r in result['records'] if not r['warmup']]
    assert [r['index'] for r in records] == list(range(24))
    for r in records:
        assert sha(result_path.parent / f"{r['index']:03}" / 'native.png') == r['files']['native.png']
    return {'result_sha256':sha(result_path),'result':result,'records':records,'protocol':protocol}

def main():
    arms = {key:load_arm(folder) for key,_,folder in ARMS}
    if arms['step20']:
        assert arms['base']
        for a,b in zip(arms['base']['records'],arms['step20']['records']):
            for key in ('prompt','seed','noise_values_sha256','generator_after_noise_sha256'):
                assert a[key] == b[key], key
            assert a["files"]["generator-after-inference.bin"] == b["files"]["generator-after-inference.bin"]
        a=json.loads((RUNS / ARMS[0][2] / 'conditioning/manifest.json').read_text())
        b=json.loads((RUNS / ARMS[1][2] / 'conditioning/manifest.json').read_text())
        assert [r['tensors'] for r in a['records']] == [r['tensors'] for r in b['records']]
        for arm, cache in zip(ARMS, (a,b)):
            for r in cache['records']:
                assert sha(RUNS / arm[2] / 'conditioning' / r['file']) == r['file_sha256']
        proof={'conditioning_tensor_pairs_equal':sum(len(r['tensors']) for r in a['records']),
               'cases_including_warmup':len(a['records']), 'same_noise_and_before_after_rng_for24':True,
               'cache_manifests_sha256':[sha(RUNS / arm[2] / 'conditioning/manifest.json') for arm in ARMS]}
        (HERE/'paired-input-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    cases=json.loads(CASE_FILE.read_text())['cases']
    sections=[]
    for c in cases:
        i=c['index'];cards=[]
        for key,label,folder in ARMS:
            if arms[key]:
                url=f'../../diffusion/runs/{folder}/{i:03}/native.png'
                cards.append(f'<figure><figcaption>{label}</figcaption><a href="{url}" target="_blank" rel="noopener"><img loading="lazy" src="{url}" alt="{label}, case {i:02}"></a></figure>')
            else:
                cards.append(f'<figure><figcaption>{label}</figcaption><div class="pending">Not generated yet</div></figure>')
        group='Varied appearance' if i<16 else 'Same prompt · fresh seed'
        sections.append(f'<section><h2>{i:02} · {group}</h2><div class="grid">{"".join(cards)}</div><details><summary>Prompt and seed</summary><p>{html.escape(c["prompt"])}</p><p>Seed {c["seed"]}; prefix PR1EST_CAL.</p></details></section>')
    summary={key:{'complete':arm is not None,'result_sha256':arm['result_sha256'] if arm else None} for key,arm in arms.items()}
    (HERE/'manifest.json').write_text(json.dumps({'cases_sha256':sha(CASE_FILE),'arms':summary,'count':24,'production_approved':False},indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Juggernaut priest adaptation</title><style>
body{margin:0;background:#111;color:#ddd;font:16px/1.6 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:24px}h1{line-height:1.2}h2{font-size:18px;margin-top:40px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}figcaption{font-size:14px;padding:8px 0}img{display:block;width:100%;height:auto}.gray img{filter:grayscale(1)}.pending{aspect-ratio:1;display:grid;place-items:center;background:#202020;color:#aaa}a{color:#ccc}details{margin:12px 0}label{display:block;margin:16px 0}@media(max-width:500px){main{padding:12px}figcaption{font-size:12px}}
</style><main><h1>Priest adaptation and identity variety</h1><p>All 24 development images: 16 varied appearance prompts, then eight fresh seeds with exactly the same prompt. Both arms use the same trigger, initial noise, text conditioning, and native 1024 sampling. Click a portrait for its original PNG.</p><p><strong>Your review: the baseline needs more photographic realism.</strong> This completed comparison records the small adapter trial; neither model is approved for production.</p><p>The small adapter trains at512; evaluation uses1024, six TCD steps, CFG1.5 and tiled decoding. This tests appearance changes and regressions, not production approval or GPU server latency. Missing runs remain marked below.</p><label><input id="gray" type="checkbox" checked> Black and white display</label>SECTIONS<p><a href="manifest.json">Image-set provenance</a> · <a href="../juggernaut-preview/">Earlier base comparison</a></p></main><script>document.body.classList.add('gray');document.getElementById('gray').addEventListener('change',e=>document.body.classList.toggle('gray',e.target.checked));</script></html>'''
    (HERE/'index.html').write_text(page.replace('SECTIONS',''.join(sections)))
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
