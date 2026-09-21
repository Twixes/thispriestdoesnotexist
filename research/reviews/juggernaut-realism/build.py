"""Render every case from the full-model realism trial, with verified originals."""
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / 'diffusion/runs'
ARMS = [
    ('Accelerated · 6 steps', 'juggernaut-hyper-tcd6-cfg1p5-native1024-v1-tiled'),
    ('Full model · 35 steps', 'juggernaut-full-v10-35step-cfg3-native1024-v1'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    arms = []
    for title, name in ARMS:
        folder = RUNS / name
        protocol_path = folder / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        entries = []
        for case in protocol['cases']:
            if case['warmup']:
                continue
            record_path = folder / f"{case['index']:03}" / 'record.json'
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
            if record['status'] != 'complete':
                continue
            assert record['protocol_sha256'] == sha(protocol_path)
            assert record['prompt'] == case['prompt'] and record['seed'] == case['seed']
            assert record['steps'] == protocol['steps']
            assert record['guidance_scale'] == protocol['guidance_scale']
            native = record_path.parent / 'native.png'
            assert sha(native) == record['files']['native.png']
            entries.append({'index': case['index'], 'prompt': case['prompt'], 'seed': case['seed'],
                            'native_sha256': sha(native), 'record_sha256': sha(record_path),
                            'url': f'../../diffusion/runs/{name}/{case["index"]:03}/native.png'})
        result_path = folder / 'result.json'
        complete = result_path.exists() and json.loads(result_path.read_text())['complete'] and len(entries) == 8
        arms.append({'title': title, 'run': name, 'complete': bool(complete),
                     'protocol_sha256': sha(protocol_path), 'entries': entries,
                     'result_sha256': sha(result_path) if result_path.exists() else None})
    rows = []
    for baseline in arms[0]['entries']:
        cards = []
        for arm in arms:
            entry = next((e for e in arm['entries'] if e['index'] == baseline['index']), None)
            label = html.escape(arm['title'])
            if entry:
                assert entry['prompt'] == baseline['prompt'] and entry['seed'] == baseline['seed']
                content = f'<a href="{entry["url"]}" target="_blank" rel="noopener"><img src="{entry["url"]}" loading="lazy" alt="{label}, case {entry["index"]}"></a>'
            else:
                content = '<div class="pending">Generation pending</div>'
            cards.append(f'<figure><figcaption>{label}</figcaption>{content}</figure>')
        rows.append(f'<section><h2>Case {baseline["index"]:03}</h2><div class="grid">{"".join(cards)}</div><details><summary>Prompt and seed</summary><p>{html.escape(baseline["prompt"])}</p><p>Seed {baseline["seed"]}</p></details></section>')
    manifest = {'purpose': 'Photographic realism first, following owner rejection of the accelerated baseline',
                'all_cases_shown': True, 'warmups_excluded': True, 'production_approved': False,
                'server_latency_proven': False, 'arms': arms,
                'comparison_limit': 'Same prompts and numeric seeds; weights, scheduler, guidance and step count change together. This does not isolate distillation or match identities.'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status = ' · '.join(f'{a["title"]}: {len(a["entries"])}/8' for a in arms)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Photographic realism comparison</title><style>
body{margin:0;background:#111;color:#ddd;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{line-height:1.2}h2{font-size:18px;margin-top:40px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{font-size:15px;padding:8px 0}img{display:block;width:100%;height:auto}.gray img{filter:grayscale(1)}.pending{aspect-ratio:1;display:grid;place-items:center;background:#202020;color:#aaa}a{color:#ccc}details{margin:12px 0}label{display:block;margin:16px 0}@media(max-width:600px){main{padding:12px}.grid{gap:8px}figcaption{font-size:12px}}
</style><main><h1>Photographic realism first</h1><p>You rejected the accelerated baseline’s realism. This comparison tests the full Juggernaut X v10 model at 35 steps, CFG 3 and DPM++ 2M Karras against Hyper at six TCD steps and CFG 1.5. Both generate native 1024-pixel portraits with tiled decoding.</p><p>All eight outputs are shown in generation order, with the same prompts and numeric seeds. Different weights and sampling settings can produce different people. Click any portrait to inspect the original at full size.</p><p><strong>Neither model is approved for production.</strong> This trial prioritizes image quality; it does not establish the server latency target. The 20-step Hyper adapter did not clearly improve photographic realism.</p><p>STATUS</p><label><input id="gray" type="checkbox" checked> Black and white display</label>ROWS<p><a href="manifest.json">Verified image provenance</a> · <a href="../juggernaut-development/">Completed adapter trial</a></p></main><script>document.body.classList.add('gray');document.getElementById('gray').addEventListener('change',e=>document.body.classList.toggle('gray',e.target.checked));</script></html>'''
    (HERE / 'index.html').write_text(page.replace('STATUS', html.escape(status)).replace('ROWS', ''.join(rows)))
    print(status)


if __name__ == '__main__':
    main()
