"""Measure two complete regularization cycles from already written metrics."""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = {row['step']: row for row in map(json.loads, (root / 'metrics.jsonl').read_text().splitlines())}
cycles = []
evidence = []
for start, stop in [(384, 400), (400, 416)]:
    selected = [rows[step] for step in range(start + 1, stop + 1)]
    counts = collections.Counter(p['phase'] for row in selected for p in row['phases'])
    assert counts == {'Gmain': 16, 'Dmain': 16, 'Greg': 4, 'Dreg': 1}
    images = rows[stop]['images_seen'] - rows[start]['images_seen']
    assert images == 1024
    seconds = rows[stop]['seconds'] - rows[start]['seconds']
    cycles.append({'steps': [start + 1, stop], 'images': images, 'seconds': seconds,
                   'phase_counts': dict(counts), 'days_per_million_linear': seconds / images * 1e6 / 86400})
    evidence.extend([rows[start], *selected])
seconds = sum(c['seconds'] for c in cycles)
images = sum(c['images'] for c in cycles)
canonical = json.dumps(evidence, sort_keys=True, separators=(',', ':'))
result = {'cycles': cycles, 'combined_images': images, 'combined_seconds': seconds,
          'days_per_million_linear': seconds / images * 1e6 / 86400,
          'evidence_canonical_json_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
          'selected_metric_rows': evidence,
          'context': 'MPS reference training with the paired-factorial CPU experiment running; the CDC GPU run had been stopped. These cycles contain no checkpoint milestone.',
          'limitations': 'Two short observed cycles only; excludes checkpoint/preview costs and future load or thermal changes. A throughput estimate is not a quality or convergence prediction.'}
(root / 'throughput-step416.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'selected_metric_rows'}, indent=2))
