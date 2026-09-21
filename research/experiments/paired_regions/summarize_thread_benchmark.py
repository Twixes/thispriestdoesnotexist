"""Compare saved benchmark results and PNGs only; no model loading."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / 'runs/paired-regions-thread-benchmark'
results = {n: json.loads((ROOT/f'threads-{n}/results.json').read_text()) for n in [1,2,4]}
reference = np.array(Image.open(ROOT/'threads-1/final-calibration-original.png')).astype(np.int16)
rows = []
for n, result in results.items():
    png = ROOT/f'threads-{n}/final-calibration-original.png'
    delta = np.abs(reference-np.array(Image.open(png)).astype(np.int16))
    rows.append({
        'threads':n, 'mean_seconds':result['timed_mean_wall_seconds'],
        'median_seconds':result['timed_median_wall_seconds'],
        'minimum_seconds':result['timed_min_wall_seconds'],
        'maximum_seconds':result['timed_max_wall_seconds'],
        'warmup_seconds':result['warmup_wall_seconds'],
        'cpu_mean_seconds':sum(r['cpu_seconds'] for r in result['updates'][1:])/3,
        'peak_rss_gib':result['peak_rss_bytes']/2**30,
        'free_memory_percent':result['memory_preflight_free_percent'],
        'speedup_vs_1thread_mean':results[1]['timed_mean_wall_seconds']/result['timed_mean_wall_seconds'],
        'initial_model_digest_matches_thread1':result['initial_student_digest']==results[1]['initial_student_digest'],
        'final_model_digest_matches_thread1':result['final_student_digest']==results[1]['final_student_digest'],
        'final_png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),
        'final_png_max_delta_vs_thread1':int(delta.max()),
        'final_png_mean_delta_vs_thread1':float(delta.mean()),
        'final_png_changed_channel_fraction_vs_thread1':float((delta>0).mean()),
        'max_loss_metric_delta_vs_thread1':max(abs(a[k]-b[k])
            for a,b in zip(result['updates'],results[1]['updates'])
            for k in ['collar_l1','rest_clothing_l1','protected_l1','fresh_preservation_l1']),
        'source_validation_at_threads':result['source_validation_discrepancy_at_training_threads'],
    })
summary = {
    'count_updates':sum(r['updates_total'] for r in results.values()),
    'all_runs_passed':all(r['finite_changed_student'] for r in results.values()),
    'all_source_frozen_checks_passed':all(r['frozen_source_preserved'] and r['frozen_student_preserved'] for r in results.values()),
    'no_checkpoint_saved':all(not r['checkpoint_saved'] for r in results.values()),
    'configs':rows,'fastest_mean_threads':min(rows,key=lambda r:r['mean_seconds'])['threads'],
    'summarizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'caveat':'Only three timed updates each, sequential order 1/2/4 and concurrent pre-existing MPS activity; no long-run speed guarantee.',
}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
