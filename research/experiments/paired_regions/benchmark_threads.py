"""Bounded fresh-source CPU thread benchmark: exactly four updates per process."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=False)
    pressure = subprocess.check_output(['memory_pressure'], text=True)
    (args.run/'memory-before.txt').write_text(pressure)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure)[1])
    if free < 25:
        raise SystemExit(f'DEFER: {free}% memory free is below required25%; no model load')
    started = time.perf_counter()
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    import trainer as t
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed = 20260921
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    manifest = json.loads(args.manifest.read_text())
    assert sum(p.get('split', 'train') == 'train' for p in manifest['pairs']) == 1
    bundle = (args.manifest.resolve().parent/manifest['source_bundle']).resolve()
    metadata = json.loads((bundle/'model.json').read_text())
    weights = bundle/'generator.safetensors'
    assert metadata['schema_version'] == 1 and metadata['truncation_psi'] == 1
    assert t.sha256(weights) == metadata['weights_sha256']
    source = t.Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
    source.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    assert source.img_resolution == 1024 and source.c_dim == 0
    source_digest = t.state_digest(source.state_dict())
    # This strict check ALWAYS runs at one CPU thread, with zero tolerance.
    verified_at = time.perf_counter()
    pairs, pair_provenance = t.load_pairs(args.manifest, source, 0, 0)
    strict_verification_seconds = time.perf_counter()-verified_at
    train_pairs = [p for p in pairs if p['split'] == 'train']
    student = copy.deepcopy(source)
    trainable = t.freeze_student(student)
    frozen_digest = t.state_digest(t.frozen_state(student))
    initial_student_digest = t.state_digest(student.state_dict())
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=1e-4, betas=(.9,.999))
    sampling = torch.Generator(device='cpu').manual_seed(seed+1)
    preservation = torch.Generator(device='cpu').manual_seed(seed+2)
    torch.set_num_threads(args.threads)
    differences = []
    # Record the numerical discrepancy introduced by the new thread setting.
    # It does not change the accepted strict CPU1 provenance or its tolerances.
    with torch.inference_mode():
        for p in pairs:
            mapped = source.mapping(p['z'], None, truncation_psi=1, skip_w_avg_update=True)
            rendered = source.synthesis(p['w'], noise_mode='const', force_fp32=True)
            delta = (rendered-p['original']).abs()
            rgb = t.quantized(rendered).astype(np.int16)
            original_rgb = t.quantized(p['original']).astype(np.int16)
            pixel_delta = np.abs(rgb-original_rgb)
            differences.append({'id':p['id'], 'w_max_abs_error':float((mapped-p['w']).abs().max()),
                                'source_float_max_abs_error':float(delta.max()),
                                'source_float_mean_abs_error':float(delta.mean()),
                                'source_uint8_max_abs_error':int(pixel_delta.max()),
                                'source_uint8_mean_abs_error':float(pixel_delta.mean()),
                                'source_uint8_changed_channel_fraction':float((pixel_delta>0).mean())})
            del mapped, rendered, delta, rgb, original_rgb, pixel_delta
    options = {'batch':1, 'clothing_weight':1, 'protected_weight':1,
               'fresh_weight':1, 'upper_fraction':.75}
    records = []
    setup_seconds = time.perf_counter()-started
    for index in range(4):
        wall = time.perf_counter(); cpu = time.process_time()
        metrics = t.update(student, source, optimizer, train_pairs, options, sampling, preservation, 'cpu')
        metrics.update(update=index+1, warmup=index==0, wall_seconds=time.perf_counter()-wall,
                       cpu_seconds=time.process_time()-cpu,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(metrics)
        with (args.run/'updates.jsonl').open('a') as log:
            log.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics), flush=True)
    final_check_started = time.perf_counter()
    assert t.state_digest(source.state_dict()) == source_digest
    assert t.state_digest(t.frozen_state(student)) == frozen_digest
    final_student_digest = t.state_digest(student.state_dict())
    assert final_student_digest != initial_student_digest
    assert all(bool(torch.isfinite(p).all()) for p in student.parameters())
    assert all(p.grad is None for p in source.parameters())
    final_check_seconds = time.perf_counter()-final_check_started
    preview_started = time.perf_counter()
    with torch.inference_mode():
        for p in pairs:
            output = student.synthesis(p['w'], noise_mode='const', force_fp32=True)
            mono = t.grayscale(output).repeat(1,3,1,1)
            Image.fromarray(t.quantized(mono)).save(args.run/f'final-{p["id"]}.png')
    preview_seconds = time.perf_counter()-preview_started
    timings = [r['wall_seconds'] for r in records if not r['warmup']]
    result = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Thread scaling of actual fresh-source region training; no model approval',
        'production_approved':False,'checkpoint_saved':False,'device':'cpu','threads':args.threads,
        'interop_threads':torch.get_num_interop_threads(), 'seed':seed,'resolution':1024,
        'updates_total':4,'warmup_updates':1,'timed_updates':3,'fresh_latent_updates':4,
        'strict_source_verification_threads':1,'source_max_uint8_error_allowed':0,'w_atol_allowed':0,
        'memory_preflight_free_percent':free, 'load_verify_setup_seconds':setup_seconds,
        'strict_source_verification_seconds':strict_verification_seconds,
        'warmup_wall_seconds':records[0]['wall_seconds'],
        'timed_mean_wall_seconds':statistics.mean(timings),'timed_median_wall_seconds':statistics.median(timings),
        'timed_min_wall_seconds':min(timings),'timed_max_wall_seconds':max(timings),
        'final_hash_finite_checks_seconds':final_check_seconds,'final_preview_seconds':preview_seconds,
        'total_process_work_seconds':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'frozen_source_preserved':True,'frozen_student_preserved':True,'finite_changed_student':True,
        'source_validation_discrepancy_at_training_threads':differences,
        'updates':records,'options':options,'lr':1e-4,'adam_betas':[.9,.999],
        'freeze':'unchanged paired_regions/base policy: mapping, all buffers/noise_strength, blocks<=32',
        'trainable_parameters':trainable,'source_state_digest':source_digest,
        'initial_student_digest':initial_student_digest,'final_student_digest':final_student_digest,
        'platform':platform.platform(),'python':platform.python_version(),
        'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','pillow','safetensors']},
        'provenance':{'benchmark_sha256':t.sha256(__file__),'manifest_sha256':t.sha256(args.manifest),
                      'source_weights_sha256':t.sha256(weights),'metadata_sha256':t.sha256(bundle/'model.json'),
                      'trainer_sources':t.source_hashes(),'pairs':pair_provenance},
        'limitations':['Three timed updates are a small scheduling-sensitive benchmark, not a long-run speed guarantee.',
                       'Update timing includes fresh forwards/backward, finite gradient checks and optimizer; checkpoint I/O excluded.',
                       'Different thread counts can change floating-point reduction results; differences are measured, not declared exact.',
                       'This single-pair, four-update preview is not a quality or generalization evaluation.'],
    }
    (args.run/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['updates','provenance','trainable_parameters']},indent=2),flush=True)


if __name__ == '__main__':
    main()
