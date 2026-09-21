"""Completed six-train/one-holdout paired evaluation; CPU only, never training."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.experiments.paired_regions import evaluate as guards

sha256 = guards.sha256


def verify_paired_manifest(manifest, actual_sha, checkpoint):
    if checkpoint['step'] != 600:
        raise ValueError('This paired evaluation requires completed step 600')
    if actual_sha != checkpoint['provenance']['manifest_sha256']:
        raise ValueError('Training manifest checksum mismatch')
    if manifest['version'] != 2 or manifest.get('production_approved', False):
        raise ValueError('Expected unapproved version 2 research manifest')
    if manifest.get('requires_trainer_feature') != 'equal_area_collar_partition_v1':
        raise ValueError('Expected the reviewed collar-region objective')
    pairs = manifest['pairs']
    ids = [pair['id'] for pair in pairs]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in ids):
        raise ValueError('Unsafe or repeated paired identity')
    if Counter(pair['split'] for pair in pairs) != {'train': 6, 'validation': 1}:
        raise ValueError('Expected six training identities and one validation identity')
    if [pair['id'] for pair in pairs if pair['split'] == 'validation'] != ['028']:
        raise ValueError('Expected unchanged holdout 028')
    if [(pair['id'], pair['split']) for pair in checkpoint['provenance']['pairs']] != [(pair['id'], pair['split']) for pair in pairs]:
        raise ValueError('Checkpoint paired split/order mismatch')
    if checkpoint['options']['source_max_uint8_error'] != 0 or checkpoint['options']['w_atol'] != 0:
        raise ValueError('This evaluation requires original zero-tolerance source provenance')


def verify_pair_provenance(actual, recorded):
    if actual != recorded:
        raise ValueError('Reconstructed paired provenance differs from checkpoint')


def split_label(split):
    if split == 'train':
        return 'TRAIN — supervised identity; not generalization evidence'
    if split == 'validation':
        return 'HOLDOUT — validation identity 028; never used by optimizer'
    raise ValueError('Unknown split')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, choices=[600], default=600)
    return parser.parse_args()


def main():
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty paired-evaluation directory')
    pressure = subprocess.run(['memory_pressure'], text=True, capture_output=True, check=True).stdout
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1))
    if free < 25:
        raise RuntimeError(f'Defer: memory free is {free}%, below 25%')
    import torch
    import PIL
    from safetensors.torch import load_file
    from research.experiments.paired_regions import trainer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with args.checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        loaded = torch.load(stream, map_location='cpu', weights_only=True)
    student_state = loaded.pop('student')
    checkpoint = {key: loaded[key] for key in ['format', 'production_approved', 'step', 'init_kwargs',
                  'provenance', 'options', 'source_digest', 'frozen_digest']}
    del loaded
    gc.collect()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = sha256(args.manifest)
    verify_paired_manifest(manifest, manifest_sha, checkpoint)
    if checkpoint['step'] != args.expected_step:
        raise ValueError('Unexpected completed step')
    configuration_path = args.checkpoint.parent / 'config.json'
    configuration = json.loads(configuration_path.read_text())
    metadata_path = args.source_bundle / 'model.json'
    metadata = json.loads(metadata_path.read_text())
    weights = args.source_bundle / 'generator.safetensors'
    weights_sha, metadata_sha = sha256(weights), sha256(metadata_path)
    guards.verify_metadata(metadata, checkpoint, configuration,
                           {'model': {'model_sha256': weights_sha}}, weights_sha, metadata_sha)
    if metadata['init_kwargs']['img_resolution'] != 1024:
        raise ValueError('Expected native 1024 source')
    if (args.manifest.resolve().parent / manifest['source_bundle']).resolve() != args.source_bundle.resolve():
        raise ValueError('Manifest and requested source bundle differ')
    guards.verify_code_provenance(checkpoint['provenance']['source_files'], guards.current_source_hashes())
    versions = {'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__}
    if versions != checkpoint['provenance']['versions']:
        raise ValueError('Runtime versions differ from exact training provenance')

    source = trainer.Generator(**metadata['init_kwargs']).cpu().eval().requires_grad_(False)
    source.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    if trainer.state_digest(source.state_dict()) != checkpoint['source_digest']:
        raise ValueError('Source state digest mismatch')
    trainer.freeze_student(source)
    if trainer.state_digest(trainer.frozen_state(source)) != checkpoint['frozen_digest']:
        raise ValueError('Frozen source state mismatch')
    source.requires_grad_(False)
    student = trainer.Generator(**checkpoint['init_kwargs']).cpu().eval()
    student.load_state_dict(student_state, strict=True)
    del student_state
    trainer.freeze_student(student)
    if trainer.state_digest(trainer.frozen_state(student)) != checkpoint['frozen_digest']:
        raise ValueError('Student altered frozen source parameters or buffers')
    student.requires_grad_(False)
    for model in [source, student]:
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('Nonfinite model state')
    student_digest = trainer.state_digest(student.state_dict())
    # Unchanged loader verifies actual one-thread CPU source PNG and mapped W exactly;
    # its original source tensors are reused, rather than generating them a second time.
    pairs, pair_provenance = trainer.load_pairs(args.manifest, source, source_max_error=0, w_atol=0)
    verify_pair_provenance(pair_provenance, checkpoint['provenance']['pairs'])
    train_z = {guards.latent_digest(pair['z'].numpy()) for pair in pairs if pair['split'] == 'train'}
    holdout_z = {guards.latent_digest(pair['z'].numpy()) for pair in pairs if pair['split'] == 'validation'}
    if len(train_z) != 6 or len(holdout_z) != 1 or train_z & holdout_z:
        raise ValueError('Duplicate or overlapping train/holdout latent identity')
    verified_seconds = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'host-memory-before.txt').write_text(pressure)
    entries, thumbnails = [], []
    with torch.inference_mode():
        for pair, provenance in zip(pairs, pair_provenance, strict=True):
            sample_started = time.monotonic()
            prediction = student.synthesis(pair['w'], noise_mode='const', force_fp32=True)
            source_gray, target_gray, predicted_gray = map(trainer.grayscale, [pair['original'], pair['target'], prediction])
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('Nonfinite student image')
            # Divide [-1,1] grayscale errors by 2 to express the nominal [0,1] scale.
            metric = lambda a, b, mask: float(trainer.masked_l1(a, b, mask)) / 2
            metrics = {'protected_source_l1_0_1': metric(predicted_gray, source_gray, 1-pair['mask']),
                       'all_clothing_target_l1_0_1': metric(predicted_gray, target_gray, pair['mask']),
                       'collar_target_l1_0_1': None, 'rest_clothing_target_l1_0_1': None,
                       'balanced_clothing_target_l1_0_1': None}
            if pair['collar'] is not None:
                metrics['collar_target_l1_0_1'] = metric(predicted_gray, target_gray, pair['collar'])
                metrics['rest_clothing_target_l1_0_1'] = metric(predicted_gray, target_gray, pair['rest'])
                metrics['balanced_clothing_target_l1_0_1'] = .5*(metrics['collar_target_l1_0_1']+metrics['rest_clothing_target_l1_0_1'])
            metrics.update(guards.region_errors((source_gray[0,0].numpy()+1)/2, (predicted_gray[0,0].numpy()+1)/2))
            images = [('source-gray', source_gray), ('student-gray', predicted_gray), ('target-gray', target_gray), ('student-rgb', prediction)]
            outputs = {}
            for label, tensor in images:
                pixels = trainer.quantized(tensor)
                if label != 'student-rgb':
                    pixels = pixels[:,:,0]
                image = Image.fromarray(pixels)
                path = args.output / f'{pair["id"]}-{label}.png'
                image.save(path)
                outputs[label] = {'path': path.name, 'sha256': sha256(path), 'dimensions': list(image.size)}
                if label != 'student-rgb':
                    thumbnails.append(image.convert('RGB').resize((320,320), Image.Resampling.LANCZOS))
            entries.append({'id': pair['id'], 'split': pair['split'], 'label': split_label(pair['split']),
                            'eligible_for_holdout_assessment': pair['split']=='validation', 'production_approved': False,
                            'metrics': metrics, 'provenance': provenance, 'outputs': outputs,
                            'seconds': time.monotonic()-sample_started})
            print(json.dumps({'id':pair['id'], 'split':pair['split'], 'metrics':metrics}), flush=True)
            del prediction, source_gray, target_gray, predicted_gray
    if trainer.state_digest(source.state_dict()) != checkpoint['source_digest'] or trainer.state_digest(student.state_dict()) != student_digest:
        raise ValueError('Evaluation mutated model state')
    contact = Image.new('RGB', (960, len(entries)*348+30), (24,24,24))
    draw = ImageDraw.Draw(contact)
    draw.text((5,5), 'DIAGNOSTIC ONLY: source | actual student | edited target; TRAIN is not unseen evidence', fill='white')
    for index,row in enumerate(entries):
        y=30+index*348
        for column in range(3):
            contact.paste(thumbnails[index*3+column], (column*320,y))
        draw.text((5,y+324), f'{row["id"]} {"TRAIN" if row["split"] == "train" else "HOLDOUT"}: source | student | target', fill='white')
    contact.save(args.output/'comparison.png')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    report={'purpose':'Completed seven-pair reconstruction/holdout comparison; training examples are not unseen successes',
            'production_approved':False, 'checkpoint_step':checkpoint['step'], 'checkpoint_sha256':checkpoint_sha,
            'checkpoint_path':str(args.checkpoint.resolve()), 'student_state_sha256':student_digest,
            'source_weights_sha256':weights_sha, 'source_metadata_sha256':metadata_sha,
            'source_state_sha256':checkpoint['source_digest'], 'frozen_state_sha256':checkpoint['frozen_digest'],
            'manifest_path':str(args.manifest.resolve()), 'manifest_sha256':manifest_sha,
            'config_sha256':sha256(configuration_path), 'training_provenance':checkpoint['provenance'],
            'runtime':{'platform':platform.platform(),'python':sys.version,**versions,'cpu_threads':1,'interop_threads':1},
            'memory_preflight_free_percent':free,'load_and_verify_seconds':verified_seconds,
            'total_seconds':time.monotonic()-started,'peak_rss_bytes':rss,'count':len(entries),
            'train_count':6,'holdout_count':1,'script_sha256':sha256(__file__),
            'guard_helper_sha256':sha256(guards.__file__),'shared_metric_helper_sha256':sha256(guards.shared.__file__),
            'contact_sha256':sha256(args.output/'comparison.png'),'entries':entries,
            'rendering':'Native 1024 actual student forwards at original W, constant noise, FP32; no compositing. Source generated exactly; edited target resized from 1254 to 1024 via unchanged loader Lanczos, no crop/warp.',
            'metrics_caveat':'Unclipped grayscale errors on nominal [0,1] scale, not quality or identity scores. Protected regions use original sources; clothing regions use imperfect edited targets. Validation 028 lacks a traced collar mask: collar/rest metrics remain null, never invented. TRAIN results are reconstruction evidence only. A single difficult beard holdout cannot establish population generalization; use separate disjoint source30 evaluation too.'}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['count','train_count','holdout_count','total_seconds','peak_rss_bytes']},indent=2))


if __name__=='__main__':
    main()
