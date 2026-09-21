"""Exact 10-to-500 NADA-style CPU continuation worker. Imported only after the supervisor guards."""
import copy
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time


def run(root, output, pins, restore_only=False):
    # Selection's environment remains untouched. Only these pinned, missing
    # package roots may be resolved from the existing research environment.
    class ExistingPackageFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if path is None and fullname in pins['borrowed_package_roots']:
                return importlib.machinery.PathFinder.find_spec(fullname, [str(root / pins['borrowed_site'])])
            return None
    sys.meta_path.append(ExistingPackageFinder())
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(202609210604)
    np.random.seed(20260921)
    random.seed(20260921)
    sys.path.insert(0, str(root / pins['clip_vendor']))
    import clip
    require(Path(clip.__file__).resolve().is_relative_to(root / pins['clip_vendor']), 'Wrong CLIP source')
    base_path = root / 'research/experiments/paired_edit/trainer.py'
    spec = importlib.util.spec_from_file_location('pinned_paired_helpers', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    # base imports the pinned native NVIDIA Generator and CPU reference ops.
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'inputs.json').read_text())
    with np.load(here / 'latents.npz', allow_pickle=False) as archive:
        latent_arrays = {name: archive[name].copy() for name in archive.files}
    require({k: list(v.shape) for k, v in latent_arrays.items()} == manifest['array_shapes'], 'Latent shape changed')
    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in latent_arrays.values()), 'Bad latent dtype/value')
    require(len({row.tobytes() for name in ['source_z', 'previous_train_z', 'train_z', 'eval_z'] for row in latent_arrays[name]}) == 540,
            'Original50/fresh490 z overlap')
    latents = {name: torch.from_numpy(value) for name, value in latent_arrays.items()}
    bundle = root / 'research/runs/inference-cpu/ffhq1024/baseline-bundle'
    metadata = json.loads((bundle / 'model.json').read_text())
    source = base.Generator(**metadata['init_kwargs']).eval().requires_grad_(False)
    source.load_state_dict(load_file(str(bundle / 'generator.safetensors'), device='cpu'), strict=True)
    require(source.img_resolution == 1024 and source.num_ws == 18, 'Wrong source architecture')
    student = copy.deepcopy(source)
    student.synthesis.b4.const.requires_grad_(True)
    conv_names = []
    for resolution in student.synthesis.block_resolutions:
        block = student.synthesis._modules[f'b{resolution}']
        for key in ['conv0', 'conv1']:
            if key in block._modules:
                block._modules[key].requires_grad_(True)
                conv_names.append(f'synthesis.b{resolution}.{key}')
    trainable = {name: p for name, p in student.named_parameters() if p.requires_grad}
    require(len(conv_names) == 17, 'Expected all 17 synthesis convolutions')
    require(all(name == 'synthesis.b4.const' or any(name.startswith(c + '.') for c in conv_names)
                for name in trainable), 'Unexpected trainable parameter')
    require(not any('torgb' in name or name.startswith('mapping.') for name in trainable), 'Frozen layer became trainable')
    encoder, _ = clip.load(str(root / 'research/selection/models/ViT-B-32.pt'), device='cpu', jit=False)
    encoder.float().eval().requires_grad_(False)
    require(all(p.dtype == torch.float32 for p in encoder.parameters()), 'CLIP must be FP32')
    initial_source_hash = base.state_digest(source.state_dict())
    initial_clip_hash = base.state_digest(encoder.state_dict())
    frozen_hash = base.state_digest(base.frozen_state(student))
    require(base.state_digest(student.state_dict()) == initial_source_hash, 'Student/source initialization differs')
    optimizer = torch.optim.Adam(list(trainable.values()), lr=.002 * .8, betas=(0., .99 ** .8))
    continuation = manifest['continuation']
    parent_result = json.loads((root / continuation['previous_result']).read_text())
    require(parent_result['complete'] and parent_result['updates'] == 10
            and parent_result['actual_optimizer_and_rng_reload'], 'Prior smoke incomplete')
    checkpoint = root / continuation['checkpoint']
    require(base.sha256(checkpoint) == continuation['checkpoint_sha256'] == parent_result['checkpoint_sha256'],
            'Wrong trusted checkpoint010')
    prior = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    require(prior['format_version'] == 1 and prior['step'] == 10, 'Wrong checkpoint schema/step')
    parent_recipe = prior['recipe']
    require(parent_recipe['source_state_sha256'] == initial_source_hash
            and parent_recipe['clip_state_sha256'] == initial_clip_hash
            and parent_recipe['frozen_student_sha256'] == frozen_hash, 'Frozen initialization provenance differs')
    require(parent_recipe['trainable_parameters'] == list(trainable)
            and parent_recipe['trainable_convolutions'] == conv_names, 'Parameter ordering/trainability changed')
    require(parent_recipe['lr'] == .0016 and parent_recipe['betas'] == [0., .99 ** .8]
            and parent_recipe['threads'] == 1 and parent_recipe['dtype'] == 'float32', 'Prior recipe mismatch')
    recipe = {**parent_recipe, 'continuation': continuation, 'input_pins': pins,
              'updates': 490, 'start_step': 10, 'end_step': 500,
              'first_update': 'Already completed in checkpoint010; no bootstrap repeats',
              'updates_11_to_500': 'Unchanged directional cosine loss',
              'fresh_source_features': 'Per-step no-grad, overhead included and separately measured'}
    dump(output / 'recipe.json', recipe)

    def unit(value):
        require(bool(torch.isfinite(value).all()), 'Nonfinite feature')
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def encode(image):
        gray_rgb = base.grayscale(image).expand(-1, 3, -1, -1)
        scaled = F.interpolate((gray_rgb + 1) / 2, size=(224, 224), mode='bicubic',
                               align_corners=False, antialias=True)
        mean = scaled.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = scaled.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        return unit(encoder.encode_image((scaled - mean) / std))

    def generate(model, z):
        w = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
        return model.synthesis(w, noise_mode='const', force_fp32=True)

    # Preserve the exact already-verified centroid features/direction from checkpoint010.
    direction = prior['direction'].detach().clone()
    target_features = prior['target_features'].detach().clone()
    source_features = prior['source_features'].detach().clone()
    target_unit = prior['target_unit'].detach().clone()
    require(tuple(target_features.shape) == (24, 512) and tuple(source_features.shape) == (32, 512)
            and tuple(direction.shape) == tuple(target_unit.shape) == (1, 512), 'Feature shape mismatch')
    require(all(bool(torch.isfinite(x).all()) for x in [direction, target_features, source_features, target_unit]),
            'Nonfinite preserved features')

    def rng():
        n = np.random.get_state()
        return {'torch': torch.get_rng_state(), 'python': random.getstate(),
                'numpy': [n[0], n[1].tolist(), *n[2:]]}

    def restore(state):
        torch.set_rng_state(state['torch'])
        random.setstate(state['python'])
        n = state['numpy']
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))

    def nested_hash(value):
        h = hashlib.sha256()
        def visit(item):
            if torch.is_tensor(item):
                t = item.detach().cpu().contiguous()
                h.update(str((str(t.dtype), list(t.shape))).encode()); h.update(t.numpy().tobytes())
            elif isinstance(item, dict):
                for k in sorted(item, key=str):
                    h.update(repr(k).encode()); visit(item[k])
            elif isinstance(item, (list, tuple)):
                h.update(type(item).__name__.encode())
                for v in item: visit(v)
            else: h.update(repr(item).encode())
        visit(value)
        return h.hexdigest()

    def frozen_checks():
        require(base.state_digest(base.frozen_state(student)) == frozen_hash, 'Frozen student state changed')
        require(all(p.grad is None for p in source.parameters()) and all(p.grad is None for p in encoder.parameters()), 'Frozen source/CLIP gradient')

    previews = []
    def snapshot(step):
        state_rng = rng()
        state_hash = base.state_digest(student.state_dict())
        rows = []
        folder = output / f'preview-{step:03}'
        folder.mkdir(exist_ok=False)
        try:
            with torch.no_grad():
                for i, z in enumerate(latents['eval_z']):
                    original, generated = generate(source, z[None]), generate(student, z[None])
                    for label, tensor in [('student-rgb', generated), ('student-gray', base.grayscale(generated).expand(-1,3,-1,-1))]:
                        file = folder / f'{i:03}-{label}.png'
                        Image.fromarray(base.quantized(tensor)).save(file)
                        previews.append({'step': step, 'latent_index': i, 'kind': label,
                                         'path': str(file.relative_to(output)), 'sha256': base.sha256(file)})
                    rows.append(np.concatenate([base.quantized(base.grayscale(original).expand(-1,3,-1,-1)),
                                                 base.quantized(base.grayscale(generated).expand(-1,3,-1,-1))], axis=1))
            Image.fromarray(np.concatenate(rows, axis=0)).save(folder / 'source-student-native.png')
            require(nested_hash(rng()) == nested_hash(state_rng), 'Snapshot consumed RNG')
            require(base.state_digest(student.state_dict()) == state_hash, 'Snapshot mutated student')
            frozen_checks()
        finally:
            restore(state_rng)

    require(nested_hash(prior) == parent_result['full_checkpoint_nested_sha256'], 'Trusted full checkpoint payload mismatch')
    student.load_state_dict(prior['student'], strict=True)
    optimizer.load_state_dict(prior['optimizer'])
    restore(prior['rng'])
    require(nested_hash(student.state_dict()) == nested_hash(prior['student']), 'Initial student restoration differs')
    require(nested_hash(optimizer.state_dict()) == nested_hash(prior['optimizer']), 'Initial Adam restoration differs')
    require(nested_hash(rng()) == nested_hash(prior['rng']), 'Initial RNG restoration differs')
    require(all(float(state['step']) == 10 for state in optimizer.state.values()), 'Adam step not10')
    frozen_checks()
    snapshot(10)
    expected_previews = {(entry['latent_index'], entry['kind']): entry['sha256']
                        for entry in parent_result['previews'] if entry['step'] == 10}
    require(len(expected_previews) == 16 and len(previews) == 16, 'Incomplete original step10 evaluation')
    for entry in previews:
        require(entry['sha256'] == expected_previews[(entry['latent_index'], entry['kind'])],
                f'Initial restored step10 native PNG bytes differ: {entry["path"]}')
    dump(output / 'initial-restore.json', {'complete': True, 'checkpoint_sha256': base.sha256(checkpoint),
        'full_checkpoint_nested_sha256': nested_hash(prior), 'student_adam_rng_exact': True,
        'native_png_sha256_matches': 16, 'same_evaluation_latents': True, 'no_training_update_yet': True,
        'source_centroid_features_preserved': True, 'bootstrap_repeated': False})
    del prior
    if restore_only:
        dump(output / 'worker-result.json', {'complete': True, 'restore_only': True,
            'updates': 0, 'optimizer_calls': 0, 'start_step': 10, 'total_step': 10,
            'initial_native_png_matches': 16, 'actual_optimizer_and_rng_reload': True,
            'previews': previews, 'production_approved': False, 'quality_approved': False})
        return

    checkpoints = []
    def save_checkpoint(step):
        # Commit only complete atomic checkpoints. Each gets actual model/Adam/RNG reload checks.
        optimizer.zero_grad(set_to_none=True)
        state = {'format_version': 2, 'step': step, 'student': student.state_dict(),
                 'optimizer': optimizer.state_dict(), 'rng': rng(), 'recipe': recipe,
                 'direction': direction, 'target_features': target_features, 'source_features': source_features,
                 'target_unit': target_unit, 'next_train_index': step-10,
                 'latents_npz_sha256': manifest['latents_npz_sha256']}
        expected = nested_hash(state)
        path = output / f'checkpoint-{step:03}.pt'
        base.atomic_save(state, path)
        del state
        restored = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        require(nested_hash(restored) == expected, 'Full checkpoint serialization changed state')
        # Non-vacuous reload; deliberately perturb live parameter, Adam and each RNG first.
        with torch.no_grad():
            next(iter(trainable.values())).flatten()[0].add_(1)
            next(iter(optimizer.state.values()))['exp_avg'].flatten()[0].add_(1)
        torch.rand(1); random.random(); np.random.rand()
        student.load_state_dict(restored['student'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        restore(restored['rng'])
        require(nested_hash(student.state_dict()) == nested_hash(restored['student']), 'Checkpoint student reload differs')
        require(nested_hash(optimizer.state_dict()) == nested_hash(restored['optimizer']), 'Checkpoint Adam reload differs')
        require(nested_hash(rng()) == nested_hash(restored['rng']), 'Checkpoint RNG reload differs')
        require(all(float(item['step']) == step for item in optimizer.state.values()), 'Unexpected Adam step')
        with torch.no_grad():
            check = base.quantized(generate(student, latents['eval_z'][0:1]))
        with Image.open(output / f'preview-{step:03}/000-student-rgb.png') as image:
            require(np.array_equal(check, np.array(image)), 'Checkpoint reload changed native preview')
        require(nested_hash(rng()) == nested_hash(restored['rng']), 'Reload preview consumed RNG')
        require(base.state_digest(source.state_dict()) == initial_source_hash, 'Frozen source changed')
        require(base.state_digest(encoder.state_dict()) == initial_clip_hash, 'Frozen CLIP changed')
        frozen_checks()
        record = {'step': step, 'path': path.name, 'sha256': base.sha256(path),
                  'full_checkpoint_nested_sha256': expected, 'actual_student_adam_rng_reload': True,
                  'native_preview_reload_equal': True, 'source_clip_and_frozen_student_unchanged': True,
                  'previews': [item for item in previews if item['step'] == step]}
        dump(output / f'checkpoint-{step:03}-review-ready.json', record)
        checkpoints.append(record)
        del restored

    metrics = []
    for index, z in enumerate(latents['train_z']):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        before = base.state_digest(trainable)
        source_started = time.monotonic()
        with torch.no_grad():
            current_source_feature = encode(generate(source, z[None])).detach()
        source_seconds = time.monotonic() - source_started
        generated = generate(student, z[None])
        generated.retain_grad()
        features = encode(generated)
        delta = features - current_source_feature
        delta_norm = float(delta.detach().norm())
        require(delta_norm > 1e-6, 'Degenerate edit direction in continuation; bootstrap is forbidden')
        loss = 1 - (unit(delta) * direction).sum(-1).mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite loss')
        loss.backward()
        require(generated.grad is not None and bool(torch.isfinite(generated.grad).all()) and float(generated.grad.abs().max()) > 0,
                'Missing/nonfinite/zero upstream image gradient through grayscale/resize/CLIP')
        gradients = [p.grad for p in trainable.values()]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients), 'Missing/nonfinite upstream parameter gradient')
        gradient_norm = float(torch.stack([g.detach().norm() for g in gradients]).norm())
        require(0 < gradient_norm < float('inf'), 'Invalid global gradient norm')
        image_gradient_norm = float(generated.grad.norm())
        frozen_checks()
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in trainable.values()), 'Nonfinite updated parameter')
        require(all(bool(torch.isfinite(v).all()) for state in optimizer.state.values()
                    for v in state.values() if torch.is_tensor(v)), 'Nonfinite optimizer state')
        after = base.state_digest(trainable)
        require(after != before, 'Optimizer did not change student')
        frozen_checks()
        metric = {'step': index+11, 'loss': float(loss.detach()), 'edit_norm': delta_norm,
                  'bootstrap': False, 'source_feature_seconds': source_seconds, 'gradient_norm_before_update': gradient_norm,
                  'image_gradient_norm': image_gradient_norm, 'finite_before_sanitization': True,
                  'gradient_sanitization_or_clipping': False, 'updated_state_sha256': after,
                  'seconds': time.monotonic()-started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        metrics.append(metric)
        with (output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metric)+'\n')
        print(json.dumps(metric), flush=True)
        del generated, features, delta, loss, gradients, current_source_feature
        if index+11 in [50, 100, 250, 500]:
            snapshot(index+11)
            save_checkpoint(index+11)

    require(base.state_digest(source.state_dict()) == initial_source_hash, 'Frozen source changed')
    require(base.state_digest(encoder.state_dict()) == initial_clip_hash, 'Frozen CLIP changed')
    frozen_checks()
    dump(output / 'worker-result.json', {'complete': True, 'updates': 490, 'optimizer_calls': 490, 'restore_only': False,
        'start_step': 10, 'total_step': 500, 'raw_generator_only': True,
        'actual_optimizer_and_rng_reload': True, 'checkpoints': checkpoints,
        'source_and_clip_unchanged': True, 'frozen_student_unchanged': True,
        'snapshot_rng_and_state_checks': [10, 50, 100, 250, 500], 'initial_native_png_matches': 16,
        'checkpoint_reloaded_native_preview_equal': True, 'bootstrap_repeated': False,
        'uninterrupted_vs_resumed_training_equivalence_tested': False,
        'latents_sha256': manifest['latents_npz_sha256'], 'previews': previews,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'quality_approved': False, 'production_approved': False})



def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
