"""Ten-update NADA-style CPU worker. Imported only after the supervisor guards."""
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


def run(root, output, pins):
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
    require(len({row.tobytes() for name in ['source_z', 'train_z', 'eval_z'] for row in latent_arrays[name]}) == 50,
            'Source/train/evaluation z overlap')
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
    recipe = {'method': 'image-guided NADA-style variant; not exact official reproduction',
              'device': 'cpu', 'threads': 1, 'dtype': 'float32', 'updates': 10,
              'batch': 1, 'lr': .0016, 'betas': [0., .99 ** .8], 'epsilon': 1e-6,
              'first_update': 'global target-image-centroid cosine bootstrap',
              'updates_2_to_10': 'source/student CLIP feature-difference cosine to image-centroid direction',
              'clothing_or_face_masks': False, 'identity_preservation_loss': False,
              'discriminator': False, 'ema': False, 'auto_layer_selection': False,
              'noise_mode': 'const', 'truncation_psi': 1, 'style_mixing': False,
              'postprocessing': 'differentiable grayscale in loss; retain complete native RGB and grayscale previews',
              'trainable_convolutions': conv_names, 'trainable_parameters': list(trainable),
              'trainable_count': sum(p.numel() for p in trainable.values()),
              'source_state_sha256': initial_source_hash, 'clip_state_sha256': initial_clip_hash,
              'frozen_student_sha256': frozen_hash, 'input_pins': pins}
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

    targets, source_features, reproduction = [], [], []
    with torch.no_grad():
        for entry in manifest['targets24']:
            tensor, rgb, _ = base.rgb_tensor(root / entry['image'], 1024, resize=False)
            require(bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])), 'Target must be RGB grayscale')
            targets.append(encode(tensor))
        for index, entry in enumerate(manifest['source32']):
            z = latents['source_z'][index:index+1]
            w = source.mapping(z, None, truncation_psi=1, skip_w_avg_update=True)
            w_error = float((w - latents['source_w'][index:index+1]).abs().max())
            require(w_error == 0, f'Exact CPU1 W reproduction failed for {entry["id"]}: {w_error}')
            original = source.synthesis(w, noise_mode='const', force_fp32=True)
            _, rgb, _ = base.rgb_tensor(root / entry['image'], 1024, resize=False)
            pixel_error = int(np.abs(base.quantized(original).astype(np.int16) - rgb.astype(np.int16)).max())
            require(pixel_error == 0, f'Exact source PNG reproduction failed for {entry["id"]}: {pixel_error}')
            source_features.append(encode(original))
            reproduction.append({'id': entry['id'], 'w_max_error': w_error, 'rgb_max_error': pixel_error})
        target_features = torch.cat(targets).detach()
        source_features = torch.cat(source_features).detach()
        target_mean = target_features.mean(0, keepdim=True)
        direction_raw = target_mean - source_features.mean(0, keepdim=True)
        require(float(target_mean.norm()) > 1e-6 and float(direction_raw.norm()) > 1e-6, 'Degenerate centroid direction')
        direction = unit(direction_raw).detach()
        target_unit = unit(target_mean).detach()
        train_source_features = torch.cat([encode(generate(source, z[None])) for z in latents['train_z']]).detach()
    dump(output / 'source-reproduction.json', {'complete': True, 'entries': reproduction,
        'direction_norm_before_normalization': float(direction_raw.norm()), 'centroid_targets': 24, 'centroid_sources': 32})

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
                    if step == 0:
                        require(torch.equal(original, generated), 'Initial source/student image differs')
                    for label, tensor in [('student-rgb', generated), ('student-gray', base.grayscale(generated).expand(-1,3,-1,-1))]:
                        file = folder / f'{i:03}-{label}.png'
                        Image.fromarray(base.quantized(tensor)).save(file)
                        previews.append({'step': step, 'latent_index': i, 'kind': label,
                                         'path': str(file.relative_to(output)), 'sha256': base.sha256(file)})
                    if step == 0:
                        for label, tensor in [('source-rgb', original), ('source-gray', base.grayscale(original).expand(-1,3,-1,-1))]:
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

    snapshot(0)
    metrics = []
    for index, z in enumerate(latents['train_z']):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        before = base.state_digest(trainable)
        generated = generate(student, z[None])
        generated.retain_grad()
        features = encode(generated)
        delta = features - train_source_features[index:index+1]
        delta_norm = float(delta.detach().norm())
        if index == 0:
            require(delta_norm == 0, 'Bootstrap expected identical initial generator features')
            loss = 1 - (features * target_unit).sum(-1).mean()
        else:
            require(delta_norm > 1e-6, 'Edit direction remains degenerate after bootstrap')
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
        metric = {'step': index+1, 'loss': float(loss.detach()), 'edit_norm': delta_norm,
                  'bootstrap': index == 0, 'gradient_norm_before_update': gradient_norm,
                  'image_gradient_norm': image_gradient_norm, 'finite_before_sanitization': True,
                  'gradient_sanitization_or_clipping': False, 'updated_state_sha256': after,
                  'seconds': time.monotonic()-started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        metrics.append(metric)
        with (output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metric)+'\n')
        print(json.dumps(metric), flush=True)
        del generated, features, delta, loss, gradients
        if index+1 in [5, 10]: snapshot(index+1)

    require(base.state_digest(source.state_dict()) == initial_source_hash, 'Frozen source changed')
    require(base.state_digest(encoder.state_dict()) == initial_clip_hash, 'Frozen CLIP changed')
    frozen_checks()
    optimizer.zero_grad(set_to_none=True)
    state = {'format_version': 1, 'step': 10, 'student': student.state_dict(),
             'optimizer': optimizer.state_dict(), 'rng': rng(), 'recipe': recipe,
             'direction': direction, 'target_features': target_features, 'source_features': source_features,
             'train_source_features': train_source_features, 'target_unit': target_unit,
             'latents_npz_sha256': manifest['latents_npz_sha256']}
    expected = nested_hash(state)
    base.atomic_save(state, output / 'checkpoint-010.pt')
    del state
    restored = torch.load(output / 'checkpoint-010.pt', map_location='cpu', mmap=True, weights_only=True)
    require(nested_hash(restored) == expected, 'Full checkpoint serialization roundtrip changed state')
    # Demonstrate actual restoration, not a vacuous re-read of unchanged objects.
    with torch.no_grad():
        next(iter(trainable.values())).flatten()[0].add_(1)
        first_state = next(iter(optimizer.state.values()))
        first_state['exp_avg'].flatten()[0].add_(1)
    torch.rand(1); random.random(); np.random.rand()
    student.load_state_dict(restored['student'], strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    restore(restored['rng'])
    require(nested_hash(student.state_dict()) == nested_hash(restored['student']), 'Generator reload differs')
    require(nested_hash(optimizer.state_dict()) == nested_hash(restored['optimizer']), 'Optimizer reload differs')
    require(nested_hash(rng()) == nested_hash(restored['rng']), 'RNG reload differs')
    with torch.no_grad():
        check = base.quantized(generate(student, latents['eval_z'][0:1]))
    with Image.open(output / 'preview-010/000-student-rgb.png') as image:
        require(np.array_equal(check, np.array(image)), 'Reloaded generator changed native preview')
    require(nested_hash(rng()) == nested_hash(restored['rng']), 'Reloaded preview consumed RNG')
    frozen_checks()
    dump(output / 'worker-result.json', {'complete': True, 'updates': 10, 'optimizer_calls': 10,
        'raw_generator_only': True, 'actual_optimizer_and_rng_reload': True,
        'full_checkpoint_nested_sha256': expected, 'checkpoint_sha256': base.sha256(output/'checkpoint-010.pt'),
        'source_and_clip_unchanged': True, 'frozen_student_unchanged': True,
        'snapshot_rng_and_state_checks': [0, 5, 10], 'checkpoint_reloaded_native_preview_equal': True,
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
