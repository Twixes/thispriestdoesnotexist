"""Local smoke-run evidence and fail-fast guards; no training objective changes."""
import hashlib
import json
import math
from pathlib import Path
import time

import psutil
import torch
from safetensors import safe_open


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def initialize(args, accelerator):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    if accelerator.num_processes != 1 or accelerator.device.type != 'mps':
        raise ValueError('This bounded adapter smoke is single-process MPS only')
    if args.max_train_steps != 20 or args.gradient_accumulation_steps != 1:
        raise ValueError('Smoke requires exactly20steps and accumulation1')
    if args.with_prior_preservation or args.resume_from_checkpoint or args.push_to_hub:
        raise ValueError('Smoke excludes prior generation, resume and hub publication')
    if args.validation_prompt or args.final_validation_prompt or not args.skip_final_inference:
        raise ValueError('Inference must run separately after this smoke')
    if args.checkpoints_total_limit is not None:
        raise ValueError('Retain every smoke checkpoint')
    if args.mixed_precision != 'fp16' or not args.cache_latents or not args.offload:
        raise ValueError('Smoke requires fp16, cached latents and offloaded encoder')
    if args.bnb_quantization_config_path or args.do_fp8_training or args.fsdp_text_encoder:
        raise ValueError('No quantization, FP8 or FSDP in this smoke')
    state = {'output': Path(args.output_dir), 'started': time.monotonic(),
             'initial_swap_used': psutil.swap_memory().used, 'initial_base': None,
             'initial_adapter': None, 'steps': 0}
    state['output'].mkdir(parents=True, exist_ok=True)
    for name in ('run_training.py', 'smoke_guards.py', 'patch-provenance.json'):
        source = Path(__file__).parent / name
        (state['output'] / name).write_bytes(source.read_bytes())
    resources(state, 'before_model_load', 0)
    return state


def resources(state, phase, step, **extra):
    torch.mps.synchronize()
    vm = psutil.virtual_memory()
    entry = {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic()-state['started'],
             'rss_bytes': psutil.Process().memory_info().rss,
             'mps_allocated_bytes': torch.mps.current_allocated_memory(),
             'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory(),
             'system_available_bytes': vm.available, 'system_total_bytes': vm.total,
             'system_available_fraction': vm.available/vm.total,
             'system_swap_used_bytes': psutil.swap_memory().used, **extra}
    with (state['output']/'worker-resources.jsonl').open('a') as stream:
        stream.write(json.dumps(entry)+'\n')
        stream.flush()
    reason = None
    if entry['rss_bytes'] > 24*1024**3:
        reason = 'worker RSS exceeded24GiB'
    elif entry['mps_driver_allocated_bytes'] > 22*1024**3:
        reason = 'MPS driver allocation exceeded22GiB'
    elif vm.available/vm.total < .25:
        reason = 'system available memory below25%'
    elif entry['system_swap_used_bytes']-state['initial_swap_used'] > 512*1024**2:
        reason = 'system swap grew by more than512MiB during smoke'
    elif entry['elapsed_seconds'] > 1800:
        reason = '30minute smoke deadline'
    if reason:
        write_json(state['output']/'guard-failure.json', {'reason': reason, 'resources': entry})
        raise RuntimeError(reason)
    return entry


def finite(state, tensor, label, step=0):
    if not bool(torch.isfinite(tensor).all().item()):
        write_json(state['output']/'nonfinite-failure.json', {'label': label, 'step': step,
            'dtype': str(tensor.dtype), 'shape': list(tensor.shape)})
        raise FloatingPointError(f'Nonfinite {label} at step{step}')


def tensor_digest(tensor):
    # At most one tensor copied to CPU at a time; no second full model retained.
    value = tensor.detach().cpu().contiguous()
    header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
    digest = hashlib.sha256(header)
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def state_digests(state, model, phase):
    base, adapter = {}, {}
    for name, value in model.state_dict().items():
        if '.lora_' in name:
            adapter[name] = tensor_digest(value)
        else:
            base[name] = tensor_digest(value)
        resources(state, phase, state['steps'], tensor_name=name)
    return base, adapter


def start_evidence(state, model, target_modules):
    if model.config.num_layers != 5 or model.config.num_single_layers != 20:
        raise ValueError('Expected klein4B config with5dual and20single blocks')
    trainable = {name: {'shape': list(p.shape), 'dtype': str(p.dtype), 'elements': p.numel()}
                 for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any('.lora_A.' not in n and '.lora_B.' not in n for n in trainable):
        raise ValueError('Only LoRA matrices may be trainable')
    if any(v['dtype'] != 'torch.float32' for v in trainable.values()):
        raise ValueError('FP16 smoke requires FP32 trainable LoRA matrices')
    matched = [name for name, module in model.named_modules() if hasattr(module, 'lora_A')]
    if any(not any(name == target or name.endswith('.'+target) for name in matched) for target in target_modules):
        raise ValueError('A requested LoRA target matched no module')
    state['initial_base'], state['initial_adapter'] = state_digests(state, model, 'initial_state_hash')
    write_json(state['output']/'adapter-initialization.json', {'target_modules': target_modules,
        'matched_modules': matched, 'trainable_parameters': trainable,
        'trainable_elements': sum(v['elements'] for v in trainable.values()),
        'frozen_tensor_hashes': state['initial_base'], 'initial_adapter_tensor_hashes': state['initial_adapter'],
        'production_approved': False})


def gradient_check(state, model, step):
    total_norm = 0.0
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError('Frozen parameter received gradient: '+name)
            continue
        if parameter.grad is None:
            raise RuntimeError('Trainable parameter has no gradient: '+name)
        finite(state, parameter.grad, 'gradient:'+name, step)
        norm = float(torch.linalg.vector_norm(parameter.grad.float()).item())
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm:'+name)
        total_norm = math.hypot(total_norm, norm)
        count += 1
    if count == 0:
        raise RuntimeError('No adapter gradients')
    resources(state, 'after_backward', step, gradient_tensor_count=count, unclipped_gradient_norm=total_norm)
    return total_norm


def after_optimizer(state, model, step, loss, started):
    for name, p in model.named_parameters():
        if p.requires_grad:
            finite(state, p, 'updated_parameter:'+name, step)
    state['steps'] = step
    resources(state, 'after_optimizer', step, loss=float(loss.detach().item()),
              step_seconds=time.monotonic()-started)


def finish_evidence(state, model):
    base, adapter = state_digests(state, model, 'final_state_hash')
    unchanged = base == state['initial_base']
    changed = [name for name in adapter if adapter[name] != state['initial_adapter'].get(name)]
    result = {'steps': state['steps'], 'frozen_base_exactly_unchanged': unchanged,
              'adapter_key_set_unchanged': set(adapter) == set(state['initial_adapter']),
              'changed_adapter_keys': changed, 'final_adapter_tensor_hashes': adapter,
              'final_frozen_tensor_hashes': base, 'production_approved': False}
    write_json(state['output']/'training-state-evidence.json', result)
    if not unchanged or not result['adapter_key_set_unchanged'] or not changed:
        raise RuntimeError('Frozen-base/changed-adapter evidence failed')
    if state['steps'] != 20:
        raise RuntimeError('Incomplete20update smoke')


def saved_adapter_evidence(directory, expected_keys):
    directory = Path(directory)
    path = directory/'pytorch_lora_weights.safetensors'
    records = {}
    with safe_open(path, framework='pt', device='cpu') as source:
        for name in source.keys():
            value = source.get_tensor(name)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Saved adapter contains nonfinite tensor:'+name)
            records[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_digest(value)}
    expected = {'transformer.'+name for name in expected_keys}
    if set(records) != expected:
        raise RuntimeError('Serialized adapter keys differ from expected PEFT state')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(directory/'saved-adapter-evidence.json', {'path': path.name, 'sha256': digest,
        'keys_exactly_match_peft_state': True, 'tensors': records, 'production_approved': False})
