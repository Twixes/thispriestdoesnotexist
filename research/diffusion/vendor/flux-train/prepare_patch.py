"""Reproduce the documented patch from the pinned unchanged upstream source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
original=(HERE/'upstream.py').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='83bccc8c041496aa33939b6ff1f95d0929c1c71a9a29cdc64ecacd4a503fccfe'
s=original

def replace(old,new,count=1):
    global s
    assert s.count(old)==count, (old,s.count(old))
    s=s.replace(old,new)

replace('import warnings\n','import warnings\nimport time\n\nimport smoke_guards as smoke\n')
replace('        accelerator.native_amp = False\n','        accelerator.native_amp = False\n\n    smoke_state = smoke.initialize(args, accelerator)\n')
replace('args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant\n',
        'args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,\n        torch_dtype=weight_dtype,\n')
replace('    text_encoder.requires_grad_(False)\n','    text_encoder.requires_grad_(False)\n    smoke.resources(smoke_state, "models_loaded_cpu", 0)\n')
replace('    text_encoder.to(**to_kwargs)\n','    text_encoder.to(**to_kwargs)\n    smoke.resources(smoke_state, "models_placed", 0)\n')
replace('for i in range(24)', 'for i in range(transformer.config.num_single_layers)')
replace('    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))\n',
        '    smoke.start_evidence(smoke_state, transformer, target_modules)\n    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))\n')
replace('    precompute_latents = args.cache_latents or train_dataset.custom_instance_prompts\n',
        '    smoke.write_json(Path(args.output_dir) / "loaded-dataset.json", {\n        "count": train_dataset.num_instance_images, "captions": train_dataset.custom_instance_prompts,\n        "custom_captions_used": bool(train_dataset.custom_instance_prompts)})\n    if not train_dataset.custom_instance_prompts:\n        raise ValueError("Smoke requires the reviewed per-image captions")\n    precompute_latents = args.cache_latents or train_dataset.custom_instance_prompts\n')
replace('        return prompt_embeds, text_ids\n',
        '        smoke.finite(smoke_state, prompt_embeds, "cached_prompt_embeddings")\n        smoke.finite(smoke_state, text_ids, "cached_text_ids")\n        smoke.resources(smoke_state, "prompt_cached", 0)\n        return prompt_embeds, text_ids\n')
replace('        for batch in tqdm(cache_dataloader, desc="Caching latents"):\n',
        '        for batch in tqdm(cache_dataloader, desc="Caching latents"):\n            smoke.resources(smoke_state, "before_cache_batch", 0)\n')
replace('                        latents = vae.encode(batch["pixel_values"]).latent_dist.mode()\n',
        '                        latents = vae.encode(batch["pixel_values"]).latent_dist.mode()\n                        smoke.finite(smoke_state, latents, "cached_vae_latents")\n                        smoke.resources(smoke_state, "image_latent_cached", 0)\n')
replace('    del text_encoder, tokenizer\n    free_memory()\n',
        '    del text_encoder, tokenizer, text_encoding_pipeline\n    free_memory()\n    smoke.resources(smoke_state, "encoder_released", 0)\n')
replace('        for batch in train_dataloader:\n',
        '        for batch in train_dataloader:\n            smoke_step_started = time.monotonic()\n            smoke.resources(smoke_state, "before_training_step", global_step + 1)\n')
replace('                accelerator.backward(loss)\n',
        '                smoke.finite(smoke_state, loss, "loss", global_step + 1)\n                smoke.resources(smoke_state, "before_backward", global_step + 1)\n                accelerator.backward(loss)\n                smoke.gradient_check(smoke_state, unwrap_model(transformer), global_step + 1)\n')
replace('                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n',
        '                    smoke_clip_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n                    smoke.finite(smoke_state, smoke_clip_norm, "clipped_gradient_norm", global_step + 1)\n')
replace('                optimizer.step()\n',
        '                optimizer.step()\n                smoke.after_optimizer(smoke_state, unwrap_model(transformer), global_step + 1, loss, smoke_step_started)\n')
replace('    # Save the lora layers\n',
        '    smoke.finish_evidence(smoke_state, unwrap_model(transformer))\n\n    # Save the lora layers\n')
replace('''                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
''','''                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
            smoke.saved_adapter_evidence(output_dir, transformer_lora_layers_to_save.keys())
''')
replace('''            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
''','''            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        smoke.saved_adapter_evidence(args.output_dir, transformer_lora_layers.keys())
''')
ast.parse(s)
(HERE/'run_training.py').write_text(s)
(HERE/'upstream.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream.py',tofile='run_training.py')))
changes={'upstream_sha256':hashlib.sha256(original.encode()).hexdigest(), 'patched_sha256':hashlib.sha256(s.encode()).hexdigest(),
         'helper_sha256':hashlib.sha256((HERE/'smoke_guards.py').read_bytes()).hexdigest(),
         'changes':['Qwen initial load uses existing weight_dtype; transformer already had this upstream.',
                    'VAE initial FP32 load retained because upstream computes batch-normalization mean/std before later casting; no normalization math change.',
                    'LoRA single-block targets follow actual num_single_layers config (20for4B).',
                    'Delete text_encoding_pipeline after caching to release its text encoder reference.',
                    'Finite cached latent/prompt, loss, gradient and updated adapter checks.',
                    'Resource checks and logs at load/cache/forward/backward/update boundaries; external watchdog still required for operation peaks.',
                    'Hash every frozen state tensor before/after20steps and require no changes; record changed adapter tensors and all saved safetensor keys.'],
         'training_or_import_execution_performed':False,'production_approved':False}
(HERE/'patch-provenance.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps(changes,indent=2))
