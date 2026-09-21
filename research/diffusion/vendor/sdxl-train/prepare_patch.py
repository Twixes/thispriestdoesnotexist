"""Derive a bounded cached-input SDXL trainer from retained official source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
original=(HERE/'upstream.py').read_text()
assert hashlib.sha256(original.encode()).hexdigest() == '7a00615e035804162fa35a092e6b3cfd02cf560b43dddaed7b38a72b51c23822'
s=original

def replace(old,new,count=1):
    global s
    assert s.count(old)==count,(old,s.count(old))
    s=s.replace(old,new)

def between(start,end,new):
    global s
    assert s.count(start)==1 and s.count(end)==1
    a=s.index(start);b=s.index(end,a)
    s=s[:a]+new+s[b:]

replace('def main(args):','import guards as smoke\nfrom priest_cache import validate_contract, load_cache, collate_fn\n\n\ndef main(args):')
replace('    if input_args is not None:\n','    parser.add_argument("--priest_cached_data", type=str, required=True)\n    if input_args is not None:\n')
replace('    # Make one log on every process with the configuration for debugging.\n',
        '    smoke_state = smoke.initialize(args, accelerator)\n    cache_contract = validate_contract(args)\n\n    # Make one log on every process with the configuration for debugging.\n')
between('    # Load the tokenizers\n','    if args.enable_npu_flash_attention:',
'''    # Encoders and VAE were run separately; this process loads only UNet.
    weight_dtype = torch.float16
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        variant=args.variant, torch_dtype=weight_dtype, device_map={"": "mps"},
        low_cpu_mem_usage=True, local_files_only=True)
    unet.requires_grad_(False)
    smoke.resources(smoke_state, "unet_loaded_direct_mps", 0)

''')
between('    def load_model_hook(models, input_dir):\n','    if args.gradient_checkpointing:',
'''    accelerator.register_save_state_pre_hook(save_model_hook)

''')
replace('    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs\n',
        '    smoke.start_evidence(smoke_state, unet, ["to_k", "to_q", "to_v", "to_out.0"])\n\n    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs\n')
between('    # Get the datasets: you can either provide your own training and evaluation files (see below)\n',
        '    # DataLoaders creation:\n',
'''    train_dataset = load_cache(cache_contract, smoke, smoke_state)
    vae_scaling_factor = cache_contract[1]["vae_scaling_factor"]

''')
between('                # Convert images to latent space\n','                # Sample noise that we\'ll add to the latents\n',
'''                step_started = time.monotonic()
                smoke.resources(smoke_state, "before_forward", global_step)
                # Match upstream posterior sampling, scaling, then FP16 cast.
                posterior_noise = torch.randn_like(batch["latent_mean"])
                model_input = (batch["latent_mean"] + batch["latent_std"] * posterior_noise)
                model_input = (model_input * vae_scaling_factor).to(weight_dtype)
                smoke.finite(smoke_state, model_input, "sampled_scaled_latents", global_step)

''')
replace('''                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                )
''','''                prompt_embeds = batch["prompt_embeds"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]
''')
replace('                # Backpropagate\n                accelerator.backward(loss)\n',
'''                # Backpropagate; fail before propagating nonfinite values.
                smoke.finite(smoke_state, loss, "loss", global_step)
                smoke.resources(smoke_state, "before_backward", global_step, loss=float(loss.detach().item()))
                accelerator.backward(loss)
                smoke.gradient_check(smoke_state, unet, global_step)
''')
replace('                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)\n',
'''                    clipped_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                    smoke.finite(smoke_state, clipped_norm, "clipped_gradient_norm", global_step)
''')
replace('                optimizer.step()\n                lr_scheduler.step()\n',
'''                optimizer.step()
                if accelerator.optimizer_step_was_skipped:
                    raise FloatingPointError("Optimizer update was skipped; no successful step recorded")
                smoke.after_optimizer(smoke_state, unet, global_step + 1, loss, step_started)
                lr_scheduler.step()
''')
replace('                        accelerator.save_state(save_path)\n',
'''                        accelerator.save_state(save_path)
                        smoke.saved_adapter_evidence(save_path,
                            convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(unet))))
''')
replace('        unet = unwrap_model(unet)\n        unet_lora_state_dict',
        '        unet = unwrap_model(unet)\n        smoke.finish_evidence(smoke_state, unet)\n        unet_lora_state_dict')
between('        del unet\n','    accelerator.end_training()\n',
'''        smoke.saved_adapter_evidence(args.output_dir, unet_lora_state_dict)
        # Inference is a separate process after releasing all training allocations.

''')
# Module time is used only for per-step timing.
replace('import argparse\n','import argparse\nimport time\n')
ast.parse(s)
(HERE/'run_training.py').write_text(s)
(HERE/'upstream.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream.py',tofile='run_training.py')))
manifest={'upstream_commit':'9f1246971270c84dcbe71233edb7a519596a5d02',
    'upstream_path':'examples/text_to_image/train_text_to_image_lora_sdxl.py',
    'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
        ['upstream.py','LICENSE','run_training.py','guards.py','priest_cache.py','upstream.patch']},
    'objective':'Unchanged ordinary DDPM noise prediction MSE, uniform random training timesteps, fresh stochastic VAE posterior samples; not adversarial diffusion distillation.',
    'risk':'Ordinary denoising LoRA can degrade distilled SDXL-Turbo one-step behavior. Separate frozen-seed one-step before/after native image review required.',
    'scope':'Single-process MPS, FP16 frozen UNet, FP32 rank8/16 attention LoRA; 20/100/250 updates only, no resume/in-process inference/publication.',
    'executed':False,'production_approved':False}
(HERE/'patch-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
