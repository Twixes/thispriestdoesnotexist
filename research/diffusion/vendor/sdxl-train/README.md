# Bounded SDXL-Turbo LoRA experiment

`upstream.py` is the unmodified official Diffusers
`examples/text_to_image/train_text_to_image_lora_sdxl.py` at commit
`9f1246971270c84dcbe71233edb7a519596a5d02`. Its Apache license is retained.
`prepare_patch.py` verifies the upstream SHA and deterministically generates the separate
`run_training.py`, complete `upstream.patch`, and hash-bound `patch-provenance.json`.

This adaptation preserves the upstream ordinary denoising objective: a fresh stochastic
VAE posterior sample, VAE scaling, uniform random DDPM training timestep, independent
Gaussian noise, scheduler noise addition, and epsilon/v-prediction MSE according to the
frozen scheduler config. It does **not** reproduce Turbo's adversarial diffusion
 distillation. Ordinary denoising LoRA may harm one-step generation even when loss falls.
Unmodified Turbo and every candidate require fixed-noise before/after review at native
512, including face/eye integrity, priest clothing, no hats, appeal and diversity. The
20-update smoke proves compatibility only. No model is approved for production here.

The trainer loads only the FP16 UNet, directly to MPS using `device_map={"": "mps"}` and
`low_cpu_mem_usage=True`. No text encoder or VAE is loaded. Rank 8 or 16 attention LoRA
uses the original targets `to_k`, `to_q`, `to_v`, `to_out.0`, Gaussian initialization and
alpha equal to rank; all trainable matrices are FP32. Gradient checkpointing is required.
The optimizer is original AdamW; use explicit learning rate, constant scheduler and zero
warmup for reproducible bounded comparisons. Inference runs separately after training.

## Cache contract

`--dataset_name` is the absolute reviewed dataset's `train` directory.
`--priest_cached_data` is an absolute cache directory containing `manifest.json`:

- `complete: true`, `resolution: 512`
- `dataset_manifest_sha256`, `model_provenance_sha256`
- `vae_scaling_factor` equal to the frozen VAE config (currently 0.13025)
- `transform`: exactly `Bilinear Resize512,CenterCrop512,normalize[-1,1],no flip` or
  `Lanczos Resize512,CenterCrop512,normalize[-1,1],no flip`, matching the explicit CLI
  `--image_interpolation_mode bilinear` or `lanczos`. Upstream defaults to Lanczos;
  the earlier cache v1 explicitly used bilinear. Neither is silently substituted.
- `entries`: one entry for every reviewed training file, each with `file_name`, `caption`,
  relative `path`, `sha256`, `original_size: [H,W]`, `crop_top_left: [y,x]`.

Each entry's safetensor has exactly these tensors:

| Key | Shape | Dtype |
|---|---|---|
| `prompt_embeds` | 1×77×2048 | FP16 |
| `pooled_prompt_embeds` | 1×1280 | FP16 |
| `latent_mean` | 1×4×64×64 | FP32 |
| `latent_std` | 1×4×64×64 | FP32 |

Text uses the two original frozen FP16 CLIP encoders: concatenate their penultimate
hidden states, and retain encoder 2's pooled output, matching upstream `encode_prompt`.
No classifier-free negative prompt is needed for the training cache. Run both text
encoders in a separate cache process, release them, then load the VAE in FP32. VAE
posterior mean/std are cached **before scaling**, rather than fixing one sampled latent.
Training draws fresh `mean + std * randn_like(mean)` each update, multiplies by scaling,
then casts to FP16 as upstream does. This preserves the sampling distribution; it does
not promise bitwise identity with an uncached run's random-number sequence.

Validation checks the reviewed train metadata and image hashes, exact image/caption set,
model provenance, safetensor paths/hashes, dimensions/dtypes, finite tensors, nonnegative
posterior standard deviations and explicit interpolation. Entry order need not match a
separate dataset enumeration: each row carries its complete image/text/time-ID metadata.
No validation or test files are loaded.

## Execution boundaries

Root owns environment/import preflight, the independent supervisor and actual runs. Use
Python 3.11 from `research/.venv-flux-train`. Preparation has only parsed syntax and checked
source hashes; it has not run models. A representative smoke configuration is:

```text
--variant fp16 --resolution 512 --center_crop --image_interpolation_mode bilinear
--train_batch_size 1 --dataloader_num_workers 0 --max_train_steps 20
--gradient_accumulation_steps 1 --gradient_checkpointing --mixed_precision fp16
--rank 8 --learning_rate 0.00001 --lr_scheduler constant --lr_warmup_steps 0
--checkpointing_steps 10 --seed 20260921 --report_to tensorboard
```

Supply absolute model, dataset, cache and new output paths. Choose interpolation to match
that cache. Allowed horizons are exactly 20, 100 or 250 updates. Each starts from the frozen
base; resume is deliberately disabled, so a longer run repeats the prefix rather than
loading a prior optimizer. Keep seed, LR and constant schedule identical for comparison.
All snapshots are retained, and an adapter is also saved at successful completion.

Worker boundaries check 24 GiB RSS, 22 GiB MPS driver allocation, at least 20% system
available memory, at most 512 MiB swap growth, and a 30-minute deadline. Torch uses two
CPU threads, one inter-op thread, MPS allocation fraction 0.55. The independent supervisor
must catch peaks between checks; unified-memory counters must not be added together.

The guards require finite cache/latents/loss/gradients/gradient norm/updated matrices,
fail on a skipped optimizer update, and verify only FP32 LoRA A/B matrices train. They
hash the full frozen UNet state before and after, require unchanged base tensors and a
changed adapter, and verify saved safetensor keys against the converted Diffusers adapter
state. Saved adapters are not automatically reloaded or visually approved. Checkpoint
save hooks retain the upstream optimizer/scheduler/RNG state, but this bounded launcher
still prohibits resume. No Hub publishing, text-encoder training or in-process validation
is allowed.
