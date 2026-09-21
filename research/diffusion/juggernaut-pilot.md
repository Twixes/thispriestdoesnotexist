# Bounded Juggernaut priest adaptation

This pilot starts from the exact Juggernaut-X-Hyper revision `42fee7475922d8de7246cdcbe5e3a5e22c0ecf61`, exported into `models/juggernaut-x-hyper-components`. The component provenance SHA256 is `8e1c607e04bef6fcee8e778e9e4fc5d163ae19f29d67d80149debd61ad78774d`; all2641 exported tensors passed the exporter's reload comparison. `juggernaut_pilot.py` checks that fixed provenance, every exported file, the source checkpoint lineage, the development manifest and the unchanged shared trainer before compute.

`juggernaut-pilot-protocol.json` fixes the20-update compatibility/style pilot, then a fresh100-update run only if the native visual-benefit gate passes. There is no250 checkpoint, no failed Turbo adapter reuse, no resume, and no automatic retry. Ordinary noise-prediction LoRA may harm Hyper distillation; loss reduction is not a quality result. Training512 and inference1024 is an explicit cross-resolution experiment. Native1024 comparisons decide continuation.

## Exact-base cache

Only the20 reviewed training portraits/captions are loaded. Five validation and five test portraits remain unopened. Fresh exact-base dual CLIP embeddings use FP16. Both text encoders are released before loading the exact-base VAE in FP32. Cache entries contain:

- `prompt_embeds`: FP16 `[1,77,2048]`
- `pooled_prompt_embeds`: FP16 `[1,1280]`
- `latent_mean` and `latent_std`: FP32 `[1,4,64,64]`, before VAE scaling
- Original image dimensions,512 center-crop coordinates, exact captions and hashes

Lanczos resize/center crop, RGB normalization to[-1,1], no flip or augmentation. Training samples a fresh posterior Gaussian every update, applies VAE scaling and FP16 cast, then independent diffusion noise and an ordinary uniform DDPM timestep. No Turbo cache is accepted: both model provenance and this protocol hash must match.

Run commands from the repository root with the existing training environment, **serially after root's inference work is finished**:

```sh
research/.venv-flux-train/bin/python research/diffusion/cache_juggernaut_training.py --output research/diffusion/runs/juggernaut-train-cache-v1
research/.venv-flux-train/bin/python research/diffusion/train_juggernaut_pilot.py --updates 20 --cache research/diffusion/runs/juggernaut-train-cache-v1 --output research/diffusion/runs/juggernaut-lora-smoke20-v1 --prepare-only
research/.venv-flux-train/bin/python research/diffusion/train_juggernaut_pilot.py --updates 20 --cache research/diffusion/runs/juggernaut-train-cache-v1 --output research/diffusion/runs/juggernaut-lora-smoke20-v1
```

`--prepare-only` checks real artifacts and prints the exact trainer command without creating the run or loading a model. It requires the completed cache, so it cannot pass before cache generation.

The shared `vendor/sdxl-train/run_training.py` remains unchanged and hash-pinned: direct MPS FP16 frozen UNet, FP32 rank16/alpha16 attention LoRA, LR1e-5, constant schedule, batch1, checkpointing, AdamW and seed2026092199. It does not load CLIP or VAE. Finite cache/loss/gradient/adapter checks and full frozen-base before/after hashes remain in force.

## Native control and continuation

Root owns `compare_juggernaut.py` and its evaluator scheduling. The fixed cohort is `juggernaut-development-cases.json`:16 varied hot-priest prompts and8 identical generic prompts, seeds202609220000–202609220023. The primary arms are base-trigger and adapter-trigger, both with `PR1EST_CAL. ` prefix. Native inference remains TCD6, CFG1.5, eta0.3,1024, tiled VAE512/25%, adapter strength1. The base run is `juggernaut-development-base-v2-cached`; adapter comparisons must use `--cached-text` too. Preserve identical positive/negative conditioning values, initial noise and post-noise RNG states across arms. Text-cache startup is excluded from timing; transfer is included. This is not a server-latency proof.

The adapter20 evaluation is intended at `juggernaut-development-step20-v1`, using `juggernaut-lora-smoke20-v1/pytorch_lora_weights.safetensors`. No evaluation command is executed by the training launcher.

A100-update launch requires a recorded decision file:

```sh
research/.venv-flux-train/bin/python research/diffusion/train_juggernaut_pilot.py --updates 100 --cache research/diffusion/runs/juggernaut-train-cache-v1 --continue-decision research/diffusion/runs/juggernaut-lora-smoke20-v1/continue100-decision.json --output research/diffusion/runs/juggernaut-lora-pilot100-v1
```

Decision fields: `protocol_sha256`, `continue100:true`, and `artifacts:[{kind,path,sha256}]` (repo-relative paths), including kinds `training20_evidence`, `adapter20`, `base_development_result`, `adapter20_development_result`, and `native_visual_review`. Its `summary` must contain `reviewed_cases:24`, `reviewed_primary_arms:2`, `photographic_wins`, `photographic_losses`, `calendar_wins`, `calendar_losses`, `base_joint_pass`, `adapter_joint_pass`, `base_face_pass`, `adapter_face_pass`, `new_severe_face_or_identity_regression:false`, and `strong_teacher_copies:0`. The launcher verifies hashes, changed adapter/unchanged base evidence, at least one photographic win with wins greater than losses, no face/joint-count drop, and no calendar or severe identity regression. Root must derive these judgments from every native output. Compatibility or low loss alone cannot authorize100.

## Resource and archival boundaries

No model has been run by this preparation. Static parsing and provenance/file hash preflight passed. Root owns heavy scheduling and launch. Both cache and trainer require35% initial available memory,20% runtime floor,24GiB RSS,22GiB MPS driver allocation,512MiB maximum swap growth, Torch2 CPU threads/1 inter-op and MPS fraction0.55. Independent process-group supervisors preserve failed runs and stop at20-minute cache/30-minute training deadlines. Model component files retain upstream license; site/application code remains MIT. No production approval or held-out-test execution is included.
