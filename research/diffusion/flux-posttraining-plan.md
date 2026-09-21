# FLUX.2 klein: concrete post-training plan

2026-09-21. Original plan; subsequent execution updates are recorded in diffusion/runs. The finalized dataset is20train/5validation/5test in ../data/flux-priest-domain-v1/manifest.json; it supersedes the provisional split below. The isolated environment was installed and import/pip checks passed. Smokev1 stopped before model loading because its original seed exceeded NumPy's32-bit range; smokev2 uses2026092181. The commands below now use that valid seed.

Original planning status: research and source inspection only. No training, installation, weight download, credentials, or paid compute was started. This plan is conditional on the root's pretrained klein comparison demonstrating a worthwhile quality advantage. It does not assume that klein meets the requested below-500ms server-generation budget.

**Preferred implementation: the maintained Diffusers klein LoRA script, in a separate environment, training base-4B and evaluating the adapter on distilled 4B.** This avoids a custom trainer and reuses our existing Diffusers model/artifact pipeline. BFL's recommended AI-Toolkit route is also viable in principle, including experimental MPS support, but introduces a different original-weight representation, text-encoder/VAE sources and a much larger dependency tree. CUDA is not intrinsically required for an initial local compatibility test.

## Exact supported adapter route

BFL explicitly recommends training against `black-forest-labs/FLUX.2-klein-base-4B` and loading the resulting adapter onto `black-forest-labs/FLUX.2-klein-4B` for four-step inference. This is author-endorsed transfer, not our speculation that any diffusion adapter works on a distilled model. Their reported small style experiment is not evidence that our photographic domain will work at the same step count or cost. Both 4B checkpoints use Apache 2.0. [BFL guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora).

Pinned model revisions already established in this repository:

- Training base: `a3b4f4849157f664bdbc776fd7453c2783562f4d`.
- Distilled inference model: `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Diffusers source: `9f1246971270c84dcbe71233edb7a519596a5d02`.
- Dedicated trainer: `examples/dreambooth/train_dreambooth_lora_flux2_klein.py`, SHA-256 `83bccc8c041496aa33939b6ff1f95d0929c1c71a9a29cdc64ecacd4a503fccfe`.

The [trainer source](https://github.com/huggingface/diffusers/blob/9f1246971270c84dcbe71233edb7a519596a5d02/examples/dreambooth/train_dreambooth_lora_flux2_klein.py) saves PEFT transformer adapters through `Flux2KleinPipeline.save_lora_weights`. The distilled `Flux2KleinPipeline.load_lora_weights` route is exactly the family BFL demonstrates. Validate actual saved-key loading without unexpected keys, and a nonzero change versus adapter-disabled inference; API compatibility is not proof of quality. No re-distillation is needed by this documented route, but preservation of four-step quality is still an experimental question.

## Why not paste the guide's YAML unchanged?

The [linked worked example](https://docs.bfl.ai/flux_2/flux2_klein_training_example) currently shows **base-9B** configurations for a painterly style, not a ready 4B priest recipe. Its guidance about data, captions, and visual checkpoint selection is useful; copying its model name, schedule and aesthetic-specific timestep weighting would be a different experiment. We need a multi-identity portrait domain, not a character LoRA.

AI-Toolkit inspected revision: `a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7`. Its `Flux2Klein4BModel` uses `arch: flux2_klein_4b`, loads original `flux-2-klein-base-4b.safetensors`, `Qwen/Qwen3-4B`, and `ai-toolkit/flux2_vae/ae.safetensors`. It does not directly consume our existing Diffusers directory as-is. Its manager has an explicit experimental macOS/MPS backend, skips CUDA accelerators, and its model config maps unsupported MPS qfloat8 requests to a different quantizer. Thus calling this trainer “CUDA-only” would be incorrect. A simple MPS configuration would use ordinary `adamw`, no quantization initially, checkpointing, cached latents and cached/unloaded text encoding; that exact klein backward path remains untested here. [Trainer](https://github.com/ostris/ai-toolkit/tree/a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7), [Mac environment](https://github.com/ostris/ai-toolkit/blob/a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7/manager/spec.py), [klein loader](https://github.com/ostris/ai-toolkit/blob/a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7/extensions_built_in/diffusion_models/flux2/flux2_klein_model.py).

## Dataset: reuse high-resolution teachers, avoid rejected outputs

Use `research/data/plain-background36-whole-image/1024/`, not the 256px aligned134 set, restored rejected GAN outputs, or hard clothing composites. Its 36 teachers have prior root acceptance as prospective calendar-style training data, preserved source provenance, full collar/framing, and no image restoration. Dataset manifest SHA-256: `5a9e6558f5bdb4a2fe53d86c19f34cbb4221b3f1030255f43d3d8415b51961bd`. This assessment rechecked all 36 native1024 derivative hashes; no exact-byte duplicates exist. That does not establish unique identities.

Keep the already rejected 167,169,172,178 excluded. Additionally exclude 159 and161 from this pilot because prior native review explicitly flags face-family resemblance149/159 and151/161. Retain149 and151; do not split a known related pair between training and evaluation. Proposed fixed split of the remaining34:

- **Train24:** 141,142,143,145,147,148,149,151,152,154,155,156,157,160,162,163,165,166,170,173,174,176,177,180.
- **Validation5:** 144,150,158,168,175.
- **Final held-out teacher5:** 146,153,164,171,179.

Before making this a training manifest, review the remaining native faces together for additional close identity families and eye/clothing defects. Freeze any additional exclusions/group assignments **before** training and keep them hash-bound. This memo has not established that all other pairs are unrelated. Hold-out teachers are never put in the training directory; they support coverage/nearest-neighbor review rather than supplying a fake quantitative photographic-quality score. Do not augment one face into many supposed identities.

Create a new `research/data/flux-priest-domain-v1/train/` with exactly the24 approved1024 images and `metadata.jsonl`, one line per image:

```json
{"file_name":"141.png","text":"PR1EST_CAL. An adult man with short dark wavy hair, clean-shaven face, looking at the camera, wearing a clerical shirt and white collar, with a plain background."}
```

That line is an illustrative schema, **not a verified caption for141**. Write each caption from its actual image: hair, beard, pose, expression, framing, clothing and background. Describe varying appearance instead of labeling every image as one character. Treat `PR1EST_CAL` as a photographic-domain trigger. Do not condition each identity on a unique name. Legitimate broad clerical collars count as priests; a narrow Roman tab is not a user requirement. Use diverse inference prompts to retain age/face/pose coverage. Uniform backgrounds and related faces in this small dataset remain overfitting risks.

For this script, use `--dataset_name /absolute/path/to/train --image_column image --caption_column text`. Its local `--instance_data_dir` branch opens every file as an image and discards custom captions; placing `.txt` or metadata files there would fail despite a misleading nearby source comment. Hugging Face imagefolder metadata through the dataset branch is the concrete per-image-caption path.

## Minimal dependencies and command

These are commands to run **after dataset approval and supervised resource availability**, not work already performed. Keep this environment separate from `research/diffusion/.venv` and the shared research environments: the script requires `diffusers>=0.41.0.dev0`, while our inference lock is0.40.0. Install its matching pinned source instead of deleting the version check.

```sh
cd /Users/twixes/Developer/thispriestdoesnotexist
uv venv --python 3.11 research/.venv-flux-train
uv pip install --python research/.venv-flux-train/bin/python \
  torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 \
  accelerate==1.15.0 peft==0.21.0 datasets==5.0.1 \
  pillow==12.3.0 numpy==2.4.6 safetensors==0.8.0 \
  psutil==7.2.2 tensorboard==2.21.0 \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@9f1246971270c84dcbe71233edb7a519596a5d02'
```

Archive the exact trainer source at `research/diffusion/vendor/flux-train/train_dreambooth_lora_flux2_klein.py`, verify the SHA above, preserve its Apache notice, and save the resolved `pip freeze`. Transitive package resolution/import compatibility has not been executed; these pins are source-compatible candidates, not an installation-tested lock. Check `--help`, `pip check`, and imports before loading weights. No `bitsandbytes`, `torchao`, FlashAttention, xFormers, credentials, or paid API is needed for the proposed ordinary MPS path.

Minimal20-update **smoke**, after materializing the reviewed train folder:

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
research/.venv-flux-train/bin/python \
  research/diffusion/vendor/flux-train/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
  --revision a3b4f4849157f664bdbc776fd7453c2783562f4d \
  --dataset_name /Users/twixes/Developer/thispriestdoesnotexist/research/data/flux-priest-domain-v1/train \
  --image_column image --caption_column text \
  --instance_prompt 'PR1EST_CAL. A portrait of an adult male priest.' \
  --output_dir research/diffusion/runs/flux-base4b-lora-smoke20-v1 \
  --resolution 512 --center_crop --repeats 1 \
  --train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing --cache_latents --offload \
  --mixed_precision fp16 --optimizer AdamW \
  --rank 16 --lora_alpha 16 \
  --learning_rate 5e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps 20 --checkpointing_steps 10 \
  --dataloader_num_workers 0 --seed 2026092181 \
  --skip_final_inference --report_to tensorboard
```

Before this command, archive the required base4B Diffusers weights through our pinned download/LFS-chunk process; do not rely on an unrecorded HF cache. Training base transformer adds about7.75GB of raw weights. Reuse existing VAE/text-encoder files only after verifying exact hashes against base metadata. Do not download duplicate original-format single-file weights for this Diffusers route. More than2GB must follow the repository's LFS part/reassembly convention.

## MPS portability and actual changes needed

The dedicated Diffusers script explicitly supports an MPS route: it disables native AMP, accepts fp16, keeps LoRA parameters float32, provides plain AdamW, and uses checkpointing and cached text/image latents. It also **explicitly rejects bf16 on MPS** in two guards. This may be conservative relative to modern M5 inference support, but it is the script's actual behavior. Use its fp16 path first; do not claim local bf16 training from successful inference or remove guards without a separate backward test. No CUDA FP8, NF4, TF32, FSDP or bitsandbytes flags belong in the local command.

No mathematical port appears necessary for that initial fp16 path. Two concrete memory safeguards are advisable in a separately hashed patch before the first model load:

1. Pass `torch_dtype=weight_dtype` when loading the base transformer and Qwen text encoder, to avoid their initial default-float32 CPU materialization. Preserve the already explicit casting of trainable LoRA matrices to float32. Verify nonfinite output/gradient checks with this change; lower-precision loading is not assumed numerically interchangeable with another path.
2. After cached captions are built and `text_encoding_pipeline` is moved to CPU, delete the pipeline itself as well as the local `text_encoder`/`tokenizer` names. The inspected script keeps a pipeline reference to the encoder; deleting only the names does not release its roughly8GB half-precision weights from unified memory. There are no later uses of that pipeline in the inspected script. Re-run a checkpoint load/save smoke and save the patch.

The script defaults its LoRA target enumeration to24 single-transformer block indices, while the pinned4B transformer config explicitly has5 dual blocks and20 single blocks. Replace the hardcoded `range(24)` with `range(transformer.config.num_single_layers)` in the archived patch, or provide an explicit list of existing targets. This removes nonexistent block20–23 names; it does not add trainable capacity. Enumerate and save actual matched modules before the smoke. The fixed pilot must retain the same chosen modules, rank and optimizer.

Use a supervising process that records RSS, MPS allocated/driver memory **separately** (do not sum overlapping unified-memory counters), system memory pressure, wall time, step time, and exit status. Suggested initial bounds: one heavy job at a time, at least25% free system memory, a30-minute smoke deadline, and stop before sustained memory pressure/swap. Do not enable silent CPU operator fallback to disguise an unsupported MPS backward. A failed fp16 numerical/kernel smoke is a real result; a CUDA run can then use BF16 with the same dataset and adapter design. This task did not establish local training speed or memory.

## Bounded useful pilot and unbiased comparison

After smoke verifies finite gradients, changed adapter weights, frozen base weights, all24 captions/images loaded, and exact save/reload behavior, run one fresh **750-update** rank16 pilot at512 with checkpoint snapshots250/500/750. At batch1 this is31.25 sample presentations per training image on average, not a million-image scratch run. Use the same5e-5 LR, cached data and no random crop/flip. Allocate at most two hours measured from the new run start; if smoke timings predict a much longer run, report that before choosing local time versus a bounded GPU job. Do not silently truncate and call20 updates a successful domain adaptation. No hyperparameter sweep is proposed.

Evaluate the **distilled** model, not only the training base:

- Before training, save the eight existing comparison prompts/seeds with and without the new trigger to separate trigger-token effects from trained effects.
- At each snapshot, compare adapter strength1.0 against disabled adapter using identical saved latent tensors, four steps, guidance1.0, fixed resolution and unchanged prompts. Save every output. Use the eight comparison cases plus16 predeclared development seeds for checkpoint selection.
- Inspect original native outputs for eyes, anatomy, photographic texture, recognizable priest clothing, adult male appearance, no hats, calendar appeal and repeated identity families. Review nearest training faces manually; similar CLIP embeddings are diagnostic rather than proof of memorization or distinct identities.
- After choosing one checkpoint on that development panel, evaluate32 fresh predeclared test seeds that have never appeared in checkpoint previews. Do not retune the adapter strength/selection threshold on them. Store all rejects too. Keep held-out teacher images out of gradient updates throughout.
- If training collapses diverse faces into one teacher family or loses the base's photographic quality, reject it even if collars or loss improve. If the unadapted model was already stronger, retain the negative pilot result rather than deploying worse trained weights merely to tick a training box.

512px is an explicitly lower-resolution pilot, not native1024 detail. Compare512/768/1024 inference as separate configurations only after root selects the resolution that satisfies displayed quality. A good pilot can justify a higher-resolution follow-up; it does not prove one is necessary. Adapter fusion may remove extra low-rank inference operations, but fuse/unfuse equivalence and actual resident GPU response timing need measurement. Include request queueing, rejection attempts and encoding when verifying the user's latency scope. Four denoising steps or BFL's hardware-specific speed claim alone does not prove the site's500ms requirement.

This plan is a trained diffusion/flow model route, not a GAN architecture. It meets the fresh-noise generation behavior if successfully trained and served, while the architecture difference remains explicit. Dataset curation, actual local backward, distilled adapter quality, server latency, hosting cost and CI deployment are still outstanding.
