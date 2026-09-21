# Staged-cache smoke trainer

`run_training_cached.py` accepts `--priest_cached_data /absolute/cache/directory`.
`prepare_cached_patch.py` deterministically derives it from `run_training.py`, copies
the independent resource helper, and records hashes in `cached-patch-provenance.json`.
`cached.patch` preserves the complete trainer delta. Python 3.11 is required.

The trainer never loads Qwen, a tokenizer, or a text pipeline. It briefly loads the
original VAE in FP32 solely to read the same BN mean and standard deviation as upstream,
then releases it before loading the transformer. It performs no VAE forward pass.
Cached raw latents still receive upstream patchification and BN normalization in the
training loop. The flow-matching loss, optimizer, frozen base and FP32 LoRA parameters
are unchanged.

Following v4's stop during CPU-to-MPS transformer placement, the transformer loader
now passes `device_map={"": "mps"}` and `low_cpu_mem_usage=True`, retaining FP16
`weight_dtype`. The installed Diffusers loader initializes meta parameters and converts
and places individual tensors directly, avoiding a complete converted CPU model before
MPS placement. Its subsequent same-device/same-dtype `.to(...)` remains in the upstream
code. Accelerate's device-map check permits this single-entry map. Diffusers still performs
a temporary MPS allocator warmup of half the model's element count; this change does not
claim zero transient memory or establish successful loading. The failed v4 trainer and
run are preserved. All resource guards are unchanged for v5.

V5 completed direct model placement and cache loading, then stopped before the first
update because TensorBoard rejected the new argument's `Path` value. The argument now
parses as a primitive string for tracker configuration; `validate_contract` converts it
to `Path` before all existing directory and hash checks. V5's source and failure remain
preserved. This logging compatibility correction changes no training math or guard.

The cache manifest must be complete and match the reviewed dataset manifest SHA,
resolution, sequence length, text hidden-state layers, image file set and exact captions.
The loader verifies reviewed image hashes, metadata hash, safe cache paths and tensor
file hashes. It maps each actual Hugging Face dataset filename to its cache entry;
directory or manifest order is never assumed. Repeats, caption dropout and random flip
are prohibited, and center cropping is required. Precomputation uses the unchanged
upstream `DreamBoothDataset` transform: bilinear resize, center crop and [-1, 1]
normalization.

Each image has one safetensor containing `prompt_embeds` (FP16, 1×512×7680), `text_ids`
(1×512×4, original dtype), and `latents` (FP16, 1×32×64×64). Latents are raw VAE encoder
mode, before patchification and BN normalization. Shapes, embedding/latent dtypes and
finiteness are checked before use. Actual dataset order, cache hashes and tensor metadata
are retained in `external-cache-evidence.json`.

Pass `--text_encoder_out_layers 9 18 27` explicitly for this 4B cache. The pinned
training script defaults to `[10, 20, 30]`, while the klein 4B inference pipeline and
cache use `[9, 18, 27]`. Cached smoke v3 correctly stopped on this mismatch before
training. The failed run is retained; v4 supplies the explicit layers. The loader's
equality check remains strict rather than silently adapting cached tensors.

Cache v3 computes frozen Qwen captions in native BF16 and casts finite results to FP16;
the cast is also checked for finiteness. VAE encoding remains FP16. This is an explicitly
recorded numeric recipe change, not a claim of equivalence to all-FP16 Qwen encoding.

The cached helper uses a 20% minimum system-available-memory fraction, 24 GiB worker RSS,
22 GiB MPS driver allocation, and a 512 MiB swap-growth stop. This measured policy revision
followed a BF16 encoder probe using 8.74 GiB MPS driver allocation and 0.49 GiB RSS while
10.87 GiB system memory remained available. Prior 25% guard failures are retained, and
the original helper still uses 25%. The independent supervisor remains necessary for
peaks between worker checks. Unified-memory counters must not be summed.

The twenty-update experiment remains a compatibility and resource smoke test. Completion
does not establish output quality, generalization, server latency, or production approval.
