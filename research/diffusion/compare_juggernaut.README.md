# Bounded offline Juggernaut X Hyper comparison

`compare_juggernaut.py` has now executed the four-step native1024 and512 comparisons. It performs no training or model conversion at import time. Use the existing pinned `research/.venv-flux-train` environment.

Native quality comparison:

```sh
research/.venv-flux-train/bin/python research/diffusion/compare_juggernaut.py --resolution 1024 --tiled-vae --output research/diffusion/runs/juggernaut-hyper-tcd4-native1024-next-tiled
```

The initial protocol is **TCD,4 steps, CFG1, eta0.3**. A separately labeled reference arm permits `--steps 6 --guidance-scale 1.5` at native1024, retaining TCD and eta0.3. Four-step TCD and CFG1 are within the publisher's ranges; **eta0.3 is our explicit choice**, because the author does not prescribe eta. Scheduler construction uses the pinned publisher scheduler config via `TCDScheduler.from_config`; its complete resulting config and actual timestep list are retained. We do not use the publisher's original scheduler class implicitly, add an acceleration LoRA, or alter settings after inspecting images.

The exact eight historical comparison prompts and seeds, plus the original warmup prompt/seed, come from a hash-pinned result file. Each run generates fresh float32 CPU noise with the correct latent shape:512→`[1,4,64,64]`,1024→`[1,4,128,128]`. Same numeric seeds across resolutions do not imply matched identities.512 additionally checks equality against the original noise artifact. The same generator continues from its post-noise state into TCD; before/after states are retained.

Every image is retained at `000..007/native.png` and `warmup/native.png`, with monochrome `display.webp`, latent arrays and hash-bound records. A contact sheet is for overview only; review native1024 files individually. Raw NaN arrays/failure artifacts survive a stopped run. No selectors, image-level retries, repairs, hires fix or new prompts. No Turbo caption/VAE cache is used.

Timing records fresh RNG construction/noise plus transfer, uncached text encoding, denoising, original VAE and PIL conversion. WebP conversion/encoding is measured separately and added. PNG/latent/provenance writes are excluded; WebP's `.save()` includes its disk write. Per-step memory/finite guards are included, so these are conservative local timings, not a clean server benchmark. Warmup is retained and excluded from warm median/p95/max. A UNet pre-forward hook records actual calls and requires the selected4 or6 calls; CFG1 leaves the unconditional branch disabled, whereas CFG1.5 enables it. Cold model loading is reported separately. HTTP, queueing and any future quality retries remain unmeasured.

## Loading and resource bounds

Inspection of the installed Diffusers source found that `from_single_file` reads safetensors using mmap, iterates components sequentially, and uses Accelerate meta initialization when `low_cpu_mem_usage=True`; CLIP conversion also uses meta initialization. The script passes an explicit local config, `local_files_only=True`, FP16 dtype and mmap enabled. It assembles on CPU, lets the loader's temporary checkpoint dictionary go out of scope, collects garbage, then transfers text encoders, UNet and VAE to MPS one at a time. This avoids a full random FP32 initialization and a second complete pipeline. It does **not** prove a particular peak RSS; mappings, converted tensors and device allocations can overlap.

There is no custom conversion framework, saved converted model, offload hook or automatic lower-quality fallback. Before launch require35% available system memory. Worker/supervisor stop on RSS>24GiB, MPS driver allocation>22GiB, available memory<20%, swap growth>512MiB or20minutes. The single-file CPU load is covered by the external supervisor even before component guards run. If it exceeds limits, retain the failure and plan a separate component conversion; do not silently increase limits.

Source files inspected: installed `diffusers/loaders/single_file.py`, `single_file_model.py`, `single_file_utils.py` and `models/model_loading_utils.py`. Runtime stores their hashes, TCD source hash and all relevant package versions. Full checkpoint/config/license hashes are verified before loading. The checkpoint and its LFS parts are already verified by the separate model preparation; this evaluator never downloads or reconstructs missing files.

## Optional512 tradeoff

Only after the native result looks promising, record a review decision:

```json
{
  "native_1024_promising": true,
  "source_result": "/absolute/path/to/native1024-run/result.json",
  "source_result_sha256": "actual SHA256 of that completed result"
}
```

This is a recorded research judgment, not a user-approval requirement. Then run the same command with `--resolution 512 --native-review /absolute/path/to/decision.json` and a **new** output directory. The script verifies the referenced result is a completed native1024 run of this base/revision.512 is a separate non-native quality/latency experiment. Neither local timing nor a promising base means production approval or proven server generation below500ms.

Validation performed during preparation: Python compilation/AST parsing, pinned provenance and historical-result hash checks. Those preparation checks did not include execution; subsequent local runs are recorded below.

## Executed local comparisons

- `runs/juggernaut-hyper-tcd4-native1024-v1`: full-image VAE decode crossed the20% system-memory reserve after all four warmup denoising steps. The supervisor terminated it; no complete image.
- `runs/juggernaut-hyper-tcd4-native1024-v2-tiled`: all eight cases plus warmup completed using `--tiled-vae`. Original checkpoint VAE with automatic precision upcast,512 sample tiles /64 latent tiles /25% overlap. This changes decoded pixels and timing versus full-image decoding; it is not a bit-equivalent memory optimization. No gross tile seams observed in initial review.
- `runs/juggernaut-hyper-tcd4-512-v1`: all eight plus warmup completed, full-image VAE, based on hash-bound exploratory native review.

The six-step/CFG1.5 arm uses the publisher's suggested step count and CFG midpoint but retains TCD rather than its concrete DPM++SDE example. It is a quality reference with two CFG branches per call; six calls are not equivalent work to six CFG1 calls. All variants retain original prompts, seeds, outputs and distinct run directories. See each completed result.json for measured timings; none proves server performance.

## Fixed development cohort and staged conditioning

`--case-manifest juggernaut-development-cases.json --prompt-prefix 'PR1EST_CAL. ' --steps 6 --guidance-scale 1.5 --resolution 1024 --tiled-vae --cached-text` opens the fixed24-case development cohort (16 varied appearance prompts, eight fresh seeds of one generic prompt). CLI sampling/prefix must match the frozen manifest. No sealed test cases are opened.

Cached mode now uses the exact tensor-verified `models/juggernaut-x-hyper-components` export: load only both CLIPs, save positive/negative/pooled conditioning on CPU, verify saved tensor values, release text encoders and clear the unused MPS pool, then load UNet and VAE directly on MPS. This avoids the earlier single-file CPU-to-MPS peak. The final denoising callback clears unused MPS allocations before decoding. Both actions and the exported component provenance are recorded; these conservative local timings include cache-clear overhead and exclude initial text preparation.

`--lora /absolute/path/pytorch_lora_weights.safetensors` adds a hash-pinned UNet-only adapter at strength1 after conditioning preparation. Text encoders remain unchanged. Paired review must compare case/prompt/noise/generator-state and all conditioning tensor-value hashes, not merely the same seed number. The preview builder enforces these checks. All native outputs and failed resource-bound attempts remain archived.
