# Portable reference-phase trainer prototype

`trainer.py` is a separate experimental outer loop around the **unmodified** NVIDIA loss, augmentation, network, image dataset and infinite-sampler iterator. It does not change `research/train.py` or either running training job. The goal is a clearly recorded reference-style baseline, not another silently modified portable recipe.

## Reuse and deliberate differences

| Component | Implementation |
| --- | --- |
| Logistic, path-length and R1 objectives | Imported `training.loss.StyleGAN2Loss`, unchanged |
| Pixel, filtered geometry and color augmentation | Imported `training.augment.AugmentPipe`, upstream `bgc` settings |
| Source architecture/weights | Existing trusted base raw G, D and G_ema, loaded through upstream `legacy` |
| Data normalization and image loading | Upstream `ImageFolderDataset`; CPU uint8, each microbatch moved to the chosen device |
| Data order | Upstream `InfiniteSampler.__iter__`, unchanged; constructor shim removes the obsolete `Sampler(data_source)` call rejected by PyTorch 2.14 |
| Schedule and optimization | Adapted single-device outer loop from reference `training_loop.py`, preserving four phases, gains, separate optimizer steps and lazy compensation |
| Logging | Upstream collectors with detached reports moved to CPU to support their float64 counters on MPS |
| Precision | Explicit FP32 call wrappers; no mixed precision/custom CUDA kernels |
| Training state | Full atomic durable checkpoint with optimizer, loss/controller/sampler and RNG state, rather than network-only transfer pickle |

This prototype follows the reference's **sum of microbatch mean gradients**: `gain=phase.interval` on every accumulation round, with no division by the number of rounds. At defaults, batch 32 / microbatch 4 means eight rounds. It also preserves PL's moving-mean update per regularization microbatch. Adding a conventional round normalization would change this reference loop's optimizer trajectory.

`Gmain`/`Greg` share one Adam and make separate steps; `Dmain`/`Dreg` likewise. G interval is 4, D interval 16, both regularizers run at batch index zero. Starting from base LR 0.0025 and betas (0, 0.99), G uses ratio 4/5, D 16/17 for both LR and beta exponent. Epsilon is 1e-8. The reference gradient NaN/Inf replacement remains in the normal loop; `--verify-updates` checks and rejects nonfinite gradients **before** this replacement, then checks parameter finiteness and a real update.

Defaults use paper256 loss/optimizer values: gamma 1, PL weight 2/decay .01/shrink 2, style mixing .9, one G rate, EMA 20 kimg and no transfer ramp-up. Batch 32 / microbatch 4 is the explicitly requested local configuration, differing from paper256's global batch 64. Source architecture is retained rather than reconstructing a preset. FreezeD defaults off; `--freeze-d-layers` is a recorded optional ablation using immutable input-block buffers. For a different source resolution, choose its recipe parameters explicitly; the data resolution must match loaded G.

ADA starts at p=0 by default, targets .6, updates every four global batches and uses transfer speed 100 kimg. It uses the actual upstream real-sign collector and update formula. Probability is **lower-clamped only**, with no local .85 or 1.0 cap. Snapshot rendering preserves all RNGs and model train/eval flags, uses fixed constant-noise latent grids, and saves raw psi=1, EMA psi=1 and EMA psi=.7 separately.

## Checkpoints and resume

`resume.pt` includes G/D/G_ema, both complete Adam states, PL moving mean, augmentation module state/p, phase/global batch index, real-image counter, fixed snapshot latents, CPU/MPS/Python/NumPy RNG, source/config hashes and pending ADA/metric statistics. The pending-window state includes the pinned upstream global counters and each collector's cumulative/moment state; this avoids resetting ADA on a checkpoint between four-step boundaries.

The sampler is restored exactly by reconstructing its local NumPy-seeded upstream iterator and replaying the consumed image count. It does not consume global Torch/NumPy RNG during replay. This avoids forking/reimplementing the iterator, at the cost of O(images_seen) resume work. Its seed/count and dataset checksum are checkpointed; a future efficient sampler-state adapter would need its own exact replay test.

A resume rejects changes to the recorded mathematical recipe, source hashes, data hash, batch/microbatch, device or seed. Device changes are not represented as exact resumes. Output directory, requested total steps and checkpoint/preview frequency may change. It does not silently replace restored optimizer hyperparameters from CLI defaults.

Writes serialize to a temporary file in the same directory, flush/fsync it, atomically replace the prior checkpoint, then fsync the containing directory. The previous checkpoint remains intact until replacement. Trusted local research checkpoints are loaded with `weights_only=False`; this is not an untrusted-model ingestion interface.

## Commands

Default prototype configuration, **not a launched long run**:

```sh
research/.venv/bin/python -u research/experiments/reference_phases/trainer.py \
  --run research/runs/reference256-candidate --device mps \
  --batch 32 --microbatch 4 --pl-batch-shrink 2 \
  --data research/alignment/collar-only/eyes42 --base research/models/ffhq256.pkl
```

All image pixels remain CPU/disk-side until their microbatch is needed. No full 110-image FP32 dataset is copied to MPS. Before any full run, benchmark the intended batch/PL settings with a fresh memory check and inspect augmentation/output previews. A 256px batch-1 smoke does not certify memory for 1024px training. This prototype is not authorized to select an unreviewed 1024 alignment candidate automatically.

The bounded CPU verification is invoked with:

```sh
research/.venv/bin/python -u research/experiments/reference_phases/check_trainer_resume.py
```

It uses actual 256px networks, batch 2 / microbatch 1 (two accumulation rounds), PL shrink 1, FreezeD 4 and p=.25. It compares five uninterrupted updates against one update plus a four-update resumed continuation, including a resumed PL phase and an ADA update across a checkpoint taken before the ADA boundary. It records all artifacts under `prototype-tests/` and refuses to overwrite completed evidence.

## Provenance and license

The scheduling/controller/EMA adaptation is based on the vendored NVIDIA `training/training_loop.py`; its exact hash and those of the loss, augmentation, dataset, sampler and statistics files are in every run config. Direct upstream files remain unchanged. The small execution adapters and state orchestration are project code; imported NVIDIA source/weights and adaptations retain the upstream research/evaluation terms. See `research/THIRD_PARTY.md`, `research/sources.json` and the retained vendor `LICENSE.txt`.

Primary references: [training loop](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py), [loss](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py), [augmentation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/augment.py), [presets](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py), [sampler](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/misc.py).

## Verification result

The actual CPU comparison **passed bitwise** across **823 tensors / 171,199,888 tensor elements**, covering G/D/G_ema, both optimizers, PL mean, full augmentation state, fixed latents, CPU/Python/NumPy random states and pending statistics. Five uninterrupted updates matched one update plus four resumed updates. Both paths processed exactly ten real-image presentations using two microbatch accumulation rounds per phase.

The resumed path executed the second PL phase at batch index 4. ADA also crossed its four-step boundary, changing p from .25 to **.2499199957**. PL mean ended at **.0173758399**. All gradient/update assertions and frozen-D invariance checks passed, and atomic saves left no temporary checkpoint behind. Total subprocess wall times were 33.47 s (continuous), 14.37 s (first update) and 19.46 s (resumed continuation), including model/data loading, validation and checkpoint writes. Exact evidence: `prototype-tests/resume-comparison.json`, per-run configs, logs and actual checkpoint files.

A separate actual-network preview check also passed: all G/EMA state hashes, training flags and CPU/Python/NumPy RNG states stayed unchanged while writing three 1024×256 grids of four 256px images. Evidence is in `prototype-tests/snapshots/snapshot-check.json` and `prototype-snapshots.log`; implementation is `check_snapshots.py`.

Two initial setup failures are retained for provenance: `prototype-initial-sampler-error.log` explains the constructor compatibility shim, and `prototype-initial-adam-error.log` records PyTorch's requirement that both beta arguments be floats (corrected to `0.0`). Neither reached an optimizer update. The completed tests ran the final trainer hash recorded in their JSON.

The separate upstream four-phase MPS smoke passed as described in `README.md`. The subsequent complete-trainer intended-size MPS update is documented below. Exact full MPS resume equivalence remains untested. No new long run or production model was launched. These correctness results do not establish improved output quality.


## Intended-size MPS update

`check_intended_mps.py` executed **exactly one full actual update** in `research/runs/reference256-mps-intended-smoke`, with batch 32 / microbatch 4 / PL shrink 2 and `--verify-updates`. Fresh preflight reported **22% system memory free** against the required 20% gate. Both earlier training jobs were concurrent. The wrapper only observed phase/optimizer/snapshot memory and synchronized timings; it did not change loss or optimizer math.

All four phases ran eight microbatch rounds and made four finite optimizer updates. All six initial/final raw/EMA grids had valid 1024×1024 dimensions. The actual **685,989,929-byte** full checkpoint reloaded successfully, included positive PL state and all RNG/controller/optimizer state, and had finite G/D/G_ema tensors.

| Phase | Time |
| --- | --- |
| Gmain | 11.25 s |
| Greg | 36.44 s |
| Dmain | 9.87 s |
| Dreg | 102.04 s |
| Whole heavy regularized update | **159.65 s** |
| Setup, previews, update, save and checkpoint validation | **169.61 s** |

This is the **initial heavy regularized update under concurrent load**, not a steady training-rate estimate. R1 and PL do not run at every ordinary iteration. Finite/update checks and instrumentation also add synchronization. No extrapolated completion promise should be based on this one result.

Maximum **observed** allocator snapshots were **988,329,728 bytes current / 4,818,665,472 bytes driver**. Process peak RSS was **1,101,807,616 bytes before reloading the checkpoint**, rising to **1,187,332,096 bytes including reload**. Samples were taken after each backward, optimizer step and snapshot; they are not true instantaneous device peaks. Host free memory transiently reached 17% while the job ran, then recovered after completion. No OOM occurred.

Evidence: `intended-mps-preflight.txt`, `intended-mps.log`, `intended-mps-result.json` and the actual `research/runs/reference256-mps-intended-smoke/` artifacts. This test exercised trainer hash `c639223e8ceb05c2cc57651c63d16f0ee82009d8b6fc0acd92f0c18db4e9ccdf`.

## Safe exporter integration

The existing offline exporter expects a weights-only-loadable generator snapshot with `G_ema` and positive `step`; it cannot consume the full training resume with NumPy RNG objects directly. A subsequent **serialization-only** addition writes `generator-STEP.pt` beside each full checkpoint via the same durable `atomic_save` helper. It contains CPU EMA tensors, positive step, images_seen and JSON-safe config/recipe. The JSON round trip removes non-primitive subclasses such as TorchVersion. Full resume format and training math stay unchanged.

`check_export.py` extracted the EMA from the actual one-update MPS checkpoint with the same new writer, then loaded it using `weights_only=True`. All **144 EMA tensors / 24,943,026 elements** matched exactly. The unmodified `research/export_model.py` exported safetensors plus metadata into `prototype-tests/export/bundle/`. The normal inference `Model` constructor rejected that bundle because its review is unapproved. No server was started, no review approval was fabricated, and nothing was deployed.

Evidence: `export-result.json`, `export-check.log`, `prototype-tests/export/`. The serialization-enabled trainer hash is `00b5266875dbe07169025eb7b38b22a412230019c52d1bae77eeb8daf0045b13`; the underlying tested-training hash remains recorded separately. No repeated heavy MPS training was needed for this serialization change. Exporter hash is `311421b3d539a9e90f05c30a5aa674f5c4123e6178aae00c6856c11027a9f3e1`.
