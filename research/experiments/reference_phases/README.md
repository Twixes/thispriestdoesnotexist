# Reference StyleGAN2 phases: bounded portability experiment

**The actual NVIDIA StyleGAN2 loss and full `bgc` augmentation passed all four training phases on CPU and MPS**, using the existing 256px FFHQ raw G/D. Path-length and R1 second derivatives were finite, and each phase made a real parameter update. This supports reusing the reference objectives instead of writing more custom loss code. It does not establish useful priest images or a converged model.

`research/train.py`, vendored source, active runs and production were not changed. No long run was launched.

## What ran

`smoke.py` imports the vendored `training.loss.StyleGAN2Loss` and `training.augment.AugmentPipe` directly. It executes the initial reference-style schedule, when both regularizers are due:

| Phase | Gradient gain | Optimizer update |
| --- | --- | --- |
| Gmain | 1 | G Adam |
| Greg (path length) | 4 | Same G Adam, separate step |
| Dmain | 1 | D Adam |
| Dreg (R1) | 16 | Same D Adam, separate step |

For G, the lazy ratio is 4/5: base LR 0.0025 becomes **0.002**, and beta2 becomes **0.99^(4/5) = 0.99199197**. For D, the ratio is 16/17: LR becomes **0.00235294**, beta2 **0.99058546**. Beta1 remains zero; epsilon is 1e-8. Parameters are enabled only for their phase. Every phase asserts finite, nonzero gradients, finite resulting parameters, and a changed parameter. It does not sanitize NaNs into a passing result.

The smoke uses paper256 loss settings: style mixing 0.9, PL weight 2/decay 0.01, R1 gamma 1. For bounded memory it uses **batch 1 and PL shrink 1**, rather than the usual larger batch and PL shrink 2. The `bgc` pipeline is the actual upstream combination of pixel blitting, filtered geometry and color transforms; each transform uses its upstream random gate. Probability is fixed at 0.5 for this portability test. An initial color-only run also passed, followed by the full `bgc` CPU run. Neither smoke is an ADA-controller training experiment.

Two small execution adapters are explicit:

1. An `nn.Module` call wrapper adds `force_fp32=True` when the unchanged loss invokes synthesis or D. No weights or network formulas are rewritten.
2. On MPS, `training_stats.report` receives a detached CPU copy for logging, because upstream float64 statistics counters are unsupported on MPS. The wrapper returns the original loss value; optimization tensors and graphs stay on MPS. CPU uses unchanged logging.

Upstream convolution/resampling operations choose their available PyTorch reference paths. `grid_sample_gradfix` warns that its old custom workaround only supports older PyTorch versions and falls back to `torch.nn.functional.grid_sample`. **On the installed PyTorch 2.14.0 this fallback passed double backward on both CPU and MPS**, first in a 32px probe and then in the full 256px R1 phase through `bgc`. Do not infer current incompatibility from the warning alone or replace the sampler without reproducing a failure.

## Measurements

CPU uses two threads. MPS ran once after `memory_pressure` reported **29% free**, exceeding the required 15% gate; the preflight output is retained. Both existing training jobs continued. Timings include backward, optimizer update, finite checks and gradient norm collection, and are single-run measurements with cold-start and contention effects.

| Phase | CPU full bgc | MPS full bgc |
| --- | --- | --- |
| Gmain | 0.724 s | 5.650 s |
| Greg | 1.651 s | 2.846 s |
| Dmain | 1.211 s | 1.078 s |
| Dreg | 2.075 s | 3.988 s |
| Whole four-phase loop | 5.663 s | 13.572 s |

CPU cumulative peak RSS was **2,656,206,848 bytes (2.47 GiB)**, including model loading and smoke allocations. MPS post-Dreg allocator snapshots were **787,913,472 bytes current / 2,549,809,152 bytes driver**; process RSS peak was 1,379,975,168 bytes. MPS snapshots are not peaks and process RSS does not include all device allocation. These figures do not predict a full batch-8 run's peak or imply CPU is intrinsically faster; inputs and augmentation draws differ between backends, and this was not a controlled throughput or numerical-equivalence benchmark.

Both regularizers had nonzero gradient norms and updated parameters. PL moving mean rose from zero to 0.00407273 on CPU and 0.00374271 on MPS. Full evidence, hashes, losses, norms and memory snapshots are in:

- `smoke-cpu-color.json` and matching log.
- `smoke-cpu-bgc.json` and matching log.
- `smoke-mps-bgc.json` and matching log.
- `mps-preflight-memory.txt`.

```sh
research/.venv/bin/python -u research/experiments/reference_phases/smoke.py --device cpu --augmentation bgc
# Run only after checking host memory and coordinating active jobs:
research/.venv/bin/python -u research/experiments/reference_phases/smoke.py --device mps --augmentation bgc
```

## Recommended next implementation

Build a **separate reference-phase trainer** around these existing upstream loss/augmentation classes. Preserve the current portable and CDC runs as named experiments; do not mutate their recipes while they run. This smoke removes the immediate PL/R1/backend feasibility concern, but the following loop work is still required before claiming a reference-style baseline:

- Replace the reference loop's hard-coded CUDA device/events, distributed setup and pinned-memory assumptions with a single-device MPS/CPU outer loop. Keep its phase ordering, shared optimizers, lazy LR/beta compensation, interval gains and schedule (regularizers run at batch index zero). The loss bodies need no rewriting.
- Use the upstream data sampler, augmentation configuration and ADA real-sign collector/control formula. Restore its probability state and accumulator/counters on resume. A fixed p=0.5 smoke is not evidence for the whole ADA controller. Decide a documented preset rather than retaining current arbitrary augmentation caps/cutout gates.
- Restore `loss.pl_mean` as well as G/D/EMA, both optimizers, phase index/kimg, sampler state and CPU/MPS/Python/NumPy RNG. Test exact save/resume plus finite next PL/R1 phases; the current CDC resume test does not cover this new state.
- For a paper256-oriented baseline, use one G optimizer rate (mapping already has its network LR multiplier), correct lazy compensation, gamma 1 and the preset EMA half-life of 20 kimg with no transfer ramp-up. Batch/microbatch and discriminator minibatch-std settings must be recorded; batch-1 smoke has different statistics from paper256. If freezing D or changing gamma/batch to match current controls, label those deliberate deviations.
- Use batch/microbatch at least 2 with PL shrink 2, or explicitly keep shrink 1 if microbatch 1 is necessary. Test memory at the intended batch and schedule before a long run. The four-phase initial iteration is a useful conservative memory gate, but gradient accumulation introduces its own semantics that must match the upstream loop.
- Reuse the existing matched raw/EMA/untruncated evaluations and add fixed augmented-real previews. Compare at equal kimg and inspect facial detail and identity variation; no passing smoke justifies deploying an unreviewed model.

A faithful loop port is now more defensible than assuming the upstream algorithms cannot run locally. It remains a controlled quality experiment, not evidence that any one earlier recipe difference caused the visible artifacts. NVIDIA's reference training-duration guidance still applies as context: short runs test feasibility and trends, not promised convergence.

## Sources and licensing

The new harness is independent project orchestration code under the repository MIT license. It **imports**, rather than copies/relicenses, the vendored NVIDIA implementation and FFHQ weights; their research/evaluation license remains in force. No vendor edits were made. Source hashes are captured in every JSON, and the pinned vendor commit is recorded in `research/sources.json`.

- [Official loss and PL/R1 implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py).
- [Official phase schedule, lazy optimizer scaling, ADA and EMA](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py).
- [Official augmentation implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/augment.py).
- [Official paper256 and augmentation presets](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py).
- [NVIDIA license](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt); local `research/THIRD_PARTY.md`.
