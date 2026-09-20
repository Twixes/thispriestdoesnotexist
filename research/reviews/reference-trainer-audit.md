# Reference-phase trainer audit

2026-09-21. Read-only comparison of `research/experiments/reference_phases/trainer.py` with the vendored NVIDIA loop, loss, sampler, statistics and relevant operation adapters. No training, tests, model loading, process control or code edits were performed. Existing evidence was read, not rerun.

**Conclusion:** no material mathematical error found that justifies invalidating or stopping the current `reference256-paper-b64` run. One nonblocking provenance gap matters before future resumes with changed code. This audit does not establish convergence, photorealistic output, or exact MPS continuation.

Reviewed trainer SHA-256: `00b5266875dbe07169025eb7b38b22a412230019c52d1bae77eeb8daf0045b13`. It exactly matches the active run's recorded hash. The active recipe matches its one-update starting checkpoint's recipe: MPS, batch 64, microbatch 8, PL shrink 2, mirror on, FreezeD off, gamma 1, LR .0025, EMA 20 kimg, ADA speed 100 kimg. There are 110 source portraits and 220 virtual mirrored entries, not 220 independent images.

## Finding: resume provenance is incomplete (P2, future-resume scope)

`trainer.py:197–199` hashes the loss, augmentation, loop, dataset, sampler and statistics files. `trainer.py:232–254` compares the resulting recipe on resume. However, `trainer_sha256` and `torch_version` are recorded only in `config` (`trainer.py:240–247`), which is **not** compared. The imported `FP32Call` implementation (`smoke.py:35–41`), `training/networks.py`, and numerical operation modules are also absent from that source-hash gate.

Consequently, editing accumulation/EMA logic in the outer trainer, the adapter, a network operation, or changing PyTorch can still pass recipe equality when resuming. Optimizer/RNG restoration alone would then not make the continuation mathematically identical. This is not happening in the current run: its trainer hash and launch recipe agree with the reviewed artifacts. Before a future changed-code resume, compare this provenance explicitly or introduce a reviewed compatibility decision; do not silently assert exact continuation. No change to the running trainer is warranted solely by this finding.

## Training correctness checks

| Area | Assessment and exact references |
| --- | --- |
| Accumulation and phase schedule | `trainer.py:225–229,281–303` matches upstream `training/training_loop.py:197–208,269–292`: one zero-grad and optimizer step per phase, interval gain on **each** microbatch, no division by accumulation rounds, and regularizers at zero-based batch indices divisible by 4/16. Adding a conventional mean-over-rounds factor would change this upstream single-device recipe. |
| Lazy Adam compensation | Both G and D share their main/reg optimizer. Ratios 4/5 and 16/17 affect LR and beta exponents (`trainer.py:225–229`), matching upstream `training_loop.py:201–208`. |
| Losses and PL moving mean | The unmodified loss is invoked directly (`trainer.py:220–222,293–294`). PL shrink, its per-microbatch moving-mean update, and the interval multiplier match `training/loss.py:76–92`. R1 and logistic terms remain the upstream implementation (`loss.py:94–131`). FP32 wrappers only add `force_fp32=True` (`smoke.py:35–41`). |
| EMA | `trainer.py:309–314` uses the same half-life formula, parameter interpolation direction, and direct buffer copy as `training_loop.py:296–305`, with the explicitly selected no-ramp-up policy. It updates once per complete global batch. |
| ADA | `trainer.py:315–319` matches `training_loop.py:307–315`: increment image/batch counters, update every four batches, use the real-sign collector, scale by global batch × interval / ADA kimg, and lower-clamp only. Upstream real-sign reporting includes Dmain and Dreg (`loss.py:108–114`); that behavior is preserved. |
| Multiple statistics collectors | Updating metrics each batch does **not** consume the ADA window. Global totals are cumulative and every collector maintains its own watermark (`training_stats.py:147–168,234–266`). Saving both global state and collector watermarks (`trainer.py:132–147,334–335`) preserves an unfinished four-batch ADA interval. |
| Resume and sampler | Model/optimizer/PL/augmentation state, counters, fixed latents, statistics and CPU/MPS/Python/NumPy RNG are restored (`trainer.py:250–264`). Sampler replay uses NVIDIA's private seeded `RandomState`, not global NumPy state (`misc.py:125–140`), so skipping consumed indices does not perturb the restored training RNG. |
| Preview and checkpoint side effects | Preview temporarily enters eval mode and restores flags/RNG (`trainer.py:112–129`), preventing mapping-average/RNG drift. Durable checkpoint serialization is synchronous and atomic (`trainer.py:46–70,329–337`); the added generator-only export does not make an optimizer update. |

The active batch/microbatch combination also preserves the loaded discriminator's minibatch-standard-deviation group size 8. The paper256 preset specifies batch 64, mbstd 8, LR .0025, gamma 1 and EMA 20 (`vendor/stylegan2-ada-pytorch/train.py:154–193`). This is a **single-device reference-loop adaptation**, not bitwise reproduction of the paper's eight-GPU run: accumulation scaling, per-microbatch PL moving-mean updates, RNG consumption and FP32/MPS execution differ from a distributed run even though they follow NVIDIA's single-device loop.

## Existing evidence and remaining limits

- `experiments/reference_phases/prototype-tests/resume-comparison.json` records bitwise CPU uninterrupted-versus-resumed equality across 823 tensors, including optimizer/RNG/PL/controller/statistics state, with an ADA interval crossing and resumed PL phase. It tested an earlier trainer hash; the subsequent documented change adds serialization-only generator snapshots. This is useful evidence, not an MPS equivalence test.
- `experiments/reference_phases/b64-m8-result.json` records the current trainer hash passing a full actual batch-64/microbatch-8 MPS update, four phase optimizer steps, finite/update assertions, checkpoint reload and snapshot/export checks. It does not demonstrate multi-day stability or exact uninterrupted-versus-resumed MPS equality.
- The active run disables `--verify-updates`. Its NaN/Inf gradient replacement matches upstream (`trainer.py:296–303`; `training_loop.py:287–292`), but that does not certify ongoing finiteness. Actual previews, checkpoint tensor health and loss/controller behavior still determine whether continuing is useful.
- PL/R1 statistics persist on steps when their phase did not run, because the collector defaults to `keep_previous=True` (`training_stats.py:129–139,160–168`). Read the `phases` field before interpreting a repeated regularization metric as a fresh measurement. This affects reporting, not optimization.

No loss normalization, ADA cap, EMA change, or other speculative adjustment is recommended on the basis of this audit. Quality conclusions require the run's actual subsequent outputs.
