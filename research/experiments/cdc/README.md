# CDC-only experimental regularizer

This independent helper implements a bounded cross-domain distance consistency experiment inspired by Ojha et al., *Few-shot Image Generation via Cross-domain Correspondence* (CVPR 2021). The paper transfers relative similarities between source instances into the adapted domain and also uses anchor-dependent discrimination. This directory implements only the distance-consistency idea, not that full recipe. [Primary paper](https://arxiv.org/abs/2104.06820), [official implementation](https://github.com/WisconsinAIVision/few-shot-gan-adaptation).

The helper diagnostics do not update any weights or change the production model. A later opt-in trainer integration and its single-update MPS smoke are documented below; the active training process was not modified.

## Computation

`cdc_loss.py` uses frozen source G and differentiable target G with the same unmixed batch of at least four latent vectors. Mapping uses no truncation and explicitly skips `w_avg` updates. Synthesis uses constant noise and the same explicit unfused convolution mode for source and target. Temporary hooks collect corresponding block activations and are removed even on error; model training flags are untouched.

For each selected block, pool features to 8×8, flatten and L2-normalize each example. For each example, compute cosine similarities to the other examples, excluding the diagonal. Softmax at temperature 1 gives source and target neighbor distributions. The returned scalar is KL(source || target), summed over neighbors, averaged over examples, then averaged over blocks. Source features have no autograd graph. Target feature gradients reach its upstream synthesis/mapping parameters.

Defaults select blocks 32, 64, and 128. This fixed multi-block choice and adaptive average pooling differ from the paper implementation's randomly selected unpooled features. Our explicit reduction also differs from its default elementwise `KLDivLoss` mean. Do not copy the paper code's weight 1000 without calibration: with batch 4, summing over the three neighbors already changes scale by a factor of three for otherwise identical features, and pooling/block averaging changes it further. The regularizer returns an unweighted loss.

Batch 2 is rejected because each row would have only one neighbor and its softmax would always equal one. The helper requires a frozen source rather than silently changing the caller's optimizer or flags. It does not introduce anchors, patch discrimination, path regularization, or any other part of the full algorithm.

## Reproduce the smoke benchmark

From the repository root:

```sh
research/.venv/bin/python -u research/experiments/cdc/smoke_cpu.py
```

The test loads the existing FFHQ 256px G, freezes source, clones target, and uses four fixed latents with two CPU threads. It performs these checks against the actual model, not a toy replacement:

- Identical source/target: loss exactly **0** for all three blocks.
- Perturb target `synthesis.b32.conv0.affine.weight` by seeded noise at 0.15 times its standard deviation: loss **2.836e-7**, with finite nonzero gradients in 72 target tensors; total gradient L2 norm **3.645e-5**.
- Source parameters and buffers unchanged by full state hash; no source gradients.
- Target parameters unchanged by forward/backward (no optimizer), target buffers and training modes unchanged, RNG unchanged, and all temporary hooks removed.
- Batch 2 rejected.

Recorded on local macOS arm64, PyTorch 2.14.0, CPU, two threads:

| Measurement | Result |
| --- | --- |
| Identical-model forward (first call) | 1.653 s |
| Perturbed forward (source + target) | 1.169 s |
| Target backward | 0.635 s |
| Perturbed forward + backward | 1.804 s |
| Process cumulative peak RSS before calls | 0.760 GiB |
| Process cumulative peak RSS after backward | 3.333 GiB |
| Increase in recorded process peak | 2.573 GiB |

`smoke-cpu.json` contains exact hashes, per-block losses, feature dimensions, timings, and checks. `smoke-cpu.log` preserves command output. RSS is a process lifetime high-water mark including model loading, state hashing, both diagnostic phases, and allocation reuse; it is **not** isolated activation memory or an estimate of MPS memory. The timings are one bounded benchmark, not a throughput guarantee or a full GAN training iteration. CPU cost cannot simply be added to the active MPS trainer's timing.

## How to use the result

The correctness evidence establishes a differentiable penalty for relationship changes; it does not establish better priest images or repaired identity diversity. The small observed loss/gradient makes logging and coefficient calibration necessary. Pooling can emphasize broad appearance and weaken identity-specific details. This is a fallback experiment if the aligned-110 FreezeD run still contracts.

For a future controlled run, retain the existing source checkpoint, pin these helper settings, log weighted and unweighted CDC plus gradient contribution, and compare matched raw-G/EMA samples and face-focused diversity. Use the same unmixed latent batch for both sides; do not pass the adversarial step's independently mixed styles. The source may contain non-male/non-adult appearance or incompatible framing, so overly strong correspondence can impede target adaptation. Like other smooth regularizers, this should preserve diversity during adaptation; do not assume it can recover an already completely collapsed generator.

## Provenance and licensing

This helper and benchmark were written independently in this repository from the mathematical method. No Adobe implementation was copied or imported. The new helper follows the repository's MIT license. `sources.json` records the primary references and inspected official-code commit.

The compared official code uses the Adobe Research License (noncommercial research, with its redistribution and attribution terms); it is not MIT. [Adobe license](https://github.com/WisconsinAIVision/few-shot-gan-adaptation/blob/main/LICENSE.txt).

The actual model execution imports our existing vendored NVIDIA StyleGAN2-ADA network and uses its existing weights. Their upstream research/evaluation-only terms remain applicable and are not replaced by the helper's license. [NVIDIA license](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt), local `research/vendor/stylegan2-ada-pytorch/LICENSE.txt` and `research/THIRD_PARTY.md`.


## Calibration against pilot step 1,000

`calibrate_pilot1000.py` loads the archived raw G and D at step 1,000, preserving the FFHQ raw G as frozen source. It uses batch 4, two CPU threads, corresponding unmixed latents, constant noise, and the pilot's augmentation probability. The adversarial comparison uses the pilot discriminator and logistic generator loss with DiffAugment/flip. Constant noise and unmixed styles are deliberate controls; this is not a measurement averaged across the trainer's usual random style mixing. There are no optimizer updates.

```sh
research/.venv/bin/python research/experiments/cdc/calibrate_pilot1000.py --device cpu --mode calibration
research/.venv/bin/python research/experiments/cdc/calibrate_pilot1000.py --device mps --mode cdc-benchmark
```

CPU unweighted CDC loss is **0.00266048**, versus adversarial loss **2.14183**. Gradient L2 norms are **0.00294254** and **63.2707** respectively; their cosine is **0.00297428**.

| CDC weight | CDC gradient / adversarial gradient |
| --- | --- |
| 100 | 0.465% |
| 1,000 | 4.651% |
| 10,000 | 46.507% |

Global equality would require about 21,502, but that is not an appropriate automatic setting: at weight 1,000 the CDC mapping gradient is already **3.07 times** its adversarial counterpart. Blocks through 128 are constrained; block 256 receives no CDC gradient. A conservative controlled trial starts at **1,000**, with 10,000 reserved for a strong-constraint ablation. Recalibrate across several batches on the current aligned checkpoint; the collapsed old pilot and its discriminator are not the new run. These magnitudes establish neither better images nor recovery from collapse.

CPU CDC forward/backward took **2.173 s** and adversarial forward/backward **2.987 s**. Process cumulative peak RSS was **5.10 GiB**, including all loaded checkpoints/models, hashes, and graph allocations. Full metrics and unchanged-source/target assertions are in `calibration-pilot1000.json`.

The bounded MPS CDC forward/backward took **1.554 s**, with finite gradients, unchanged source/target weights and buffers, and no source gradients. Post-call MPS allocator snapshots were **295,931,392 bytes current** and **2,439,905,280 bytes driver**. These are process allocator snapshots, not peaks or total system memory. Main training was concurrent. MPS and CPU used device-specific latent RNGs, so their different numerical losses are **not** a backend-equivalence comparison. `benchmark-mps-pilot1000.json` holds full results. The preserved `benchmark-mps-pilot1000-initial-error.log` records an initial diagnostic-only float64 norm reduction unsupported on MPS; moving that reduction to CPU fixed it. It was not a non-finite training-gradient failure.

## Opt-in trainer integration

`research/train.py --cdc-weight 1000 --cdc-batch 4` enables this same helper; the default weight 0 retains the existing training objective. Source G is copied from the original base raw G **before** loading any adapted resume checkpoint. It is permanently frozen and excluded from optimizers. The adversarial graph is backpropagated and released, then a separate CDC graph is backpropagated, followed by exactly one G optimizer step. Correspondence uses its own CPU generator seeded 20260921, and its state is saved/restored in resume checkpoints. Batch/block/source mismatches are rejected when continuing an enabled CDC checkpoint. Changing the coefficient intentionally remains allowed and is recorded in the run config. Existing FreezeD resume checks remain intact.

Run config records the helper checksum, source checksum, blocks, pooling, temperature, seed, weight and batch. Metrics record unweighted/weighted CDC and per-block details. The MPS smoke uses a fresh base, adversarial batch 1, CDC batch 4, FreezeD 4 and weight 1,000, including the actual lazy-R1 discriminator update. Its source and frozen-D checks compare bitwise hashes/tensors; it also asserts finite G gradients/weights, a changed G parameter, unchanged global CPU/MPS RNG across the CDC term, and no source gradients. This bounded smoke is correctness/performance evidence, not a quality evaluation.


Fresh-base trainer smoke passed in **6.338 s** for its actual G/D/R1 training iteration (excluding setup and preview grids). CDC loss was exactly zero, as expected while source and target are identical before the first G update. The earlier perturbed/pilot diagnostics establish positive nonzero CDC gradients. All seven frozen D tensors and the full frozen source remained unchanged; G and an unfrozen D tensor updated with finite gradients/weights. CDC preserved both global RNG states. Post-update MPS allocation was **1,080,187,392 bytes current** and **3,827,138,560 bytes driver**; after CDC backward it was **585,481,984 current / 3,782,983,680 driver**. These snapshots are not a guarantee about two complete training runs' combined peak memory. Evidence: `research/runs/cdc-mps-smoke/smoke-test.json`, `metrics.jsonl`, and this directory's `trainer-smoke-mps.log`. The trainer's subsequent change removed a redundant RNG replay assertion only; the training path is the one exercised by this smoke.

```sh
research/.venv/bin/python -u research/train.py --device mps --batch 1 --threads 2 --freeze-d-layers 4 --cdc-weight 1000 --cdc-batch 4 --data research/data/hot110-256 --run cdc-mps-smoke --smoke-test
```

No long CDC training run was launched by this subagent. A subsequent full checkpoint round trip passed, as documented below.


## Full checkpoint round trip

`check_resume_roundtrip.py` executed two actual 256px MPS processes through the trainer. The first performed one update and saved its actual full checkpoint. The second restored it and performed its next update with lazy R1. Only preview-grid generation was skipped, to keep this focused and bounded; no trainer changes were required.

The test observed exact restoration of all 110 G and 31 D Adam state entries and parameter groups, identical CDC configuration, source/data checksums, learning rates, FreezeD setting, step and image count. It intercepted the next dedicated latent draw and verified both the restored generator state and the resulting tensor against the saved checkpoint. Frozen source stayed the original raw FFHQ G, distinct from resumed target, with unchanged state and no gradients. The next update was finite and retained frozen D tensors. Its CDC loss was nonzero, **5.5922e-8**, after the first adaptation update.

Create/resume phases took **6.74 / 6.79 seconds** including their trainer setup and checkpoint work, excluding outer test setup. Resumed driver allocation was **3,969,744,896 bytes**. Evidence lives in `resume-roundtrip-create.json`, `resume-roundtrip-resume.json`, matching logs, and `research/runs/cdc-resume-roundtrip-*`. This validates restoration and a finite next update; it does not claim bitwise equivalence to a separate uninterrupted MPS run. The harness refuses to overwrite existing evidence on repeat. See `research/reviews/cdc-next-experiment.md` for the recommended fresh-base paired experiment and training-horizon interpretation.


## Current experiment status

After the full resume test passed, the parent launched `aligned110-frozen4-cdc1000` from the original base with 6,000 requested steps, adversarial batch 8 and CDC batch 4/weight 1,000, using the aligned eyes42 dataset and the control's other settings. Its first 10 updates were finite in 38.4 seconds while the existing zero-CDC control continued (parent-reported). This is a live feasibility comparison; no image-quality improvement or convergence is established. No additional diagnostics remain pending. The command and review criteria are in `research/reviews/cdc-next-experiment.md`.
