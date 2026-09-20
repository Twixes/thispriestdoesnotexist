# Trainer audit against NVIDIA StyleGAN2-ADA

The read-only audit found no verified fundamental loss-sign, R1 differentiation, pretrained-network selection, style-mixing, or EMA interpolation bug. The portable trainer does materially change the reference optimization and augmentation recipe. Those differences justify controlled comparisons; none is demonstrated to cause the observed artifacts. At step 500 with batch eight, only **4,000 real-image presentations (4 kimg)** have occurred. NVIDIA gives roughly **1000 kimg** as a common transfer-learning timescale, not a guarantee for this much smaller and differently distributed dataset. Calling 500 steps a final convergence failure would be unsupported. [Official training guidance](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md).

## Evidence and scope

Reviewed `research/train.py`, the vendored NVIDIA loss/training loop/augmentation/network code, DiffAugment, the aligned run's saved configuration and metrics through step 500, and the actual step-zero raw-G, step-500 raw-G, and step-500 untruncated EMA grids. The step-500 images have visible collars and changed facial shape, but substantial ghosted facial features. Backgrounds, hairstyle, pose, and some source accessories still vary between fixed seeds. Many faces acquire similar features, which is a warning about contraction; this grid alone does not establish total identity collapse. Distortions are present in raw G as well as untruncated EMA, so neither truncation nor EMA lag alone explains them.

The audited working trainer includes optional CDC code added after the aligned run began. The saved aligned run used no CDC; this report does not attribute its step-500 result to that later feature.

- Audited trainer SHA-256: `d46d393d136ce641fd290baec422737b1f070a1c501b43c135bce1b9b8df104e`.
- Aligned run's recorded original trainer hash: `d5394da03b8db1c44fa62c18fe68f89f009066c1840f23a9cf699ce32855a879`.
- Upstream loss hash: `1bd68d7dc1c92c1b78b91f193a45a433b1ff374614f91909ea8cb88eac7bc2d7`.
- Upstream loop hash: `0b589a731712c51791eb22efe0dc1ee6856bf3e9335920db02a3e38dee909eb3`.
- Step-500 raw grid hash: `c46335643a960967d65697f6a225c7166bc083f8382e35e7ee73ae6a7b0bafca`.
- Step-500 untruncated EMA grid hash: `8353556de0120fd42e7b1591978d90dbb46405d15096b89c592ec45fda7c698a`.

## Ranked actionable findings

### 1. Restore the reference optimizer-phase semantics in a controlled baseline

Current lines 257–272 combine logistic D gradients and an interval-scaled R1 term into **one Adam step** every sixteenth iteration. NVIDIA uses a distinct regularization optimizer step and adjusts learning rate and beta for that extra step. An unbiased average regularization coefficient does not make these Adam trajectories equivalent. Simply adding the reference 16/17 learning-rate factor to the current combined update would not fix that difference. [Official phase construction](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py).

Action: add an explicitly named reference mode with separate Dmain and Dreg steps, correct lazy-regularization compensation, and R1 at the reference initial interval; alternatively evaluate non-lazy R1 every step without interval scaling or compensation. Keep gamma, dataset, and other settings fixed for this comparison. Treat this as restoring a reference experiment, not a confirmed bug fix. It is higher value than adjusting render sharpness.

The R1 formula itself is correct: differentiate the real discriminator score with respect to the original real pixels, including the differentiable augmentation chain, square/sum spatial and channel dimensions, mean over batch, and multiply by gamma/2. Interval multiplication by 16 compensates its sampling frequency. There is no missing factor of batch size. [Official loss](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py).

Observed logged regularized D losses contain large R1 contributions: at step 320, R1=0.28720 contributes 4.59522 out of D loss 4.67584; at step 480 it contributes 3.23123 out of 3.53030. This demonstrates that the phase difference is consequential enough to investigate, **not** that R1 is too strong: scalar loss magnitude is not gradient magnitude. In the earlier deterministic pretrained batch-one audit, unscaled R1 gradient norm was 0.92177 versus logistic D norm 24.96826; multiplying R1 by 16 gives 14.74825. Those are different inputs/configuration, so do not extrapolate their ratio to the trained run.

### 2. Path-length regularization was removed, leaving no reference generator regularizer

Current configuration explicitly records `path_length_regularization=false`. NVIDIA normally regularizes generator sensitivity to W every four iterations, with weight two and a smaller regularization batch. Style mixing does not replace this objective. Its omission is a significant difference when evaluating latent smoothness and diversity preservation, though it does not by itself prove the cause of a collapsed model. [Official generator regularization](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py).

Action: implement and smoke-test the reference path-length phase, its moving mean, checkpoint state, and the associated G optimizer compensation before treating the portable trainer as an ADA reproduction. Compare at equal kimg and fixed sample seeds. It adds second-order computation, so benchmark a bounded CPU/MPS step first; do not silently add it to a running experiment. CDC is a different, experimentally useful objective, not evidence that this missing reference component was harmless.

### 3. Augmentation is a different algorithm with unusually coupled occlusion

The trainer samples a 50% horizontal flip independently of ADA, then applies **all** DiffAugment color/translation/cutout operations or none using one per-image gate. At 256 pixels, the standard cutout side is 128 pixels and translations can shift by 32 pixels; translation padding is normalized zero. NVIDIA's default `bgc` independently gates several transforms, uses filtered geometric resampling/reflection padding, and does **not** include cutout. [Official augmentation implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/augment.py), [official configuration](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py).

This is valid DiffAugment-style training, but applying the ADA controller to it does not make it NVIDIA ADA. Large cutouts can hide most of a face; coupled transformations reduce the variety of partially preserved signals. These are plausible pressure changes, not demonstrated causes of the ghosting. Real and fake use the same distribution, independent draws are correct, and gradients are not detached by the `where` gate.

Action: a cheap first ablation is the same portable policy **without cutout**, keeping all other settings fixed. A faithful NVIDIA augmentation comparison is more informative but must first verify the second derivatives needed by R1 on the target backend. Do not transplant geometric sampling code into MPS assuming its double backward works. Save augmented real/fake contact sheets and separately measure D gradients to check what signal the discriminator actually receives.

ADA's target 0.6 and update magnitude are implemented consistently with the reference controller. The transfer preset explicitly uses `ada_kimg=100`. In 4 kimg, that controller can move p by at most 0.04, and the recorded p rises from 0.5 to 0.53744. That slow numerical movement is expected, **not** an arithmetic bug. Starting at 0.5 and capping at 0.85 are local choices. Earlier advice suggesting a faster controller as a correctness fix was too strong; changing its speed is a separate hyperparameter experiment.

### 4. Small-data adversarial imbalance needs measurement beyond training loss

There are 110 target images. Logged real-score sign means average 0.925 during steps 10–100, 0.900 during 110–250, and 0.990 during 260–500, all above the controller target. The last interval's mean real logit is 2.957 and fake logit -2.000. These are the saved every-tenth-step samples, not full-step averages. They establish strong separation on sampled training images; they do not distinguish useful domain learning from memorization.

Action: maintain a small, genuinely held-out target set and compare real-score distributions and nearest neighbors alongside fixed and fresh-seed output grids. Continue reporting kimg rather than assuming image reuse equals independent data. Do not judge learning rate or D strength by raw G/D loss magnitudes alone. FreezeD's input-block buffers are constructed correctly and protected from `requires_grad_` toggles; the existing smoke verifies frozen tensors remain unchanged. Freezing four layers is still a chosen hyperparameter, not an upstream default or a guarantee against overfitting.

### 5. Mapping rate and EMA choices change adaptation speed and visible volatility

Current mapping/synthesis/D rates are 0.0005/0.001/0.001. The source `paper256` reference uses one G rate before regularization compensation; mapping already has its architectural learning-rate multiplier. Reducing its optimizer rate further may slow coordinated geometry adaptation, but increasing it without a controlled comparison is not a verified remedy. The present rates are not simply an excessive version of NVIDIA's 0.0025 base rate. Batch eight versus a larger reference batch also changes updates and noise per kimg. [Official presets](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py).

The current EMA half-life is only 0.5 kimg; `paper256` uses 20 kimg. The local EMA therefore shows transient changes much sooner. Interpolation direction and copying buffers are correct. At 4 kimg only 2^-8, approximately 0.39%, of the initial EMA parameter contribution remains. Increasing the half-life can smooth presentation but cannot repair the distorted raw generator. [Official EMA update](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py).

Action: keep raw and untruncated EMA comparisons, already implemented. Expose a mapping-rate-equals-synthesis ablation rather than interpreting the split rate as required by the architecture. Preserve the short EMA for rapid diagnostics if desired, but label it and do not compare its step count directly with long-EMA reference samples. **No EMA ramp-up is correct for NVIDIA's transfer preset**, so its absence should not be reported as a correctness defect.

## Components that checked out

- Loading raw `G` for training and `G_ema` for the initial EMA matches NVIDIA transfer loading. Replacing raw G with G_ema is an experiment, not correction of a source-selection bug.
- Style mixing uses an independent second latent, a uniformly chosen nonempty layer cutoff, probability 0.9, and skips the second mapping call's W-average update. Concatenation preserves the intended gradient paths. Python versus Torch randomness changes reproducibility streams, not that distribution.
- Logistic loss signs, RGB normalization to [-1,1], train/eval usage, generator/discriminator gradient toggles, and synthesis FP32 behavior are consistent with their purposes.
- Earlier all-parameter CPU/MPS comparison found aggregate G/D/R1 gradient relative errors of 0.0757%/0.00247%/0.000535%, all finite. It excludes augmentation and batch statistics, so is supportive but not exhaustive evidence against a backend problem. See `cpu-mps-gradients/README.md`.
- The earlier fused/non-fused synthesis comparison was below 0.009 of an 8-bit pixel level. Switching that inference flag is not a credible remedy for the visible double features.

No training or model changes were made for this audit. The preferred next step is a clearly recorded reference-style optimizer/regularizer baseline and one augmentation ablation, with equal-kimg comparisons, rather than simultaneous untracked hyperparameter changes. Keep the current early result rejected for deployment while allowing sufficient training time to evaluate its trend.
