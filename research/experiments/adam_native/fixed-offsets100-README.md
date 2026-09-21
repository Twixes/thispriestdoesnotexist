# Prepared output-rank1 fixed-noise-and-synthesis-bias stability test

Prepared only. This100-iteration native1024 stability test starts again from verified original pretrained G_ema/D weights and fresh output-rank1 factors/optimizers. It does not resume a failed probe or represent a quality-approved model.

```sh
research/.venv/bin/python research/experiments/adam_native/probe_output_rank1_fixed_offsets100.py --output research/runs/adam-native1024-output-rank1-fixed-offsets100-v1
```

The new runner derives from frozen `probe_output_rank1_fixed_noise100.py` SHA `3a81f0931152deecf7a3f883f3d0075bc3d5db17a56cfccf4a449cac2be1f755`. No existing runner or modulation helper changed. Compared with that parent, the only additional training-policy change is to freeze the17 additive synthesis activation-bias offsets, alongside the17 already-frozen additive noise-strength offsets. At native installation the runner asserts exactly17 entries in each family and34 total.

The bias family matches only `synthesis.b<resolution>.conv0/conv1.parametrizations.bias.0.b_vector`. Mapping FC, style-affine and ToRGB bias offsets remain trainable, as do u/v weight factors. All original noise strengths and synthesis bias vectors remain bit-exact. The34 additive offsets remain zero, gradient-disabled and absent from the optimizer; dedicated gradient toggles never temporarily enable them. G EMA retains the same originals and zero offsets. Before/after each optimizer update, after EMA updates, around previews and after checkpoint restoration, assertions enforce parameter values, `grad is None`, disabled gradients and optimizer exclusion. No repairing or clamping a violation occurs.

This deviates from AdAM's trainable additive offsets and uses the explicitly declared output-rank1 convolution layout rather than released source_flattened behavior. Random spatial synthesis noise is still sampled in training at original strengths. The loss formulas, adversarial/R1/path schedule, CPU1/batch1 native1024 resolution, all20 training images, seeds/style mixing, LR/betas and EMA decay match the prepared parent.

The existing four latent tensors are hash-pinned and tensor-compared. Retain baseline4 and all four raw/EMA outputs at steps0,10,25,50,100, with complete hash manifests; no child/woman/hat filtering or replacement. Expected totals are125 G and107 D optimizer updates. Checkpoint100 retains complete raw/EMA model, optimizer, RNG, path-mean and count state with atomic serialization/exact restoration checks. Final raw-G folding must match all four native outputs. There is no Fisher estimator, mask selection or main adaptation.

Guard limits remain35% initial/20% runtime available memory,12GiB process-tree RSS,512MiB swap growth and2400seconds under an owned process group. Root schedules execution only after other heavy workers finish and fresh preflight passes. Earliest trained review is step10; preparation does not establish stability, realism or server latency.

Four tests passed in0.414seconds, CPU1: all34 native-named exclusions in a tiny scalar parameter tree, actual random32px Adam/EMA updates and repeated toggles, preservation of deliberately nonzero original strengths/biases, corruption detection, explicit continued training of mapping/style-affine/ToRGB offsets, and unchanged loss/generation code and review schedule. No native1024 model, pretrained checkpoint or native render was loaded/run in preparation.

The completed bias-family inference ablation motivates this restriction: resetting noise plus synthesis-bias offsets improved the observed contrast/texture relative to noise-only resets, while other defects remained. A reset diagnostic on a failed checkpoint does not establish the outcome of fresh constrained training. All native outputs still require review.

## Native launch and early review

The revised native run is executing at `research/runs/adam-native1024-output-rank1-fixed-offsets100-v1`, exec session51876; live supervisor/worker were independently observed at launch. The first10 steps and all four baseline/raw/EMA comparisons passed early review without the old severe grain/contrast failure. Mild color shifts remain and no priest conversion is established. It continues to its bounded100-step endpoint under the original guards. The progressive review is `research/reviews/adam-fixed-offsets100/`; preparation-only JSON above remains a historical record of prelaunch checks.

## Completed100-step result

The run completed all100 iterations in1362.19s, peak RSS9.93GiB, minimum available memory35.29%. All34 protected offsets/originals held exactly;125G/107D optimizer steps and full checkpoint-state restoration passed. Folded native outputs equal the parametrized model exactly on all four retained latents. Final raw images avoid the severe source-control grain/dark-red failure but show cooler color drift and texture changes; source identities including children/headwear remain. EMA stays near the source. This is successful bounded mechanics/preservation evidence, not successful priest adaptation or production approval.
