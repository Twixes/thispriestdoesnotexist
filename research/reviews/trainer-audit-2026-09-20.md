# Pilot trainer audit: early geometry transfer, no identified fatal implementation bug

## Evidence inspected

- Current `research/train.py`, vendored NVIDIA `training/loss.py`,
  `training/training_loop.py`, `training/networks.py`, and upstream `train.py`.
- Step-250 MPS fixed grid and the immutable resume snapshot hashed
  `d1f2c3e360c7a360545ec0de168df6c343a38d5ce2c32a79bd89fc9808807767`.
- CPU-only, two-thread rendering of eight identical new latents through base G,
  base EMA, trained G, and trained EMA. Reproduction script:
  `compare_generators.py`; exact seed, timings, state differences, and checksum:
  `pilot250/comparison.json`; four grids and combined comparison beside it.
- Existing `research/runs/benchmark-mps.json`: actual MPS generator/discriminator
  backward steps completed, R1 second derivative backward completed with finite
  penalty. This is portability evidence, not proof of exact CPU/MPS gradient
  equivalence.
- Live training PID 52466 was confirmed at step 380, 3.04 kimg; finite logged
  losses, augmentation probability 0.52848. The review did not stop or modify it.

## What the images show

At step 250 (2 kimg), **both raw G and EMA are unacceptable**: duplicate or displaced
facial features, source-domain hair/age/gender characteristics, and damaged facial
structure. Raw G nevertheless has recognizable black clerical clothing and white
collars in all eight diagnostic samples. EMA lags visibly, often representing that
clothing transition as a dark patch across the source face. This establishes that
updates are reaching the generator and learning target-specific appearance; it
does not establish that continued training will converge.

The pretrained G and EMA preserve the same identities and framing and produce
normal faces in this comparison. They are not numerically identical (brightness
and other detail differ). There is no evidence that an incompatible pretrained
G/EMA pair caused the ghosts. CPU evaluation reproduces the malformed trained
appearance, so it is not merely an MPS output/rendering issue.

The target portraits preserve upper torso and collars, with heads smaller and
higher than FFHQ's closely aligned face framing. The generator is moving facial
geometry as well as learning grayscale, male-adult appearance, clothing, and Roman
backgrounds. The forehead/eyes/mouth ghosting is consistent with this large early
geometry change. This is an inference from the images, not a demonstrated causal
ablation. Do not crop away the collar merely to make an easier face-only task.

## Implementation review

1. Logistic loss signs match NVIDIA: generator softplus(-D(fake)); discriminator
   softplus(D(fake)) + softplus(-D(real)). Real and fake receive differentiable
   augmentation, including during generator updates.
2. Style mixing at probability .9, skipping the second mapping's w-average update,
   and using raw synthesis during training match the reference structure.
3. EMA update direction is correct: lerp from EMA toward G by 1-beta, with beta
   `0.5 ** (batch / (ema_kimg * 1000))`; buffers are copied as in NVIDIA. At 2 kimg
   and a 1-kimg half-life, the original EMA contribution is still 25%. Fast early
   geometry movement plus this lag can worsen the visible intermediate mixture.
   It is not a swapped-weight or frozen-EMA bug.
4. Lazy R1 multiplier is correct for gamma 2 and interval 16: gamma/2 * 16 = 16.
   However, the 16/17 optimizer LR/beta adjustment is copied from a reference that
   performs a separate D regularization optimizer step; this trainer combines R1
   into the ordinary step. For this combined variant, use unscaled D LR=.001 and
   beta2=.99. The current ~6% mismatch is minor and is not a plausible sole cause
   of severe ghosting.
5. G mapping uses .0001 while synthesis uses .001, in addition to the network's
   internal mapping LR multiplier .01. That is deliberately much more conservative
   than reference equal optimizer LR groups; it can retain source latent semantics
   longer. It is not evidence of a missing gradient.
6. ADA target .6 and transfer speed 100 kimg match NVIDIA's transfer setup. Combining
   the ADA controller with this smaller DiffAugment family is a custom variant,
   not an exact reproduction of NVIDIA ADA. Starting p=.5 and capping .85 are also
   choices. Over only 24 kimg, adaptation from this initial p is relatively slow.
7. The omitted path-length penalty is a training-method deviation and may affect
   latent smoothness. It is not necessary to add its costly second derivatives
   immediately to explain the first 2 kimg of geometry transfer.
8. Resume saves Torch CPU/MPS RNG but not Python random or NumPy RNG. Save/restore
   both for exact continuation; Python controls style mixing and NumPy is used in
   augmentation probability updates. This affects reproducibility, not the initial
   failed grid. The resume checkpoint contains a NumPy scalar; the offline audit
   uses trusted `weights_only=False`, while the generator-only export safely uses
   `weights_only=True` and the runtime remains pickle-free.

No fatal sign, EMA-direction, parameter-order, or style-mixing implementation bug
was identified. This audit is bounded; it does not prove every gradient is correct.

## Concrete next run recommendation

Continue the existing run to its next 500/1000-step visual checkpoint. Preserve it
as the 50-image baseline. Once the expanded, visually reviewed dataset is ready,
resume its actual optimizer/model state into a new run directory with the dataset
manifest pinned and these explicit settings:

| Setting | Suggested next value | Reason |
| --- | --- | --- |
| Batch / resolution | 8 / 256 | Already portable and fast locally |
| Synthesis LR | .001 | Keep an established variable fixed |
| Mapping LR | .0005 | Increase adaptation of source latent semantics by 5x, still below synthesis |
| D LR / beta2 | .001 / .99 | Correct combined-R1 optimizer schedule |
| R1 gamma / interval | 2 / 16 | Close to NVIDIA auto heuristic for 256px, batch 8; already finite on MPS |
| EMA half-life | .5 kimg | Reduce diagnostic lag during rapid domain adaptation |
| EMA rampup on resume | None | Preserve resumed EMA; do not reset it to FFHQ |
| ADA target / speed | .6 / 100 kimg initially | Reference transfer settings; log real-score sign mean to guide changes |
| DiffAugment / style mix | Current family / .9 | Avoid changing everything simultaneously |
| Snapshot | Every 250 steps | Render both raw G and EMA with identical seeds |

For a **new start**, an optional EMA rampup `.05` means
`ema_nimg=min(ema_kimg*1000, current_nimg*.05)`; it makes early EMA follow raw G more
closely. It cannot repair malformed raw G and should not be represented as a
quality fix. NVIDIA explicitly disables rampup on transfer resume; keep that
behavior unless an isolated comparison justifies otherwise.

If logged real-score sign mean remains above .6 and D loss remains very low for
the next 250–500 steps on the expanded set, try a separate continuation with
`ada_kimg=20` (5x faster response), same cap .85, and leave other variables fixed.
This is an experimental acceleration, not the published NVIDIA transfer default.
If geometry still fails after several more checkpoints, prefer a controlled
FreezeD ablation (first three high-resolution discriminator layers frozen) over
blindly increasing all learning rates. Preserve early generator layers' ability
to move face geometry; freezing them would work against the collar/torso framing
change. Record all these branches and visual comparisons.

Do not promise that 3000 updates (24 kimg) is enough. The current images establish
progress toward the domain but not an accepted model. Fifty examples are below
the 100-image few-shot setup highlighted by DiffAugment, and far below the
thousands-of-images headline ADA result; expanding data addresses an actual
generalization constraint. Actual diversity, memorization checks, adult male
appearance, collars, and no hats remain release gates.

## Primary sources

- [NVIDIA StyleGAN2-ADA implementation](https://github.com/NVlabs/stylegan2-ada-pytorch):
  the vendored pinned implementation supplies the loss, EMA, transfer ADA speed,
  and lazy-regularization comparisons above.
- [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676):
  discriminator overfitting motivates adaptive augmentation for limited datasets.
- [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738):
  augment both real and fake differentiably; the paper includes 100-image results.
- [Freeze the Discriminator](https://github.com/sangwoomo/FreezeD) and its
  [paper](https://arxiv.org/abs/2002.10964): lower discriminator layer freezing is
  an established transfer-learning ablation, not proof it will solve this target.
