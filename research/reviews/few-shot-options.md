# Few-shot transfer options for the 110-priest dataset

Reviewed 2026-09-20. Recommendation: **a fresh FFHQ-initialized 110-image FreezeD ablation, freezing the first four discriminator layers**. This is the smallest established change compatible with our portable trainer. Keep cross-domain distance consistency (CDC) as the next targeted diversity experiment if FreezeD still contracts identities. Neither method guarantees that 110 examples will produce an acceptable model.

## Current evidence and comparison

Inspected `research/train.py`, the vendored NVIDIA discriminator, pilot configuration and metrics, prior step-250/500 reviews, and `research/runs/pilot256/samples-000750.png` (SHA-256 `53782106b0251b3e06117afccc1abf05738b13557bc71965405ad744f1906d07`). The 750-step grid has nearly repeated facial structure and composition across latents, plus ghosted/malformed features. This is strong visual evidence of diversity contraction, not a quantified identity-collapse result. The training data contained 50 portraits at that point; it does not evaluate the expanded 110-image set.

Our logistic adversarial loss, R1, EMA, style mixing, and differentiable augmentation can improve training stability and discriminator generalization. They do not explicitly constrain the adapted generator to retain the source model's pairwise diversity. Increasing augmentation alone therefore does not establish that this failure is fixed. The existing audit also identifies a large face-position/framing shift between FFHQ and the collar-visible target; keep the collar, and evaluate the prepared alignment experiment separately.

| Method | What changes | Fit to this failure | Relative local cost (estimate) |
| --- | --- | --- | --- |
| Current logistic/R1/DiffAugment | All G and D layers can adapt | Already learns black clothing/collars, but pilot contracts faces | Baseline |
| FreezeD | Keep input-side D features fixed; train remaining D and G | Limits discriminator overfitting without preventing G from changing face framing | Approximately baseline or slightly cheaper; no extra networks |
| CDC-only addition | Preserve pairwise similarities between source and adapted G features for matching latent batches | Explicitly resists loss of source diversity | Roughly 1.2–1.8× baseline; measure first |
| Full cross-domain correspondence recipe | CDC plus anchor-dependent patch/image discrimination | Stronger few-shot recipe, substantially more integration work | Higher and unbenchmarked here; official CUDA recipe reports about 20 GB at batch 4 |

## FreezeD: smallest implementation

The original paper freezes lower, input-side discriminator layers during transfer, retaining generic features while adapting later discrimination. Its experiments support the method as a strong simple baseline; they do not prove success for this priest dataset. The authors' StyleGAN implementation is separate from our NVIDIA StyleGAN2 implementation. [Paper](https://arxiv.org/abs/2002.10964), [official repository](https://github.com/sangwoomo/FreezeD).

NVIDIA already implements this operation through `DiscriminatorBlock(freeze_layers=...)`: the layer order in our 256px residual discriminator starts with `b256.fromrgb`, `b256.conv0`, `b256.conv1`, and `b256.skip`. Frozen weights/biases are buffers, so the trainer's repeated `d.requires_grad_(True)` cannot reactivate them. Merely setting parameter flags once would be incorrect in our loop. Reconstruct D with its original constructor arguments plus `block_kwargs.freeze_layers=4`, copy the complete source state, then construct its optimizer from parameters. Preserve gradients with respect to input images through frozen convolutions: neither the generator update nor R1 should detach at the frozen block. [Official implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py), [official training flag](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py).

Four is our bounded whole-first-block choice, not a paper-prescribed optimum. Compare against zero frozen layers using the same fresh pretrained weights, 110-image manifest, seeds, learning rates, and image budget. Freezing early **generator** layers is a different intervention and could obstruct the necessary collar/torso framing change. An optional later 3-versus-4-layer comparison isolates the residual skip branch.

A clean fresh run is recommended because preserving the already contracted step-750 model can also preserve its failure. A continuation remains a useful recovery experiment, but should be labeled as such. Frozen weights must be bitwise unchanged after a real update including R1, while at least one unfrozen D weight changes. Checkpoint and resume must record the freeze count and refuse incompatible D optimizer layouts rather than silently discard or misassign momentum.

## CDC: direct diversity preservation, second experiment

Ojha et al. preserve relative similarities of source and target instances and combine this with anchor-dependent discrimination. The official implementation computes row-wise softmax distributions from cosine similarities between different examples' intermediate G features, and minimizes KL(source || adapted) for the same latent batch. It uses a frozen source G, an extra differentiable target G pass, feature-consistency batch 4, and a default KL weight 1000. Its default `KLDivLoss` reduction matters: blindly replacing it with `batchmean` changes the scale. The full algorithm also uses patch discrimination away from latent anchors. [Paper](https://arxiv.org/abs/2104.06820), [official trainer](https://github.com/WisconsinAIVision/few-shot-gan-adaptation/blob/main/train.py), [official README](https://github.com/WisconsinAIVision/few-shot-gan-adaptation).

Our proposed CDC-only experiment would be an explicitly partial adaptation, not a reproduction of that full algorithm. It can use forward hooks on corresponding source/target synthesis blocks, one unmixed latent batch, and constant noise so random texture does not contaminate the comparison. Source features are detached; target features retain gradients. Cosine similarity and softmax operate on off-diagonal pairs. Batch 2 is unsuitable for this loss because each row has just one comparison and its softmax is always one; use at least 4. Measure the unweighted loss and gradient contribution before choosing an effective weight.

Risks specific to us: pretrained FFHQ includes identities outside the adult-male target, and its close-cropped faces differ in geometry. Strong correspondence can preserve unwanted source appearance or hinder framing adaptation. Relative feature relationships are therefore preferable to a hard pixelwise source-output matching loss. Review the actual adaptation rather than assuming a larger diversity metric is always better.

## Compute estimate and evaluation budget

Measured from the local pilot's logged intervals: median **2.77 seconds/update**, and **2.44 seconds/update** over the last ten intervals, batch 8 at 256px on MPS. Those are real full-loop timings; the much faster isolated warm benchmark is not a reliable full-run estimate. At unchanged throughput, 500 updates cost about 20–23 minutes and 1,000 about 41–46 minutes, excluding checkpoint/review overhead. FreezeD adds no forward passes, so budget the same initially. Run a short actual-update smoke test before launching the next run.

CDC-only at the estimated 1.2–1.8× multiplier would put 500 updates around 24–42 minutes. This is an engineering estimate from its extra source forward and target forward/backward, not a measurement. A frozen extra G needs another copy of its weights and temporary features; feature activations, not merely parameter storage, must be profiled. The official full method's GPU-memory figure cannot be treated as an MPS requirement or promise. CPU training speed has not been established by the inference-only CPU benchmark; the local Apple GPU remains the evidenced training path.

Use fixed matched raw-G/EMA grids and fresh random latents at 250/500/1,000 updates. Include truncation 1.0 in diversity diagnostics, since the current fixed grid uses 0.7. Inspect adult male appearance, facial integrity, bare heads, and visible collars. Compare face-focused perceptual/identity variation and nearest target neighbors, rather than letting Roman-background variation inflate diversity. Preserve train/holdout split provenance if used. FID from 110 targets alone is too weak to establish novelty or lack of memorization. Do not release a model on loss curves or a single attractive image.

## Code and license provenance

- FreezeD's `stylegan/LICENSE` is MIT (Kim Seonghyeon); its repository snapshot inspected here is `b1725b5b65acecdf103ab917055f571bf161bfdf`. We need not import that implementation because NVIDIA already provides the feature. [StyleGAN license](https://github.com/sangwoomo/FreezeD/blob/master/stylegan/LICENSE).
- The CDC repository uses the **Adobe Research License**, restricting use and redistribution to noncommercial research and requiring its license and notices. Do not label imported CDC code MIT. [License](https://github.com/WisconsinAIVision/few-shot-gan-adaptation/blob/main/LICENSE.txt).
- Our existing NVIDIA StyleGAN2-ADA code has its own research/evaluation-only noncommercial terms; those remain in force for the reused code and derivatives regardless of the application repository's MIT license. [NVIDIA license](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).
- Vendored DiffAugment has BSD two-clause-style source/binary attribution terms. [Official license](https://github.com/mit-han-lab/data-efficient-gans/blob/master/LICENSE.txt).

## Implementation validation

The follow-up implementation adds optional `--freeze-d-layers` (default 0), reconstructs D with NVIDIA's official buffer-based freezing, and preserves the pretrained state and RNG. Checkpoints record the freeze count; incompatible freeze counts are rejected before restoring optimizer state. Existing resume counters and RNG restoration remain intact.

The requested fresh-base CPU smoke test passed with the actual 110-image dataset, batch 2, two CPU threads, and four frozen layers. One real G/D update including R1 took 6.62 seconds. All seven frozen first-block weight/bias tensors stayed bitwise identical; `b128.conv0.weight` changed; no buffers entered Adam; G/D weights and R1 remained finite. Evidence: `research/runs/frozen-d-smoke/smoke-test.json`, configuration, metrics, and the sibling log. This proves the freezing operation works for that update, not image quality or convergence.

A separate tiny 8px NVIDIA discriminator regression performs two real updates with R1 across an in-memory model/Adam checkpoint round-trip. It verifies preserved frozen buffers, restored optimizer moments, and unchanged initialization RNG. Script: `research/reviews/check_freeze_d_resume.py`; result: `research/runs/frozen-d-smoke/resume-regression.json`. This is targeted serialization evidence, not a full 256px GAN convergence or full-trainer resume test.

Snapshots now preserve the existing EMA psi=0.7 filename and add matched `-raw.png` (raw G, psi=1.0) and `-untruncated.png` (EMA, psi=1.0) grids at initialization and checkpoints. Rendering temporarily uses evaluation mode, then restores every module's prior mode and CPU/MPS RNG. A bounded check using the actual pretrained 256px raw G in training mode passed: all parameters and buffers, CPU RNG, and module modes were unchanged. Script and evidence: `research/reviews/check_snapshot_state.py` and `research/runs/frozen-d-smoke/snapshot-state-check.json`. MPS RNG restoration is implemented but was not separately exercised by this CPU-only check.

The next research run can start fresh from FFHQ on the prepared aligned 110-image dataset with FreezeD4. Preserve its dataset manifest and compare raw G/EMA against the unfrozen reference before claiming that diversity is repaired. No extended training or production deployment was performed by this subtask.
