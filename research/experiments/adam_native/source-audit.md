# AdAM source audit and native1024 probing smoke

Retrieved 2026-09-21. This is a source audit and proposed compatibility protocol, not an executed experiment or a claim of reproduction. No GAN checkpoints, external datasets, or compute resources were acquired. The complete 1.31MB source archive includes upstream's small noise and LeNet fixtures; neither was loaded.

Official repository: [yunqing-me/AdAM](https://github.com/yunqing-me/AdAM), pinned to **6428e99cfb36bc8bda3506350824f0c7dac9a5ad**. GitHub's commit API independently returned that SHA, dated2023-12-21. Source is in `upstream/source/`; original archive, commit metadata and per-file SHA256 provenance are retained in `upstream/`. The upstream MIT license remains verbatim at `upstream/source/LICENSE`. NVIDIA code and weights retain their own licenses.

The [NeurIPS2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b122d0a0dcb1a86ffa25ccba154652b-Paper-Conference.pdf) proposes target-aware importance probing followed by constrained adaptation of important kernels and fine-tuning of less-important ones. The details below are independently checked against the pinned executable source, including deviations and apparent implementation defects. Published few-shot results do not demonstrate success on priests, native1024, this checkpoint, or the latency requirement.

## Modulation, exactly as implemented

[model_adam.py](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/gan_training/models/model_adam.py):

- FC: `W_eff = W * (1 + outer(v_out, u_in))`, `b_eff = b + b_vector` (lines17–48). Both factors start as independent normal samples multiplied by1e-7; additive biases start at zero.
- Conv: `outer(u_flat, v_out).view(out,in,k,k)` (lines174–217), or the same view with a leading singleton for styled G convolutions (lines381–464). **The outer-product order differs from FC.** This flattened permutation is not ordinary per-output-channel rank-one modulation when the convolution is flattened as `[out,in*k*k]`. A native port must label whether it preserves this source layout or corrects it to `outer(v,u)`; the latter is a methodological change.
- G modulation happens before the original style multiplication and demodulation. Style-affine FCs are separately modulated. The noise parameter receives a learned additive scalar; activation biases receive additive channel vectors; ToRGB biases receive additive RGB vectors.
- Both1e-7 factors make the initial multiplicative change about1e-14, which is usually rounded away in FP32, but this is not an algebraic identity guarantee. A centered form `W*(1 + M(u,v)-M(u0,v0))` guarantees zero residual at initialization while retaining nonzero factor derivatives. That is an explicitly documented reparameterization. Initializing both factors to zero produces zero gradients to both and must be avoided. A changed initialization scale also changes gradient/Fisher magnitudes.
- Upstream `KML_fc.scale` uses `1/sqrt(out_dim)` whereas the ordinary FC and NVIDIA FC use `1/sqrt(in_dim)`. This differs for nonsquare style affines. **Preserve NVIDIA's existing scaling in the native port**, rather than inheriting this apparent upstream defect and changing the source function before learning.
- Upstream `FusedLeakyReLU_kml` conditionally applies its learned bias only when `b_vector.requires_grad` is true (`op/fused_act.py:85–103`). Copying that condition into inference would drop trained offsets after freezing. Native effective parameters and folding must remain independent of requires-grad flags.

## Exact probing trainable families

The [probing script](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/AdAM_importance_probing.py) freezes G parameters except names containing `u_vector`, `v_vector`, or `b_vector` (lines497–520). D additionally trains ordinary parameters whose names contain `final`.

| Component | Probing behavior | Native NVIDIA mapping |
|---|---|---|
| G mapping FCs | Modulate all eight; base weights/biases frozen | `G.mapping.fc*` |
| G first4x4 and subsequent synthesis convolutions | Conv factors, style-affine factors/bias, activation-bias offset, noise-strength offset | All `G.synthesis.b*.conv*`, including b4.conv1 |
| G ToRGB at every active resolution | Conv factors, affine factors/bias and RGB-bias offset | All `G.synthesis.b*.torgb` |
| G learned constant | Frozen, no modulation | b4.const |
| D input/fromRGB | Conv factors and activation-bias offset | Actual active `D.b*.fromrgb` |
| D residual blocks | Factors for conv0/conv1 and skip; biases where present | All actual residual blocks |
| D final convolution and two FCs | **Ordinary full training**, no KML module | `D.b4.conv`, `D.b4.fc`, `D.b4.out` |

Inventory actual modules and parameter shapes; do not assume names or layer counts from the256px implementation. NVIDIA stores styled-convolution weights as4D tensors; upstream uses5D with a singleton batch axis. NVIDIA combines noise/bias inside SynthesisLayer and keeps convolution scaling/resampling in its original forward. It also uses its own minibatch-standard-deviation configuration; source sets group25. Unconditional labels remain empty. Do not modify mapping moving-average buffers inadvertently while testing frozen-source equivalence.

## Probing objective and updates

Source defaults and recipe: batch4, image256, Adam base LR0.002; style mixing probability0.9; source weights loaded from `g`, `d`, with separate `g_ema` loaded from source `g_ema` and `d_ema` from `d`. Starting our trainable G from the original G_ema is a deliberate source difference, retaining the chosen photographic prior.

One source loop iteration performs:

1. Generate a fresh mixed-style fake batch. D adversarial objective is `mean(softplus(-D(real))) + mean(softplus(D(fake)))`; update D.
2. At `i % 16 == 0`, update D again with R1: `gamma/2 *16 * mean(sum(grad_x D(real)^2))`, gamma10. This includes iteration0.
3. Generate fresh noise again. G objective is `mean(softplus(-D(G(z))))`; update G.
4. At `i % 4 == 0`, generate another fresh batch of size `max(1,batch//2)` and update G with path regularization, coefficient `2*4`. Image-space path noise is divided by `sqrt(H*W)`; latent gradient norm is `sqrt(mean_over_layers(sum_over_w_channels(grad^2)))`. Moving target uses decay0.01. This includes iteration0.
5. Update both G and D EMAs, with fixed decay `0.5**(32/(10*1000))`, approximately0.997784. This formula is fixed in the source, not rescaled to its batch4.

Adam compensates lazy regularization: G LR0.002*(4/5), betas `(0,0.99**(4/5))`; D LR0.002*(16/17), betas `(0,0.99**(16/17))`. These ratios matter if regularizers are omitted: an omitted-regularizer smoke must record the changed optimizer choice rather than imply exact fidelity.

The recipe does not enable optional ADA/non-leaking augmentation. Dataset transforms do include Resize, CenterCrop, RandomHorizontalFlip, normalization to[-1,1]. The source computes unnecessary cross-network gradients because alternating requires-grad calls are absent; detaching fake for D and freezing D parameters while retaining its image Jacobian for G is a safe memory optimization. Detaching D's fake prediction during the G update is not.

## Fisher measurement and later adaptation

The [published launch script](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/_bash_importance_probing.sh) requests500 probing iterations, frequency500,250 Fisher batches. It contains a `.py.py` filename typo, even at this pinned revision. The Python loop runs `fisher_iter+5` iterations: evaluation at i500 occurs after500 updates, then five more update iterations execute. Do not use this loop unchanged for an exactly-ten-iteration smoke.

Fisher is estimated on EMA G and D. Evaluation enables gradients for all their parameters, processes one real/fake pair at a time, computes G non-saturating loss and D logistic loss, and accumulates squared parameter gradients. The source divides by `num_batch_fisher*batch`:250*4 gives1000 evaluated examples if fixture sizes match. This is an empirical squared-loss-gradient importance estimate; **mean of per-example squared gradients is not square of a batch-average gradient**. Its magnitude depends on parameterization and normalization. The real-image reshape is hardcoded to `[1,3,256,256]`; fixture noise uses unconditional `.cuda()`. Native1024/CPU must fix both. The checked ten-step smoke must not describe a handful of gradients as a stable importance ranking.

[Main adaptation](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/AdAM_main_adaptation.py), lines705–849, builds three pooled percentile thresholds: G convolution, G style-affine FC, D convolution. For convolution without a bias contribution, score per output is `mean(F_u)+F_v`; with affine/activation bias the implementation uses `(mean(F_u)+F_v+F_b)/2`. Values strictly above threshold select modulation; values at or below select fine-tuning. Recipe quantiles are50 for Babies and75 for AFHQ-Cat; those are not calibrated priest settings.

Selection is narrower than probing: G uses twelve `convs` and their affines, excluding first4x4, ToRGB and mapping from this threshold calculation; D uses six residual blocks including skip, excluding input/fromRGB and final layers. These are hardcoded256px ranges. Native1024 requires dynamic enumeration. Low-importance selected `v`/bias entries are zeroed and their gradients masked; high-importance original weight rows/bias entries have gradients masked. Shared `u` remains trainable. This output-channel masking sits uneasily with the source's permuted conv outer layout; do not silently pretend it implements a clean per-output kernel partition.

Main adaptation **reloads original source weights with fresh modulation factors**, not the probed G/D checkpoint. Probing supplies importance statistics. G's original activation bias and noise-strength gradients are explicitly zeroed during main adaptation. D's input original weights/bias are frozen, final layers remain trainable. A native main-adaptation implementation needs separately audited masks and optimizer state; the present smoke does not implement or validate that policy.

## Exactly ten native compatibility iterations

Root-owned executable should consume the verified original native1024 G_ema and D, with an immutable target training manifest and no held-out/test access. CPU one-thread and batch1 are explicit resource deviations. Preserve1024 resolution, source random style mixing, stochastic synthesis noise and target flip policy, with recorded RNG states/seeds.

Before training, inventory every trainable tensor and frozen tensor, and assert G and D output equivalence to untouched originals on fixed inputs/noise. Verify both rank factors receive nonzero gradients and that every intended additive bias participates. Keep source scaling, resampling, activation and minibatch-statistic behavior unchanged.

Execute loop indices0–9: ten D logistic updates plus R1 at0; ten G adversarial updates plus path updates at0,4,8. That is **11 D optimizer steps and13 G optimizer steps**, not ten total optimizer steps. Record losses, finite gradients, norms and actual wall time for each phase, memory guard status and optimizer-step counts. If second-order regularization is infeasible, stop or explicitly label a reduced objective compatibility smoke; do not silently omit it and call it source-faithful probing.

Afterward, assert original frozen weights/buffers have not changed, intended factors and D epilogue have changed finitely, and source files/checkpoint hashes remain unchanged. Fold modulation on a copy and verify output agreement with the unfused trained model under identical latent/noise and eval flags. Preserve all outputs and failures. Optionally collect a small, explicitly diagnostic per-example squared-gradient sample after ten iterations; do not select importance thresholds or claim an adaptation result from it.

The smoke establishes computational compatibility only. Neither finite losses nor exact folding demonstrate photographic quality, diversity, priest correctness, or sub500ms deployment generation. Current owner feedback rejects the previous diffusion baseline's photographic realism; that remains unresolved until a new native preview is reviewed.
