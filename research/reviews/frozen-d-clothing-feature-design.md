# Optional frozen-discriminator clothing reconstruction

**Wait for the six-pair step-600 evaluation. Prefer a small boundary loss first if the remaining defect is only a soft collar edge.** A frozen-D feature term is a reasonable separate experiment if correct white tabs and dark shirts appear but their local shape/texture remains poor despite improved pixel losses. Neither term fixes missing supervision, invalid masks, or absent unseen-seed collars by itself. This is a design, with no implementation, large-pickle load, inference or training performed.

## What JoJoGAN actually does

The [official notebook](https://github.com/mchong6/JoJoGAN/blob/ff33d3f541a8deccc6df85f053286ac8c88f7526/stylize.ipynb) computes target discriminator features under `no_grad`, generated-image features with autograd, then averages L1 means across feature layers. Its optimizer updates G; it does not use D's final real/fake score for this loss. The notebook does not explicitly turn off D parameter gradients, so our version should do so to avoid wasted gradient storage. Its style mixing and LR are separate choices we should not inherit.

The [official D implementation](https://github.com/mchong6/JoJoGAN/blob/ff33d3f541a8deccc6df85f053286ac8c88f7526/model.py) selects module indices `(1,3,4,5)` in its own convolution list and stops there. At 1024px these are spatial outputs 512,128,64,32. These indices are **not NVIDIA block names**. Our localized, area-normalized, shallower proposal below is an adaptation of the feature-reconstruction idea, not a JoJoGAN reproduction.

## Available source and exact feature path

The existing `research/models/ffhq1024.pkl` provenance records official NVIDIA weights SHA256 `a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c`. The native loader validates G, D and G_ema members (`legacy.py:20–41`), but current exported bundle metadata describes only G_ema. **This review did not inspect the actual D constructor/tensors.** A future guarded extraction must verify D's unconditional RGB1024 identity, architecture and state, record constructor/channel/buffer metadata, and strictly reconstruct it. Do not substitute the separately trained 256px D or import JoJoGAN's incompatible checkpoint.

Use the original pretrained D, `eval().requires_grad_(False)`, never an optimizer or adversarial update. In `networks.py:556–582`, each NVIDIA block returns `(x, img)` after downsampling. Call the prefix explicitly:

1. `b1024(None, image, force_fp32=True)` → 512px features.
2. `b512(x, img, force_fp32=True)` → **256px features**, first selected tap.
3. `b256(x, img, force_fp32=True)` → **128px features**, second tap; stop.

Do not call `D.forward` or `b4`: the epilogue adds minibatch statistics (`networks.py:615–659`). Explicit traversal avoids hooks retaining unrelated tensors and avoids batch-dependent features. With channel_base 32768/channel_max 512, expected selected shapes are `[1,128,256,256]` and `[1,256,128,128]`; these are conditional expectations to assert against the extracted D, not verified checkpoint facts. Keep the source architecture and resampling kernels intact.

## Loss inputs, normalization and gradients

Let S be the exact original source, T the edited target resized by the existing rule, Y the full student output, M the unchanged clothing mask, C the effective collar intersection, and R=M−C. All images use the existing differentiable grayscale in `[-1,1]`, replicated to RGB for D. Do not quantize, clamp, rescale a crop, or change G's forward settings.

```text
student_D_input = M * Y + (1-M) * S.detach()
target_D_input  = M * T + (1-M) * S.detach()
```

These composites exist **only inside the loss**. Previews and serving retain the complete generated Y. Both D inputs share precisely the same protected source pixels. Consequently the feature term's gradient with respect to Y is zero outside M; the existing protected-pixel and fresh-latent losses remain responsible for preservation. Parameter sharing can still change the generated face, so this is not an identity guarantee. Target/source mismatch at a hand-drawn mask boundary can itself affect features; masks must remain independently reviewed.

Compute/cache target features with `torch.no_grad()` and detach all targets, masks and normalization constants. Do **not** use `no_grad`, `detach`, or inference mode on the student feature branch: frozen D weights still need input gradients. No graph or D gradients should accumulate in a cache.

For each tap l, use area-resized fractional masks `C_l` and `R_l`. Normalize each region by `channels_l * sum(region_mask_l)` per image, then average the batch. Require positive masses; do not silently substitute another region. Preserve the current equal collar/rest weighting:

```text
A_l(Q) = sum(Q * abs(F_l(student_input)-F_l(target_input)))
         / (channels_l * sum(Q))
s_l = max(1, RMS(F_l(target_input), weighted over clothing))  # detached scalar
L_feature = mean_l [ (0.5*A_l(C_l) + 0.5*A_l(R_l)) / s_l ]
L_total = existing_region_protected_fresh_losses + 0.05 * L_feature
```

The detached RMS floor and coefficient 0.05 are explicit proposed experimental choices, not JoJoGAN defaults or tuned values. The floor avoids amplifying near-zero feature scales. Log raw/normalized components and scales. Fractional masks preserve tiny-region mass on downsampling; do not threshold them back to binary. Receptive fields mix surrounding context, so feature masks localize a comparison approximately, while the input composite exactly controls where image gradients enter.

## Memory and compute

The current CPU 2 baseline averaged 4.62 seconds/update with 5.29 GiB peak RSS in the small thread benchmark. That does not estimate D-augmented performance. Even frozen D retains high-resolution activations for input gradients. Conditional on the channel sizes above, detached selected features alone take 48 MiB per pair in FP32; six cached targets take 288 MiB, excluding G, D, intermediate activations and convolution workspaces. The first unselected512px output alone is 64 MiB. This is a storage calculation, not a peak prediction.

For a first smoke, cache only the selected target pair, run paired pixel+feature backward together, then release that graph before the unchanged fresh-preservation forward/backward. Do not retain a graph between updates. Load/extract the original pkl once in a separate guarded process, then persist a research-only D state and constructor/provenance to avoid keeping G/D/G_ema pickle objects beside the student. Keep upstream licenses. Activation checkpointing or separate recomputed feature backwards are possible later tradeoffs; introducing them in the first test would confound the baseline comparison.

## Cheaper alternative and decision after 600

A collar-boundary term needs no extra model. Define an 8px band around C, intersect it with M, and compare horizontal/vertical first differences of Y and T only where **both** neighboring pixels lie in M and the edge touches the band. Normalize by valid edge count, with a fixed modest coefficient (e.g. 0.05); reject empty support. This directly tests whether an otherwise correct tab is merely blurred. It will not supply fabric semantics and can overfit imperfect traced edges.

At 600, inspect native source/student/target for all six train seeds, untouched028, and the existing fixed unfiltered random set. Record tab/rest/protected pixel errors, boundary error and visible collar geometry separately. If collars remain absent on unseen seeds, do not interpret a sharper training collar as the solution. If only boundary softness remains, test the boundary term first. Choose frozen-D features only for broader local clothing structure/detail failure that the pixel/boundary explanation does not account for, with faces otherwise preserved.

## Conditional ten-update smoke after the current evaluation

The existing goal authorizes this local research; the decision to run it depends on the completed six-pair evidence and resource guard, not another user permission prompt. First add tiny CPU tests for target detach, nonzero finite student-input gradient confined to M, unchanged D/source/frozen state, feature shapes, empty-mask rejection, and no validation sampling. Then, after competing evaluation finishes and a fresh ≥25% memory guard, make **two isolated branches from the same immutable six-pair 600 student/optimizer/RNG state**, retaining the original source teacher. Record this as an objective-change experiment with explicit parent lineage, never an exact resume of the old objective.

- Baseline branch: five unchanged updates.
- Feature branch: five updates with the fixed feature term; same predetermined calibration pair and same fresh-latent draws. **Ten optimizer updates total**, no parameter sweep.
- Measure full update time, peak RSS, separate components, finite gradients, exact D/source/frozen-state hashes, and native full-output previews at 0/1/5. Verify protected-input feature gradients are zero in the tiny test; compare actual protected-face drift here.
- Stop on nonfinite values, changed supposedly frozen state, or peak RSS exceeding a proposed 10 GiB smoke ceiling. The ceiling is an abort criterion, not a preallocation guarantee. Do not automatically retry larger.
- A longer comparison is justified only if the feature branch beats the matched baseline on tab/local-structure error and native detail **without visible face degradation**, protected MAE rises by no more than 0.005 on the `[-1,1]` scale, and timing/memory are acceptable (proposed ≤2× baseline update time and ≤10 GiB peak). These are predeclared engineering gates, not proof of perceptual quality. Failure rejects this particular configuration; success still requires a longer controlled held-out/unseen-seed evaluation before any adoption.

## Static provenance

Read only: region trainer `7842039b65b0230dcc745c023761350da7bdeb49720fdac30f679fdaa38cb821`; base helpers `e17a9618ffc3e44dc57ef41f2a43cd41b08c2a41b468eb5956c7a722b58ea8ba`; NVIDIA networks `df49fc78a0364015a09b592bb43546af83a6cdaad753d89a2a1a69bd6e171c24`; legacy loader `6496d8ebe84d1e515832eada2c9688968ce23c6e13b1e2bc14e673ad96bb562a`. Official JoJoGAN commit `ff33d3f541a8deccc6df85f053286ac8c88f7526`, model.py SHA `f3d41ead5619deb859cb6cae393ea6e4d9ca79a1a353cc83e07a28cbf4f7cdbe`, notebook SHA `cc004c7be12fdcc5d985531452648759ae084de63bb03deaed727ca6dc3354a2`. Existing NVIDIA/source-asset licenses remain unchanged. No active file or production configuration was modified.
