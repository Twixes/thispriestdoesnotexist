# Optional 512-pixel face restoration after a successful generator

Recommendation: retain GFPGAN v1.4 as the first bounded enhancement experiment **after** the generator produces diverse, anatomically sound adult male priests. CodeFormer is a useful comparison with an explicit fidelity control, but its non-commercial license and larger dependency surface make it less attractive as the first production candidate. Neither has been installed, downloaded, benchmarked, or used to approve an output here. This review fetched only documentation, source text, and release metadata.

A pipeline that samples a fresh latent, runs our trained StyleGAN, then applies a fixed pretrained restorer remains actual generative inference on every successful request. Its final output would be jointly determined by both models. It should be described and versioned that way, with both weight hashes and preprocessing settings included in the review record. It does not turn a 256-pixel generator into a natively trained 512-pixel generator.

## What these models offer

| Candidate | Relevant mechanism | Main concern for this project |
| --- | --- | --- |
| GFPGAN v1.4, clean architecture | One forward pass through an encoder and a pretrained facial GAN prior; restores a 512 × 512 face | Added facial detail may alter identity or produce a smooth, generic face; it does not learn priest clothing |
| CodeFormer | Predicts discrete facial codes with a transformer; `w` controls fidelity versus perceptual quality | Lower fidelity can replace features; a sharper result can disguise the underlying generator's failures |
| Lanczos 2× control | Interpolation without a second learned face prior | Adds no genuine texture; useful to establish whether restoration visibly helps rather than merely enlarges |

GFPGAN's authors describe restoration through a generative facial prior and a single forward pass, rather than image-specific optimization. Their implementation has a clean variant without custom CUDA extensions. The v1.4 update is intended to improve detail and identity over v1.3, but this is not evidence about our generated images. [GFPGAN project](https://xinntao.github.io/projects/gfpgan), [official model documentation](https://github.com/TencentARC/GFPGAN/blob/master/README.md).

CodeFormer's fidelity parameter ranges from zero to one: larger values favor preserving input information, while smaller values favor its learned prior. An initial comparison should use 0.7 and 1.0, not optimize solely for sharpness. [CodeFormer project](https://shangchenzhou.com/projects/CodeFormer/).

These are restoration models, not domain or attractiveness classifiers. There is no reason to expect them to add a correct white collar, remove every hat, enforce adult male appearance, or ensure the requested aesthetic. A deterministic restorer cannot recover latent variation that a collapsed generator has stopped expressing. Even when it changes appearances, those changes are not evidence that the source generator regained diversity. These are consequences of the proposed pipeline, not claims of a tested outcome.

## Framing and integration

Both reference implementations restore aligned 512-pixel facial crops. Our collar-preserving portraits deliberately contain more torso than a canonical close face crop. Passing the whole portrait as `has_aligned=True` would therefore test a different input geometry, potentially changing collar and hair content. A legitimate full-image evaluation should restore only the aligned facial region and map it back into a 2× interpolated original, keeping the collar and background available for direct inspection. Boundary seams, inconsistent grain, changed apparent age, and artificial skin require review at the final displayed size. [GFPGAN implementation](https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/utils.py).

The stock helper constructors initialize detector/parser machinery even when aligned inputs are supplied; using their command-line scripts can trigger auxiliary downloads. The smallest **core-model** test should instantiate only the restoration architecture and use the existing YuNet alignment tooling. A later end-to-end test must include the actual crop, mask, inverse warp, monochrome handling, and WebP encoder. Disable Real-ESRGAN background enhancement initially; it adds a separate model and CPU cost. Stock scripts can catch restoration failures and return the input crop, so a benchmark must explicitly record failures instead of treating that fallback as a successful restored image. [CodeFormer inference code](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py).

## Licensing and artifact size

GFPGAN's top-level code is Apache-2.0 **with third-party exceptions**. Its license file includes separate BasicSR/StyleGAN, NVIDIA, DFDNet, facexlib, and other notices; the heading alone is not sufficient to label the whole dependency or model bundle Apache-2.0. Preserve applicable notices, record the exact upstream revision and weight source, and check the components actually packaged. This repository's MIT license does not replace those terms. [GFPGAN license](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE).

CodeFormer is under S-Lab License 1.0, permitting redistribution and use for non-commercial purposes with retained notices; commercial use requires contacting the contributors. Do not present a CodeFormer-containing distribution as wholly MIT or assume the personal site's future uses are unrestricted. [CodeFormer license](https://github.com/sczhou/CodeFormer/blob/master/LICENSE).

Official GitHub release metadata reports:

- `GFPGANv1.4.pth`: **348,632,874 bytes** (about 332.5 MiB). [Official release](https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4).
- `codeformer.pth`: **376,637,898 bytes** (about 359.2 MiB). [Official release](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0).

Both exceed the present 50 MB download limit, so neither checkpoint was downloaded. These are checkpoint-file sizes, not runtime-memory measurements. A subsequent experiment should use only the official release, record its SHA-256, preserve its license/provenance, and store any retained model via Git LFS.

## Runtime expectations: unmeasured

GFPGAN's helper defaults to CUDA when available and otherwise CPU; MPS must be supplied explicitly. CodeFormer's official device helper explicitly selects available MPS. Neither observation establishes compatibility or speed with this project's PyTorch 2.14 environment. [GFPGAN device selection](https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/utils.py), [CodeFormer device selection](https://github.com/sczhou/CodeFormer/blob/master/basicsr/utils/misc.py).

For scheduling a first experiment only, reserve roughly **1–10 seconds per 512-pixel crop on two CPU threads**, **0.2–3 seconds warm on MPS**, and **1–4 GiB of additional process/device memory** for either candidate. These deliberately broad engineering allowances are **not measured estimates of these models on this machine**, guarantees, or paper benchmark results. Initialization, compilation, auxiliary models, thread contention, and allocator behavior can exceed them. A more precise number before running the model would be misleading.

The existing 3 GiB hosting candidate may not accommodate generator plus restorer; total resident and peak memory must be measured with both loaded. MPS results would inform local experimentation only, since the proposed inexpensive hosting uses CPU. Added CPU time increases request queuing and billable active use, so the current sub-$15 hosting preference cannot be validated from generator-only measurements.

## Smallest useful benchmark once downloading is authorized

1. Start with only GFPGAN v1.4, an isolated environment, CPU FP32, two threads, batch one, inference mode, and no background upsampler. Download only its official 349 MB checkpoint; prevent implicit helper/model downloads. Do not change active training or serving code.
2. Use four fixed existing pilot outputs: two of the least malformed samples plus two obvious failures. Save their generator checkpoint hash and seeds when available; if extracting tiles from a saved sample grid, record the grid hash and tile positions instead. Label all four as unapproved pilot images. Their purpose is to expose the restorer's behavior and cost, not to rehabilitate a rejected checkpoint.
3. Reuse current YuNet landmarks to prepare aligned facial crops; save the transforms. Compare 512-pixel Lanczos interpolation with a single GFPGAN pass. Run one cold invocation, two warmups, then ten timed single-image passes cycling over the four crops. Record initialization time separately, all per-image latencies, median and maximum, peak RSS, parameter bytes, package versions, and failure count. Ten samples do not establish a reliable p95.
4. Save paired originals, interpolated controls, restored crops, and reconstructed full portraits. Review eyes, teeth, ears, hairline, facial asymmetry, changed apparent age/identity, skin texture, collar boundaries, monochrome consistency, and crop seams. Inspect the two malformed inputs explicitly; a cosmetically smoother failure remains a failure.
5. Only if this improves already-sound faces at acceptable measured cost, consider CodeFormer as a second independent comparison. After a generator passes its own quality and diversity review, evaluate at least 64 newly sampled seeds through the complete deployed pipeline, including the encoded WebP. Compare diversity and nearest-neighbor behavior before and after restoration; pairwise image metrics alone do not prove unique identities or attractiveness.

Production approval must cover the exact generator, restorer, transforms, output encoding, and an end-to-end random sample review. Sharpness scores, face-detector success, or a selected attractive example are insufficient. None of the current rejected pilots should be promoted on the strength of a restoration demo.
