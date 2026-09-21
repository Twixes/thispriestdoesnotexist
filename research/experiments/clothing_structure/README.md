# Unused clothing-structure loss components

These functions are isolated from all trainers and deployment paths. They implement the optional feature/boundary terms in `research/reviews/frozen-d-clothing-feature-design.md`. No real weights are loaded here. The six-pair evaluation must determine whether either experiment is useful before integrating or running it with FFHQ1024.

`losses.py` exposes:

- `FrozenDPrefix(D)`: freezes/evaluates the supplied, independently verified NVIDIA RGB1024 discriminator and retains references to `b1024`, `b512`, `b256`. Selected outputs are the last two blocks' spatial256/128 features. The full D is frozen **in place**; pass only a separately owned research model. No hooks, final logits, minibatch-statistics epilogue, optimizer or loader are present. Actual constructor/channel/weight provenance still needs verification before a real-weight experiment.
- `feature_reconstruction_loss(prefix, generated, target, source, clothing, collar)`: returns an **unweighted** loss and detached scalar diagnostics. Both feature inputs share detached source pixels outside clothing. Target features run under `no_grad`; the generated feature branch retains image gradients through the frozen prefix. Source/target/masks are detached. Each feature region is independently normalized by its fractional area and channel count, with equal collar/rest weights, detached per-image target RMS normalization floored at1, then layer and batch means. Full generated tensors are not modified or replaced. This minimal component recomputes target features per call and retains no cache/graph between calls; target caching is a later optimization requiring provenance checks.
- `collar_boundary_loss(generated, target, clothing, collar, radius=8)`: returns unweighted first-difference error and valid-edge counts. The radius8 dilation/erosion band is intersected with original clothing; compared horizontal/vertical edges must have both endpoints inside clothing and touch that band. It uses detached target/masks and no extra network.
- `region_mean(values, mask)`: per-image, channel-and-area normalization, including fractional downsampled masks.

Inputs must be finite FP32 NCHW tensors, with matching dimensions/devices and one or three channels. Grayscale uses the same fixed `.299/.587/.114` conversion as the existing paired trainer. No clipping, image warping or quantization is introduced. Binary input masks must define nonempty collar, remaining clothing and protected regions per image, with collar a strict subset of clothing. Empty feature mass and empty valid boundary edges fail explicitly. Feature masks use area interpolation without thresholding.

The proposed coefficient **0.05** for either optional term remains an unvalidated experiment choice and is deliberately not applied inside these functions. No caller or active trainer was added. Existing protected/fresh losses, LR, generator freeze policy and complete generated output are outside this component's responsibility and remain unchanged.

## Tiny verification

Run from the repository with its existing research environment:

```sh
research/.venv/bin/python research/experiments/clothing_structure/test_losses.py
```

Seven tests exercise real nonzero finite gradients through a randomly initialized **64px NVIDIA discriminator with at most16 channels**, plus an identity feature fixture for exact normalization checks. An epilogue that raises on invocation proves it is not executed. They verify:

- Image gradients are exactly zero outside clothing and nonzero inside it.
- D parameters/buffers, source/target tensors and complete generated tensors remain unchanged; D/source/target/masks receive no gradients.
- A one-pixel collar and much larger rest region each receive half the summed gradient for equal feature error. Fractional masks and unequal batch areas normalize correctly.
- Invalid/empty/full/nonfinite masks, zero-support edges, wrong discriminator taps/resolution, and accidentally trainable/train-mode prefixes fail.
- Boundary loss is invariant to a constant intensity offset, as a first-difference comparison should be.

No FFHQ source, G, pretrained D, MPS device or optimizer is used. `test.log` and `test-evidence.json` retain results, CPU1/interop1 settings, peak RSS, and hashes of components/tests plus the full vendored Python/native source inventory. These tests establish mechanics, not real-weight memory requirements, collar quality or generalization.

The algorithm review pins official JoJoGAN sources and explains deviations; these components do not copy JoJoGAN network code. They rely on PyTorch and accept the existing NVIDIA architecture, whose upstream license remains intact. No model or dependency was downloaded. Source and asset licenses are not reclassified.
