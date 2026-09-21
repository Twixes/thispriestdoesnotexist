# Convolution layout decision

The current `adam-native1024-probing500-v1` run is a control following the released source's convolution layout. It is not a completed paper-aligned importance-selection experiment. Its running code and objective remain frozen.

The [paper, pages 6–8](https://openreview.net/pdf/44f72b6c163a1eaea103348545b78dfbc262d4d5.pdf) defines each convolution kernel by its output channel. Its modulation uses an outer product of an output-channel vector with a flattened input/kernel vector. Thus one output-vector entry controls one kernel. Main adaptation preserves important base kernels through restricted modulation and directly trains the other kernels.

Our independently checked source uses the reverse outer-product order before reshaping. With two outputs and four flattened inputs, its rows are `[u0*v0, u0*v1, u1*v0, u1*v1]` and `[u2*v0, u2*v1, u3*v0, u3*v1]`. Both rows depend on both output-vector entries. A source-style mask on `v0` therefore does not isolate original weight row zero.

Decision: keep this source-layout run as an explicitly labeled control. For literal per-output kernel importance and selective adaptation, use the existing `output_rank1` implementation in a separate, fresh-source, pinned experiment and recompute importance there. Do not transplant or transpose the current run's learned factors or scores. Keep source-model, data, seeds, native resolution, optimizer, losses, and resource limits matched for the comparison. This changes training parameterization; inference remains an ordinary unconditional generator after folding.

Before launching that separate experiment, validate its native initial equivalence, trainable-factor gradients, frozen weights, regularizers and exact folded export. A successful source-layout control does not prove those outcomes for the alternative. Neither finite losses nor either layout guarantees photographic quality or adult priest appearance.

## Checkpoint-100 visual evidence

The source-layout control has reached its first verified checkpoint. All four raw outputs show strong grain and color/exposure degradation; EMA remains closer to the prior, without establishing priest conversion. Tensor diagnostics find a large change in the additive noise-strength offset at the highest resolution. This is a plausible separate failure mechanism, not proof of cause. The layout correction remains mathematically necessary for literal output-kernel masks, but should not be assumed to fix the visible grain. Before another long probing run, perform a controlled G-only noise-offset ablation after the current job is terminal and memory preflight passes. Preserve the matched raw render and every ablation image.

The remaining control was explicitly stopped after128 completed iterations. Its verified100-iteration checkpoint, full terminal logs and reason are preserved. This is an early quality-driven stop, not a completed500-iteration control. The next experiment isolates noise offsets using the saved checkpoint.

## Completed noise-offset intervention

`adam-native1024-noise-ablation100-v1` completed with exact unchanged raw reproduction for all four retained latents. Resetting only the highest-resolution conv0 noise offset substantially reduces dense fine grain in the inspected native images. Resetting all17 offsets does not restore original photographic quality: residual tone, contrast and texture problems remain, as does the non-priest source population. All original pretrained noise strengths remain present; only learned additive offsets were reset.

Next bounded stability experiment: fresh original weights, corrected output_rank1 layout, and17 G noise offsets fixed at their initial zero values throughout optimization. This deliberately differs from upstream AdAM. It tests whether preserving pretrained noise avoids this observed failure while the other adapters learn; it is not a claim that noise was the only cause. Changing layout and noise policy together does not isolate their separate training effects. Preserve all original evaluation latents and early native previews; do not reinterpret this diagnostic as full500-iteration probing or main adaptation.

## Bias-family intervention and revised stability run

The completed five-arm bias diagnostic isolates a second substantial contribution: across all four fixed latents, resetting synthesis activation-bias offsets on the common noise-reset background produces the largest recovery of natural shading and detail. ToRGB bias contributes a smaller global cast; affine/mapping bias resets do not materially rescue these images. Full evidence is under `research/reviews/adam-bias-ablation/`. This remains a conditional inference result, not evidence of successful training.

Consequently the prepared noise-only100 variant was not launched. The new `probe_output_rank1_fixed_offsets100.py` starts from original pretrained weights and holds all34 noise/activation-bias offsets at zero while keeping the original parameters and remaining adapters. It uses the same native resolution, four review latents and full regularized probing losses, with snapshots at10/25/50/100. Layout and offset policy both differ from the stopped source control; do not claim single-factor attribution from the training comparison.
