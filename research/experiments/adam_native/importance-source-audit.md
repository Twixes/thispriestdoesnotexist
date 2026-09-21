# AdAM importance and adaptation-mask audit

2026-09-21. This is a read-only source analysis and prospective design. No model was loaded, no gradient/model calculation or training was run, and no running source was changed. A small index-support proof used Python lists only. The live500-iteration source-layout probe remains an experiment with unknown outcome; this document does not infer completion from its launch.

**Recommendation:** retain the current `source_flattened` probe as an executable-source control. Before claiming paper-aligned output-kernel importance or applying paper-aligned masks, run a separately pinned fresh-source `output_rank1` probe and estimate its own importance. Do not transpose or relabel current factor scores. Generic accumulation mechanics can be shared, but learned scores, checkpoints and mask policies must identify their layout and checkpoint precisely.

## Evidence and scope

Official source is pinned to `yunqing-me/AdAM@6428e99cfb36bc8bda3506350824f0c7dac9a5ad`, archived with license and file hashes in `upstream/`. Main evidence:

- [Model definitions](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/gan_training/models/model_adam.py): convolution factors at199–207 and432–459; FC factors at23–38; empirical gradients at794–817 and1096–1119.
- [Importance probing](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/AdAM_importance_probing.py): EMA evaluation and accumulation180–267, EMA updates412–414, initial checkpoint keys492–495, data sampler/loader53–60 and562–583.
- [Main adaptation](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/AdAM_main_adaptation.py): initial zero masks181–191; masks on D logistic/R1 and G adversarial/path updates233–254,297–318,347–382,411–446; score construction713–812; fresh-source reload827–834 and trainable families836–866.
- [Paper](https://openreview.net/pdf/44f72b6c163a1eaea103348545b78dfbc262d4d5.pdf), sections4.1–4.3: the output-length factor assigns one scalar to each output kernel, combined with the input/spatial factor through an outer product. This gives row-wise support when convolution weights are flattened as output-by-input/spatial. The mathematical proof below is our independent analysis of that definition against the actual source, not a reported paper experiment.

The earlier `source-audit.md` covers the full probing objective and known native-port differences. This audit distinguishes three things: empirical gradients in a specific parameterization, heuristics for reducing those gradients into scores, and the actual support of a mask on effective convolution weights. Their names are not evidence of equivalence.

## The convolution permutation changes what a factor controls

Let a convolution have `O` output channels and `K=I*k*k` flattened input/spatial coordinates. The native tensor is indexed `W[o,t]` after flattening. Let `u` have lengthK and `v` lengthO.

Paper-aligned output-row modulation is:

```text
M_row[o,t] = v[o] * u[t]
W_eff[o,t] = W[o,t] * (1 + M_row[o,t])
```

The pinned convolution code instead forms `outer(u,v)` and directly reshapes its contiguous storage to `[O,K]`. Consequently:

```text
n = o*K + t
M_source[o,t] = u[n // O] * v[n % O]
```

This is not an ordinary transpose into output-row semantics. WithO2,K4, the support is:

```text
source row0: u0*v0  u0*v1  u1*v0  u1*v1
source row1: u2*v0  u2*v1  u3*v0  u3*v1

paper  row0: u0*v0  u1*v0  u2*v0  u3*v0
paper  row1: u0*v1  u1*v1  u2*v1  u3*v1
```

Changing `v0` affects both source rows, versus only row0 in the paper layout. The Python-list proof enumerated `(u_index,v_index)` for each flat position and returned source support `[0,1]` and paper support `[0]`. No tensor library was imported for this proof.

For many real network convolutions, `K` is a multiple of `O` (for example512 input/output channels with3x3 kernels givesK4608,O512). Then `v[n%O]` selects strided coordinates within **every** output row. Zeroing `v[j]` and freezing original rowj does not isolate output kernelj: other factors still alter that protected row, and factorj also alters low-importance original rows. Effective-weight support, not vector length, is decisive.

Source FC uses `outer(v,u)` and already has output-row support; the discrepancy concerns convolution modulation. It does not mean the source-layout probe cannot learn useful changes. It means calling its `v` coordinates output-kernel identities, or its masks literal preservation of chosen kernels, is unsupported.

**Separate controlled rerun is necessary for the paper-aligned claim.** Changing layout after probing changes forward functions, Jacobians and the training trajectory. Even identical initial factor seeds do not make learned factors or their Fisher estimates interchangeable. A simple reshaping/transposition of scores cannot repair this. An alternative estimator based on actual effective-weight output support would be a different, separately specified method; it is not a shortcut to claiming the current factor estimates are paper-aligned.

## What the source actually estimates

After500 probing updates, the script evaluates **EMA G and EMA D**, with gradients enabled, one fake and one real image at a time. It does not estimate on raw G/D by default. One source sample uses one512-dimensionalGaussian z; the generator receives a single-style list, so **Fisher evaluation does not use training's0.9 style mixing**. Synthesis noise remains random by default. The real input comes from the same training loader, including horizontal-flip augmentation. No R1, path penalty, perceptual loss or image reconstruction term is included in the Fisher objective.

For pair `(z_n,x_n)`:

```text
fake_n = Gema(z_n)
L_G,n = softplus(-Dema(fake_n))
L_D,n = softplus(-Dema(x_n)) + softplus(Dema(fake_n))
F_theta = (1/N) * sum_n [grad_theta(L_n)]²
```

Square is elementwise; this estimates diagonal empirical squared-loss-gradient importance, not a full Fisher matrix. Source `estimate_fisher` differentiates the loss and squares each returned gradient; it does not build cross-parameter covariances. The saved mean gradients are separate diagnostic arrays and are not substituted for squared-gradient means.

For D, **the real and fake gradient terms are summed before squaring**. Squaring them separately drops their cross term and changes the estimator. Similarly, squaring a mean gradient across several examples can cancel opposing gradients; it is not the mean of their squares. G's loss uses D's image Jacobian, but only gradients for G parameters contribute to G importance. D importance treats the fake image as fixed with respect to D parameters; detaching it after G's gradient calculation is a memory optimization, provided both losses use the same fake and D state.

The source evaluates250 noise fixtures of shape[4,512], intendedN1000, and divides by `num_batch_fisher*batch`. A native evaluator should assert matching real/fake sample counts and divide by **actual successful pair count**, exactly once. Missing gradients and nonfinite gradients must not silently create zeros or increment counts. A mathematically zero but present finite gradient is a valid observation. Reject an invalid sample atomically so one parameter cannot receive a partial update before another parameter fails validation.

The original fixture script has an absolute `/_noise/...` output-path typo; the evaluator contains hardcodedCUDA loads and a256px reshape. These must not be copied. Use a separately frozen deterministic evaluation sample manifest with z hashes, real-image IDs/hashes, flips and synthesis-noise seeds. Keeping1000 pairs gives the published example count, but runtime must be measured separately. A smaller diagnostic prefix must disclose its actualN and must not be called an equivalent completed estimator.

A reproducible balanced plan cycling shuffled training IDs is reasonable; it differs from blindly continuing the loader position used during probing. The current native probe itself uses uniform-with-replacement image selection, whereas the original loader uses shuffled RandomSampler passes with drop-last batches. State that distinction rather than claim identical source sampling. No held-out image should enter importance estimation.

Factor importance is also parameterization-dependent even when the represented function is unchanged: replacing `(u,v)` with `(c*u,v/c)` leaves their product fixed, but scales the corresponding gradient squares by reciprocal factors. Pin initialization scale, factor convention and checkpoint; do not compare absolute importance magnitudes across arbitrary rescalings or call them invariant properties of a kernel.

The source accumulates NumPy arrays from model-gradient dtype, normallyFP32. CPUFP64 accumulation can reduce summation error without changing the mathematical objective, but must be labeled a numerical deviation. Save sums, counts, shapes, dtype and reduction version; normalization should be a derived view rather than destructive division that could accidentally occur twice after reloading.

## Raw, EMA and reset semantics

Original probing loads trainableG from source`g`, EMA G from source`g_ema`, and both D models from source`d`. It exponentially averages **parameters**, including factor vectors, after each probing iteration with fixed decay `0.5**(32/10000)`. An EMA of factor vectors is not generally equal to an EMA of folded effective weights, since modulation is bilinear.

Our native probe deliberately starts both G models from the verified original G_ema export and both D models from the verified D export. Native scaling is preserved; source FC's nonsquare scaling discrepancy is not copied. Both are already disclosed source differences. Do not change this initialization halfway through a controlled layout comparison. A new output-row probe should use those same original exports, twenty-image manifest, seeds, objective and horizon, changing only the explicitly pinned layout.

Importance evaluation must load the complete **Gema/Dema parameterized states** from the completed checkpoint and the matching modulation metadata. The final folded raw-G safetensors export is insufficient: it has neither EMA identity nor the factor coordinates needed for this estimator. Bind every accumulator to checkpointSHA, state key, layout, factor initialization, source/config hashes, sample plan and objective. Raw-G estimates may be a separately labeled diagnostic; do not pool them with EMA estimates or choose whichever looks more convenient afterward.

Main adaptation then **starts again from original source weights and fresh factors/optimizers**, loading only importance statistics from probing. It does not continue the probed raw model, EMA model or probing optimizer. In the native version, preserving original G_ema as the reset prior is a deliberate carry-forward of our disclosed initialization choice. D's trained probing epilogue also resets to original D. Starting from the probed model would be a separate warm-start variant.

## Score reduction and selection

The source creates three pooled score distributions; it does not use one threshold per layer, one threshold for the entire GAN, or a uniform layer-weighted average.

| Pool | Per-coordinate source score | Included source layers |
|---|---|---|
| G convolution | `mean(F_u) + F_v` |12 subsequent synthesis convolutions |
| G style-affine FC | `(mean(F_u) + F_v + F_b)/2` |Affine FCs of those12 convolutions |
| D convolution |Bias-bearing:`(mean(F_u)+F_v+F_b)/2`; bias-free skip:`mean(F_u)+F_v` |Two main convolutions plus skip in each of six residual blocks |

The three-term expression really divides by2, not3. Preserve it if reproducing executable source. D skip and bias-bearing scores share a pool despite different formulas/scales. `mean(F_u)` contributes the same offset to every output coordinate of one layer, so it affects cross-layer ranking; without bias, within-layer ordering is determined byF_v. Pooling concatenated vectors weights layers in proportion to their output-coordinate count.

The script applies `np.percentile(..., q)` and labels strictly-greater scores important/modulated; ties at or below the threshold enter the fine-tuned set. Thusq75 means approximately the top25%, not75% retained. Ties can produce fewer important coordinates, including an empty protected set when all scores match. Pin percentile interpolation behavior and report counts, ties and degenerate distributions rather than silently forcing a quota. The launch scripts useq50 andq75 for different datasets; neither is a validated priest threshold. Threshold selection must be a separate frozen decision after inspecting numerical validity and development-only evidence.

## Native1024 layer mapping and exact mask policy

For the current skip-G/resnet-D architecture, the natural resolution-generalized source selection would contain16 G synthesis convolutions at resolutions8–1024 and their16 affines, plus eight D residual blocks with24 convolution/skip tensors. These counts are derived from this architecture; discover and verify actual inventory rather than hardcoding them. Exclusions remain important:

- G mapping FCs, first4x4 convolution and all ToRGB layers receive modulation during probing/main adaptation but are not in the source's threshold pools. Their original weights remain frozen. The learned constant remains frozen.
- In selected G convolutions, high-score original weight output rows are frozen; low-score original rows may fine-tune. Affine weight/bias rows use their independently computed affine-pool mask, not the convolution's mask. All original synthesis activation biases and noise strengths remain frozen, while their additive modulation offsets remain trainable. They are not included in the Gconv score.
- D fromRGB original parameters stay frozen and its modulation remains trainable. Selected residual convolution/skip original high-score rows are frozen and low-score rows fine-tune. For bias-bearing residual convolutions, the same mask applies to the original activation-bias entries and additive bias offsets. D epilogue original conv/fc/out remains fully trainable, unmodulated and outside threshold pools.

At initialization, selected low-score `v` and bias-offset entries are set to zero. Their gradients are then zeroed after **every** relevant backward, including D R1 and G path regularization. Original high-score weight/bias gradients are zeroed. Shared `u` factors remain trainable. Under the output-row layout, zero low-scorev entries keep modulation absent on low-score rows while those original rows fine-tune. Under source_flattened, that row interpretation is false for the proven support reason.

Source uses fresh Adam, no weight decay, and fixed masks. In that specific setting, zero-gradient protected slices have zero optimizer moments and remain fixed. Reusing momentum from a prior optimizer, switching to AdamW, or changing masks can move supposedly frozen slices even with zero current gradients. Prefer an explicitly masked parameter construction or verify/protect parameter values and optimizer moments after each step. Test actual protected **original** rows for equality; important effective rows are allowed to change through their permitted modulation. Reusing the probing helper's whole-original-weight frozen checksum would incorrectly reject legitimate fine-tuning of low-score rows.

NVIDIA's4D convolution storage and unified bias parameters require an explicit inventory mapping; source name replacements and leading-singleton indexing cannot be copied. Every threshold-pool coordinate must map to exactly one target original output row and its intended factor/bias entries in the paper-aligned variant. Verify disjoint high/low sets, full coverage, bounds and matching dimensions. Mapping, ToRGB, first4x4, fromRGB and epilogue need explicit separate policies, not accidental fallback behavior.

## Validation that would support the next decision

The generic accumulator should be validated with tiny deterministic arrays: opposing gradients expose square-of-mean mistakes; real/fake cancellation exposes missing cross terms; partial batches expose nominal-count normalization; invalid samples expose partially updated accumulators; serialization/reload must preserve unnormalized sums and counts. A source-formula reducer should expose whether its output is a factor score or verified output-kernel score.

Before any main-adaptation run, a tiny output-support/Jacobian test must establish that varying selectedv_j only changes modulation rowj in the chosen layout. A masked update should keep protected original rows and low-score modulation entries exactly unchanged across adversarial and regularizer steps, including optimizer moments. Round-trip and folded-output checks should preserve the active offsets independently of requires-grad flags. These tests are prospective; no such main-adaptation implementation was executed in this audit.

For real estimates, report per-family/per-layer finite fractions, zeros, magnitude ranges, actual sample counts and selection counts. Use deterministic disjoint prefixes/halves to inspect rank and selected-set stability before treating a small estimate as reliable. Such checks diagnose estimator instability; they do not turn adaptation-loss gradients into an attractiveness or photographic-quality score.

Finally, review every fixed development-preview image at native display size against the original prior, with hashes and no cherry-picking. Assess eyes/facial structure, skin/hair realism, recognizable clergy appearance, hats, repeated identities and potential training-image copying. Fixed four-image previews are useful non-blocking feedback, not proof of broad diversity. Training targets remain training-only; final held-out evaluation waits until a candidate and its policy are frozen. Neither importance stability, finite optimization, collar learning nor exact folding overrides the owner's visual-quality rejection or proves sub500ms server generation.

**Decision boundary:** current source-layout results may inform whether the code's modulation is useful. They must not supply relabeled paper-aligned masks. A fresh output-row probe and its own EMA-based importance estimate are the required controlled precursor to a paper-aligned main-adaptation experiment. This audit implements and launches neither.
