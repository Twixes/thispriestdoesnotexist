# White-tab weighting diagnosis and optional objective proposal

**The broad clothing objective substantially dilutes the small white tab's pixel-gradient budget.** This supports a controlled region-balanced loss experiment; it does not prove weighting is the only reason the tab failed to appear. No trainer was edited, model evaluated, gradient computed, or training run started for this analysis. Only existing PNGs were read.

Native inspection of the original paired outputs at steps 150, 175 and 200 shows a recognizable intact face and progressively darker clothing, but an open dark neckline rather than a distinct white clerical tab. The target has a clearly light tab immediately below the chin. This is consistent with the parent's step-200 review.

## Measured error

The target is resized from native 1254 to 1024 with Lanczos, then source/target PNG values use the trainer's fixed .299/.587/.114 grayscale coefficients. The white-tab polygon was traced manually on the target. Its effective region is intersected with the **unchanged approved clothing mask**, protecting the original jaw; the remainder is clothing minus this intersection.

| Paired-output step | Tab MAE (0–255) | Rest-clothing MAE | Broad-clothing MAE | Generated tab mean | Tab share of total clothing absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 150 | 136.41 | 13.94 | 20.18 | 39.82 | 34.45% |
| 175 | 128.91 | 12.23 | 18.18 | 47.31 | 36.15% |
| 200 | 119.09 | 11.19 | 16.69 | 57.14 | 36.37% |

Target mean in this effective tab region is **176.22**, so the brighter step-200 neck pixels still fall well short. At step 200 the tab contributes **6.07** of the broad **16.69** MAE; the rest contributes **10.62**. The tab's mean error is **10.64 times** the rest. Its 7,198 pixels occupy only **5.097%** of the 141,212-pixel clothing mask. This is a 0.687% region of the full image. Full manually traced tab MAE at step 200 is 117.62; 90.85% of that trace lies inside the existing clothing mask.

These are post-update, clipped uint8 PNG measurements, **not exact pre-update unclipped training-tensor losses**. `analysis.json` records both 0–255 and [-1,1]-equivalent scales, means, area/error contributions, input/source hashes and all three snapshots. `analysis-before200.json` preserves the initial partial audit before the last PNG was available. `analyze.py` reproduces the PNG-only measurement.

## Minimal proposed objective

Let M be the existing clothing mask, T the manual tab trace, C = M intersection T, and R = M minus C. Keep source-protected and fresh-latent preservation losses and their weights unchanged. Optionally replace the single clothing MAE with:

`L_clothing_balanced = 0.5 * mean_abs_error(C) + 0.5 * mean_abs_error(R)`

Each mean is independently area-normalized per image, then averaged over batch. Require nonempty C and R; reject missing training traces or empty intersections rather than silently falling back. This is equivalent to collar/rest weights 1/1 with an overall clothing multiplier of .5. Using weights 1/1 **without** that .5 would also double the total clothing coefficient against face preservation, confounding the first comparison.

Under current plain L1, nonzero-error pixels have equal-magnitude output-space derivatives. The tab therefore receives only about 5.1% of their aggregate magnitude despite its large residual error. The balanced version assigns 50% to each region: tab per-pixel derivative magnitude becomes **9.81 times** baseline, rest becomes .527 times. This is output-space loss algebra, **not measured parameter-gradient dominance**: network Jacobians can alter the actual update direction. The tab is already 36% of the scalar residual, so it is inaccurate to say the current loss completely ignores it.

The frozen mapping/blocks <=32 preserve the coarse source prior. Blocks 64–1024 still receive derivatives through all the masked pixels and can alter clothing/detail, but freezing may also make a new neckline shape harder to represent. This measurement does not distinguish limited capacity, shared-weight conflicts, insufficient steps, or L1 texture blur from region imbalance. A balanced objective adds no model or additional forward pass; compare its tab brightness/shape and protected-face drift against the pixel baseline before relaxing any frozen blocks or masks. Only a visible, well-formed tab on the full generated image demonstrates the intended improvement.

## Proposed manifest, without changing approved inputs

`manifest-proposed.json` copies all approved `paired4` source/target/latent paths, splits and clothing polygons unchanged. It adds manual `collar_polygons` only to the three training entries. The held-out 028 record is **exactly unchanged** and gets no new beard/tab target. This proposal uses **version 2 intentionally**, so the existing version-1 trainer rejects it instead of silently ignoring the new objective. A separate explicitly enabled trainer experiment must implement/validate this feature before the manifest can be used. The current original trainer and both existing manifests are untouched.

| Training pair | Effective tab pixels | Tab/clothing area | Trace retained after protecting original face | Balanced tab derivative multiplier |
| --- | ---: | ---: | ---: | ---: |
| calibration-original | 7,198 | 5.097% | 90.85% | 9.81x |
| 000 | 4,248 | 3.482% | 55.40% | 14.36x |
| 030 | 4,204 | 6.470% | 85.48% | 7.73x |

`tab-contact-sheet.png` and native-size per-target diagnostics were visually reviewed: cyan traces existing clothing boundaries, green the target tab, yellow the effective intersection. 000 loses much of the tab's upper portion because the existing source-jaw protection is deliberately retained. This may limit collar height; it is not permission to train through the chin. The traced regions follow the visible cropped tabs and impose **no whole-collar-in-frame requirement**. 028's original beard and hard held-out status remain intact.

The candidate is research-only and pending the parent's objective decision. The evidence supports changing how clothing pixels are weighted, while still requiring separate validation of unseen identities and adult/hot-only sampling. It does not establish production quality or generalization from these three training identities.
