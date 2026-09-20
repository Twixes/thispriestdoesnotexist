# Held-out alignment review

All twelve new 1254 × 1254 originals were processed into `aligned256/` using the **same** recipe as the current 110-image training set: PIL Lanczos resize to 256, the same YuNet model/settings, the same collar heuristic and fallback, eyes42 affine geometry, OpenCV Lanczos4 resampling, and reflected borders. The transform implementation was checked against every accepted training matrix: maximum absolute difference **0.0 across all 110**. Pure helper functions are reused from `research/alignment/analyze.py` without executing that script's top-level dataset writes.

`alignment-manifest.json` contains input/output hashes, input dimensions, face confidence/landmarks, collar detections, exact transforms, crop metrics, reflection fractions, code/model hashes, and dependency versions. Every output was opened and verified as 256 × 256 RGB; all twelve file hashes differ. This directory is explicitly marked `validation_only`, with `training_use_permitted: false`. No training data, source-original manifest, model, or running training process was modified.

All twelve native-resolution before/after panels were visually inspected in `alignment-comparison.png`. They retain visible white collars and bare heads without new facial warping. The collar detector failed for **003**, whose existing geometric fallback still leaves its collar visible. No special correction was applied.

The shared reflection policy creates a meaningful diagnostic caveat:

| ID | Reflected output fraction | Observation |
| --- | ---: | --- |
| 002 | 11.27% | Detached reflected hair at the upper edge |
| 004 | 5.33% | Reflected background boundary |
| 005 | 16.26% | Reflected hair at the upper edge; greatest padding |
| 006 | 9.67% | Detached reflected hair at the upper edge |

Mean reflected area is **4.60%**, compared with **1.88%** in the accepted eyes42 training manifest. The other eight held-out images are at or below 5%; 003 uses the collar fallback and has 4.95% reflected area. These input-dependent differences remain even though the algorithm is identical. There were no quiet crop changes, inpainting, or adoption of the separate zero-reflection geometry.

The dataset is suitable for a transparently reported preliminary held-out discriminator diagnostic, with per-image results and a padding-stratified comparison alongside the full-twelve aggregate. A train-versus-held-out score difference can reflect these border/pose differences as well as memorization; twelve synthetic portraits are not a broad independent generalization benchmark. Do not silently exclude the four harder images or treat their lower scores as conclusive overfitting evidence. This is not approval of the reflected images as production outputs.

Reproduce from the repository root:

```sh
research/alignment/.venv/bin/python research/data/validation12/align_validation.py
```

The contact sheet, comparison sheet, and `alignment-review.json` preserve the review scope and caveats. The originals, prompts, and parent-managed source provenance remain untouched.
