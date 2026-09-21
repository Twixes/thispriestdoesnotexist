# Completed 50-update collar-surround probe

Neither branch produces publication-quality priests. The surround branch is useful diagnostic evidence that emphasizing the exterior collar ring changes the intended region, but it is not a visual winner or justification for a long extension.

Both branches completed exactly 50 updates from the same step-600 student and copied Adam state, with all 50 pair draws and fresh latents equal. The launcher exited 0 in 585.09 seconds; baseline and variant workers completed in 305.03 and 279.17 seconds. Their reported peak RSS was 5.43 and 5.41 GiB respectively. These are local research training measurements, not serving memory estimates. No restart, additional model inference, or production approval occurred during this review.

`compare_saved.py` independently verified the parent checkpoint SHA, all six complete checkpoint file hashes, all 132 native PNG hashes/dimensions/modes, both 50-draw schedules, matching initial student states, and exact step-zero evaluation rows/images. Runner-recorded source/frozen invariants and final full-checkpoint deserialization checks passed. This audit itself never imports Torch. The full evidence is in `comparison-audit.json`; gray images are RGB-mode grayscale. `comparison-diagnostic.png` is a labeled resized contact sheet, never a generated output or input to training.

## Region errors

Equal-image means over the six TRAIN pairs, luminance L1 on [-1,1]. Lower is closer to the annotated target in that mask; protected uses the original source. Multiply by 127.5 for uint8-equivalent MAE. These are not identity, realism or collar-recognition scores.

| Branch / added updates | White tab | Exterior 30px ring | Remaining clothing | Protected source |
|---|---:|---:|---:|---:|
| Shared parent / 0 | 0.06298 | 0.70478 | 0.22065 | 0.02124 |
| Baseline / 25 | 0.06028 | 0.69421 | 0.20730 | 0.02154 |
| Surround / 25 | 0.12640 | 0.51737 | 0.19375 | 0.02330 |
| Baseline / 50 | 0.06096 | 0.67924 | 0.19494 | 0.02183 |
| Surround / 50 | 0.11911 | 0.40530 | 0.18128 | 0.02409 |

At 50, the variant reduces ring error 40.3% and remainder error 7.0% versus the baseline, while increasing tab error 95.4% and protected error 10.4%. Every one of the six training pairs shows that same direction in all four regions. The ring is the square radius-30 exterior of the raw tab trace, clipped to clothing; this is one ring region, not separate 5/15/30px diagnostics.

HOLDOUT 028 is reported separately: clothing L1 goes 0.84948 at the shared start, 0.83795 at baseline50, and 0.78779 at surround50. Protected L1 is 0.02693, 0.02630 and 0.02791 respectively. It has no tab annotation, so no tab/ring/remainder claim is possible. Four fixed unseen portraits have no target-region metrics.

## Native visual review

All 11 saved grayscale portraits at both 25 and 50 for both branches were viewed individually at 1024px. Hash-scoped per-image observations are in the four `visual-review-*.json` files. None shows a new broad collapse of facial anatomy across these fixed samples. This small comparison does not establish distributional diversity or exact identity preservation.

The baseline retains broad white neck patches and rough cloth boundaries. The variant darkens the edges of those patches, especially calibration-original, but retains skin-like texture and original garment remnants instead of forming clean fabric and a distinct tab. On 030, pronounced pitted/scratch-like dark neck texture remains and becomes darker/more extensive in the variant; the left-edge extra person predates both branches. b2-055 retains a ragged bright strip and glossy smeared garment. b2-051 remains the closest partial collar-like patch, but its rough skin/garment edge is not a quality pass. 000 and b2-020 still have bright skin/neck regions without clear clerical construction.

Held-out 028 gains no convincing clerical tab. None of the four fixed unseen samples gains a convincing collar. Hats and extra people remain in unseen000 and unseen003; age is not established for unseen003. These are diagnostic samples, not eligible/hot-priest successes.

## Ranked next-step judgment

1. Do not promote either branch or call the surround objective a winner. The substantial ring improvement came with worse tab agreement in every training pair and no clear visual collar improvement. Do not extend this variant merely because its aggregate clothing error is lower.
2. Before another long run, review neck/clothing boundaries against source-face preservation and the edited target geometry, particularly calibration-original and 030. Darkening textured source neck pixels can lower ring error without synthesizing plausible cloth; this is a plausible explanation, not a proven cause. Preserve raw target/source evidence and test mask changes separately rather than silently modifying the current manifest.
3. If a further loss-only probe is chosen, a controlled alternative is tab 0.5 + surround 0.25 + remainder 0.25, keeping tab emphasis at the parent's weight and splitting the remaining half. Start both controls from the same completed parent with identical draws and compare the same regions/native images after a short fixed budget. This is an untested hypothesis, not permission to run or an assurance it fixes texture. Require recognizable cloth/tab improvement, no worsening protected facial/neck anatomy, and held-out improvement before scaling.
4. Generalization and eligibility remain separate unsolved gates. More pairs may help coverage, but these six-pair results do not prove that adding pairs will solve the texture or unseen-collar failures. Keep the held-out and unfiltered unseen evaluations, and do not substitute a curated success catalog for a generator.
