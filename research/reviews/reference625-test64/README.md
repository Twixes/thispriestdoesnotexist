# Frozen reference625 restoration selector: held-out test

**The selector fails the required visual-quality gate.** At its preselected threshold of 0.9,
9 of 64 fresh attempts pass, but 2 of those deliveries have visible eye defects.
No model, threshold, labels, or predictions were changed after examining this test.

| Measure | Result |
|---|---:|
| Attempts, including failures | 64 |
| Successful restorations | 61 |
| Manually acceptable across all attempts | 43/64 |
| Automatically accepted | 9/64 (14.06%) |
| Accepted and manually acceptable | 7/9 (77.78% precision) |
| False accepts among deliveries | 2/9 (22.22%) |
| Acceptable portraits retained | 7/43 (16.28% recall) |
| Acceptable deliveries per attempt | 7/64 (10.94%) |

Accepted indices: 015, 024, 044, 045, 051, 054, 055, 057, 061.
Restoration failures 013, 022, 033 remain in the denominator and are forced rejects.

- **051** (score 0.928720): Image-right iris appears largely blank/white with no convincing pupil.
- **061** (score 0.977028): Image-right iris lacks convincing pupil, markedly washed out against other eye.

The calibration cohort had five accepts with zero observed false accepts at this same threshold.
The independent test contradicts an inference that the calibrated threshold guarantees defect-free output.
The two failed accepted portraits must remain in the review preview and archived results.

## Diversity and interpretation

Reviewers described recurring face families in 023/024/028, 016/026/030, and 017/020/025;
037/038, 035/045, and 032/044 also share facial templates. Hair, beard, pose and expression
vary, but this does not establish the broad identity diversity requested for deployment.
These are subjective visual observations, not face-recognition measurements. Unique image hashes
are provenance evidence only.

The 64 existing score-hidden visual labels are unchanged. They were produced by four agent
reviewers, one per portrait, rather than independent human production approval. Small sample
size, subjective eye-quality judgments and no inter-rater assessment limit the estimate.
Acceptability also does not prove parity with thispersondoesnotexist.

## Reproduce and inspect

Run `python3 research/reviews/reference625-test64/evaluate.py` from the repository checkout.
The standard-library script verifies every source image, all 64 restoration records, all
497 record-listed outputs, reviewed PNG and scored WebP hashes, the feature archive,
the frozen weights and calibration selection. Any missing LFS object or changed hash fails.
It evaluates frozen predictions and never executes a model.

[Machine-readable report](report.json) contains all 64 decisions, reasons, source paths and
SHA-256 references. Original labels remain in
[the restoration cohort](../../restoration/runs/reference625-test64-v1/).
The [frozen scores](../../selection/restored-test64-scores-v1/result.json) and
[calibration selection](../../selection/restored-calibration32-scores-v1/threshold-selection.json)
remain unchanged. [Non-blocking preview](../latest-preview/index.html) contains every automatic accept.

Production approval remains **false**. This report does not establish service latency,
Linux memory use, monthly operating cost, or readiness for release.
