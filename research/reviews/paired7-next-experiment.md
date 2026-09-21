# Six-pair adaptation recommendation

A fresh, bounded six-identity run with the **existing region-balanced objective** is justified as the next diagnostic experiment. Do not simultaneously change loss weights, unfreeze coarse blocks, add an adversarial loss, or resume the already specialized single-pair model. Six supervised identities can test whether the previous result was a single-example shortcut; they cannot establish broad collar generalization or production quality.

This recommendation uses the local executed experiments, code, mask overlays, and recorded image reviews. No new paper claim, model computation, or trainer change was needed for this comparison. The parent has now launched the corresponding run at `research/runs/paired-regions1024-sixpair-600`, with exact invocation and memory evidence in the sibling `.launch.json` and `.memory-before.txt` files. This note does not claim that run has completed.

## Evidence and limitations

- Region200 preserved coherent source faces on the 32-image contact and five native-reviewed examples, but none of those five acquired a recognizable collar. The training image learned a soft white neck patch rather than the target's crisp rectangular tab. This supports keeping preservation while testing supervision across different necks and clothing; it does not prove a capacity or loss bug.
- At step 200, the single-pair pre-update tab L1 was 0.02675 while rest-clothing L1 was 0.25845, in the trainer's unclipped grayscale range. A low error inside the traced white region did not establish a complete collar structure. Increasing the tab coefficient again is poorly motivated: it may reward another white patch while further reducing attention to the surrounding black garment.
- The reviewed v2 manifest SHA256 is `9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6`. It contains six training identities (calibration-original, 000, 030, b2-055, b2-051, b2-020) and unchanged validation identity 028. The new three-pair overlay contact and prior three-pair tab contact were explicitly inspected. Root's native v2 mask approval is preserved in `research/data/paired7/root-review-v2.json`.
- The source face supervises protected pixels. Edited faces are not identity ground truth. New 055 adds an example with a larger, clear garment region; new 051 has a tab at the frame edge. New 020 retains only 44.63% of its target tab after protecting the source chin; old 000 retains 55.40%. Those two examples have a real geometry conflict and should be identified separately in reviews. Do not expand masks into source chins to make losses easier.
- The corrected 055 mask retains 90.62% of its tab. The targets span neck/pose/clothing differences, but all tabs are low in frame and the sample remains tiny. A complete collar inside the frame is not a user requirement; recognizable white clerical clothing is.
- CPU thread benchmarking favors two threads (4.62s mean versus 5.72s for one, only three measured updates per setting). Exact teacher rendering was checked at two threads for the calibration source only; the seven-pair preflight and training loader retain zero-tolerance checks. Do not relax provenance tolerances if another pair differs.

## Exact bounded run

Use the original trusted 1024 source bundle, a fresh student and Adam state, reviewed v2 manifest, CPU device, two intra-op threads and one inter-op thread. The launched parameters are:

| Parameter | Value |
| --- | --- |
| Total steps / batch | 600 / 1 |
| Learning rate / Adam betas | 0.0001 / (0.9,0.999) |
| Clothing, protected, fresh weights | 1, 1, 1 |
| Clothing loss | 0.5 × mean collar-intersection L1 + 0.5 × mean rest-clothing L1 |
| Fresh preservation |top 75% rows, explicitly not face segmentation |
| Frozen state |mapping, buffers/noise strengths, synthesis blocks at 32px and below |
| Rendering |constant noise, FP32, truncation psi 1, grayscale loss |
| Seed | 20260921 |
| Checkpoints / random previews |every 100 steps / 4 fixed latents |
| Source verification |exact uint8 PNG and exact W, zero tolerance |

At batch 1, 600 updates give 100 presentations per training identity **in expectation**, not a balanced guarantee: the current sampler draws uniformly with replacement. Count actual `pair_ids` in the log before interpreting per-pair failure. The update-only estimate is 46.2 minutes from the short benchmark; loading, seven-pair verification, saving, previews and system contention add overhead. Fresh free memory must be at least 25%; never restart because an observation call times out.

## Stop and review criteria

1. Stop immediately for nonfinite state, provenance/frozen-state rejection, or clear newly introduced facial damage (double features, broken eyes/mouth, collapsed identities) in saved previews. A new persistent neckline stain is a negative outcome to record, not evidence of a collar. Preserve the checkpoint/log; do not repair the run by relaxing source checks or changing live inputs.
2. Review saved previews at 100-step intervals. The existing helper previews **only the first training pair**, the one validation pair, and fixed random latents. It does not show all six training identities. Do not infer all-pair success from the calibration image. Additional teacher/student rendering should happen only when the training process has stopped, avoiding concurrent heavy compute.
3. Hard stop at 600. Evaluate all six supervised identities and the validation028 pair against source and target, plus the disjoint source30 set below. Inspect actual tab boundaries and adjacent dark garment, not merely tab mean brightness. Record individual facial preservation and clothing outcomes.
4. Do not automatically extend if several training examples still produce white skin patches or if none of the unseen eligible male portraits has a recognizable tab. That result favors diagnosing the surrounding clothing/edge objective and capacity in a separate experiment rather than buying more identical updates. An isolated held-out failure 028 is not decisive because its beard leaves very little visible neck; assess the other unseen males too.
5. A follow-up run requires visible progress on multiple training identities **and** at least one clearly unseen adult male with convincing collar structure while retaining coherent faces. This is a research continuation criterion, not a production threshold or statistical guarantee. Production still requires broad unseen quality, diversity and adult-male/no-hat eligibility, which this loss does not solve.

If the six-pair run again matches tab brightness without forming a boundary, the most directly motivated next ablation is a small, explicitly traced collar-surround/boundary region with its own error reporting, retaining source-face protection. This is a hypothesis to investigate after these outcomes; do not implement it now. Whole-image target perceptual losses would penalize the source face for imperfect imagegen identity preservation, and a new adversarial loss would confound the controlled comparison. Relaxing blocks at 32px or lower could alter coarse face geometry; it is not justified before measuring this run.

## Disjoint evaluation prepared

`research/data/paired7/evaluation-source30.json` derives from the unchanged original source32 manifest. It removes 000 and 030 by exact Z identity because both are now training examples; original calibration reproduction was never part of the 32 entries. All three b2 training Z values were also checked for overlap. Validation 028 remains included. The prepared script verifies all source PNG/NPZ hashes, actual float32 Z digests, six distinct training seeds, no train/validation overlap and exactly 30 distinct retained evaluation seeds. No images or models are generated by preparation.

The existing `paired_regions/evaluate.py` already sizes every loop, contact and report from the actual entries. There is no required count 32 or special treatment that would incorrectly reclassify 000/030 as held out. Its independent training-Z exclusion remains a second safeguard. **No evaluator edit was necessary**, preserving the SHA of the already executed region200 evaluator. The previous usage document describes its original 32 invocation; this run intentionally uses 30.

After terminal process exit, final configured step 600 and a fresh memory preflight, use:

```sh
research/.venv/bin/python research/experiments/paired_regions/evaluate.py \
  --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle \
  --checkpoint research/runs/paired-regions1024-sixpair-600/resume.pt \
  --sources research/data/paired7/evaluation-source30.json \
  --expected-step 600 \
  --output research/runs/paired-regions1024-sixpair-600/evaluation30
```

The command above is documentation only; evaluation has not run. Expect 30 results and zero additional exclusions. For native review use 004, 025, 028, 017 and 024. Root's later cross-check marks 004 as age-uncertain with an additional person: retain it only as a difficult face-preservation/neckline case, and exclude it from eligible-adult or hot-success counts. The clearly adult masculine examples in this list are 017, 024, 025 and 028; beards limit visible neck space on 017 and 028. 000 and 030 belong in the training review, never the unseen-success count. The generic fixed top75/bottom 25 MAE remains a preservation diagnostic, not identity, attractiveness or collar recognition. It does not provide target loss for all six training pairs; that end-of-run paired review must be added separately without changing the trainer.

## Superseding eligibility clarification

The later native manual96 review supersedes the earlier region200 description of all five selected subjects as adult male. Source004 has uncertain age and an additional person, so it must not count as an eligible adult/hot priest success. Historical pixel/face/clothing observations remain recorded unchanged; this correction concerns eligibility only. The unfiltered source30 manifest remains unchanged to preserve honest diagnostics, including ineligible and ambiguous cases. No age classifier or biometric proof is implied by manual review.
