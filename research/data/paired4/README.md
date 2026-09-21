# Four existing edited sources: prospective masked experiment

Prepared for parent review only. `manifest.json` uses the paired prototype's schema and exact FFHQ1024 source bundle. **No generator forward, training, imagegen call, source/target modification, warp or alignment was performed.** This is a fresh-run candidate, not an input replacement for the active single-pair continuation. Do not resume the old run with this changed manifest.

| Pair | Split | Final mask diagnostic | Boundary decision |
| --- | --- | --- | --- |
| calibration-original | train | `calibration-original-overlay.png` | Exact unchanged reuse of the original reviewed v2 polygon. Its hash is retained. |
| 000 | train | `000-overlay-v2.png` | Covers white shirt and lower neck; refined narrow neck-side shirt edges after overlay inspection. Mouth, chin and upper face stay source-supervised. |
| 030 | train | `030-overlay-v2.png` | Covers added shirt and lower neck below the jaw. Bottom-left x<.16 stays protected to avoid painting over the partial second person inherited from the source. |
| 028 | **validation only** | `028-overlay.png` | Conservatively protects the original beard. The central beard nearly reaches the bottom, leaving almost no supervised collar area there. This is intentionally a difficult held-out seed. |

`mask-contact-sheet.png` labels the four final diagnostics by ID and split. Red indicates the edited-clothing loss region; unshaded regions use the **regenerated source**, not the edited face. Each separate diagnostic has 768px source and target panels with a JSON sidecar recording input hashes and polygons. Raw files remain native 1024px sources and 1254px targets. Overlays are diagnostic drawings, not training targets or delivered generator output.

The three new source/target pairs were inspected at native size, then each proposed mask was inspected on both images. Initial 000 and 030 masks were refined; draft masks/overlays/manifest/provenance remain archived. `prepare.py` records the original construction and `refine_masks.py` the two manual corrections. These polygons are approximate human-selected clothing/lower-neck boundaries, **not automatic segmentation or exact anatomy measurements**. Thin neck strips near collar tops remain protected where extending the mask would threaten the original jaw; this conservative choice may resist learning the target's higher collar edge. Parent visual review remains pending.

## Edited-face differences are not supervision

The existing matched-grayscale geometry review in `../ffhq-clothing-edits3/review.md` reports upward mean mouth shifts of .02037 image heights for 000 and .01415 for 030, with maximum landmark displacements .02121 and .01771. Both exceed the earlier strict .015 displacement threshold. Their mouths are outside the clothing masks: those target changes are ignored by the pixel objective, while the original source face remains the preservation target. These are clothing targets, not claims of exactly identity-preserving whole-image edits.

028 passed the five-landmark threshold but that detector has no beard/chin outline landmarks. Native inspection shows a potentially different lower beard contour and only a very short target tab. We keep the **original beard** protected even where that excludes most of the target's small central white strip. This means full-image edited-target similarity—and even masked clothing loss alone—would be misleading as a collar-quality metric. Review the full generated image against its source and separately assess whatever collar is visible. Do not shorten the beard, move the face, or expand the mask into facial hair to manufacture success. 028's complete latent identity and edited target are excluded from optimizer sampling by its validation split.

030 also contains a partial second person at the left boundary, inherited from its source. This dataset does not solve that composition defect, FFHQ's broader demographics, or hot-only sampling. Four existing edited identities cannot establish generalization.

## Reproducibility and next gate

`provenance.json` records exact paths/hashes for source PNGs, target PNGs and NPZ z/W, the original v2 mask hash, preparation/refinement source hashes and final manifest hash. Paths are relative to this directory; arrays are finite float32 z `[1,512]` and W `[1,18,512]`. Lightweight checks confirmed four distinct latent files, exactly three train/one validation IDs, overlay/manifest polygon equality and exact original-mask reuse. `review.json` records these checks and explicitly says no heavy source preflight has run.

Source bundle: `../../runs/inference-cpu/ffhq1024/baseline-bundle`, weights SHA256 `f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d`. The original calibration target was edited from a Linux WebP while its paired source is a native lossless regeneration; the earlier documented small codec/backend discrepancy remains. The three new edits use the exact stored source PNGs.

After the CPU-200 experiment finishes and the parent reviews its result and these masks, a prospective fresh run can first use the trainer's strict CPU source/W verification on this manifest. That verification must regenerate every source, not just trust these file hashes. Neither that preflight nor a new training run is started by this preparation. NVIDIA source/weight research licensing remains unchanged, and nothing here is production-approved.

## Root review and exact source preflight

Root reviewed the four source/target mask panels (`root-review.json`). A fresh CPU1 preflight then verified all four saved latent/style pairs and original source PNGs with **zero W discrepancy and zero uint8 pixel discrepancy**. It performed no backward or optimizer updates, left source state unchanged, and took 4.187 seconds for source verification (peakRSS3,759,423,488bytes; host29%free before launch). See `preflight/preflight.json` and `preflight.log`. The original candidate manifest remains immutable; approval does not make it a trained model. The separate `../paired4-tabs/` analysis and controlled single-pair proposal address the missing white tab before dataset expansion.
