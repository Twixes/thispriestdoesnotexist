# Optional collar-region paired adaptation

This is an isolated research variant of `paired_edit/trainer.py`. That base file remains unchanged. It imports the base's source network, grayscale conversion, area-normalized L1, polygon rasterizer, freeze policy, RNG/checkpoint helpers, source-image helpers, and full-generated-image previews. Its own training entry point records both trainer files and all vendored Python/native implementation hashes. It never modifies imported module globals.

## Why this experiment exists

The PNG-only diagnosis in `research/data/paired4-tabs/analysis.json` finds that the original calibration's effective white tab occupies 7,198 of 141,212 clothing pixels (5.097%). At step 200, its tab MAE is 119.09/255 versus 11.19/255 for remaining clothing. This supports investigating spatial loss dilution. It does not prove that the new objective will fit the tab, generalize collars, or preserve photographic quality. These are post-update rendered PNG measurements, not exact floating-point training losses or parameter-gradient measurements.

The only objective change is:

```text
clothing = 0.5 * mean_abs_error(tab) + 0.5 * mean_abs_error(rest_clothing)
paired = clothing_weight * clothing + protected_weight * protected
total gradients also include the unchanged optional fresh_weight term
```

Each region is normalized independently per image, then averaged across the batch. Equal regional errors produce the same scalar clothing loss as the old broad mask; the 0.5/0.5 weights preserve the overall clothing coefficient, while reallocating pixel-gradient mass toward the small tab. The numerical loss on unequal regional errors is deliberately different.

`tab = raw_collar_trace ∩ original_clothing_mask`; `rest = original_clothing_mask − tab`. Raw trace area, effective area, excluded area, retained fraction, and tab fraction of clothing are recorded. Missing train annotations and empty/full intersections fail. Tab must be a nonempty **proper** subset of clothing, and rest must be nonempty. A trace that crosses protected pixels is clipped to the unchanged clothing boundary. The protected mask is always the original mask's complement; no protected face pixels are reclassified as clothing.

## Manifest and run boundaries

The manifest must have `version: 2`, `requires_trainer_feature: "equal_area_collar_partition_v1"`, and one or more train entries with normalized `collar_polygons`. Existing source/target/latent and clothing fields retain their meanings. Validation annotations are optional; validation records are never sampled for optimization. The proposed held-out 028 record is unchanged and needs no tab field.

Default LR remains `1e-4`; mapping, synthesis blocks through 32px, all buffers, and noise-strength parameters remain frozen. Source verification, grayscale, protected loss, fresh-latent preservation, optimizer, sampling, and preview behavior retain the base algorithm. Arbitrary positive training-pair counts are supported.

Use a fresh initial experiment from the original source bundle. Broad-mask checkpoints are explicitly rejected. Exact resume is supported only for this new checkpoint format, with identical options, source/data/code hashes and dependency versions, into another new empty output directory. There is no production approval or export path.

The next controlled 1024 experiment should use only the same calibration pair with the new collar annotation and preserve the prior LR, freeze/protected/fresh settings. Expand to three pairs only after assessing whether the tab can be fitted. No 1024 preflight, inference, or training was run while implementing this extension; the parent task chooses those runs separately.

Example source/mask preflight after the parent selects its manifest (this performs actual source forwards):

```sh
research/.venv/bin/python research/experiments/paired_regions/trainer.py \
  --manifest PATH_TO_REVIEWED_V2_MANIFEST \
  --run NEW_PREFLIGHT_DIRECTORY --device cpu --threads 1 --preflight-only
```

## Completed validation

`check_cpu.py` created tiny, actual-vendor 64px/16-channel generator fixtures. The complete check passed in 8.17 seconds on one CPU thread. It performed five continuous updates and an independent 2+3 resumed sequence, including fresh preservation and previews. The resulting model, optimizer, RNG states, provenance and final PNG previews matched exactly.

Checks also establish real finite student changes; unchanged source, mapping, low-resolution blocks, buffers and noise strengths; validation exclusion with no validation tab annotation; equal summed tab/rest pixel gradients; rejection of missing/empty/full tab regions; unchanged protection after intersection; full generated output previews; source mismatch rejection; changed option/imported-helper hash rejection; old-objective resume rejection; and unavailable production approval.

`cpu-evidence.json`, the check logs, and `tiny-fixtures/` retain evidence. No FFHQ model was loaded, no MPS device was used, and these tiny tests do not establish actual 1024 resource use or visual quality.

## Actual1024 single-pair smoke

Root reviewed the implementation and tab overlays; an independent code review found no actionable blocker. Ten CPU1 updates on the original calibration pair completed with exit0 in73.79s, peakRSS5,546,328,064bytes, zero reported swaps. Frozen/source checks passed at steps5/10. The native step10 face remains coherent; open white shirt persists and collar quality is not established. See `../../runs/paired-regions1024-cpu-smoke/completion-review.json`. Root is starting an exact continuation to200total steps with unchanged LR/freezing/protected/fresh settings and the same two fixed unseen preview latents, into a separate outputdirectory.

## Completed single-pair 200 and next dataset

The continuation exited 0 after 200 total updates (1,069.57 seconds for steps 11–200; peak RSS 5,803,442,176 bytes). The controlled saved-image comparison fits the small tab's brightness much better than broad-mask L1, but the native output is an irregular white neck patch. The 32-source evaluation preserves faces yet yields no recognizable clerical collar on the five unseen adult men inspected natively. This is a negative generalization result, not production approval. See `../../runs/paired-regions1024-cpu-200/completion-review.json`, `comparison200/`, and `evaluation32/`.

The next experiment uses six training identities and one held-out identity in the reviewed `paired7/manifest-proposed-v2.json`, starting from the original pretrained source. This supersedes the earlier prospective three-pair suggestion. Keep the region objective and freeze policy unchanged so the new test measures the effect of more diverse examples. The separate 12-update thread benchmark tentatively favors CPU2 (4.62 s/update versus CPU1 5.72 s); it is not a long-run timing guarantee. All three tested thread counts produced identical final checkpoint-state digests and preview PNGs for the benchmark pair.
