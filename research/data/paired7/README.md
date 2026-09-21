# Prospective seven-pair collar-region dataset

`manifest-proposed.json` is a **new version-2 candidate** for the separate `paired_regions` trainer: six training seeds (calibration-original, 000, 030, b2-055, b2-051, b2-020) and the original held-out 028. It is not a continuation of a run with an old data hash. It must not be selected automatically before root reviews these masks and the single-pair region-loss result.

All four prior records are preserved as exact JSON objects from `../paired4-tabs/manifest-proposed.json`, including every clothing/collar coordinate and the complete held-out record. Because both manifests are in sibling directories, their relative paths resolve to identical files without rewriting. The source bundle also resolves identically. Assertions verify these properties and seven distinct latent files; old manifests, assets, and running inputs are untouched.

The three new clothing polygons were manually drawn around the **source** lower neck/jaw. Their target-tab polygons were manually traced from the raw edits, then intersected with the clothing masks. Face, eyes, glasses, mouth and chin remain supervised by the source. These are human-style manual annotations, not automatically inferred segmentations. No image pixels were changed; resizing and overlay drawing happen only in diagnostic files. `b2-*-overlay.png` shows source and target side by side, each at 1024 diagnostic resolution. Red indicates clothing, cyan its boundary, green the raw tab trace, and yellow the effective intersection. All three overlays were visually inspected individually and in `mask-contact-sheet.png`.

| New pair | Clothing pixels | Raw tab pixels | Effective tab pixels | Raw trace retained | Tab fraction of clothing |
| --- | ---: | ---: | ---: | ---: | ---: |
| b2-055 | 252,686 | 6,895 | 6,015 | 87.24% | 2.380% |
| b2-051 | 90,269 | 6,302 | 4,924 | 78.13% | 5.455% |
| b2-020 | 314,638 | 4,011 | 1,790 | 44.63% | 0.569% |

The 020 edit places part of its tab above the source chin boundary. The effective mask deliberately drops that part instead of allowing a target clothing loss into the original chin. Its remaining tab is small and is a harder supervision example; this tradeoff needs explicit review. The 051 tab reaches the bottom edge; neither the mask nor objective demands an entire collar below the crop. The 055 original high white neckline is traced close to the jaw, with conservative face protection; narrow boundary strips may retain some original brightness. No mask includes glasses or facial features.

All three new edits pass the existing matched-grayscale geometry checks, as recorded in `../ffhq-clothing-edits-b2/geometry.json`, but are not exact facial replicas. Small edited expression/texture changes are ignored through source-face supervision. The 020 crowd/harsh-light caveats remain; no background cleanup has been performed. Existing 028 validation keeps its difficult beard/collar tradeoff unchanged.

`prepare.py` is PNG/NumPy-only and creates the manifest, overlays, region counts, and hashes. `provenance.json` records immutable-base checks; `regions.json` records source, target and latent hashes. No model preflight, latent regeneration, forward/backward pass, image generation, or training was run while preparing this dataset. No production approval is implied.

## Reviewed revision

`manifest-proposed-v2.json` is the selected research revision. Only b2-055's clothing boundary changes: it includes the original white sweater strips while retaining source skin/chin protection. Its effective tab retains 6,248 of 6,895 traced pixels (90.62%). All other six records and collar polygons remain identical. `root-review-v2.json` records the native overlay review. The initial draft and intermediate refinement remain as research history; they are not the chosen training manifest. No production approval follows from mask approval.
