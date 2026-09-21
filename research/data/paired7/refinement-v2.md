# 055 clothing-boundary refinement

The new candidate is `manifest-proposed-v2.json`, SHA256 `9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6`. Only b2-055's `clothing_polygons` changed. The other six records, source/target/latent paths, split, bundle, and every tab polygon are unchanged. In particular, 020 still protects its source chin and retains only 44.63% of its raw tab trace. No active input was modified and no training or model inference was run.

The boundary now follows the native source skin/white-sweater edge more closely, using 23 manually placed pixel-coordinate points. The face, chin and glasses remain outside the clothing mask. Native source and both versions of the side-by-side overlay were inspected; `b2-055-source-boundary-grid.png` preserves the enlarged source reference used for tracing. These are visual manual annotations, not an automatic skin classifier or a pixel-perfect semantic guarantee.

Compared with the preserved first candidate, **15,109 previously protected pixels** now belong to clothing: 7,230 on the left (x<450), 7,577 on the right (x≥580), and 302 around the central lower boundary. These regions are chiefly the previously visible white sweater strips along both jaw sides, highlighted blue in `b2-055-overlay-v2.png`. Two isolated pixels leave the clothing mask because of polygon rasterization; the net clothing increase is 15,107 pixels, from 252,686 to 267,793.

The raw tab trace is unchanged at 6,895 pixels. Its effective intersection increases from 6,015 to 6,248 pixels, or 90.62% retained. The new boundary does not expand the tab loss into the source face to satisfy the edited target.

`refinement-v2.json` records exact hashes, counts, and hand-traced points. `refine055_v2.py` reproduces the candidate and diagnostics using Pillow/NumPy only. `mask-contact-sheet-v2.png` combines the refined 055 overlay with unchanged 051/020 overlays. The original manifest, overlays, preparation script, and provenance remain intact. An intermediate conservative trace is also preserved with `v2a` filenames; it still left cloth slivers and is superseded by v2. Review **v2**, not v2a, for the prospective run.
