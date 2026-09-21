# Fixed FFHQ sources for paired clothing-edit research

Generated exactly 32 new, deterministic native 1024px RGB PNG sources plus one separate reproduction of the already edited source. This is an offline research source set, not a production seed catalog, an approved training set, or a priest generator. No active dataset, training run, or deployment was changed.

## Results and selection

The fresh preflight reported 32% system-wide memory free, exceeding the requested 25% threshold. The 32-source batch took 22.27 seconds; total model load, batch, reproduction, and artifact preparation took 24.02 seconds. Peak native macOS process RSS was 3,936,387,072 bytes (3.67 GiB). PyTorch used one CPU thread and one interop thread; no GPU was used. A later check after completion reported 28% free.

`contact.png` contains all 32 newly derived sources, in row-major order 000–031. All were reviewed on the contact sheet. The following nine were additionally inspected individually at native resolution: 000, 004, 005, 016, 017, 024, 025, 028, 030.

Four recommended first clothing-edit candidates are below. Attractiveness is a subjective selection judgment, not a measured property. Each appears adult, male, bareheaded, and photographic, with an unobstructed lower neck edge. None is approved as an edited training target yet.

| ID | Reason to try | Limits to preserve or inspect |
| --- | --- | --- |
| 000 | Rugged short-haired man, clean frontal face, white open shirt and visible lower neck. | Hard sunlight and squint are part of the original; do not redraw the eyes or beautify the face. |
| 004 | Smiling dark-haired adult man with room at the lower neckline. | Busy background includes a partial bystander; avoid a general background rewrite and inspect teeth after editing. |
| 025 | Photographic mature man with clear shirt/neck boundary and a different three-quarter pose. | Keep the existing pose and expression; a thin collar must fit the sloping neckline. |
| 030 | Clear eyes, short reddish hair, clean facial detail, accessible lower neck edge. | Only a narrow band fits below the chin; a partial background person is already present. |

Two attractive secondary candidates are **017** and **028**, both adult men with beards. Their facial quality is strong, but their beards reach almost to the lower edge and leave very little central neck. They are harder tests of the thin-collar edit, not equally strong clear-neck sources. Do not zoom out or shorten the beard to force a collar to fit. Prefer the first four for initial tests. The separate `reproduction-sample002.png` retains the already demonstrated source composition and is another available control, not one of the 32 new seeds.

Rejected for this initial edit selection: 008, 019, and 026 are children; other youthful or ambiguous-age portraits are not selected. 021, 023, and 027 have headwear; 023 also has hands near the head. 005 has no useful central neck margin and noisy facial detail. 016 and 024 are adult bareheaded men but offer less useful composition for this specific attractive-priest calibration. The remaining contact-sheet entries are not selected as adult male sources. Rejection here does not authorize removal from the recorded batch; retaining all 32 documents the sampling process.

## Exact source and latent recipe

For zero-based decimal indices 0 through 31, the seed is SHA256 of UTF-8 `priest-paired-source-v1:` followed by the index. All 256 bits are interpreted as a big-endian integer and passed to NumPy `default_rng`; the latent uses `standard_normal((1,512), dtype=float32)`, exactly matching the service. It does **not** draw float64 and cast afterward.

The research-only baseline bundle is loaded through the actual `inference.server.Model`, with `allow_unreviewed=True` explicit, one thread, expected safetensors SHA256 `f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d`, truncation 1.0, constant noise, and FP32 synthesis. The mapping forward hook captures the actual style vectors used by each generator forward. The PNG pixels use the service's exact `(image + 1) * 127.5`, clamp, and uint8 conversion.

Every NPZ contains only non-pickled float32 arrays `z` with shape `[1,512]` and `w` with shape `[1,18,512]`. The latter contains the complete mapped synthesis styles, not an average or inferred inversion. All 18 styles are identical for these unmixed psi-1 sources; this is checked and recorded. `manifest.json` records tensor hashes, PNG and NPZ file hashes, code/runtime versions, per-image timings, source model provenance, and hashes of all constant-noise buffers. `source_path` and `latent_path` are relative to the manifest directory. There are no target paths, clothing polygons, or train/validation assignments yet: those require reviewed edits.

The completed run used `generate-executed-v1.py`; its hash is the generation-time hash recorded for `generate.py` in the manifest's original `code_sha256` section. A subsequent schema-only normalization renamed the captured full `ws` array to the trainer's `w` key, preserving all tensor values and PNG hashes. `normalize_latents.py` and `schema_normalization` record the old/new NPZ hashes and this transition. The current `generate.py` writes the final schema directly on a future explicitly authorized run. Do not rerun merely to rename metadata: the current batch is already complete.

## Existing-seed reproduction

`images/reproduction-sample002.png` and its NPZ use seed `25a5656ee7914252f9e6a81f81018d338c20e5ca7075fb495a1eb1c54cc6ad20`. The original source is the Linux/amd64 benchmark WebP at `research/runs/inference-cpu/ffhq1024/docker-4g/sample-002.webp`. The new PNG is a fresh native Mac CPU forward using the same weights, latent recipe, and inference settings. The script also encodes a WebP with the service's quality 90 / method 4 settings for comparison.

The WebP is **not byte-exact** across these environments. Comparing their decoded WebP RGB channels gives mean absolute difference **0.5112/255**, maximum **15/255**, and RMSE **1.0942/255**; 29.53% of channels differ. Comparing the new uncompressed PNG pixels with the original decoded WebP gives MAE **1.4857/255**, maximum **19/255**, and RMSE **2.0383/255**; that comparison also includes the original lossy encoding. Neither is declared a tolerance pass. The original pre-encoding Linux RGB array was not retained, so these artifacts alone cannot separate CPU backend rounding from encoder-version effects. Exact figures and hashes are in the reproduction record.

The prior successful built-in edit used the original WebP. Pairing it with this native PNG is therefore a documented close source reconstruction, not a byte-identical input pair. Newly requested edits should use the saved PNGs directly to avoid this ambiguity.

The generator code and NVIDIA source weights retain their upstream research/evaluation license; exporting these images and tensors does not relicense the underlying model as MIT. No model was downloaded, no further random batch was generated, and no production approval is implied.
