# Proposed paired10 clothing dataset

Research proposal only: nine training seeds and the unchanged held-out028 seed. Root has accepted the clothing masks for research use in `root-review.json`; strict source-model reproduction preflight remains required. No training, inference or production approval was performed. The earlier `review.json` records the proposal before root review. The existing seven records are copied exactly from `../paired7/manifest-proposed-v2.json`; the source bundle, all prior paths and masks, and held-out split are unchanged. New training candidates are b2-042, b2-058 and b2-059. No active manifest or original image was modified.

`manifest-proposed.json` uses the existing version2 region-loss schema. `raw-polygon-proposals.json` preserves manual source-pixel traces. Clothing masks are manually drawn against1024px source anatomy; everything outside is the source-preservation region. Each target white tab is independently traced, then intersected with clothing. Intersection never expands into protected source skin. These are manual annotations, not automatic segmentation or proof of exact anatomy.

| New source | Clothing pixels | Raw tab | Effective tab | Retained | Effective tab / clothing |
| --- | ---: | ---: | ---: | ---: | ---: |
|042|44,928|5,397|2,016|37.35%|4.49%|
|058|130,594|6,315|4,787|75.80%|3.67%|
|059|81,328|4,262|4,262|100%|5.24%|

042 deliberately protects the source's low chin/upper-neck transition even where the edited tab sits higher. 058 preserves the original chin, mustache, lips and nose; its target mouth/nose movement is excluded from target supervision. 059 allows the lower neck while protecting the chin/stubble. 042 and059 target tabs reach the image edge; complete in-frame tabs are not required. Expanded target shoulders outside the original garment are not used to overwrite source background.

Review `mask-contact-sheet.png` and the three full-size `b2-*-overlay.png` panels: red is allowed clothing/lower neck, cyan its boundary, green the raw tab, yellow its effective intersection. Native source images are untouched; native1254px targets are resized only inside diagnostics. `review.json` records visual findings and artifact hashes. `regions.json` has exact new-region pixel counts; `audit.json` checks all ten file hashes, saved latent shapes/finite values, region counts, and unchanged old records. The old seven resolve to the same files because both manifest directories are siblings.

`prepare.py` and `finalize.py` reproduce the proposal and consistency audit with NumPy/Pillow only. Run from the repository root using `research/.venv/bin/python research/data/paired10/prepare.py` then `research/.venv/bin/python research/data/paired10/finalize.py`. `source_grids.py` creates coordinate guides. `draft-v1/` retains first outlines. `draft-v2a/` retains a rejected intermediate042 polygon with incorrect bottom closure; the final polygon explicitly includes the bottom-left corner. Do not use archived drafts for training.

The exact FFHQ source model and upstream licenses remain those of the original bundle; repository MIT does not relicense third-party weights. Built-in imagegen provenance and native edit geometry are retained in `../ffhq-clothing-edits-b3/`. These faces are clothing teachers with varied age/appearance, not approved final hot-priest outputs. A fresh controlled experiment would be required if root approves the masks; this dataset cannot be substituted into an existing run's input manifest.
