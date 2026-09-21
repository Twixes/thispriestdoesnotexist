# Three prospective clothing edits from source batch 2

Built-in `image_gen` produced independent edits for 055, 051 and 020. Native outputs are preserved as received at 1254×1254; exact inputs were stored 1024×1024 PNGs from `../ffhq-paired-sources64-b2/`. There was no crop, warp, compositing, pixel repair, generator inference, or training during review. All three outputs were inspected at native resolution.

`prompts.json` preserves the exact prompts and original tool output paths. `manifest.json` records source/target/NPZ checksums and hashes of prompts, measurement code, geometry and diagnostic comparison. Source z/W remain unchanged. The files are prospective research pairs only; they have not been added to an active training manifest and have no masks or training eligibility. They are imagegen edits, not output from an adapted StyleGAN model.

| ID | Native visual assessment | Maximum matched-grayscale landmark shift | Interocular change |
| --- | --- | ---: | ---: |
| 055 | Face, glasses and hair remain close to source; complete clear white Roman tab with black shirt. Minor skin/expression differences remain. | 0.005705 of image width | −1.220% |
| 051 | Mature face and hair largely preserved; clear white Roman tab reaches lower frame edge. Slight eye/skin-detail changes visible. Do not require a full collar below the frame. | 0.001784 | +0.644% |
| 020 | Pose and smile largely preserved; complete white Roman tab beneath chin. Harsh illumination and blurred background people remain inherited source limitations. Mouth geometry changes slightly. | 0.008250 | −1.876% |

All three pass the existing prospective thresholds: maximum five-landmark displacement ≤0.015 image width and absolute interocular change ≤5%. These thresholds measure coarse geometry, not exact identity, skin texture, chin/beard preservation, or final model quality. Source-to-edit changes remain despite a pass. Protected upper-face supervision should continue to use regenerated source pixels rather than the edited faces.

Measurement reuses the exact helper functions from `../ffhq-clothing-edit1/measure.py`, with a saved copy of the previous three-pair audit adapted for this batch. Inputs are resized to 256×256 with Pillow LANCZOS for the detector only; **both source and edit receive the same grayscale conversion**. YuNet runs on CPU with one thread. Original color-vs-edit results and the source-only grayscale control are retained in `geometry.json`; the unmatched comparison fails the landmark threshold for 051 and 020, illustrating why the grayscale control matters. No training models were loaded.

Run the small PNG-only audit with:

```sh
research/alignment/.venv/bin/python research/data/ffhq-clothing-edits-b2/measure.py
```

The first attempt with `research/.venv/bin/python` stopped immediately because that environment lacks OpenCV; the existing alignment environment succeeded. No dependencies or models were downloaded. `comparison.png` is a labeled diagnostic of source RGB, source grayscale, and raw edit, not a training or serving composite.

These assets inherit the original FFHQ source model's provenance and research/evaluation license context recorded in the source manifest; the website's MIT license does not relicense NVIDIA code or weights. Source selection and these edits do not constitute production approval.
