# Three additional prospective clothing targets

Raw native1254×1254 edits for b2-042, b2-058 and b2-059 are saved unchanged. The expansion agent generated042/058 using built-in `image_gen`; root generated059. Each `ID-prompt.json` preserves the exact prompt, raw tool output path and source/target/latent hashes. Root's059 PNG and individual prompt file were only read, not edited. `prompts.json` aggregates the three records for the shared audit without replacing their individual provenance.

All three raw outputs were inspected at native size. They preserve recognizable mature faces, hair/facial hair, pose and glasses where present, and show black clerical shirts with clear white Roman tabs. No hats were added.042 and059 tabs reach the lower image boundary;058's tab is fully visible. The images are plausible clothing teachers with small facial texture/expression changes, not exact facial replicas and not necessarily hot candidates. Final attractiveness eligibility remains a separate requirement.

| Source | Native review caveat | Maximum matched-gray landmark displacement | Interocular change | Mean mouth-y shift |
| --- | --- | ---: | ---: | ---: |
| 042 | Natural tooth spacing and glasses remain recognizable; skin texture and smile details changed slightly. Tab cropped by bottom edge. | 0.005991 image width | −0.153% | +0.003209 |
| 058 | Mustache/eyes/mouth remain plausible, but this has the largest geometry shift and warrants careful source-jaw masking. Complete tab visible. | 0.013141 | +1.739% | +0.004784 |
| 059 | Face, glasses, smile and mustache remain recognizable; teeth/skin details changed slightly. Tab meets bottom edge. | 0.005062 | −0.433% | −0.003520 |

All pass the existing prospective criteria of maximum five-landmark displacement≤0.015 and absolute interocular change≤5%. Passing these coarse checks does not establish exact identity, skin texture fidelity, chin/beard preservation, or suitability for training without mask review. Protected-face supervision should continue to use regenerated source images rather than edited facial pixels. Source-face preservation has priority over forcing a complete collar into a tight crop.

`measure.py` adapts the existing batch2 audit, reuses the same original calibration helper, and runs YuNet on CPU with one thread. Both original source and raw edit are converted through the same256px grayscale preprocessing for the matched comparison. Original color comparisons and source-only grayscale controls are retained in `geometry.json`. There is no alignment warp, padding, reflected boundary, crop repair or pixel compositing. `comparison.png` is a diagnostic only; it shows source RGB, source grayscale, and raw edit.

The audit verifies unchanged source PNG/latent checksums against the source64 manifest, records raw output and audit hashes in `manifest.json`, and completed successfully using the existing alignment environment:

```sh
research/alignment/.venv/bin/python research/data/ffhq-clothing-edits-b3/measure.py
```

No generator forward, training, checkpoint load or additional image generation was performed during the audit. No active training manifest was changed. These records remain `training_eligible:false` and `production_approved:false` until prospective masks and data provenance are reviewed. Source-model license/provenance remain as recorded in the original FFHQ source manifest; the application's MIT license does not relicense NVIDIA research code/weights.
