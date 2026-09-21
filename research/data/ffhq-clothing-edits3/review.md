# Three clothing edits: useful appearance, mixed geometry preservation

All three raw edits look photographic at native resolution and at the 256px diagnostic size. They depict adult men with bare heads and recognizable white clerical tabs. All three tabs extend through the bottom image boundary, so **visible collar** passes visual inspection but **complete collar inside the frame** does not. Complete-collar framing was a calibration preference, not a hard user requirement: the requirement is a recognizable white clerical collar, which all three visibly provide. These are built-in imagegen edits of unadapted FFHQ sources, not output from a trained priest generator. They remain research-only and excluded from active datasets by this review.

## Matched-grayscale measurements

The same YuNet detector measured each RGB source, an in-memory grayscale source, the raw edit, and an in-memory grayscale edit. Both sides of the main comparison therefore have the same color treatment, following the earlier control showing that color alone affects detector positions. The source and edit are resized to 256-square with the same Pillow Lanczos path; there is no alignment, warp, padding, or reflection.

| ID | Maximum landmark displacement | Limit .015 | Interocular change | Limit 5% | Mean mouth Y change |
| --- | ---: | --- | ---: | --- | ---: |
| 000 | .02121 | Fail | +0.225% | Pass | −.02037 |
| 028 | .00784 | Pass | +1.002% | Pass | −.00568 |
| 030 | .01771 | Fail | −2.002% | Pass | −.01415 |

Displacement is Euclidean distance in units of square image width. At the 256px detection size, the maxima correspond to approximately 5.43, 2.01, and 4.53 pixels. The prospective .015 limit corresponds to 3.84 pixels. Both mouth landmarks shift upward for 000 and 030, so these two do not pass the strict five-landmark preservation criterion despite preserving interocular scale. Do not describe them as proven exact clothing-only changes.

| ID | Grayscale source eye Y → edit | Grayscale source mouth Y → edit | Source interocular → edit |
| --- | --- | --- | --- |
| 000 | .48808 → .48192 | .74590 → .72554 | .25438 → .25495 |
| 028 | .47054 → .46674 | .75287 → .74719 | .27564 → .27840 |
| 030 | .47580 → .47796 | .74044 → .72629 | .27681 → .27127 |

The edits retain the large-face framing associated with these FFHQ sources, substantially unlike the earlier calendar portraits with much higher eyes and smaller faces. Preservation of source geometry is a distinct criterion from whether a target is useful as an FFHQ-framed priest example. A failed arbitrary detector threshold does not by itself prove a visually unusable target, and a passed threshold does not prove identical anatomy. The complete unadjusted comparisons and color-only controls are retained in `geometry.json`.

## Native and 256px visual review

- **000:** Photographic adult man, bare head, original squint and general expression retained, convincing black shirt and white tab. The face appears closely preserved overall, but the measured upward mouth shift prevents a strict preservation pass. The lower edge of the tab is cropped. No obvious gross zoom or recrop is visible.
- **028:** Photographic adult man with the same broad smile and beard. The five measured landmarks pass both limits. The beard nearly touches the bottom and leaves only a short white strip; its lower outline may differ subtly, and the detector cannot assess that outline. The full collar is not visible. Do not shorten the beard or move the chin merely to fit a larger tab.
- **030:** Photographic adult man with closely similar eyes, pose, expression, and short hair. The measured mouth shift is just outside the strict displacement tolerance; interocular scale passes. The black shirt and thin white tab are clear, but the tab is cut off at the bottom. The partial second person at the left edge was already present in the source and remains present; it is a source-composition limitation, not a newly added person.

The five-landmark YuNet model has eye, nose, and mouth landmarks but **no chin or beard-outline landmark**. No numerical chin-preservation result is claimed. The side-by-side images show no obvious gross repositioning of the head or chin, but small local changes cannot be ruled out by this detector. A confirmed chin shift would fail the requested face-preserving recipe even if all five available landmarks passed. The native/256 review therefore supplements, rather than replaces, the measured limits.

No hats, reflected padding, duplicated anatomy, or new border is visible. The existing backgrounds are broadly retained, including their imperfections. All three edits prioritize a recognizable clipped collar over changing the square composition. The collar clipping is a documented composition tradeoff, not an automatic rejection against the user's recognizable-collar requirement.

## Interpretation and reproducibility

This is three-example evidence that native FFHQ geometry can support recognizable priest clothing while retaining photographic faces. It is not evidence of diversity or generalization in a trained generator. Keep 028's geometric pass separate from its small visible collar area, and keep 000/030's visually plausible appearance separate from their strict landmark failures. These candidates require any further data-use decision to consider both properties explicitly.

For the proposed paired training method, the original source face would supervise the protected region, while the edited target would supervise only a reviewed clothing mask. The 000/030 failures mean they are unsuitable as strictly identity-preserving whole-image ground truth; they do **not** automatically disqualify those images as clothing-only targets. All three are potentially useful for masked supervision pending review that the clothing polygons exclude the face, chin, and beard boundaries. This review does not approve masks or activate any pair for training, and neither prospective threshold has changed.

Run from the repository root:

```sh
research/alignment/.venv/bin/python research/data/ffhq-clothing-edits3/measure.py
```

`comparison.png` shows source RGB, source grayscale, and raw edit at exactly 256px in each row. `geometry.json` stores all four measurements, method/versions, helper and detector hashes, thresholds, and per-landmark shifts. `manifest.json` records the original sources, targets, and latent-file hashes against the fixed source-batch manifest, plus hashes of the prompts, script, geometry, and comparison. Raw source/target images are read only. Two OpenCV CPU threads were used; no PyTorch model, GPU computation, training mutation, or production change occurred.
