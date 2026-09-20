# Tight-framing calibration: larger faces, unresolved vertical framing

The two new prompts materially increase face size, but do not solve the vertical mismatch with the FFHQ reference. Keep these originals as calibration evidence, excluded from training. Reject both derived alignment variants: they introduce obvious reflected anatomy, and the full FFHQ-like crop also removes most of the detected white collar.

Measurements are fractions of the square image dimension. Interocular distance is Euclidean distance between the detected eyes. Reference numbers are the previously measured FFHQ baseline sample means, not a universal anatomical target.

| Group / variant | Eye Y | Interocular distance | Mouth Y | Reflected area |
| --- | ---: | ---: | ---: | ---: |
| FFHQ reference mean | .466 | .252 | .729 | — |
| Existing 110 raw mean | .342 | .161 | .500 | 0% |
| Existing 110 eyes42 mean | .420 | .200 | .618 | 1.88% |
| New 001 raw | .367 | .273 | .629 | 0% |
| New 002 raw | .335 | .282 | .583 | 0% |
| New 001 eyes42 | .420 | .234 | .644 | 26.58% |
| New 002 eyes42 | .420 | .229 | .622 | 34.43% |
| New 001 FFHQ-like | .476 | .250 | .716 | 21.13% |
| New 002 FFHQ-like | .478 | .250 | .699 | 27.49% |

The raw pair's mean interocular distance is .278, versus .161 in the original training set and .252 in the reference. Their mean mouth position improves from .500 to .606. However, mean eye position moves only from .342 to .351, still .115 image heights above the reference. The prompt successfully asks for closer faces, but has not reliably moved the eyes down. Two examples are insufficient to establish a general prompt success rate or training benefit.

## Visual review and collar tradeoff

The original portraits look photographic, depict adult men with bare heads, and show complete white collars near the bottom edge. The second source naturally crops some top-of-head hair. Both are materially tighter than the original calendar-like composition.

The eyes42 variants preserve the detected collar boxes, but create conspicuous duplicated hair at the top, white collar fragments at the bottom, and reflected shoulder shapes along the sides. Their mean reflected area is 30.51%, compared with 1.88% for the accepted 110-image eyes42 dataset. These are visibly unsuitable training crops despite the improved face-size metric.

The FFHQ-like variants bring the eyes and mouth substantially closer to the reference, while reflecting 21–27% of the output. Only 34.2% of image 001's and 44.4% of image 002's detected collar-box area remains inside the crop. The white collar becomes a clipped strip at the bottom. They fail both the border-integrity and full-collar checks.

Moving these already-close faces downward needs image content above the source frame while pushing the collar below the output. Shrinking the source enough to retain the collar exposes still more missing background. The existing reflection strategy fills that missing area with duplicated anatomy. Neither alignment recipe is a usable correction for these sources.

## Reproduction and limits

Run from the repository root:

```sh
research/alignment/.venv/bin/python research/data/tight-framing2/measure.py
```

The script uses two OpenCV CPU threads and the existing YuNet five-landmark detector. It verifies original image hashes, resizes the 1254-square originals to 256 using Pillow Lanczos, and applies unchanged accepted eyes42 and existing FFHQ-like helper functions with OpenCV Lanczos4 / BORDER_REFLECT_101. No original pixels, existing dataset, training configuration, or model were changed. The rendered candidates are 256-square diagnostic images, not new training data.

`geometry.json` records source/output hashes, source dimensions, detector/helper/script hashes, versions, landmarks, collar boxes, and exact affine transforms. `geometry-comparison.png` shows raw, eyes42, and FFHQ-like images in that order for each source. All six displayed variants were visually reviewed.

The FFHQ-like transform is the existing five-landmark approximation, not the canonical full 68-landmark FFHQ preparation pipeline. Collar detection is a bright-region heuristic, corroborated by visual inspection; the retained-area metric measures its box rather than segmenting every collar pixel. Reflected area is the fraction of nearest-neighbor validity-mask samples outside the source, so it does not count every fractional Lanczos boundary contribution. Tiny retained-area values above one in the raw JSON are floating-point polygon-intersection error.

Recommendation: do not expand this prompt family or add these aligned candidates to active training on the strength of this pair. A further source-framing experiment would need eyes lower in the untransformed original and enough real upper background, while keeping the collar above the bottom edge. Evaluate that condition directly before making a larger dataset.
