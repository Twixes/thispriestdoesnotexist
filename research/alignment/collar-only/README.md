# Collar-only crop comparison

This follow-up removes the earlier, inferred requirement to keep the entire top
of the head inside the crop. The user requires no hats and visible priest collars;
an ordinary tight-portrait crop through the top of the hair is acceptable.

Both variants retain the earlier detected collar plus four source pixels of
margin within 96% of output height. They aim for 24% interocular distance and
center the eyes at 40% or 42% of output height, reducing zoom only as required by
the collar. No prior dataset is overwritten.

| Candidate | Mean eye Y | Mean interocular | Mean mouth Y | Mean reflected border |
| --- | ---: | ---: | ---: | ---: |
| Earlier whole-head partial, 110 combined | .400 | .184 | .583 | 2.6% |
| Collar-only, eyes40 | .400 | .207 | .605 | 0.6% |
| Collar-only, eyes42 | .420 | .200 | .618 | 1.9% |
| Pretrained FFHQ reference | .466 | .252 | .729 | n/a |

The whole-head guard was materially limiting zoom. Removing it produces visibly
tighter portraits and a 12% larger average interocular distance in eyes40, or 8%
larger in eyes42. Both remain smaller than source FFHQ faces because keeping the
collar imposes a real geometric limit.

**Prefer eyes42 for the next fresh pretrained restart.** Its eye and mouth
positions are closer to the source geometry than eyes40, with slightly smaller
interocular distance as the tradeoff. This recommendation is not a claim that one
variant has already produced a better GAN; neither has been trained by this audit.

I visually inspected all 110 candidates in each full contact sheet. Both preserve
visible white collars, visible bare heads, and normal facial structure. Some cut
through the top of the hair in the same ordinary way as tight FFHQ portraits; no
new hats or facial distortion were observed. These are accepted training
preprocessing candidates, not a production model approval.

- `eyes40/` and `eyes42/`: 110 separate 256×256 RGB PNGs per variant.
- `eyes40-contact.png`, `eyes42-contact.png`: all candidates, labeled by source.
- `comparison-contact.png`: original / whole-head partial / eyes40 / eyes42 rows.
- `manifest.json`: source hashes, exact affine matrices, output hashes, settings,
  script hash, source manifest hash, and metrics.
- Reproduce with `research/alignment/.venv/bin/python research/alignment/collar_only.py`.

Collar localization remains heuristic (108 automated detections and two geometric
fallbacks), so contact-sheet review is part of acceptance. Reflected border pixels
remain a small preprocessing artifact to monitor. The transform is a rotation,
uniform scale, and translation; it does not warp facial proportions or serve as
an inference image-morphing substitute.
