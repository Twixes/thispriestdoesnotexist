# Deterministic crop correction without reflected content

The separate `eyes42-1024-zero-reflection/` candidate contains all 110 portraits at 1024 × 1024 RGB, sampled directly from the original 1254-pixel sources. It removes the reflected hair fragments identified in the exact-transform candidate without image generation, inpainting, local warping, or changing any existing 256-pixel dataset. Total PNG size is 92,493,629 bytes (approximately 88.2 MiB).

The correction retains each accepted rotation and permits only uniform scale and translation. A linear program constrains all inverse-mapped output corners to remain at least four pixels inside the original source, keeping the Lanczos4 interpolation support inside real source pixels. It also retains all five facial landmarks with a 64-pixel output margin and the existing detected collar box plus its original four-reference-pixel bottom guard. The two existing geometric collar fallbacks remain fallbacks; collar detection was not silently replaced.

The solver first seeks the exact accepted eye position. If that is infeasible, it minimizes absolute eye-height change, then horizontal eye displacement, then scale change, with 0.0001-pixel numerical slack between objectives. Eye X is restricted to 40–60% of the frame, eye Y to 30–49%, and scale lies between the source-inscribed-square minimum and twice the accepted scale. Observed scale changes are only 1.000–1.119×. These stated bounds define the feasibility claims; they are not a proof about every imaginable crop policy.

| Check | Result |
| --- | ---: |
| Zero reflected content with face/collar constraints | 110 / 110 feasible |
| Exact accepted eye X/Y retained | 27 / 110 |
| Cannot retain eye Y within 40–44% under the stated constraints | 46 / 110 |
| Corrected eye-height range | 35.74–42.15% |
| Mean corrected eye height | 40.19% |
| Maximum combined eye displacement | 122.38 output pixels (`synthetic-099`) |
| Unique output files | 110 / 110 |

The accepted target appears as 42.15% in integer pixel-center coordinates at 1024 because the original 42% at 256 is transported with the half-pixel resize convention. This small difference is not a new alignment target.

Rendering uses one Lanczos4 affine resampling of the original. Every corrected image was rendered both with a magenta constant border and with `REFLECT_101`; the resulting RGB arrays were **bitwise identical for all 110 images**. Together with the inverse-corner bounds, this verifies that neither reflected nor constant padding contributes to the output. All images were opened and checked for 1024 × 1024 RGB.

I inspected the complete contact sheet, the side-by-side comparisons for the three highest-padding originals (`synthetic-122`, `synthetic-102`, `synthetic-099`), and corrected `synthetic-122` at full resolution. The detached reflected hair and duplicated border content are absent in the reviewed examples. Facial proportions and visible white collars remain intact; some crops cut more of the hair at the image edge or shift the face horizontally. These are ordinary crop changes rather than generated repairs. The variant is reasonable as a separately identified preprocessing experiment, not a production model approval.

There is a real alignment tradeoff: moving eyes upward makes their average position farther from the original FFHQ reference's roughly 46.6% eye height. Removing synthetic border cues is preferable to teaching the model detached hair, but it is not evidence that this geometry will train better. For a clean resolution comparison, first test the corrected framing at 256 in another explicitly separate dataset; do not silently replace the currently running accepted data or jump to 1024 training because these images look sharper.

Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 research/.venv/bin/python research/alignment/solve_1024_bounds.py
research/alignment/.venv/bin/python research/alignment/render_1024_bounds.py
```

The first command uses SciPy's HiGHS solver and writes `eyes42-1024-zero-reflection-proposal.json`, retaining the parent manifest hash, per-image constraints/results, and inverse corners. The second uses the alignment environment's OpenCV and writes the images, manifest, contact sheet, and comparison sheet. The manifest includes source/output hashes and rendering evidence. The review JSON records the scope of visual inspection and known framing compromise. No training, model, hosting, or production configuration changed.
