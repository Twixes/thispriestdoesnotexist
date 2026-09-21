# SDXL-Turbo pilot preview

Run `research/.venv-flux-train/bin/python research/reviews/sdxl-pilot-preview/build.py`
from the repository root. The builder automatically includes the standard development
base run, step20, and step100/250 when their directories exist. Override with
`--base-run PATH --checkpoint 20=PATH --checkpoint 100=PATH` as needed.

For the frozen fallback, use `--steps 4 --output research/reviews/sdxl-pilot-4step-preview`.
This copies the UI assets and builds separate data without replacing the one-step data.
Run records must match the selected step count. Explicitly pass retry paths with
`--base-run`/`--checkpoint`; the builder never silently selects a different run revision.
The two views link to one another and label their inference configuration. Below900px,
each case becomes a labeled2×2 image grid so every arm is visible without horizontal
scrolling. The checkpoint selector initially chooses the latest included horizon.

Serve the repository root and open `/research/reviews/sdxl-pilot-preview/`, or use the
existing research preview server route `/reviews/sdxl-pilot-preview/`. No network API,
model inference, or image transformation is required by this preview. The grayscale
control is a CSS display filter over the unchanged native PNG. The full-image link always
opens original RGB; 512px native images display at their native dimensions on wider screens.

All24 development cases and all four arms remain visible. Completed image files, records,
runtime and protocol hashes are checked. Every displayed matched row must use equal noise
value hashes, initial generator state and post-inference generator state across its arms.
Pending and failed outputs remain explicit rather than being filtered out. Partial runs
can be rebuilt safely; an incomplete set is never described as a completed evaluation.
`manifest.json` binds every included source/output to its hash; `data.js` mirrors that data
for a static browser without fetch requirements. No timing estimate, quality verdict,
production approval, or image selection is inferred by the builder.

`review-step20-first12.json` is an agent visual review of all four arms for legacy00–07
and dev00–03. It records source hashes and conservative per-image labels. Root reviews
dev04–15 separately; this half-review does not establish full-set diversity or rule out
teacher copying.
