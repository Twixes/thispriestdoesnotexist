# Prospective expanded 256px dataset

This directory is a candidate input for an explicitly declared dataset fork. It is not an accepted training dataset, an exact same-dataset continuation, or model output. No training was launched.

- `aligned110--*.png`: all110 existing `research/alignment/collar-only/eyes42` PNGs, copied byte-for-byte.
- `whole24--141.png` through `whole24--164.png`: all24 previously reviewed whole-image grayscale256 derivatives, copied byte-for-byte.

All134 filenames are unique and all images were decoded to verify RGB256×256 dimensions. Every output hash equals its source hash. All24 new images have exactly equal grayscale RGB channels. Source hashes were checked before and after assembly; the110-image source digest matches `3a6e258a9b9ddae8d415b70c3be540ff39cf44e8a37d80ab71540058f1ee0553` using the archived reference trainer's exact filename-plus-file-bytes algorithm.

The expanded dataset digest is `2d50e93823001b08ae30eb4e0e1086050b4c8f6808cf791e57e60e176a44ebeb`. The110 aligned crops and24 unaligned full-frame images intentionally mix framing distributions: eye positions, face scales and tilts are not uniform. Changed image count, filenames and content require a declared dataset fork; no resume compatibility is implied.

`manifest.json` maps every source/output path and hash, records versions and script provenance, and preserves the original generation ancestry of the24 additions. No images were cropped, resized, recolored or re-encoded during assembly. No automatic removals or held-out quality metrics were invented.

Reproduce from the repository root into a new destination:

```sh
python3 research/reviews/assemble-priests134-expanded256.py --output research/data/priests134-expanded256-repeat
```

The script refuses an existing output directory. Originals and active training inputs remain unchanged.
