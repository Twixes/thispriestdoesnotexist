# Isolated completed-run evaluation

`evaluate.py` is prepared for evaluation **after the active CPU training process finishes**. Preparation checks have not loaded a real checkpoint or run new generator inference. Do not run it concurrently with the training process merely because previews exist.

From the repository root, once the paired run has completed at step 200:

```sh
research/.venv/bin/python research/experiments/paired_edit/evaluate.py \
  --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle \
  --checkpoint research/runs/paired1024-cpu-200/resume.pt \
  --sources research/data/ffhq-paired-sources32/manifest.json \
  --expected-step 200 \
  --output research/runs/paired1024-cpu-200/evaluation32
```

The output directory must be new or empty. There is a fresh macOS `memory_pressure` guard requiring at least 25% available, and the checkpoint must match both the requested step and its sibling configuration's completed total. The latter is a completion guard, not a process-exit detector: the caller must still wait for the training process to exit.

The script hashes and loads the same open checkpoint inode with `weights_only=True`, releases optimizer data before model construction, and verifies source weights/metadata, architecture, checkpoint configuration, trainer/vendor code and library versions, source/frozen state digests, and finite model state. It uses one PyTorch CPU thread and one interop thread. There is no CUDA/MPS use, training update, model export, or production approval.

Only the fixed source manifest's 32 `entries` are enumerated. Its separate original-training-seed reproduction is excluded. Any exact Z overlap with a checkpoint training pair is also excluded by tensor hash. Before evaluating each remaining source, the script checks saved source/latent hashes, finite float32 Z/W shape, exactly recomputed mapped W, and exact source PNG uint8 regeneration. A mismatch stops evaluation rather than loosening a threshold.

Each sample receives actual sequential source/student synthesis forwards at its saved W, with constant noise and FP32. Frozen mapping verification ensures this W is also the student's current mapping output. Saved artifacts include native student RGB PNG, native source/student grayscale PNGs, hashes, a clearly labeled diagnostic contact sheet, per-image timings, source/checkpoint provenance, runtime/RSS, and fixed-region grayscale MAE. The raw RGB student is never composited with source pixels. Grayscale is the same fixed differentiable luminance operation used by the trainer; contact-sheet concatenation is diagnostic only.

Top 75% and bottom 25% are horizontal image regions, **not** detected face and clothing masks. Their unclipped luminance MAE is not an identity score, collar detector, attractiveness assessment, or quality approval. Raw out-of-range generator values may produce errors above one. No new clothing ground truth or class predictions are fabricated. Unseen children, women, hats, or poor faces remain in the evaluation rather than being silently filtered out.

Memory has not yet been measured for this evaluator. Previous native Mac generator measurements exceeded 3 GiB; two models plus a checkpoint may require more. The script records actual RSS and has no claim to fit a 3 GiB container. The 32 sources are processed sequentially, and only small contact thumbnails remain resident between samples.

Lightweight preparation tests run without importing PyTorch or generating images:

```sh
research/.venv/bin/python -m unittest discover -s research/experiments/paired_edit -p test_evaluate.py -v
```

These tests cover metadata rejection, incomplete-run rejection, exact-latent identity by content, and region statistics. They do not substitute for the deferred real-checkpoint evaluation.
