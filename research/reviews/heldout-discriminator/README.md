# Held-out discriminator diagnostic

This is a CPU-only diagnostic of the paused `aligned110-frozen4` discriminator at step 1000. It scores the 110 accepted eyes42 training portraits and all twelve newly generated, separately aligned held-out portraits. It does not train, use MPS, update an optimizer, assess generator output quality, or approve a production model.

Reproduce from the repository root, after the held-out alignment review is complete:

```sh
research/.venv/bin/python -u research/reviews/heldout-discriminator/diagnose.py \
  > research/reviews/heldout-discriminator/run.log 2>&1
```

The script requires at least 20% free memory reported by macOS `memory_pressure`. It verifies the official pretrained source hash, paused step/freeze settings, training dataset hash, aligned-image hashes, held-out source hashes, and alignment-manifest lineage. It checks the resume file hash again after evaluation. The pretrained NVIDIA discriminator architecture is reconstructed with the run's four frozen input-side layers, then strictly loaded from the task's own discriminator state. The resume contains a NumPy scalar unsupported by the restricted weights-only loader; `weights-only-loader-error.log` preserves that first failed attempt. Full deserialization is used solely for this trusted locally produced resume, mapped to CPU. No optimizer object is instantiated or state applied to one.

Two contexts are reported separately:

1. **Batch one:** every target is scored alone. Minibatch variance is zero, so NVIDIA's epsilon produces a constant standard-deviation feature of `sqrt(1e-8) = 0.0001`. Both splits use precisely this context, but it differs from the minibatch context used during training.
2. **Three common anchors plus target:** the first three sorted training images are fixed anchors in positions 1–3, and each scored target occupies position 4. The three anchors themselves are excluded from the training targets in this secondary protocol, giving 107 training versus 12 held-out targets. They are still included in the primary batch-one protocol. The identities and hashes are recorded. This is one chosen common context, not an average over possible anchor sets.

All inputs use their existing 256 × 256 RGB pixels divided by 127.5 and shifted by −1, without augmentation or resizing. Evaluation uses float32, inference mode, and two CPU threads. Mean, median, range, standard deviation, positive fraction, and sign mean are descriptive **raw logit** statistics, not calibrated probabilities or measures of attractiveness/photorealism. Scores are not directly comparable with augmented training-log means.

The [alignment review](../../data/validation12/alignment-review.md) documents higher reflected padding in the held-out set: 4.60% average versus 1.88% for training. Results retain all twelve held-out portraits and additionally describe ≤5% and >5% padding strata. The strata are small, observational subsets; they do not remove pose, generation, prompt, or other domain differences. A gap between splits alone cannot establish memorization or explain the generator's quality failure.

`results.json` is the complete numerical record, including every target, both protocols, per-image padding, strata, exact checkpoint/source/code/provenance hashes, elapsed time, and memory use. `summary.md` interprets those measurements with the above limitations. No generated image or production model is selected by this diagnostic.
