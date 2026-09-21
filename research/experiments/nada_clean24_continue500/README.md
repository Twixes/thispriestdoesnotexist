# Checkpoint010 → total500 NADA continuation

Prepared only: no models loaded, continuation or restore-only execution not run. This is a separate derivative of the successful ten-update smoke; original files and checkpoint are unchanged. `review.diff` shows the complete source delta.

The ten-update smoke demonstrated CPU feasibility (77.5s total, 6.61GiB worker peak, 2.13s mean update), exact32 source reproductions and functional Adam/RNG restoration. Native review found preserved faces but skin speckling/contrast, no priest collars, and persistent children/headwear. These motivate a bounded experiment with early review, not any quality claim.

## Unchanged mathematics and exact restoration

CPU1/FP32, native1024 source, frozen mapping/ToRGB/source/CLIP, all17 synthesis convolutions plus4px constant trainable, batch1, Adam lr0.0016/betas(0,0.99**0.8), constant noise and psi1 stay unchanged. Identical source mapping, grayscale→224 antialiased bicubic→official CLIP normalization, unit-feature and directional cosine functions are checked against the original syntax trees. There is no identity loss, discriminator, masks, augmentation, EMA, clipping or gradient sanitization.

The pinned trusted checkpoint `ce683dd7e79a4038abdea2c70b3addd056cbc306eeea435c090a17999d20a6fd` is loaded with CPU mmap/weights_only. Its entire nested payload must match the prior successful result. Actual student, complete Adam and Python/NumPy/Torch RNG are restored and compared exactly. Frozen source/CLIP/student state digests and trainable parameter order must match. Stored24 target and32 source centroid features/direction are reused exactly. The bootstrap is never repeated; degenerate edit direction fails.

Before the first update, all eight evaluation latents reproduce both native RGB and grayscale step10 PNG hashes (16 exact encoded files). `initial-restore.json` records this gate. It is an execution requirement, **not yet verified for this new program**. Normal execution already enforces it; a separate restore-only run is optional and exits before any optimizer call.

`latents.npz` stores490 new PCG64(seed202609210800) training vectors, disjoint from all original50 source/train/eval vectors; the same8 evaluation vectors, original32 source z/W and original10 training z are retained. No filtering/search is performed. Targets and source manifests are unchanged. Each update computes its fresh source feature under no-grad; `source_feature_seconds` records overhead and `seconds` includes it. This avoids a long upfront cache delay and uses tiny transient feature storage.

## Review checkpoints and resource policy

- Full atomic checkpoint files and eight native RGB+gray previews at total steps50,100,250,500. Checkpoints contain complete student/Adam/RNG, exact features/direction, recipe/provenance, next latent index and NPZ hash. A `checkpoint-NNN-review-ready.json` appears only after successful serialization/hash checks, deliberate live-state perturbation followed by actual student/Adam/RNG restoration, and native preview reproduction. It includes the preview hashes. Partial preview folders are not completion evidence.
- Snapshot checks ensure RNG and model state stay unchanged. Frozen/finite/nonzero-gradient checks run throughout, and all Adam counters must match total step at checkpoint review. The reload checks do not claim uninterrupted-versus-resumed training equivalence.
- Fresh35% memory required by supervisor and worker, CPU1,12GiB sampled peak RSS,60-minute original absolute deadline including setup, no automatic retries or overrides. Competing known model jobs refuse launch and are checked every2s during training; metadata-only validator commands also conservatively count, so do not run them concurrently. Only the newly owned child process group can be terminated/reaped. These are sampled guards, not OS memory reservation or a hard real-time scheduler.
- Root can inspect50/100/250 and stop if faces fail or texture changes grow without priest/collar progress. There is no automatic quality classifier or production promotion.

## Proposed commands — not executed

Static pins only, no model imports:

```sh
research/selection/.venv/bin/python research/experiments/nada_clean24_continue500/runner.py --check-only
```

Normal continuation (includes exact step10 restoration gate before update11):

```sh
research/selection/.venv/bin/python -u research/experiments/nada_clean24_continue500/runner.py \
  --execute --output research/runs/nada-clean24-cpu-continue500
```

Optional independent zero-update restoration review, only if needed:

```sh
research/selection/.venv/bin/python -u research/experiments/nada_clean24_continue500/runner.py \
  --execute --restore-only --output research/runs/nada-clean24-restore10-check
```

All output directories must be new. There are258 file pins covering actual new source/input paths, the immutable prior checkpoint/results/16 step10 PNGs, unchanged original sources and imported vendor/dependency closure. Six lightweight preparation checks and five harmless child cleanup tests passed (postspawn record failure, SIGINT, SIGTERM, deadline, RSS); these loaded no real models. Lifecycle evidence retains source/harness hashes. Actual continuation restoration, timing/memory and visual outcomes remain unexecuted.
