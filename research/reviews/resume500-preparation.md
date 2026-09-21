# Reference500 resume preparation

Prepared 2026-09-21, while factorial C/D runs. **Provenance currently passes; no source/data/runtime mismatch was found.** This is a prepared continuation option, not a launch or quality approval. Only the standard-library guard, package metadata, JSON and streaming file hashes were read. No Torch import, tensor deserialization, model inference, training, signal or download occurred. Detailed observations/hashes are in `resume500-preparation.json`.

## Verified inputs and limits

- Preserved `research/runs/reference256-paper-b64/resume-000500.pt` still hashes to `74d9ec9634bd814d94d418925443e8b25bd33c31319361d7efdf0b9b78205f9c`. It matches the prior CPU numeric validation: exact batch_idx500 /32,000 image presentations, 989 finite tensors, all144 EMA tensors equal to the dedicated generator export. These are prior numeric results, not a new checkpoint load.
- The documented guard's `check(...)` passed all32 source/config/vendor files, its own hash, Python/platform/package versions and recorded numeric environment. Torch2.14.0 is installed; it was not imported. All allowlisted numeric environment overrides remain unset. The original baseline was recorded after launch and cannot retrospectively certify an unrecorded startup environment or guarantee bitwise MPS replay.
- Source `research/models/ffhq256.pkl` matches recipe SHA `7aa4ddeee38e007ce92a1a0bccd386fc6abba6b5c8692bdbe813d3f1987c0722`. The trainer still loads this trusted base before applying resume state, so it remains required even with a full checkpoint.
- Dataset `research/alignment/collar-only/eyes42` matches the trainer's exact filename+PNG digest `3a6e258a9b9ddae8d415b70c3be540ff39cf44e8a37d80ab71540058f1ee0553`:110 original PNGs,220 virtual records with mirroring. No change to crop, resolution, masks or target set is implied.
- Root paused the old process at metrics512, terminal130. Resuming500 deliberately replays12 previously completed but unsaved updates, plus any interrupted in-flight update. Use the preserved verified file, not mutable `resume.pt`, provisional candidate metadata, or generator-only export.

The config's deviations list contains the stale generic text “batch32/microbatch4 default”; actual arguments and recipe correctly record64/8, and D's minibatch-statistics group is8. Do not edit the archived config to clean up that label: it is pinned provenance. This is a documentation inconsistency, not a recipe mismatch.

## Concrete candidate command — do not execute yet

A bounded decision point is total step625:125 further updates /8,000 more presentations, ending at40,000. This is not an automatic continuation to the original15,625-update horizon. The new path below does not currently exist. Wait for factorial work to finish and root's visual/resource decision; recheck free memory then. No numerical environment overrides should be inserted between the guard and launch.

From the repository root:

```sh
test ! -e research/runs/reference256-paper-b64-resumed500-to625 && \
research/.venv/bin/python research/experiments/reference_phases/resume_preflight.py check \
  --run research/runs/reference256-paper-b64 \
  --manifest research/experiments/reference_phases/resume-provenance-reference256-paper-b64.json && \
research/.venv/bin/python -u research/experiments/reference_phases/trainer.py \
  --run research/runs/reference256-paper-b64-resumed500-to625 \
  --resume research/runs/reference256-paper-b64/resume-000500.pt \
  --data research/alignment/collar-only/eyes42 --base research/models/ffhq256.pkl \
  --device mps --threads 2 --batch 64 --microbatch 8 --pl-batch-shrink 2 --mirror \
  --freeze-d-layers 0 --lr 0.0025 --r1-gamma 1 --ema-kimg 20 \
  --augment-p 0 --ada-target 0.6 --ada-kimg 100 --seed 20260920 \
  --steps 625 --checkpoint-every 125 --snapshot-every 125
```

`--steps` is the total, including the restored500, not625 additional updates. Recipe fields stay unchanged; only output path, resume path and total horizon change. `augment-p0` is the initial constructor recipe value; saved augmentation probability, PL mean, optimizer moments, G/D/EMA, RNG and statistics are restored from500. The sampler replays its recorded seed/32,000 consumed items. Thread count is explicit because its CLI default is8 and is excluded from the trainer's recipe comparison.

The guard does not hash checkpoints, so root should reconfirm the preserved file's listed SHA adjacent to launch. It also is not a memory/resource guard. If any preflight mismatch appears, stop and review it; never overwrite the baseline with `record` to force a pass. Keep the original run/config intact. After a new run has its own archived trainer/adapter/config, a deliberately recorded new provenance baseline would be required for a subsequent resume; do not repurpose the old manifest for the new directory.

## Evidence for and against continuing

**For a bounded test:** most of raw500's fourteen intact portraits have plausible black shirts/tabs and somewhat clearer facial features than375. The numeric state is sound. At32k presentations this is still a short reference-style transfer trial. With a constant20k-image EMA half-life, the approximate initial-parameter contribution is33% at32k and25% at40k; EMA's lagging appearance is unsurprising. This arithmetic is not a quality forecast.

**Against open-ended continuation:** raw r4c3/r4c4 have the same face-erasing architectural patterns at375 and500, while both EMA variants retain severe mouth/chin/eye deformation. These are persistent visual failures, not fixed by numeric validity or collar presence. A sixteen-seed grid does not establish unseen diversity/adult-only/hot eligibility, and native256 output has not met the requested photoreal presentation. Identity preservation is not the rejection criterion; malformed faces are.

This evidence does **not** prove irreversible mode collapse or that more training can never help. It also does not justify spending the remaining million-presentation horizon unchanged. If root resumes,625 should be a fresh review point, with the same raw/EMA/untruncated sheets and particular attention to the two catastrophic latents. Do not change recipe simultaneously and call it exact continuation.

The recent logged interval384→512 took3,661.6s for128 updates under the then-concurrent workload. Linear scaling gives roughly60 minutes for125 updates; this is an observed-context planning estimate, not a promise after workload changes. Root must decide whether that additional bounded comparison is worth its opportunity cost after C/D. No launch, reserved resources or new output directory was created by this preparation.
