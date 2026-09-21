# Isolated capacity × feature reconstruction experiment

Resource revision 2 is prepared for root review. The first root-run real smoke was aborted by its 8 GiB RSS guard before any completed update; no factorial training has run. Six synthetic 64px tests pass again with the revised resource guard and observational tracing. Existing paired trainers, masks, checkpoints, and the unused capacity draft remain unchanged.

The predeclared question is whether supervised collar fit improves when opening b32, adding a frozen face-discriminator feature objective, or both. This is an explicit fork of original regions step600, SHA256 `0f433c7ec50d9e80026681125d7fa7e17e9be6e8e2d31b11976fb51d1b0f3fc4`, not an exact resume or a production model.

| Arm | Trainable blocks | Objective |
|---|---|---|
| A | b64–b1024 | Original half-tab/half-rest pixel objective |
| B | b32–b1024 | Same pixel objective |
| C | b64–b1024 | Same pixel objective + 0.05 frozen-D features |
| D | b32–b1024 | Same pixel objective + 0.05 frozen-D features |

Mapping, blocks through b16, all noise strengths and all buffers remain frozen. A/C also freeze b32. Existing Adam moments, step counters, parameter ordering and hyperparameters are restored exactly. B/D append only b32 non-noise parameters with empty initial Adam state and the existing group's LR/betas. The copied transfer function is tested here; the historical `paired_capacity` runner is not imported or executed.

All arms have exactly 300 updates: 50 independently shuffled cycles through the original six train identities, using the same restored parent sampling RNG. Balanced sampling is the only common sampling-policy change from the parent's random-with-replacement behavior. The parent fresh-preservation RNG is restored independently and its per-step latent hashes are checked. Protected/fresh losses, weights, top75% band, constant noise, FP32 execution and LR remain unchanged. Only the original six identities can update weights. Paired10's new 042/058/059 are relabeled `diagnostic-heldout` in memory, with no manifest edits; 028 stays validation.

## Reuse and feature provenance

The runner reuses immutable `paired_regions`/`paired_edit` loading, original masks/losses, state hashing, atomic checkpoints and RNG helpers; `paired_surround` supplies reviewed resource/provenance and 30px evaluation-region helpers. The new update is first-order, with paired pixel+feature backward followed by separate fresh preservation backward and one optimizer step. Feature-off equality to the original update is tested on identical selected pairs and fresh RNG, excluding the deliberately different pair-selection stream.

`clothing_structure/losses.py` is imported unchanged. The frozen D is reconstructed strictly from the root-verified `research/models/ffhq1024-discriminator/D.safetensors`, SHA256 `d91ebf17ce8ef94ba50db60d5452583085a4c4723eb7b294daea92510558f1a5`, with pinned metadata and extraction review. No pickle is executed by this runner. Taps b512/b256 produce spatial256/128 features. Grayscale source/target composites exist only inside the feature objective: protected source pixels match on both branches; fractional area masks and detached target-RMS normalization are the reviewed helper's behavior. D has no optimizer or gradients. Snapshot images are full student outputs, never composited with source/target.

The coefficient 0.05 is fixed and unvalidated; it is not a 5% update contribution. Feature receptive fields mix tab/rest neighborhoods. Only clothing-versus-protected **input-image** gradient support is exact; shared generator parameters can still move faces. Clipped tabs are scored within the immutable clothing masks. Feature semantics, generalization and photorealism require native visual review.

Primary method motivation and decision gates are in `research/reviews/paired650-next-experiment.md` and `paired-factorial2x2-independent-review.md`, including [JoJoGAN §3.1](https://arxiv.org/html/2112.11641v4#S3.SS1) and its [official implementation](https://github.com/mchong6/JoJoGAN). This is a limited paired adaptation experiment, not a JoJoGAN reproduction. NVIDIA code/weights retain the [NVIDIA research license](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt); the repository's MIT application license does not relicense them.

## Explicit resource revision

The original code and failed root smoke were archived in commit `a624aa0`. Run `research/runs/paired-factorial600-smoke` exited worker70/supervisor1: observed watchdog peak8,844,099,584 bytes (about8.24GiB), external peak8,844,705,792 bytes,21.79s wall, zero reported process swaps. It saved checkpoint000 and14 native previews; it did **not** finish an update or produce gradient calibration. This was the application RSS guard, not an observed OS OOM. The original `test.log` and `test-evidence.json` remain preserved.

Root explicitly approved one resource-only revision:10GiB observed RSS ceiling with a stricter35% fresh free-memory guard, retaining CPU2/inter-op1, four hours total, identical math, resolution and feature taps. There are **no further automatic cap raises**. If this retry fails, stop and assess separately tested activation recomputation rather than increasing the cap again. The new smoke uses a new output directory. Current evidence cannot establish either the revised peak or ordinary-update cost.

## Commands and resource gates

Tiny tests only, no real models:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_factorial/test_runner.py
```

After root review, the separate real smoke command is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_factorial/runner.py --execute --smoke --output research/runs/paired-factorial600-smoke-resource-v2
```

Smoke uses arm D and calibration-original, with two **independent** single updates starting from identical parent model, Adam and RNG: one measures pixel/feature gradients, then state is reset and one ordinary update runs. Their final weights, Adam tensors and RNG must match exactly. It reports raw and weighted feature versus paired-pixel gradient norms on common b64+ parameters, plus b32 separately. Raw feature norms are exactly derived as weighted norm / 0.05. Fresh-preservation gradients are excluded from this diagnostic comparison. The diagnostic has two extra autograd traversals, so its time is reported separately from actual ordinary-update duration. It does not advance any official arm. It emits before/after full previews, a research-only checkpoint and resource/invariant reports. Single-pair gradient strength does not establish all-six balance. Smoke-only `smoke-phases.jsonl` logs elapsed time and process peak RSS before/after feature forward, both diagnostic autograd traversals, combined paired pixel+feature backward, and fresh preservation backward. Timings include this small observational logging cost; no additional tensor computations or gradient changes are introduced. Official arms pass no tracing callback.

Only after smoke review, the prospective full command is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_factorial/runner.py --execute --output research/runs/paired-factorial-600
```

Four child processes run sequentially, freeing one arm's models before the next. Before each model import, macOS `memory_pressure` must show at least35% free. CPU uses two intra-op threads and one inter-op thread, no MPS. Each worker samples peak RSS every0.25s and aborts above10GiB; this is an **observed RSS abort**, not an OS allocation cap and can overshoot between checks. A four-hour wall deadline covers setup, all arms and evaluation, enforced by both the worker watchdog and parent subprocess timeout. It must not be lengthened silently. Feature arms retain two G models, CPU paired tensors, the D prefix, and first-order feature/synthesis graphs; real peak memory remains unmeasured until smoke. No real-weight execution is implicit in import or `--help`.

Output must be a new directory under `research/`. No overwrite or exact-resume mode is provided. Checkpoints are complete but deliberately use a new, unapproved factorial format; old trainers/exporters should not treat them as ordinary resume/production artifacts. If an arm fails or a between-arm memory guard defers, retain completed arms and their evidence; do not rerun them or automatically extend the deadline. A reviewed recovery can use hidden `--worker ARM --deadline ORIGINAL_UNIX --source-hashes RUN/source-hashes.json` into its still-new arm directory. The original four-hour deadline and source hashes still apply; expired/incomplete work is a failed bounded experiment, not permission to restart its budget.

## Evidence saved

Milestones0/100/200/300 retain full native RGB and true L-mode grayscale PNGs for six train, 028 validation, four fixed unseen, and three newly held-out edited seeds: 14 outputs each. Metrics include protected, broad clothing, original tab/rest and the reviewed 30px surround/remainder regions wherever annotated. Four-column contacts have enough rows for all14. Milestone snapshots assert model/RNG invariance. Complete atomic checkpoints contain student, complete Adam, all original RNG streams, fixed z, architecture/options, source/pair provenance, freeze invariants, plan/cursor and draw history. Every final checkpoint is deserialized and compared with live model/Adam/RNG state.

The full source30 panel runs **only after step300**, not at every milestone, to keep evaluation bounded. A separate static preflight verifies all32 source entries and PNG/NPZ hashes/shapes without a model; exact latent identity excludes only training000/030. At actual evaluation every remaining source PNG and W is regenerated exactly before rendering the student. All30 are retained, including ineligible faces; there is no filtering to hide failures. Its source-relative top75% MAE is a preservation diagnostic, not a collar-quality score. Source30 and paired10 newcomers stay held out throughout this experiment.

`test.log` / `test-evidence.json` record the six tests: guarded import; actual four-arm gradient/optimizer/freeze behavior with matched sampling; diagnostic-versus-ordinary equality and complete checkpoint/render roundtrip; feature-input support; feature-off equality; source30 static preflight. `source30-preflight.json` records the real data-format checks. The tests use synthetic64 models, not the real1024 models, and establish mechanics only. `pin-preflight.json` verifies frozen source/data metadata hashes without loading models. The revised real smoke timing, RSS and gradients remain a root-owned next action. `resource-v2-test.log` and `resource-v2-test-evidence.json` record the revised six-test pass; original test artifacts remain unchanged.

Predeclared review: require recognizable bounded tabs and plausible black shirts on at least four of six native train outputs, tab AND surround improvement over A, and no more than0.005 mean protected-L1 increase on[-1,1], alongside intact native faces. These are engineering gates, not established perceptual thresholds. Interpret training fit before unseen performance; six pairs cannot establish generalization or adult/hot eligibility. No production approval or export is provided.
