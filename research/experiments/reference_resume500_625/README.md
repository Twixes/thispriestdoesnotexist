# Prepared reference500 → 625 continuation

Prepared only: no model import, deserialization, inference or training was performed. `preflight-check.json` records the successful standard-library check of 32 source/config files and the existing runtime baseline, plus streaming hashes of the immutable step-500 checkpoint/base and the 110 original PNGs. The original run and checkpoint remain unchanged. This directory is separate from the proposed new training output.

From the repository root, the read-only preparation check is:

```sh
research/.venv/bin/python research/experiments/reference_resume500_625/launch.py --check-only
```

Only after root reviews this source and the completed factorial results, the proposed one-shot launch is:

```sh
research/.venv/bin/python research/experiments/reference_resume500_625/launch.py --execute
```

The execution path requires terminal successful C/D recovery and complete ABCD comparison, no existing reference/factorial training processes, at least 35% system-wide free memory, and an absent `research/runs/reference256-paper-b64-resumed500-to625` directory. It preserves the numerical environment checked by the existing provenance guard. It never launches a retry, resumes a partial new output, regenerates a baseline, or changes a trainer. The checkpoint at step 500 restores augmentation probability, optimizer, RNG, sampler, PL mean and statistics; `--augment-p 0` describes the unchanged constructor recipe, not a reset of checkpoint state.

The command retains MPS, two CPU threads, batch64/microbatch8, mirror, no frozen D layers, LR.0025, gamma1, EMA20k, ADA target.6/speed100k, PL shrink2 and seed20260920. It stops at total step625: 125 further updates and 8,000 further image presentations. It intentionally replays the 12 completed but unsaved updates after step500 in the original paused run. The initial step500 previews are reproduced before training; final625 previews and both checkpoints are saved by the unchanged trainer.

A proposed two-hour child-process timeout is an operational bound, not a training hyperparameter. On timeout, interrupt, supervisor failure or completion, the launcher reaps only its owned process group; the original checkpoint remains available. There is no automatic retry. Interrupted work since step500 may be lost because the unchanged checkpoint interval is125. The initial free-memory guard is not a continuous RSS or MPS allocation cap. Do not run a concurrent heavy job after the launch check. The supervisor must remain alive; SIGKILL or machine failure cannot run Python cleanup.

The launcher archives itself, its pins and the unchanged trainer/adapter in the new output. A successful process result still requires independent numerical validation and native visual review. It does not approve a model for production. Before any subsequent resume of the new run, deliberately record a new provenance baseline for its new config/archive; do not replace the original baseline. The baseline cannot retroactively certify unknown earlier environments, fingerprints package versions rather than full binaries, and does not establish bitwise MPS continuation.

`test_lifecycle.py` reuses the prior recovery fixture approach: temporary fake paths/preflight/command and harmless sleeping child plus grandchild, while exercising the real unchanged launcher main/wait/cleanup/signal code. All four cases passed in 1.61 seconds: timeout, SIGINT, SIGTERM and a one-time record-write failure after Popen. Each child was reaped with return code -9, both dummy PIDs and their group were absent, and no checkpoint/completion appeared. Actual model commands were never executed. Evidence is `lifecycle-evidence.json`; this establishes these bounded process-lifecycle behaviors, not trainer or MPS correctness.

## Independent configuration review

No new confirmed mathematical correctness error was found. The baseline matches the vendored NVIDIA `paper256` preset (`vendor/stylegan2-ada-pytorch/train.py:154–193`): batch64, mbstd8, half feature maps, LR.0025, gamma1 and EMA20. Its ADA100k and disabled EMA ramp-up match upstream transfer settings (`train.py:313–315`). Microbatch8 preserves the loaded D's mbstd group8. The accumulation gain, lazy regularization, optimizer compensation, EMA direction and ADA counters were checked against the upstream loop in `research/reviews/reference-trainer-audit.md`; none is silently changed for this continuation.

Concrete risks and limits remain:

- **Gradient health is incompletely observed.** With `verify_updates=false`, `trainer.py:296–306` sanitizes NaN/Inf gradients before stepping, as upstream does. Finite checkpoints do not prove every unsanitized gradient was finite. All 499 logged records from steps2–500 contain finite numbers, but this also does not test every gradient. There is no evidence here identifying gradient sanitization as the cause of the facial defects. Enabling verification would add checks/cost to a separately reviewed launch, so this proposal preserves the original setting.
- **EMA carries substantial source weights.** With no ramp-up and a constant20k half-life (`trainer.py:309–314`), the approximate retained initial-parameter coefficient is33% at32k presentations and25% at40k. This is correct for the chosen preset and helps explain why raw and EMA outputs differ, but it does not account for or repair raw-G architectural face failures. It is not a forecast that another125 steps will improve faces.
- **The target distribution is small.** There are110 independent portraits,220 mirrored entries, not220 independent subjects. The discriminator's last32 logged real-sign means average0.769 versus ADA target0.6; step500 p is0.21248. These are training-set measurements with augmentation and minibatch context, not held-out overfitting proof or calibrated quality metrics. Background/facial entanglement and limited target variety remain hypotheses, not established implementation bugs.
- **The fixed preview is insufficient.** Step500 native review rejects the model for production, including two severe raw-G face/architecture failures and broader EMA defects. Any decision after625 must include the same fixed comparisons and separately retained unseen latents; successful loss logging or finite tensors cannot substitute for that review.
- **Single-device MPS is not an eight-GPU paper reproduction.** It uses the upstream single-device microbatch accumulation behavior and FP32 adapters. No exact MPS interrupted-versus-uninterrupted equivalence test has been established. The separate guard prevents known code/config/runtime drift; it cannot remove that limitation.

The archived config has a stale generic deviation sentence saying batch32/microbatch4, while its actual recipe and arguments correctly say64/8. This is a documentation inconsistency; the immutable archive is intentionally not edited. No speculative loss, normalization, ADA, EMA, freezing or data change is proposed.
