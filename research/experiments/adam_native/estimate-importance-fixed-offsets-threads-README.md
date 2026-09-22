# Prepared CPU-thread comparison for fixed-policy EMA importance

This is a separate, minimally changed copy of the frozen fixed-policy estimator. No native benchmark or model load was performed during preparation. The parent estimator remains unchanged. The live-input target for the next comparison is the retained checkpoint250 from `adam-native1024-output-rank1-fixed-offsets100-to500-v1`; the parent process was stopped after a later photographic-quality failure, so this diagnostic does not imply a passed training or quality gate.

The root-owned comparison starts with four pairs and two CPU intra-op threads:

```sh
research/.venv/bin/python research/experiments/adam_native/estimate_importance_fixed_offsets_threads.py \
  --run research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1 \
  --step 250 --pairs 4 --threads 2 \
  --output research/runs/adam-fixed-offsets250-importance4-threads2-v1
```

Only use a fresh output directory. The explicit `--threads` choices are1,2,4,8, default1. The supervisor saves that setting in the immutable protocol; the worker reads it there, sets PyTorch intra-op threads and verifies the observed setting. Inter-op threads remain1 and are now explicit in protocol/runtime. Parent and variant source files are pinned and archived into each output. This does not run multiple processes, change batch size or schedule later benchmarks automatically.

The parent `make_sample_plan`, `pair_gradients`, checkpoint validation and complete worker body from checkpoint loading through final result are unchanged. Consequently checkpoint250 EMA weights, 34 fixed-offset policy, sample-plan seed, real IDs/flips, latent vectors, synthesis-noise seeds, phased loss gradients, precision, accumulator normalization and saved outputs use the same logic. Different CPU reduction scheduling can still produce numerical differences: compare actual images and normalized accumulator diagnostics against the completed one-thread baseline rather than assume bitwise equality or equivalent rankings.

Memory and failure policy remain unchanged:35% available memory at start,20% runtime,12GiB owned process-group RSS,512MiB swap-growth limit,1200seconds for four pairs or14400seconds for1000. No restart, deadline extension, precision/objective change, lower-resolution fallback or concurrent heavy worker is introduced. Root should try two threads first and four only if the measured result makes that useful. Eight is available for a separately scheduled experiment, not an automatic escalation.

The bounded tests validate the parent SHA; exact AST equality of sample plan, pair gradients and input validation; equality of the complete computation/render/save suffix and supervisor after only the declared configuration/path substitutions; explicit/default thread propagation into saved protocols; parent/new source archival; equality of every other protocol field; and rejection of invalid counts before input reads/output creation. These tests use temporary opaque fixtures and mocks for protocol construction, not successful native inputs or synthetic FI results. Native speed, peak memory, image equality and FI numerical agreement remain to be measured.

The main-adaptation runner's step500 gate is unchanged. This computation experiment does not make checkpoint250 an approved source for main adaptation or deployment.
