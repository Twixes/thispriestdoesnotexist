# Matched32 evaluation prepared for reference625

Prepared tooling only. No step625 checkpoint was loaded and no inference ran while preparing this directory. Old500 code, latents and outputs remain untouched. This evaluator accepts only the same110-image continuation at `research/runs/reference256-paper-b64-resumed500-to625`; the134-image candidate fork is deliberately ineligible.

## Scope

Generate exactly96 native256 PNGs: the same32 saved z vectors for raw G at psi1, EMA at psi1 and EMA at psi0.7, with constant noise and FP32 CPU inference. Preserve all outputs without selection, restoration, compositing or retries. Save three8×4 contacts, copied byte-identical latent NPZ, input/output hashes, timing/RSS and completion records. No extra32 seeds are added: these are now a longitudinal comparison against500, not a new unseen sample. Final visual review remains separate.

`evaluate.py` adapts the reviewed500 worker/supervisor explicitly. It imports only the original pinned `sha`, `write_new` and `memory_guard` stdlib helpers. Calling the original worker would require bypassing its hardcoded500 checkpoint/schema pins; instead the separate625 worker verifies its own fixed provenance and625/40000 schema. No globals or original pins are monkeypatched. The original vendor licenses continue to apply; no vendor code was modified.

## Required completion and provenance

Before any execution, root must review the completed continuation and preserve a separate `resume-000625.pt` beside its `resume.pt`. The evaluator does not capture or overwrite checkpoints. Independently supplied checkpoint and config SHA256 values are mandatory; they are not discovered and approved automatically. It checks:

- all52 pre-pinned baseline source/artifact hashes, exact Python `sys.version`, and installed Torch/NumPy/Pillow package versions;
- supplied immutable checkpoint/config hashes;
- successful continuation `launch.json` plus `completed.json` at625/40000 and matching launch output hashes;
- original recipe, architecture, trainer/adapters, source data digest and110 originals plus mirror count220;
- only the planned run/resume/step/path argument changes;
- saved500 latent NPZ provenance, exact seed order and regenerated PCG64 float32 values;
- full checkpoint config/recipe/counters and unchanged fixed16 latents upon authorized CPU load;
- finite G/EMA state and outputs, unchanged model state during inference, and stable inputs before/after.

The trusted own checkpoint needs `weights_only=False` for NumPy/Python RNG metadata, as in the500 evaluator. This is not a loader for externally supplied pickle files. A generator-only snapshot is insufficient because the comparison needs both raw G and EMA.

## Resource limits

The original500 limits are retained: CPU1, interop1, batch1, fresh macOS memory-pressure free percentage at least35% before numerical/model imports, ten-minute absolute evaluation deadline, and6GiB observed RSS abort sampled by worker/supervisor. The supervisor explicitly sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1`; the worker sets Torch intra-op and interop threads to1. Version checks and these imposed thread settings do not establish full numerical-environment parity. The RSS threshold is not a hard OS allocation cap. No MPS, downloads, credentials, training or automatic retry. Only one Generator is resident at a time; raw then EMA weights are loaded into it, alongside a CPU mmap checkpoint. Original500 evaluation completed in about10.7s at992MB sampled RSS; this is evidence from500, not a promise for625.

Output must be a NEW direct child of the resumed500-to625 run. Both `evaluation.json` and successful `supervisor-result.json` are required before treating outputs as complete.

## Commands after root review

First perform static-only verification, which is already tested:

```sh
research/.venv/bin/python research/experiments/reference625_evaluation/evaluate.py --check-static
```

Once625 exists, use the independently reviewed hashes in place of the placeholders:

```sh
research/.venv/bin/python research/experiments/reference625_evaluation/evaluate.py \
  --check-only --checkpoint-sha256 REVIEWED_CHECKPOINT_SHA256 \
  --config-sha256 REVIEWED_CONFIG_SHA256

research/.venv/bin/python research/experiments/reference625_evaluation/evaluate.py \
  --execute --checkpoint-sha256 REVIEWED_CHECKPOINT_SHA256 \
  --config-sha256 REVIEWED_CONFIG_SHA256 \
  --output research/runs/reference256-paper-b64-resumed500-to625/matched32-000625
```

Do not run these future commands before completion/root review. Do not refresh pins to accept a mismatch.

## Preparation evidence

`test_preflight.py` uses temporary stdlib fixture files, not model checkpoints, to verify the completion/config/hash gates and rejection of a mismatched Python version. Its saved-latent test uses only NumPy and the existing small NPZ. Eight tests pass; they assert Torch is not imported. `static-check-python-v2.json` verifies52 real pinned artifacts plus exact Python and Torch/NumPy/Pillow versions without Torch import. Earlier test/static logs and evidence are retained unchanged; the narrow Python-check revision is recorded in `test-python-v2.log` and `evidence-python-v2.json`. The first test attempt exposed macOS `/var` versus `/private/var` path normalization in fixtures; expected paths now resolve symmetrically. That initial log is retained. These checks do not claim successful625 model inference.
