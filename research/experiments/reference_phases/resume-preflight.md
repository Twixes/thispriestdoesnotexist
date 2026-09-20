# Separate resume provenance guard

`resume_preflight.py` closes the source/environment comparison gap without changing the running trainer. It uses only Python's standard library and installed-package metadata: no PyTorch import, model deserialization, tensor operations, training, or process control.

The current trusted baseline is `resume-provenance-reference256-paper-b64.json`, recorded on 2026-09-21 Europe/Warsaw (2026-09-20 23:09 UTC). It checks 32 files, including:

- Current `trainer.py` against the run's archived `train-source.py` and configured trainer hash.
- Current `smoke.py` FP32 adapter against archived `adapters-source.py`.
- The exact run config and all source files under vendored `training/`, `torch_utils/` (including network operation Python/C++/CUDA/header files), `dnnlib/`, plus `legacy.py`.
- The vendor hashes already stored by the trainer's recipe.
- Python/platform versions, installed Torch/NumPy/Pillow/SciPy/optional pyspng/Ninja/SymPy versions, and an explicit allowlist of numeric/runtime environment settings.
- The guard's own hash and vendor source inventory, so changed, missing or newly added source files fail closed.

The current trainer and adapter match their archives, and PyTorch is 2.14.0. The baseline records the invocation environment now; all listed numeric environment overrides were unset. **It cannot retroactively certify an unknown earlier environment or inspect environment variables inside an already-running process.** Package versions are metadata, not full binary-wheel fingerprints, and this check does not prove exact numerical MPS continuation. Those limits are distinct from the fail-closed comparison of the values actually recorded.

## Future resume command

Run from the repository root with the same research Python environment. This example uses a new output directory, preserving the original archived config. Execute it only when deliberately resuming a completed/paused checkpoint; the guard does not stop another running job.

```sh
research/.venv/bin/python research/experiments/reference_phases/resume_preflight.py check \
  --run research/runs/reference256-paper-b64 \
  --manifest research/experiments/reference_phases/resume-provenance-reference256-paper-b64.json \
&& research/.venv/bin/python -u research/experiments/reference_phases/trainer.py \
  --run research/runs/reference256-paper-b64-resumed \
  --resume research/runs/reference256-paper-b64/resume.pt \
  --device mps --threads 2 --batch 64 --microbatch 8 --mirror \
  --steps 15625 --checkpoint-every 125 --snapshot-every 125
```

A mismatch exits nonzero, preventing the second command. Do not run `record` over a mismatch to bypass the refusal; inspect and document whether the change warrants a new experiment or an explicitly reviewed compatibility decision. `record` refuses to overwrite an existing baseline. A newly selected run needs its own archived trainer, adapter, config and deliberately recorded manifest. Reusing the same output directory would replace its config on resume and consequently fail this immutable baseline on the next preflight; preserving a new output directory avoids that ambiguity.

The trainer still performs its own model/optimizer/recipe/dataset checks when it opens the checkpoint. This guard intentionally does not hash or load large model files, establish model quality, or replace those checks. Keep the guard and launch adjacent and avoid source/environment changes between them. Do not introduce launch-only numerical environment overrides after the preflight.

## Verification

The actual baseline check passed with 32 files and PyTorch 2.14.0; result: `resume-preflight-check.json`. Seven temporary small-file fixture tests passed in `resume-preflight-tests.log`: clean input, changed trainer, changed Torch version, changed numerical operation, missing archived adapter, added source file, and refusal to overwrite a baseline. The fixtures do not mutate the real trainer, archives, packages, or running jobs.

```sh
research/.venv/bin/python -m unittest discover \
  -s research/experiments/reference_phases -p 'test_resume_preflight.py' -v
```

No checkpoint was loaded and no training or deployment was started while implementing or testing the guard.
