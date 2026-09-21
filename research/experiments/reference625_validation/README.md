# Completed625 numeric validation and preservation

Prepared tooling only: no real checkpoint was loaded during preparation. This explicitly adapts the pinned `research/runs/reference256-paper-b64/verify-checkpoint-000500.py` into a separate tool for the terminal-success continuation at `research/runs/reference256-paper-b64-resumed500-to625`. The500 validator and active training inputs are unchanged.

## Required state

The tool refuses to load tensors until `completed.json` reports625/40000, continuation `launch.json` reports successful exit from500 to625, all completed output hashes match, the recorded training PID is absent, no other recognized research training/evaluation process is live, and fresh macOS memory-pressure free percentage is at least35%. The process scan covers known model scripts plus common research trainer/evaluator/runner/benchmark/service filenames. It is an observation guard, not a system-wide lock preventing another task from launching later; checks repeat in the worker and before preservation.

Static provenance checks enforce all53 pinned baseline artifacts, exact Python `sys.version`, and Torch/NumPy/Pillow package versions. The continuation must retain the original110-image digest, mirrored virtual count220, batch64, recipe, model architecture, trainer/adapters, and planned arguments. The134-image candidate fork is ineligible. Full input hashes include config, resume, generator export, all625 preview grids, completion records, and archived continuation sources/pins.

## Numeric checks and capture

The worker uses CPU1 and interop1, with five thread environment variables set to1. It does not construct a model or run inference. Trusted own `resume.pt` is loaded via CPU `mmap=True, weights_only=False` because the trainer saves NumPy/Python RNG metadata. `generator-000625.pt` uses CPU `mmap=True, weights_only=True`. External pickle files are never accepted.

It verifies schema1,625/40000 counters, exact sampler seed/consumption, recipe/config/export equality, nonempty G/D/EMA and Adam state, and fixed-latent shape/dtype. Every saved floating/complex tensor, including Adam tensors, is checked for finiteness in262144-element chunks; contiguous CPU tensors are required to avoid hidden reshape copies. Python floating scalars are also checked. EMA tensor keys/shapes/dtypes and values must match the generator export exactly, in bounded chunks.

Only after these checks and stable source hashes does it copy `resume.pt` into exclusively created `resume-000625.pt.copying`, fsync it, verify copy/source SHA256 equality, and atomically publish `resume-000625.pt` with a same-directory hard link. This operation refuses an existing destination rather than replacing it. The directory is fsynced, final bytes are verified, the temporary link removed, and the directory fsynced again. Inputs are rehashed after capture. The final filename is an immutable preservation convention, not a filesystem immutable flag.

An existing final copy, partial copying file or evidence directory causes refusal. If failure occurs after partial or final creation, preserve and inspect the exact files; do not rerun or overwrite automatically. Supervisor evidence records whether those paths exist. This avoids pretending that a final filename alone proves a completed validated run.

## Resource and result contract

Supervisor plus worker watchdog retain an absolute five-minute deadline and4GiB observed RSS abort. The threshold is sampled, not an OS allocation guarantee. Only the owned worker process group is cleaned up; the validator never signals training or other model processes. No automatic retries, downloads, credentials or MPS work.

Future evidence is created in the new run's `validation-000625/` directory: launch, worker preflight/log, detailed `validation.json`, and `supervisor-result.json`. Success requires both a passed validation record and successful supervisor completion, with preserved checkpoint hash equality. Numeric finiteness/export equality does not approve photographic quality; saved PNGs are hashed, not regenerated.

## Commands

Safe static-only preparation check (already executed; no Torch import):

```sh
research/.venv/bin/python research/experiments/reference625_validation/validate.py --check-static
```

After625 has finished and root has reviewed the tool:

```sh
research/.venv/bin/python research/experiments/reference625_validation/validate.py --check-only
research/.venv/bin/python research/experiments/reference625_validation/validate.py --execute
```

These future commands were not run during preparation. `--check-only` performs hashes/metadata/process/memory checks but no tensor imports, loads or capture. `--execute` is required for numeric validation and preservation. Once successfully preserved, do not run again; use the result hashes for the separately reviewed625 evaluator.

## Preparation evidence

Ten lightweight tests exercise completion-counter refusal, incomplete terminal state, changed dataset, missing export hash, live-process detection, exclusive byte-copy publication and no-overwrite/partial-file handling. They use metadata and ordinary byte fixtures, not fake model success. The actual numeric walker is adapted directly from the successful500 validator and remains unexecuted for625 until root approval. `static-check.json`, `test.log` and `evidence.json` record checks. No Torch import occurred in these tests or static checks.
