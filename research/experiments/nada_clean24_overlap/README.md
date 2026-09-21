# Optional, bounded NADA overlap supervisor

**Prepared only; no models were loaded and no actual smoke was launched.** This is a separate resource policy, leaving `nada_clean24` and the active reference trainer unchanged. It is feasible to offer as a guarded experiment given current observed headroom; it is not evidence that full1024 NADA backward fits or improves throughput.

Root observed reference PID27880 at62–64% free system memory across steps528–566 despite changing RSS. The read-only preparation check here observed60% free and matched the exact current reference process. Root later observed57% free atstep592, so earlier readings must not be reused for launch. These are point observations, not bounds on MPS allocations. CPU training can contend with MPS for unified memory/bandwidth and slow both jobs. Fresh execution checks, continuous pressure observations and aborts are required; the unknown NADA peak remains unmeasured.

## Unchanged worker and inputs

`runner.py`, `worker.py`, `inputs.json` and `latents.npz` are byte-identical copies of the original prepared experiment. The original entry runner is retained under its original filename because it archives/loads its inputs relative to HERE. `policy.json` records each original path/hash; `pins.json` retains original provenance and additionally pins all four **actual overlap copy paths**, the new supervisor/policy, and original pins/tests.235 files are verified. The worker still uses exactly24 accepted targets,32 source latents,10 train latents and8 evaluation latents. Newly generated targets are not included.

All mathematical settings, CPU1/FP32, gradient/frozen-state checks, bootstrap, snapshots and full optimizer/RNG reload are unchanged. See the [original experiment](../nada_clean24/README.md) for those details and remaining unexecuted gates. The new supervisor invokes the local unchanged runner's existing internal worker mode; it does not monkeypatch or bypass its pin checks, atomic readiness handshake, fresh35% worker guard,12GiB RSS watchdog or20-minute deadline. Its provenance records intentionally differ to identify the overlap policy; training mathematics does not.

## Explicit process and memory policy

- `--allowed-reference-pid 27880` is mandatory. The PID must initially exist and match the pinned full trainer argv, original start time, MPS device, resumed500-to625 run and command SHA256 `307ca34e1050453a6bc1873cb0eba596ba5cf850759eb055f92c597ee7c5da1a`. A reused PID, different command/device/run or another known heavy research process refuses launch. Live checks continue while NADA runs. If reference finishes normally, NADA may continue; any new/reused process still has to pass the checks.
- Fresh system-wide free memory must be at least55% immediately before creating the run. While the owned child lives, query `memory_pressure` every two seconds, with a one-second command timeout. Abort on any free observation below35%, parse/command failure, or a sampled observation gap above five seconds. Process-monitor failure also aborts. Scheduling is not hard real-time; stale observations cause failure rather than silently extending the cadence.
- Retain12GiB sampled NADA RSS and the original20-minute absolute deadline, including setup. No automatic retry, cap increase, launch deadline override or training change. These are sampled aborts, not OS allocation limits.
- Every termination path signals only the newly created NADA session/process group and reaps it. The allowed reference PID and its group are explicitly forbidden signal targets. The original trainer is never paused, restarted or signaled. Python cleanup cannot execute after supervisor SIGKILL/machine failure; the unchanged worker also retains its own bounded watchdog.

## Commands after root review

Read-only source/process/pressure check:

```sh
research/selection/.venv/bin/python research/experiments/nada_clean24_overlap/supervise.py \
  --check-only --allowed-reference-pid 27880
```

Proposed actual command, **not executed**:

```sh
research/selection/.venv/bin/python research/experiments/nada_clean24_overlap/supervise.py \
  --execute --allowed-reference-pid 27880 \
  --output research/runs/nada-clean24-overlap-cpu-smoke10
```

Use the exact pinned current PID; this policy cannot be reused for a later reference process without a separately reviewed preparation. Model artifacts use the new output directory. Pressure observations, allowed-process observations and overlap provenance go into its adjacent `-supervision` directory, preserving the unchanged worker's strict startup-directory handshake. Both directories must be new. Success requires worker result/provenance and the overlap supervisor result; partial files are not success.

## Lightweight evidence

`policy-test.log`: twelve tests pass, including wrong PID/command/device/run, reused start time, additional job, reference625 validation, competing overlap coordinator (with own-PID exclusion), initial54% rejection before any spawn, malformed/stale pressure, copied-input pin coverage and the actual copied worker's readiness preflight without model imports. `lifecycle-evidence.json`: seven harmless child/grandchild fixtures pass—post-spawn readiness-record failure, SIGINT, SIGTERM, deadline, RSS, low pressure and failing pressure. Every owned child/group was reaped; a separate dummy reference remained alive and all recorded signals targeted only the owned group. The fixture reference was cleaned up separately by the test harness after those assertions; no real research process was controlled.

Existing original tests/source remain untouched. Actual source PNG/W reproduction, NADA backward memory/time, trained outputs and checkpoint reload are still deferred to an explicitly reviewed smoke. Ten updates measure feasibility/initial movement only, not convergence or quality approval.
