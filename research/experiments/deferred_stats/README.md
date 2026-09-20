# Deferred statistics transport experiment

CPU equivalence passed on 2026-09-21. The adapter is experimental and is **not installed in either active trainer**. One capped MPS Gmain probe was attempted and failed during warmup; no usable MPS timing or equivalence result was obtained. NVIDIA loss, augmentation, and statistics code remain unchanged and retain their upstream license in `../../vendor/stylegan2-ada-pytorch/LICENSE.txt`; these independent harnesses reuse them through the existing reference smoke adapters.

`adapter.py` clones detached values when each report is made, preserving report-time observations despite later in-place mutations. At flush it packs values by device and original dtype, transfers each pack to CPU, then replays the original CPU reporting sink in original report order. The original CPU float32 reductions and float64 counters remain intact. Empty observations and repeated names are preserved. The queue must be flushed before collector reads/updates, ADA updates, or checkpoint capture; `assert_empty()` provides an explicit boundary check. This design avoids preserving an autograd graph and does not reduce statistics on MPS.

## CPU evidence

`check_cpu.py` ran the actual pretrained FFHQ256 Gmain/Greg/Dmain/Dreg with one CPU thread, batch 1, PL shrink 1, full bgc augmentation at p=0.25, gamma 1, lazy-reg optimizer compensation and gains 1/4/1/16. Baseline and candidate used separate subprocesses with identical model state, inputs, latents, and RNG. Actual optimizer updates were performed.

- All phase gradient and updated-parameter hashes, final G/D state, PL mean, RNG, collector statistics, ADA probability calculation, and checkpoint counters matched exactly; gradients/weights were finite.
- Four phases produced 14 report calls; deferred replay used four packs, one flush per phase.
- Fixtures passed for subsequent tensor/NumPy/list mutation, float16/32/64, int64, bool, Python scalar/list, noncontiguous values, empty values, return identity, original replay order/dtype/content, and the unflushed boundary guard. Nine fixture reports required five dtype packs.
- Baseline/candidate peak process RSS: 2,387,247,104 / 2,251,276,288 bytes. Preflight free memory: 24% / 19%.
- Baseline/candidate measured CPU phase-and-hashing time: 8.063 / 8.001 seconds. These are correctness-test timings, **not evidence of GPU acceleration**.

Evidence is in `cpu-equivalence.json`, `fixtures.json`, `cpu-{baseline,deferred}.{json,log}`, `memory-{baseline,deferred}.txt`, and `check-cpu.log`. Source hashes are included. Python compile checks passed for all three scripts.

## Capped MPS probe outcome

The authorized `Gmain --trials 1 --memory-cap-gib 4` command passed the fresh >=25% memory gate but exited 1 in the first excluded warmup, during the deferred variant's generator synthesis forward. The allocator reported **4.23 GiB allocated**, 13.33 MiB other allocations, a **4.00 GiB configured limit**, and a failed 128 MiB allocation. This is the exact reported allocation state; the configured limit should not be read as an observed all-times process-memory ceiling. No backward, A/B comparison, or measured pair completed. No optimizer exists in this harness, and no active trainer was changed.

The process exited and post-exit system memory was 27% free. The exact preflight percentage was not logged, so only successful passage of the >=25% guard is known. `mps-gmain-cap4.log` preserves the full traceback; `mps-gmain-cap4-failure.json` records the failure and source hashes; `memory-after-gmain-cap4.txt` records a subsequent memory check. **Dmain was not run**, because the authorized prerequisite was a passing Gmain probe. No larger-cap retry, Greg, or Dreg was attempted. The CPU equivalence result remains valid; MPS equivalence and performance remain unverified.

## MPS measurement harness

`benchmark_mps.py` requires an explicit `--phase` (`Gmain`, `Dmain`, `Greg`, or `Dreg`), explicit `--memory-cap-gib`, and a fresh `--output`. `--trials` is bounded to 1–3; each command runs one excluded warmup pair and that many measured A/B pairs. Microbatch is fixed at 8; PL shrink is 2. It refuses to start below 25% system memory free or without the saved CPU equivalence pass. Do not launch alongside existing training without coordination. Gmain and Dmain are the proposed first measurements; PL/R1 timing remains deferred.

Both G and D are resident, even for a single selected phase. Their state tensors total 195,778,124 bytes (about 187 MiB), with gradients for only the selected trainable model (about 95 MiB G or 92 MiB D). There is no optimizer, EMA, full dataset cache, optimizer update, or checkpoint write. Real input is one diagnostic image repeated eight times (6 MiB FP32); matched latent inputs and saved buffers/RNG are retained. Gradient copies for A/B comparison are on CPU and outside the timed region. Activations, augmentation workspaces, backend compilation/cache, and second-derivative graphs can dominate memory: model-state sizes do **not** estimate actual peak residency. A previous full training smoke reached roughly 9.1 GB driver allocation, but that included optimizer/snapshot/cache state absent here and does not predict this harness's peak.

The requested GiB limit is converted to `cap_bytes / torch.mps.recommended_max_memory()` and passed to `torch.mps.set_per_process_memory_fraction()` **before model allocation**. PyTorch documents an allocator OOM when the limit is exceeded. This bounds the MPS allocator, not total process RSS or system-wide/other-process memory. The cap changes allocation policy without changing loss math or microbatch; an OOM aborts rather than silently reducing the workload. See [PyTorch's allocation-limit API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_per_process_memory_fraction.html). Limits above 12 GiB or the recommended working set are rejected. The example 4 GiB cap below is a conservative proposed probe limit, not a claim that microbatch 8 will fit.

Original bounded commands (Gmain failed as described above; Dmain remains unrun; do not rerun automatically):

```sh
research/.venv/bin/python -u research/experiments/deferred_stats/benchmark_mps.py --phase Gmain --trials 1 --memory-cap-gib 4 --output research/experiments/deferred_stats/mps-gmain-cap4.json
research/.venv/bin/python -u research/experiments/deferred_stats/benchmark_mps.py --phase Dmain --trials 1 --memory-cap-gib 4 --output research/experiments/deferred_stats/mps-dmain-cap4.json
```

The harness restores module buffers, PL mean and CPU/MPS RNG between variants, alternates A/B order, and compares finite gradients (rtol 1e-5, atol 1e-7), exact stats/PL/RNG. Synchronization surrounds each timed loss/backward/flush/collector region; gradient copying and comparisons occur afterwards. Results include allocator/driver snapshots and peak RSS, not an unsupported claim of instantaneous GPU peak. This isolates one microbatch's reporting change, not full accumulated training throughput, and concurrent training will remain a timing confounder.
