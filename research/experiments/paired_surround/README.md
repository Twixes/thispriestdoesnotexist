# Matched collar-surround objective probe — prepared, not executed

This standalone experiment forks the immutable completed six-pair step-600 checkpoint into two sequential CPU branches,50 updates each. It tests a more localized clothing objective; it does not unlock b32 or change model capacity. The original capacity draft remains explicitly disabled in `../paired_capacity/`. No real model weights have been loaded or trained by this task, and the frozen-D extractor remains unexecuted.

The hypothesis comes from broad white neck patches and poor dark-shirt structure despite fitted tab brightness and relatively preserved faces. Assigning immediate collar surroundings their own area-normalized term may help. This remains a hypothesis: a sharper training collar does not establish unseen-seed generalization, and failure need not identify the only cause. Radius30 was selected from prior5/15/30 diagnostics because the5px band includes bright antialiasing; it is fixed here, not swept or tuned on this comparison.

## Exactly one change

Both branches restore identical step-600 student weights, complete original Adam state, source teacher, six training pairs, validation028, four fixed unseen z values, global RNG, paired-index RNG and fresh-latent RNG. Both keep mapping, synthesis blocks<=32, every noise-strength parameter and all buffers frozen. No parameter group, learning rate, beta, batch size, loss weight, target, source or existing mask changes.

Let M be original clothing, T the raw traced white-tab mask and C=T∩M the existing effective collar. Define S=(ChebyshevDilation(T,30)−T)∩M and R=M−C−S. Dilation is a61×61 square, implemented as exact separable max operations. C/S/R must be nonempty, pairwise disjoint, sum exactly to M and contain zero protected pixels. Existing C and the original rest mask are checked unchanged. R is the remaining clothing, not a replacement for the old rest-mask definition in prior evidence.

- Baseline calls the pinned original `paired_regions.trainer.update` unchanged: clothing=`0.5*L1(C)+0.5*L1(M−C)`.
- Variant uses the isolated `objective.update`: clothing=`(L1(C)+L1(S)+L1(R))/3`.

Each region L1 uses the original per-image channel/area normalization. Protected-source and fresh-latent preservation are unchanged, including the top75% fresh band and independent fresh draws. All loss weights remain1, LR1e-4, Adam betas(.9,.999), batch1, CPU2 threads/interop1, constant noise, psi1, FP32. The variant update is a small copy of the pinned upstream project update with only the clothing component and component logs changed; no monkeypatch, active-trainer edit or loss-module substitution occurs.

Actual saved-mask counts, measured without loading images or models:

| Pair | C: tab | S: surround | R: remainder | M: total |
| --- | ---: | ---: | ---: | ---: |
|calibration-original|7,198|6,183|127,831|141,212|
|000|4,248|2,251|115,498|121,997|
|030|4,204|2,593|58,178|64,975|
|b2-055|6,248|9,471|252,074|267,793|
|b2-051|4,924|2,994|82,351|90,269|
|b2-020|1,790|7,174|305,674|314,638|

## Provenance, resource and execution guards

`parent-pins.json` pins the exact completed checkpoint SHA256 `0f433c7ec50d9e80026681125d7fa7e17e9be6e8e2d31b11976fb51d1b0f3fc4`, parent config, original paired7v2 manifest, source bundle, every paired source/target/latent and all original helper/vendor source hashes. Real execution checks those bytes before model imports, then uses weights-only checkpoint loading, exact source/W/PNG regeneration checks and strict source/student state validation. Runtime library versions must match the parent. Current experiment source hashes are recorded and checked unchanged across each run and between branches.

The parent process never imports torch or keeps models. It starts the baseline child, waits for it to terminate successfully, checks fresh memory again, then starts the variant child. Process exit frees the first branch's tensors and allocator state before the second loads. A baseline failure prevents variant launch. Each child requires at least25% system memory free before imports and uses an8GiB observed peak-RSS abort watchdog, polling every0.25 seconds. This is **not a hard allocation cap**: allocation can overshoot between polls. There is no MPS path, real D load, download, deployment or production export.

The runner forecasts each next pair index and fresh z using cloned RNG objects, records their indices/IDs and z hash, then checks the actual original RNG states after the update. Forecasting does not consume live generators. Both branches must have identical initial student hashes and all50 draw records. Previews restore all RNG state and must not mutate generator weights or buffers. At0/25/50, source/frozen/buffer hashes must match the branch start; source gradients must remain absent. Updates reject nonfinite losses, gradients and trainable parameters.

Prepared command, **not run**; parent must first review and schedule it:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_surround/runner.py \
  --execute --output research/runs/paired-surround-600-probe
```

Only a new directory under `research/` is accepted.50 updates per branch and milestones0/25/50 are hard-coded; there is no step-count override. The internal worker flag exists for child-process isolation and the narrow recovery case below. Avoid unrelated worker launches when a matched comparison is intended.

## Artifacts and resume semantics

Each branch saves full native RGB and grayscale **student outputs** for all six training identities, unchanged validation028 and the same four original fixed unseen latents at0/25/50. The grayscale PNGs are RGB-mode with three replicated gray channels. The earlier PNG surround analyzer expects L-mode, so later analysis must explicitly adapt its input conversion; direct reuse has not been tested. There is no preview compositing. Contacts and per-pair protected/tab/rest/surround/remainder errors aid inspection; training fits are explicitly labeled. Validation028 has no invented tab annotation and is never sampled for updates. Component metrics remain on the original[-1,1] scale and are not perceptual-quality scores.

At the same milestones, durable atomic full checkpoints retain student, Adam, every RNG, fixed z, original architecture/options/provenance, fork lineage, source/frozen/buffer hashes, schedule and last metrics. Final serialization is loaded weights-only/mmap and compared exactly for optimizer and student state. Logs include update time, losses, sampled pairs and fresh-z hashes; completion records include peak RSS, total time and checkpoint hashes. `comparison-mechanics.json` is written only after both branches finish and their schedules match.

This is an **explicit experiment fork**, not an exact resume under the parent's recipe. The new `paired-surround-fork-v1` format is deliberately incompatible with the original trainer's resume API. This runner has no resume flag: an interrupted branch leaves checkpoints/evidence but cannot silently restart or extend. Future continuation would require an independently reviewed loader preserving this format's freeze/objective/RNG semantics. Existing checkpoints/manifests remain untouched.

## Completed tiny validation

`test_runner.py` ran four tests successfully in3.069 seconds. It uses a small64px StyleGAN network, artificial targets and nonzero frozen noise strengths; it never opens pretrained source/model/checkpoint files. Tests prove exact initial state and Adam transfer in both branches, actual finite updates, unchanged source/frozen<=32 parameters/buffers, identical actual sampling/fresh draws,50-step forecast equivalence, full tiny checkpoint serialization,11-output snapshot state/RNG preservation, exact Chebyshev support, equal-thirds area normalization, zero protected-image gradient for the clothing term, invalid-partition rejection and all six real saved-polygon partitions. JSON polygon checks use no image inference.

Reproduce:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_surround/test_runner.py
```

`test.log`, `test-evidence.json` and `mask-counts.json` retain evidence. Passing mechanics tests says nothing about1024px memory, throughput or visual quality; those remain unmeasured for this proposed fork. Upstream NVIDIA licenses and source restrictions remain unchanged.

## Recovery after between-branch memory deferral

If the baseline child completes successfully but the next memory guard defers the variant, **never rerun the completed baseline**. Keep its immutable directory/checkpoints and original plan. After a fresh memory guard passes, root can start only the pending variant into its still-new child directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 research/.venv/bin/python research/experiments/paired_surround/runner.py \
  --execute --worker collar-surround --output research/runs/paired-surround-600-probe/collar-surround
```

Before recovery, verify the original plan's runner/objective/pins hashes still match the frozen files and the baseline completed all50 updates. The worker repeats all input and memory guards and still forks the original600 checkpoint, never baseline650. After it finishes, explicitly record the same comparison that the supervisor would have made: identical initial student hashes and all50 pair/fresh-z records; exact step0 student/optimizer/RNG equality; each milestone checkpoint hash matches its own completion record; each final checkpoint's student/optimizer state matches its branch's saved completion/roundtrip evidence. Final baseline and variant weights need not match—different objectives are expected to produce different updates. Their differing final states must be correctly attributed and preserved. No recovery orchestration or automatic resume has been implemented, and the combined-success record must not be inferred merely from two directories existing.
