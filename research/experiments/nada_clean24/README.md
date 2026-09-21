# Prepared native1024 image-guided NADA smoke

**Prepared, not executed.** No generator, CLIP weights, model inference or training has been loaded/run during preparation. The separate reference MPS run remains active. The live concurrency guard correctly refused its PID27880; use this only after that job and any other heavy research work have finished and root has reviewed this source.

Read-only input/source check:

```sh
research/selection/.venv/bin/python research/experiments/nada_clean24/runner.py --check-only
```

Proposed guarded command, **not run**:

```sh
research/selection/.venv/bin/python research/experiments/nada_clean24/runner.py --execute \
  --output research/runs/nada-clean24-cpu-smoke10
```

The supervisor imports only standard-library modules. It requires a new output directory, no other known training/evaluation process, and at least35% system-wide free memory. Its original20-minute deadline includes pin checks, imports, centroid preparation, ten updates, previews and checkpoint reload. A50ms worker watchdog plus100ms supervisor polling abort above12GiB observed RSS. This is a sampled abort threshold, not an OS allocation limit; no MPS path or memory claim is provided. The supervisor kills/reaps only its own newly created process group on timeout, resource failure, interruption or launch-record failure. No retries or cap extensions. SIGKILL/machine failure cannot execute Python cleanup. Keep the supervisor alive and do not start another heavy job after its initial check.

Startup uses an atomic readiness record: after Popen the supervisor closes `child.json.tmp`, then renames it to `child.json`. Before any model import, the worker waits at most five seconds within the original deadline and verifies its PID, supervisor PID and nonce. Only then does it require exactly the launch, log and child-record files. This corrects the earlier timing-dependent directory check; the original prepared source and evidence are retained in `pre-handshake-fix/`. A temporary partial JSON write is never parsed as a readiness record.

`pins.json` binds226 source/input/weight files and ten source inventories. Dependencies include the unchanged `paired_edit/trainer.py` helpers, NVIDIA network/ops, pinned OpenAI CLIP source,24 accepted whole-image1024 targets and all32 source PNG/NPZ pairs. The existing original FFHQ1024 and official ViT-B/32 hashes were verified by streaming bytes without loading either model. NumPy preparation wrote actual arrays in `latents.npz`: source_z[32,512], source_w[32,18,512], train_z[10,512], eval_z[8,512]. All50 z rows are distinct; fresh train/eval PCG64 seeds and exact source correspondence are in `inputs.json`. No seed selection or filtering was performed.

The selection environment is unchanged. It lacks safetensors and requests dependencies required by our existing NVIDIA helper imports. A restricted import finder permits only six pinned package roots from the existing research environment: safetensors, requests, urllib3, charset_normalizer, idna and certifi. Their Python/native source files are hashed; the whole research site-packages directory is never appended. No installation, remote download, API credential or network request is part of the smoke. CLIP loads the explicit local weights path, not a model-name download path. Actual compatibility of this dependency bridge remains untested until the reviewed model smoke.

## Exact experiment

The [decision note](../../reviews/nada-clean24-decision.md) explains the hypothesis and official references. This is a **NADA-style variant**, not an exact reproduction of the Rosinality/CUDA implementation.

- Source and student begin from the identical original native1024 G. Mapping, ToRGB, all buffers, source G and CLIP remain frozen. All17 synthesis convolution modules, including their affine parameters/noise strengths, plus the4px learned constant are trainable. No previous paired checkpoint, frozen-D feature loss, clothing mask, source-face pixel loss or identity constraint is used.
- Before updating, recompute all32 recorded sources on CPU1, requiring exactly equal stored W and RGB PNG pixels. Cache normalized CLIP features from these regenerated complete sources and the24 grayscale targets. The target direction is the normalized difference between their mean normalized embeddings. Fail on degenerate centroids.
- Apply differentiable grayscale→RGB, [-1,1]→[0,1], bicubic antialiased224px resize and official CLIP normalization to both generated branches. Keep the student image graph through frozen FP32 CLIP. Save every complete native RGB and grayscale evaluation output; no restoration or output compositing.
- Update1 uses global target-centroid cosine loss because identical generators produce a zero edit direction. Updates2–10 use directional cosine loss with1e-6 norm floors and an explicit post-bootstrap nonzero-direction check. This bootstrap is our documented deviation. Adam uses actual LR.0016 and betas(0,.99**.8), batch1, CPU1, no style mixing, psi1, constant noise, no EMA. Exactly ten optimizer calls are permitted.

At each update the worker requires finite, nonzero upstream RGB-image gradients and finite gradients for every trainable parameter **before** the optimizer; it performs no NaN sanitization or gradient clipping. It checks finite parameters and Adam state, a changed trainable-state hash, and unchanged frozen state. Source/CLIP full-state hashes are compared before/after. Evaluation at0/5/10 must leave student state and Python/NumPy/Torch RNG unchanged. Initial source/student outputs must be exactly equal.

After update10 only, save the full student/Adam/RNG/features/recipe checkpoint, load it with `weights_only=True` and mmap, compare the complete nested state digest, deliberately perturb one trainable value, Adam moment and RNG, restore them, and compare exact states plus a retained native PNG. This validates actual reload, **not** interrupted-versus-uninterrupted training equivalence. No extra update or second trajectory is hidden in the reload test. Worker success, final provenance verification and supervisor success are all required; partial outputs are not a completed smoke or quality approval.

## Preparation evidence

`preflight-check.json`:226 pinned files pass, no torch import. `preparation-test.log`: fourteen small tests pass—the original seven pin/memory/concurrency/latent cases plus worker preflight with an already-ready child record, delayed atomic publication through a partial temp write, missing readiness, unexpected existing results, and wrong child PID, parent or nonce. The real preflight runs with fixture pin/memory checks; no model module is imported. `lifecycle-evidence.json`: five harmless dummy child/grandchild tests pass—post-Popen temp-record write failure, SIGINT, SIGTERM, deadline and RSS abort; each child reaped-9, both PIDs/group absent. Actual model commands were replaced with sleeps. An initial fixture-only attempt failed before spawning because macOS temporary-path normalization differed; its log is retained, and the fixture was corrected without changing the launcher.

Only AST parsing, lightweight metadata/streaming hashes, small NumPy preparation/tests and dummy-process tests ran. Actual model import, source reproduction, differentiable backward, memory/time fit and checkpoint reload remain **unverified until execution**. No result establishes a priest collar, photoreal face, adult-only sampler or subjective attractiveness. Inspect all native held-out outputs before considering a longer experiment.

Exact CPU1 W/PNG reproduction is an execution gate, not a claim established by the static NPZ tests. Ten updates test feasibility and initial visual movement only; they do not establish convergence or rule out NADA generally if movement is absent.
