# Isolated GFPGAN diagnostic

## Current result

Both original cohorts completed on local CPU: all32 reference625 portraits in112.41seconds and all8 NADA500 portraits in25.65seconds, with no restoration errors. Peak process memory was about4.04GiB. Median per-image time was2.37seconds for reference256 inputs and2.92seconds for NADA1024 inputs, including diagnostic image encoding. These are local measurements, not a Linux serving benchmark. Restoration materially improves intact faces but does not repair architecture/head mergers, hats, extra people or unsuitable age appearance. Native reviews and all failures remain under `runs/`; neither pipeline is deployed.

The successful v1 implementation and pins are preserved in `sources/diagnostic-v1.py` and `sources/pins-v1.json`. The current wrapper additionally accepts `--cohort manifest256 --manifest <completed-generation/evaluation.json>` for1–128 sequential256px images, preserving every attempt and checking input hashes. Fresh output directories archive the exact source and input generation manifest. This uses the same restoration, alignment and encoding recipe. `reference625-development128-v1` is a separate development cohort, not a held-out test or a bank of serving seeds.

The preparation notes below describe what was and was not verified **before** the first runs; actual runtime completion evidence now lives in their supervisor and worker results.

This directory contains a research-only wrapper around **unmodified** GFPGAN v1.4 clean architecture and facexlib's official face detection/alignment/parser/paste helper. No model has been loaded or run during preparation. Root must inspect this code and schedule execution after NADA and any other model work has stopped. Nothing is connected to serving or an active trainer.

## Exact assets and environment

`asset-downloads.json` and `pins.json` record every official URL, byte count and SHA256. Downloaded assets:

| Asset | SHA256 |
|---|---|
| GFPGAN commit `7552a7791caad982045a7bbe5634bbf1cd5c8679` source tarball | `690613da16e4a1c988cca654e52ce975179b2cb5db76f4c5d92ef0ec07bd76cb` |
| `GFPGANv1.4.pth` (348,632,874 bytes) | `e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad` |
| Official RetinaFace ResNet50 auxiliary (109,497,761 bytes) | `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d` |
| Official ParseNet auxiliary (85,331,193 bytes) | `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2` |

Source lives in `vendor/`; all three weights are supplied via explicit local paths under `weights/`. Preparation downloaded source/release bytes without importing ML packages. The initial system-Python archive extraction lacked `tarfile`'s safe filter API; all downloads succeeded, then the isolated Python3.11 extracted the already hashed tarball with `filter='data'`. Both initial failure and successful extraction logs are retained. No download was repeated.

The separate `.venv` uses Python3.11.16, torch2.1.2, torchvision0.16.2, NumPy1.26.4, BasicSR1.4.2, facexlib0.3.0 and Pillow10.4.0. This older torchvision still provides `functional_tensor`, avoiding the compatibility shim needed with newer torchvision. No upstream code/math was patched. Exact47 package versions are in `requirements-lock.txt`; installed BasicSR/facexlib/torchvision Python sources and metadata are hash-pinned. GFPGAN is imported from the immutable vendor snapshot rather than installing its training extras. Existing environments were untouched.

`uv pip check` has **two documented warnings, not a clean dependency result**:

- The torch2.1.2 wheel's internal WHEEL metadata says x86_64, while `file` identifies its actual `libtorch_cpu.dylib` and the interpreter as ARM64 (`architecture-static.txt`). No metadata was rewritten. Actual import/runtime compatibility is untested and must be determined by the scheduled diagnostic.
- BasicSR declares `tb-nightly`; it is deliberately absent. Its tensorboard import is local to `init_tb_logger`, which this inference wrapper never calls. Neither Lightning nor TensorBoard training/logging packages were installed. If an actual import fails, preserve the failure before a narrow reviewed dependency correction; no automatic upgrades.

GFPGAN's license has third-party exceptions. The source LICENSE, copied `GFPGAN-LICENSE`, and package notices under `licenses/` preserve the terms; `pins.json` binds upstream license bytes. This preparation does not clear the existing NVIDIA generator or combined package for production use. The site's MIT license does not replace upstream terms.

## Predeclared cohorts

Run them separately into different new directories; no combined success rate or image selection:

1. `reference625`: **all32** raw psi1 PNGs from `reference256-paper-b64-resumed500-to625/matched32-000625`, exact hashes and completion records pinned now. No generator is loaded. Known raw structural/collar/extra-person failures remain marked challenge controls and cannot become approved merely because restoration sharpens the central face.
2. `nada500`: prospectively **all8** fixed student grayscale images, indices0–7, from `nada-clean24-cpu-continue500/preview-500`. No files or result are presumed to exist. Launch requires an independently supplied checkpoint SHA matching the final checkpoint and its review-ready record; successful supervisor and worker completion at500; actual Adam/RNG reload and native-preview equality flags; all8 preview hashes; and unchanged currently pinned source/recipe/latent provenance. The parent completion evidence is trusted metadata, not a new tensor-validation claim. No model loads are used to check it. Partial250/other checkpoints are not accepted.

## Pipeline and resource limits

Only the supervised worker imports Torch. Parent and worker enforce fresh free memory>=35%, immutable source/assets and package versions, and absence of another matching research training/evaluation process. Parent imposes five CPU thread environment variables plus NumBa1; worker explicitly selects CPU FP32, Torch1/inter-op1 and OpenCV1. SixGiB sampled RSS and a600second supervisor deadline are **abort thresholds, not OS allocation guarantees**. Parent terminates/kills and reaps only its own worker on an overrun; it never stops an existing trainer. No automatic cap raise or retry.

Socket connection/DNS access is blocked before ML imports and constructors. Auxiliaries are local and hash-checked; an implicit helper download therefore fails instead of silently fetching a new model. This is an application-level Python network prohibition, not an OS network namespace. The wrapper does not call external download programs.

The wrapper uses `GFPGANv1Clean` with the constructor from official `gfpgan/utils.py`, strict `params_ema` loading and eval/inference mode. It calls the unmodified official helper's face detection/alignment/paste methods directly rather than `GFPGANer.enhance`: this exposes all detected boxes, selects exactly the center face, saves transforms, and **avoids the stock RuntimeError→input-crop fallback**. Other faces are not repaired away and remain visible for review. Detection confidence is not a quality or eligibility filter.

`randomize_noise=False` uses stored decoder noise; all RNGs are seeded1729 and deterministic algorithms enabled. Background Real-ESRGAN is absent. Stock aligned crops are512. Reference256 inputs use helper upscale4; prospective NADA1024 inputs use upscale1. Full outputs are1024; this does not establish native1024 learned information. Stock helper crop border/paste behavior is retained, `pad_blur=False`; affine matrices, parser blending and neck seams must be inspected.

Each input is attempted once and retains byte-identical `raw.png`, a Pillow Lanczos1024 grayscale control (PNG/WebP), aligned512 input, restored512 crop, color1024 reconstruction, final grayscale1024 PNG/WebP, hashes, detected boxes, transforms, per-stage latency and process peak. Failures retain raw/control and exact error, never a fake restored fallback. Final/control WebP both use quality90/method6. A complete diagnostic may contain explicitly counted image errors; **complete does not mean quality pass**. Successful worker and supervisor results are both required. Source/input hashes are checked again at completion.

Cold import/startup is reflected in supervisor wall time; model/aux load time is separate; per-image records include detection/alignment, restoration, paste/encoding and total times. The first forward is cold and subsequent inputs are warm. No duplicated warmup pass or extra inference is hidden in the32/8 counts. Inspection/gating/retry serving costs remain unmeasured.

## Reviewed launch commands, not yet run

From the repository root, after root confirms no competing model work:

```sh
research/restoration/.venv/bin/python research/restoration/diagnostic.py \
  --cohort reference625 --output research/restoration/runs/reference625-v1
```

The separate NADA command requires its **future actual** SHA; the placeholder is intentionally not executable as supplied:

```text
research/restoration/.venv/bin/python research/restoration/diagnostic.py \
  --cohort nada500 --nada-checkpoint-sha256 <independently-reviewed-500-sha256> \
  --output research/restoration/runs/nada500-v1
```

Never reuse an existing output directory, even after failure; preserve logs and partial artifacts. Root decides any rerun/correction. The prepared wrapper loads only restoration/detector/parser weights, never GAN checkpoints; cohort checkpoint checks are streaming file hashes and completion evidence.

## Preparation evidence and next gate

Eight stdlib-only tests passed (`static-tests.log`) for incomplete NADA rejection, missing independent checkpoint digest, network blocking, memory boundary, active-trainer refusal, source mismatch, absent Torch/NumPy imports and fixed resource constants. `pin-preparation.json` records402 pinned files and47 packages, with Torch absent. Syntax/source checks and these fixtures do not prove that old package imports or any model forward will work.

Root should inspect the first actual result and then all native full images, including challenge controls and the32/8 failed cases if any. There is no automatic quality approval, hotness/age classification, rejection sampler, model serving, image bank or novelty claim in this wrapper. A later fresh-latent whole-pipeline benchmark is separate work only if restoration materially improves intact portraits at acceptable measured cost.
