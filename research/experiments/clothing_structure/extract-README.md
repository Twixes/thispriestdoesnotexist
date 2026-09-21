# Guarded FFHQ1024 discriminator extraction

`extract_discriminator.py` is prepared but **has not been executed on real weights**. Six tiny/static CPU tests passed. Actual D constructor values, state counts and1024px tap shapes remain unverified until root schedules extraction after competing evaluations finish. This standalone utility does not import or alter the loss/trainer modules.

The only accepted pickle is the existing trusted local `research/models/ffhq1024.pkl`, SHA256 `a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c`, with the existing official NVIDIA provenance. No input override, download or network loader is provided. The hash is streamed before legacy/pickle imports. `extract-vendor-pins.json` pins36 NVIDIA Python/license files, including the exact legacy loader and `training.networks.Discriminator` implementation. A mismatch aborts; update/review pins deliberately rather than bypassing them.

Before torch/legacy/pickle imports, fresh macOS `memory_pressure` must report at least25% free. The check runs again after hashing. Execution sets CPU thread and interop counts to1 and provides no GPU/MPS path. A watchdog aborts the process if peak RSS exceeds8GiB or execution after guard setup exceeds300 seconds, sampled every0.25 seconds. This is an observed-process abort limit, **not** a hard allocation cap: a transient allocation can exceed it before the watchdog runs. Partial directories are deliberately retained with failure evidence; a new output directory is required for another attempt.

The loader necessarily deserializes official G/D/G_ema together. It immediately discards G/G_ema and collects garbage before reconstructing D; it never runs a generator. The D must be unconditional RGB1024/resnet with the expected32768 channel base,512 maximum and no saved FreezeD setting. Its actual fully named constructor kwargs are validated as finite JSON data, preserved exactly, and passed to the hash-pinned NVIDIA constructor. This does **not** guess uninspected FP16, clamp or epilogue settings. Unknown top-level constructor fields/positional arguments or a different architecture abort pending review.

Every source state tensor is checked for CPU FP32, shape/count, finite values, parameter-versus-buffer registration and individual SHA256. Reconstruction uses strict `load_state_dict` and exact tensor equality. Then one deterministic batch-one RGB ramp exercises only b1024→b512→b256 under `no_grad`, forced FP32; expected output shapes are `[1,64,512,512]`, `[1,128,256,256]`, and `[1,256,128,128]`. The latter two are the design's selected feature taps. No full `D.forward`, epilogue, minibatch statistics, backward or optimizer executes. All state hashes must remain unchanged and parameter gradients absent. This is an actual prefix numerical forward when scheduled, not a meta-tensor shape inference, and is not a perceptual-quality test.

A successful new directory contains `D.safetensors`, exact constructor `metadata.json`, copied upstream `LICENSE.txt`, and memory preflight output. Serialization is verified by loading safetensors and comparing every tensor exactly; weights and final metadata are renamed atomically, with `metadata.json` as the success marker. Metadata also records actual state inventory, all pins, source provenance, tap shapes, timing, peak RSS, platform and torch version. Incomplete outputs are not approved bundles. All exports remain research-only and production-unapproved. The NVIDIA upstream license applies; repository MIT does not relicense this discriminator.

Prepared command, **not run**:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 research/.venv/bin/python research/experiments/clothing_structure/extract_discriminator.py \
  --execute --output research/runs/frozen-d-ffhq1024-extraction
```

Tiny validation command already run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 research/.venv/bin/python research/experiments/clothing_structure/test_extract_discriminator.py
```

Tests cover import without torch/pickle, explicit-execution guard, low/missing memory failure, constructor rejection, vendor pins, exact tiny16px D reconstruction, prefix-only finite forward without epilogue, frozen state, safetensors roundtrip, and nonfinite-state rejection. `extract-test.log` and `extract-test-evidence.json` retain results and source hashes. No real pickle/model weights were loaded by these tests.
