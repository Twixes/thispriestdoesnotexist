# SDXL-Turbo CUDA benchmark

Prepared for an isolated GPU rental; **not executed on CUDA and not a production server**. No resources are provisioned by this script. Start with one 512×512 step, 100 fresh outputs, and five separately reported warmups. Preserve the output directory for native-resolution visual review.

## Inputs and environment

- Official SDXL-Turbo revision `71153311d3dbb46851df1931d3ca6e939de83304`, in `models/sdxl-turbo`. The script pins the existing provenance manifest SHA-256 and verifies every listed weight/config. Missing oversized originals are reconstructed only from matching local LFS parts using `restore.py`. LFS pointers or changed weights fail verification; nothing is downloaded.
- CUDA-capable Linux environment with CUDA-enabled PyTorch, Pillow WebP support, and `nvidia-smi`. Use the versions in `flux-train-requirements-lock.txt`, including the evaluator's Diffusers commit `9f1246971270c84dcbe71233edb7a519596a5d02`, PyTorch 2.14.0 and PEFT 0.21.0. The script checks Diffusers' installed `direct_url.json` commit rather than accepting any build with the same development version. Environment installation belongs to the later rental setup; no packages were installed for this preparation.
- Copy `benchmark_cuda_sdxl.py`, `compare.py` (default prompts), `restore.py`, and the verified model directory. Avoid fetching every research model from Git LFS. Optionally copy the chosen `pytorch_lora_weights.safetensors` and provide its reviewed SHA-256. There are no credentials or network API calls in the benchmark.

Run from the repository root with that environment active:

```bash
python research/diffusion/benchmark_cuda_sdxl.py \
  --output research/diffusion/runs/cuda-turbo-base-1step-v1
```

For an adapter, replace the file and digest below with the exact reviewed checkpoint:

```bash
python research/diffusion/benchmark_cuda_sdxl.py \
  --adapter /absolute/path/pytorch_lora_weights.safetensors \
  --adapter-sha256 REVIEWED_64_HEX_SHA256 \
  --output research/diffusion/runs/cuda-turbo-lora-1step-v1
```

Use a fresh directory for each variant. `--steps 4` measures four steps. `--count N`, `--warmups N`, `--seed-base N`, `--cpu-threads N` and `--cuda-device N` are configurable. `--prompts-json prompts.json` accepts a JSON string array; the actual prompts are saved in the result. The default four comparison prompts cycle across outputs. Seeds start at `202609214000`; warmups use earlier seeds.

## Timing contract

`generation_seconds` measures CPU generator creation and fresh float32 noise, transfer/cast to CUDA float16, the full Diffusers pipeline including VAE decoding and PIL conversion, grayscale conversion and in-memory WebP encoding (quality 90, method 4). CUDA is synchronized before and after GPU work. CPU RNG time, pipeline/transfer time and encoding time are also recorded. The consumed CPU generator is passed through to the scheduler, preserving the evaluator's extra-noise convention for four-step Euler ancestral inference.

Artifact disk writes and SHA calculations occur after the request timer. No server, HTTP, network, admission queue, or automated quality rejection/retry is implemented or timed. Consequently **a passing warm result is not proof of the final production latency requirement**; any eventual checks/retries and public request behavior require another benchmark. All 100 outputs are retained, including bad ones.

Cold model/adapter loading, imports, provenance verification/reassembly, conditioning setup and warmups have separate timings. Verification reads weights and can warm filesystem caches: `cold_model_and_adapter_load_seconds` means a fresh pipeline in this process, **not a cold cloud boot**. Container scheduling, image pull, downloads, driver startup and cloud initialization are not measured. Compilation is lazy, so its substantial first-call work appears in the retained warmup records.

## Optimizations and matched controls

The default caches fixed prompt embeddings after adapter loading; every request still gets newly sampled image noise. Per-prompt cache-miss encoding times are reported in setup, cache lookup is inside each request timer, and records identify hits. `--uncached-prompts` includes text encoding on every request and more closely matches the current local evaluator. There are no dynamic prompts in a cached run.

LoRA is **unfused by default**, named `priest`, scale 1.0, matching the evaluator. `--fuse-lora` fuses it before conditioning and warmup; `--adapter-scale` changes scale. `--compile-unet` uses `torch.compile(mode="reduce-overhead", fullgraph=True)`. Each optimization must use a separate output directory and receive visual comparison; speed alone does not prove unchanged quality. The original VAE force-upcast behavior remains enabled as configured, TF32 is disabled, and the invisible watermarker is explicitly disabled.

## Results and acceptance

`result.json` records completion, exact model manifest/hash, adapter/hash/scale/fusion, Diffusers commit, package versions, GPU/driver/CUDA identity, scheduler, seeds, prompts, timings and nearest-rank p95/maximum. Each measured and warmup output has `native.png`, `display.webp`, and `record.json`; `records.jsonl` preserves completed rows if interrupted. All images have SHA-256 hashes; noise has a float32-byte hash. CUDA peak allocated/reserved memory is recorded per request. Fewer than 100 samples are useful for smoke checks only.

Evaluate all outputs for face integrity, photorealism, adult male priest appearance, visible collars, no hats and diversity. Then assess the strict `<0.5s` count, p95 and maximum of complete generation, not diffusion-only timing. Record public HTTP and cold-start tests separately on the chosen deployment configuration. A partial run remains `complete: false`; never treat it or distinct hashes alone as passing.

Preparation validation: Python syntax, CLI help, argument rejection and pure summary arithmetic only. **No GPU inference, model loading, adapter fusion or CUDA compilation has been tested here.**
