# Original FFHQ1024 CPU feasibility — 2026-09-21

The official NVIDIA checkpoint loaded successfully as a 1024 × 1024 RGB generator. It has been renamed to `research/models/ffhq1024.pkl`; the adjacent `ffhq1024.provenance.json` records its origin, checksum, size, and research/evaluation license. These are **unadapted FFHQ images, not priests**, and no output or model is approved for the site.

Source: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

Original pickle SHA-256: `a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c` (381,624,121 bytes).

## Native CPU measurements

Each benchmark ran in its own process, sequentially, with one cold and three warm generations. MPS was not used. Two separate MPS training jobs were running on the same Apple Silicon machine, so these small samples include shared-system contention and cannot establish a reliable scaling advantage.

| Loading / inference path | CPU threads | Cold generation | Warm median | Warm maximum | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original pickle, G_ema only retained; forward only | 1 | 2.227 s | 0.636 s | 0.865 s | 3.60 GiB |
| Original pickle, G_ema only retained; forward only | 2 | 1.399 s | 0.782 s | 0.929 s | 3.58 GiB |
| Exported G-only safetensors, existing serving Model; forward + WebP | 1 | 2.581 s | 0.996 s | 1.393 s | 3.42 GiB |

Model load time was 0.256/0.140 seconds for the pickle processes and 0.361 seconds for the G-only serving process. Timed cold generation is the first forward after loading, not the complete application/container start. Original-model WebP encoding was separately approximately 0.049–0.086 seconds. Three warm samples do not establish a robust latency percentile.

**Every measured peak exceeds the inactive Cloudflare candidate's 3 GiB allowance.** Even G-only loading reached 3,671,277,568 bytes, so dropping raw G and D from the runtime does not by itself make that configuration fit. The candidate sizing was not changed. A Linux container benchmark of the eventual exported model, with its real memory limit and initialization path, is required before selecting resources. Neither native Mac performance nor RSS proves Cloudflare latency or memory use. Larger memory also changes the monthly cost assumptions.

`benchmark.py` retains only G_ema immediately after loading and garbage-collects the other networks. It generates with FP32, constant noise, truncation 1.0, and fixed PCG64 seeds 2026092100–2026092103. Original 1024-pixel RGB PNGs, exact pixel/file hashes, initialization timing, per-forward timing, and peak RSS are in `threads-1/` and `threads-2/`. Their RGB pixel hashes match across both thread settings. One generated original was visually inspected to confirm it is a full-resolution ordinary FFHQ face, not a priest output.

## G-only service-path follow-up

The existing offline exporter created `baseline-bundle/` from the trusted official pickle. Safetensors hash: `f802061515460f211faee6a6ff60d8803f4aa15b26cdce0edc5bbef7d88aaa2d`. Metadata preserves the complete constructor configuration, truncation 1.0, `training_step: 0`, and `review.approved: false`. NVIDIA licensing and source provenance accompany the research bundle. No production bundle was created.

A fresh process ran the existing `inference.benchmark` against this bundle using **only the explicit local `--allow-unreviewed` override**. That invokes the actual server's `Model.generate()` implementation, including fresh 256-bit random seeds, forward inference, and WebP encoding. It does not measure HTTP/network overhead. All four WebP outputs are distinct, decode as 1024 × 1024 RGB, and have their seeds and hashes in `g-only-server-threads-1/benchmark.json`. `g-only-server-verification.json` also confirms that normal production construction refuses this bundle without the override.

Reproduce the serving-path measurement from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 research/.venv/bin/python -m inference.benchmark \
  --model-dir research/runs/inference-cpu/ffhq1024/baseline-bundle \
  --output research/runs/inference-cpu/ffhq1024/g-only-server-threads-1 \
  --threads 1 --count 4 --allow-unreviewed
```

The result supports investigating 1024-pixel CPU inference; it does not establish successful priest adaptation, final visual quality, production-host feasibility, or compliance with the preferred monthly hosting budget. No active training, serving configuration, or deployment was changed.
