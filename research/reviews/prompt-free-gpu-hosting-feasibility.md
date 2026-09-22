# Personal GPU inference hosting feasibility

Retrieved 2026-09-22, Europe/Warsaw. Read-only research using official provider pages and existing local serving code. No accounts, credentials, purchases, infrastructure, or deployment were changed. The free personal Cloudflare Worker frontend is already live; GPU inference would be a separate origin behind it.

## Recommendation and latency boundary

Modal Starter with an L4 and scale-to-zero is a reasonable first measured trial for low traffic near the preferred $15/month budget. This is conditional on accepting occasional slower cold requests. No surveyed published option credibly provides a continuously warm GPU at that budget.

Two different targets must not be conflated:

- **Server model computation below 500 ms:** a warmed GPU forward pass, measured with CUDA synchronization. This excludes queueing, cold initialization, WebP encoding, proxying and transfer. It may be attainable, but has not been measured on the eventual priest model or any proposed provider.
- **Every end-to-end request below 500 ms:** from browser request through scheduling, initialization, actual fresh generation, any restoration/selection, encoding and network response. This is a much stronger requirement. Scale-to-zero cold boots and resource contention prevent treating a fast advertised snapshot resume as a guarantee. It is also unmeasured here.

Keeping model weights/process state warm is compatible with a fresh random latent and an actual generator forward on every request. No pre-generated portraits or cached image responses are proposed.

## Published prices and illustrative arithmetic

Use a 730-hour month for comparison. USD figures are pre-tax, excluding storage and any extras unless stated. Published starting rates do not establish capacity in a chosen region or personal-account eligibility.

| Option | Published rate | Continuously warm estimate |
| --- | --- | --- |
| Modal T4 | $0.000164/GPU-second | $430.99/month, GPU only |
| Modal L4 | $0.000222/GPU-second | $583.42/month, GPU only |
| Runpod A5000 Pod | $0.27/hour | $197.10/month before additional storage |
| Beam 4090 on-demand machine | From $0.44/hour | $321.20/month at that starting rate |

Official sources: [Modal pricing](https://modal.com/pricing), [Runpod pricing](https://www.runpod.io/pricing), [Beam pricing](https://www.beam.cloud/pricing).

Modal's Starter plan advertises $30/month included compute. An **illustrative**, not resource-validated allocation of L4 + one physical CPU core + 4 GiB RAM is:

`0.000222 +0.0000131 +(4×0.00000222) = $0.00024398/second = $0.878328/hour`.

Thus $15 alone buys 17.1 billed hours; $30 included compute plus $15 buys 51.2 billed hours. Continuously warm is approximately $641.18/month before credits. Region selection carries a listed 1.15–1.75× base-price multiplier, so selecting a nearby region for latency can change these estimates. Credits/account eligibility were not checked. [Modal pricing](https://modal.com/pricing).

Modal defaults to a 60-second scaledown window and permits 2–1,200 seconds. Keeping GPU containers idle within that window incurs charges. Low traffic with frequent isolated requests can therefore cost much more than summing forward-pass durations. [Modal cold-start controls](https://modal.com/docs/guide/cold-start).

Runpod's 24 GB flex class lists $0.69/hour. Illustratively, 10,000 isolated requests with 6 billed seconds each cost $11.50; 36 billed seconds each cost $69. These are billing scenarios, **not measured latency or traffic predictions**. They exclude storage and assume no overlapping retention savings. Billing spans worker startup/model load, execution and the default 5-second idle timeout, rounded up to seconds. Current docs describe active-worker discounts through sales inquiry; older blog discounts should not be treated as guaranteed personal-account rates. [Serverless rates](https://www.runpod.io/product/serverless), [billing details](https://docs.runpod.io/serverless/pricing).

Beam's current live page lists 4090 serverless at $0.000192/GPU-second plus GPU-attached CPU/RAM. Its own 2-core/16 GiB example totals $1.77/hour. This is not clearly cheaper than the Modal trial after included compute. The live pricing page supersedes older indexed search excerpts. [Beam pricing](https://www.beam.cloud/pricing).

## Cold starts and snapshots

Runpod advertises sub-200 ms FlashBoot resumes, but its explanation distinguishes paused-state hits from true cold boots taking seconds or minutes; paused allocations are released after sufficient inactivity. That infrastructure figure does not include our model forward, encoding or network. [FlashBoot behavior](https://www.runpod.io/blog/serverless-gpu-cold-starts-flashboot).

Modal GPU memory snapshots remain alpha. They may reduce initialization/JIT work, but do not inherently accelerate loading model bytes from storage and may require compatibility changes. They are not an end-to-end 500 ms guarantee. Snapshotting RNG state also requires care: create fresh request entropy after restoration instead of repeating a snapshotted deterministic stream. The existing service already samples OS entropy inside each request. [Modal snapshots and limitations](https://modal.com/docs/guide/memory-snapshots).

## Existing service limitation and next evidence

`inference/server.py` constructs the generator on CPU, loads CPU safetensors, creates CPU latents and calls `.numpy()` directly on CPU output. `inference/Dockerfile` installs the CPU-only Torch wheel. A GPU allocation alone will not accelerate this image. A CUDA-capable serving variant would need explicit device transfers, CPU output transfer for encoding, compatible NVIDIA custom-op compilation/warmup, and timing synchronized around GPU work. The existing fresh-latent/no-store and checksum-bound model interface can be retained. No such implementation was made in this task.

The previous unadapted FFHQ 1024 Mac CPU serving benchmark measured a 0.996-second warm median for generator plus WebP, across only three warm samples. This neither meets the current 500 ms aspiration nor predicts CUDA/cloud performance. See `research/runs/inference-cpu/ffhq1024/README.md`.

Before choosing a host, measure the eventual complete pipeline on the selected actual GPU: warm model compute, full origin response, end-to-end browser latency, first request after short and long idle, startup after deployment, concurrency/queueing, and billed time. Include any restoration and rejection-sampling amplification rather than reporting only a successful generator forward. Preserve fresh outputs, latents/model hashes and cold/warm classification. Warm GPU speed, cold-start distribution, visual quality and final monthly spend all remain unproven.
