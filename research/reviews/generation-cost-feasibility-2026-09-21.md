# Generation cost and feasibility — 2026-09-21

This archives the cost analysis given in the task conversation and separates measured project evidence from budget scenarios. Prices are USD estimates, not bills or resource reservations. Public provider prices were checked again on 2026-09-21. No paid resource was created for this analysis.

## Decision

Recreating a face generator from scratch is economically unnecessary because pretrained face generators already contain that work. A successful adaptation below $100 is plausible, but the current project has **not** demonstrated the requested quality. Do not buy a 70,000-image synthetic dataset or promise that another fixed number of training hours will solve the defects.

FFHQ consisted of photographs, not 70,000 GPT-generated faces. The proposed synthetic dataset is a hypothetical replacement, not an expense the original researchers incurred. Dataset cardinality and training image presentations are different: the same image can appear repeatedly during training.

## Scratch-training comparison

The original [StyleGAN repository](https://github.com/NVlabs/stylegan#training-networks) reports 6 days 14 hours on eight Tesla V100 GPUs for its default 1024px configuration: `(6 × 24 + 14) × 8 = 1,264 GPU-hours`. A single V100 was reported as 41 days 4 hours; parallel scaling is not perfect. These are V100 GPUs, not “V1 compute.”

[Lambda's eight-GPU pricing table](https://lambda.ai/instances) lists Tesla V100 16GB at $0.79 per GPU-hour, plus applicable tax. Reusing the original aggregate hours gives `1264 × $0.79 = $998.56`. This is a hardware-family budget reference, not a rental availability check, exact software/hardware reproduction, or benchmark on a modern GPU.

## Synthetic image costs

The [official OpenAI image generation guide and calculator](https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency) give GPT Image 2.5 image output pricing of $30 per million tokens. The 1024×1024 calculator observations from the preceding analysis are preserved below; this archival pass independently rechecked the token rate and the calculator's low-quality default. The other quality token counts are the preceding calculator observations, not independently exercised again in this archival pass.

| Quality | Output tokens | Output cost/image | 500 outputs | 70,000 outputs |
| --- | ---: | ---: | ---: | ---: |
| Low | 196 | $0.00588 | $2.94 | $411.60 |
| Medium | 439 | $0.01317 | $6.585 | $921.90 |
| High | 1,756 | $0.05268 | $26.34 | $3,687.60 |
| Extra high | 3,122 | $0.09366 | $46.83 | $6,556.20 |
| Maximum | 7,024 | $0.21072 | $105.36 | $14,750.40 |

Each row uses `tokens × 30 / 1,000,000`. This excludes prompt tokens, reference-image inputs, streaming partial images, retries and taxes. A hypothetical 70% acceptance yield requires 100,000 outputs for 70,000 usable images: `$0.05268 × 100,000 = $5,268` at high quality. This yield is an assumption, not a measured GPT generation yield.

Thus the high-quality synthetic scratch scenario is `$3,687.60 + $998.56 = $4,686.16`, or `$5,268 + $998.56 = $6,266.56` with that rejection assumption. Storage, input charges, extra experiments and tax are additional.

## Transfer learning and local compute

[NVIDIA's StyleGAN2-ADA guidance](https://github.com/NVlabs/stylegan2-ada-pytorch#expected-training-time) says 1,000 kimg is often enough for transfer learning. Its 1024px V100 benchmark reports 5 hours 54 minutes with eight GPUs, or 47.2 GPU-hours. At the Lambda reference rate that is `5.9 × 8 × $0.79 = $37.288` per run. One million presentations means repeated sampled training examples, not one million distinct photographs. This is a generic convergence reference, not a guarantee for priest portraits.

The earlier local 256px trainer benchmark extrapolated to 5.19 days per million presentations. At assumed average power 50–100W and $0.30/kWh, electricity would be `5.19 × 24 × (0.05…0.10) × $0.30 = $1.87…$3.74`. Neither power consumption nor electricity tariff was measured. The material cost is elapsed time and occupying the user's laptop; extrapolation can change with implementation or thermal conditions.

## Current measured quality prevents deployment

The frozen reference625 → GFPGAN → CLIP/logistic-selector test attempted 64 fresh latents. The selector accepted nine at its preselected 0.9 threshold. Blind native-size reviews accepted 43/64 overall and rejected two of the nine selected images (051 and 061) for eye defects. This implies 7/9 selected precision (77.8%), 9/64 selection rate (14.1%), and 7/43 recall (16.3%). These are small-sample manual measurements, not deployment assurance. Repeated facial resemblance remains unresolved.

Evidence is stored in `research/selection/restored-test64-scores-v1/result.json` and the four `labels-*.json` files under `research/restoration/runs/reference625-test64-v1/`. The 64 raw attempts, restoration failures and rejected outputs remain preserved. The frozen filter failed the intended obvious-defect gate; threshold tuning on this held-out cohort would contaminate it. Longer training might help but is not proven to address these defects.

## Lower-cost hypotheses to test

1. Keep the pretrained StyleGAN face unchanged and inpaint only clothing into a clerical shirt and collar. Explicitly preserve face pixels, then verify neck boundaries and complete portrait coherence. This retains the trained source of fresh identities and narrows the edit. [Diffusers inpainting documentation](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) describes masks and preserving unmasked regions. Quality and latency remain experiments.
2. Benchmark pretrained diffusion locally before training an adapter. [Diffusers supports Apple Silicon MPS](https://huggingface.co/docs/diffusers/optimization/mps). Fresh random-noise generation could avoid GAN adaptation and a large new dataset, but changes architecture and does not by itself fulfill the explicit model-training requirement.
3. If CPU hosting matters, train a small clothing-only editor from hundreds of paired synthetic examples while freezing the face generator. Five hundred high-quality teacher outputs begin at $26.34 plus input costs. This is an unproven research direction, not a implemented capability.

These experiments should precede more paid dataset generation or long training. Acceptance must cover complete full-size faces, adult male priest appearance, no hats, collar fidelity, identity variety, fresh latent generation, serving latency and the real hosting budget.

## Hosting budget scenarios

[Runpod pricing](https://www.runpod.io/pricing) currently lists RTX 4090 Pods at $0.74/hour and serverless 4090 at $1.10/hour. These products have different billing behavior and cannot be interchanged in estimates. At $1.10/hour, an always-active worker would be about $803 over a 730-hour month; it cannot meet a $15 monthly budget.

[Serverless billing](https://docs.runpod.io/serverless/pricing) includes container startup/model loading, execution and idle timeout. Flex workers can scale to zero. For **total billed seconds per delivered accepted portrait**, including rejected attempts and amortized startup/idle:

| Billed seconds/delivered portrait | Compute per 1,000 delivered portraits | Portraits for $10 compute |
| --- | ---: | ---: |
| 10 | $3.06 | ~3,272 |
| 30 | $9.17 | ~1,090 |
| 60 | $18.33 | ~545 |

Formula: `seconds × 1.10 / 3600`. A $10 compute allowance leaves $5 of the desired $15 budget for other costs, but those costs have not been verified against a chosen deployment. At 10,000 deliveries and 10 billed seconds each, compute is $30.56. No listed latency has been measured for the proposed pipeline.

A continuously replenished queue of fresh, single-use generated portraits could amortize model startup while keeping page loads quick, but generation would happen ahead of requests. It must never recycle a fixed image/seed catalogue, and is a proposed alternative rather than an implemented or user-approved change to per-load generation semantics.
