# Fast diffusion candidates for strict sub-500ms generation

Research date: 2026-09-21. The user now accepts GPU hosting and wants several pretrained bases evaluated, with server generation strictly below 500ms. This scouting pass fetched official model metadata and license text only. No weights, GPU instance or API credentials were used. No candidate has yet been benchmarked on the project's deployment hardware.

## Shortlist

| Candidate | Why evaluate | First inference settings | Published sub-500ms evidence | Adaptation risk |
| --- | --- | --- | --- | --- |
| Stability AI SDXL-Turbo | Strongest directly relevant measured latency claim; one-step SDXL-derived model | 512×512, FP16, one step, guidance 0 | Author reports 207ms on A100 including text encoding, one denoise and VAE decode | Ordinary fine-tuning could damage one-step behavior; needs careful validation or renewed distillation |
| ByteDance SDXL-Lightning | Native 1024px output and released acceleration LoRAs offer a practical path from priest SDXL LoRA to accelerated inference | Two-step checkpoint at 1024×1024 for quality; evaluate 512×512 separately for speed | No primary source inspected here establishes complete sub-500ms API execution | LoRA composition can interfere; 1024px two-step may miss latency target even if faces are better |

SDXL-Turbo is the first speed/quality probe. Lightning is the stronger adaptation comparator, not a second proven sub-500ms solution. A small SD-Turbo control is useful if SDXL cannot fit the actual latency budget.

## 1. SDXL-Turbo: strongest latency evidence

Official model: [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo/tree/71153311d3dbb46851df1931d3ca6e939de83304), revision `71153311d3dbb46851df1931d3ca6e939de83304`.

Its [author model card](https://huggingface.co/stabilityai/sdxl-turbo) specifies 1–4 inference steps, recommends 512px, and disables classifier-free guidance and negative prompting. It explicitly acknowledges imperfect photorealism and malformed faces. These caveats matter more for this portrait-only site than generic image preference scores.

The [Stability AI release](https://stability.ai/news-updates/stability-ai-sdxl-turbo) reports **207ms on A100 FP16 for 512×512**, including prompt encoding, a single denoising step and decoding; UNet alone takes 67ms. Treat this as a resident-model pipeline measurement. It does not establish CPU image transfer, WebP encoding, HTTP handling, queueing, quality-filter retries, p95/p99 latency, or cold-start time. It is not an H100, 4090 or local Mac measurement.

Potential post-training route: train a low-rank priest-domain adapter with frozen text encoders/VAE, keeping captions varied across distinct adults, appearances and backgrounds. Merge it into the UNet for serving and evaluate after small checkpoints at the exact one-step scheduler. This is a hypothesis, not an official Turbo fine-tuning recipe. A standard all-timestep noise-prediction loss is not automatically compatible with preserving the distilled sampler. If adaptation ruins one-step fidelity, train a priest teacher first and re-distill with ADD or consistency training; that introduces a materially larger engineering/training task.

The [ADD paper](https://arxiv.org/abs/2311.17042) describes the combination of teacher distillation and adversarial supervision responsible for low-step synthesis. It does not prove that inexpensive generic LoRA fine-tuning preserves that capability.

License: the pinned repository's [actual LICENSE.md](https://huggingface.co/stabilityai/sdxl-turbo/raw/71153311d3dbb46851df1931d3ca6e939de83304/LICENSE.md) is the **Stability AI Community License Agreement, July 5, 2024**, despite stale `sai-nc-community` model-card metadata. It permits research/non-commercial uses and limited commercial uses under its conditions, including the revenue threshold. Preserve the license and notices; do not relabel these weights MIT. The file SHA-256 is in the companion JSON.

## 2. SDXL-Lightning: adaptation and native-resolution comparator

Official model: [`ByteDance/SDXL-Lightning`](https://huggingface.co/ByteDance/SDXL-Lightning/tree/c9a24f48e1c025556787b0c58dd67a091ece2e44), revision `c9a24f48e1c025556787b0c58dd67a091ece2e44`; license OpenRAIL++.

[The author card](https://huggingface.co/ByteDance/SDXL-Lightning) supplies separate 1/2/4/8-step models and full-UNet or LoRA forms. Start with `sdxl_lightning_2step_unet.safetensors` for the unadapted comparison. Use Euler trailing timesteps, guidance 0, and the checkpoint's exact step count. The one-step release is explicitly experimental and less stable; it uses sample/x0 prediction rather than the multi-step epsilon prediction. It should not be selected just to win a timing test. The authors favor full UNet quality and supply acceleration LoRAs for compatible fine-tuned bases.

Practical post-training proposal: train a priest-domain LoRA on ordinary SDXL, merge it into the SDXL base, then evaluate `sdxl_lightning_2step_lora.safetensors` on the adapted base. This separates learning the domain from acceleration and reuses released acceleration weights. It remains necessary to test whether collars, identities and face detail survive composition. Do not replace a priest-adapted UNet with the full Lightning UNet and claim the priest training is retained.

The [Lightning paper](https://arxiv.org/html/2402.13929v2) studies progressive adversarial distillation. Neither its broad speed claims nor “few steps” establishes sub-500ms server latency on the intended hardware. Test 1024px quality first, then 512px as a separate configuration; upsampling 512px is not native 1024px detail.

Supporting base: [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b), revision `462165984030d82259a11f4367a4eed129e94a7b`, OpenRAIL++.

## Controls and fallback methods

**SD-Turbo speed control:** [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo/tree/b261bac6fd2cf515557d5d0707481eafa0485ec2), revision `b261bac6fd2cf515557d5d0707481eafa0485ec2`. It is a smaller SD2.1-derived one-step model, usually 512px and guidance 0. Authors recommend SDXL-Turbo for better image quality and prompting. No complete numeric latency benchmark for SD-Turbo was verified in this pass; the 207ms claim belongs to **SDXL-Turbo**. Its pinned LICENSE.md is byte-identical to SDXL-Turbo's current Community License. Use as a speed/quality control, not the presumed best portrait generator. Post-training has the same distillation-preservation concern.

**LCM-LoRA adaptation fallback:** [`latent-consistency/lcm-lora-sdxl`](https://huggingface.co/latent-consistency/lcm-lora-sdxl/tree/a18548dd4956b174ec5b0d78d340c8dae0a129cd), revision `a18548dd4956b174ec5b0d78d340c8dae0a129cd`, card license OpenRAIL++. The [author card](https://huggingface.co/latent-consistency/lcm-lora-sdxl) explicitly demonstrates composing a domain/style LoRA with the acceleration LoRA at 4–8 steps, using LCMScheduler. This is the cleanest documented composition mechanism, but likely less favorable than one-step Turbo under a hard 500ms ceiling. Its speed benchmark section is TODO; it supplies no measured API bound.

[Diffusers LoRA training](https://huggingface.co/docs/diffusers/training/lora) provides maintained SDXL/text-to-image training entry points. [Latent consistency distillation](https://huggingface.co/docs/diffusers/training/lcm_distill) provides a teacher/student route if composition is insufficient. These establish available methods, not priest-domain convergence or a low training budget. A teacher trained only to produce one named person's identity would conflict with the required diversity; use multi-identity domain training instead.

## Benchmark that can actually prove the requirement

Use fresh random noise for every timed request and archive **every** image and failure. Cache fixed prompt embeddings and keep weights resident; neither removes fresh image generation. Fuse adapters and warm compiled kernels before warm measurement. Report compilation/model-loading and cold-start times separately, rather than counting them as zero.

For each candidate report native resolution, precision, exact scheduler/steps, GPU model, software revisions, batch size 1, and at least p50/p95/p99/max across repeated distinct inputs. Time server admission to fully encoded response bytes with CUDA synchronization. Break out text encoding, denoising, VAE decoding, device-to-host copy, format encoding, quality checks and all rejected attempts. Also measure concurrent requests and the actual public endpoint separately.

A 207ms warm benchmark leaves 293ms of theoretical margin; it does not justify two sequential quality-rejection retries. A hard bound requires either a model that reliably passes on its first draw or explicit bounded failure behavior, never an infinite retry loop or hidden finite accepted-image bank. On-demand serverless cold starts have not been shown to fit 500ms; a resident GPU or an explicitly different pre-generation design would be needed if the bound applies to the first request after idle as well.

No images, timing results or deployment claims are produced by this document. The next useful evidence is matched fresh-seed portrait output from Turbo and Lightning, followed by actual GPU endpoint timings for the visually viable candidate.
