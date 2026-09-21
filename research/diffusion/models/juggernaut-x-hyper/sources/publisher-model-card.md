---
license: creativeml-openrail-m
language:
- en
library_name: diffusers
pipeline_tag: text-to-image
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
- stable-diffusion
- stable-diffusion-xl
- sdxl
- hyper-sd
- text-to-image
- photorealistic
- fast
- low-step
- juggernaut
- rundiffusion
- kandooai
---

<div align="center">

<a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=header_banner">
  <img src="https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/resolve/main/assets/Juggernaut_Banner.webp" alt="Juggernaut — by RunDiffusion" />
</a>

<h1>Juggernaut X Hyper by RunDiffusion</h1>

<p><i>Juggernaut X aesthetic at Hyper-SD speed — photo-grade output in just a few steps.</i></p>

<p>
  <a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=hero_cta"><img alt="Try Juggernaut X Hyper — Free" src="https://img.shields.io/badge/%E2%96%B6%20Try%20Juggernaut%20X%20Hyper%20%E2%80%94%20Free-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>
</p>

<p>
  <a href="https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=prompting_resource"><img alt="Prompting Guides" src="https://img.shields.io/badge/%F0%9F%93%96%20Prompting%20Guides-1f1f23?style=for-the-badge"></a>&nbsp;<a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=lineup_compare"><img alt="Compare the Lineup" src="https://img.shields.io/badge/Compare%20the%20Lineup-1971c2?style=for-the-badge&labelColor=1f1f23"></a>&nbsp;<img alt="Few-step inference" src="https://img.shields.io/badge/Inference-Few--step-EAB308?style=for-the-badge&labelColor=1f1f23">&nbsp;<img alt="License: CreativeML Open RAIL-M" src="https://img.shields.io/badge/License-OpenRAIL--M-2ea44f?style=for-the-badge">
</p>

</div>

> Juggernaut X Hyper is the **Hyper-SD speed variant** of [Juggernaut X v10](https://huggingface.co/RunDiffusion/Juggernaut-X-v10). Same ground-up retrain, same GPT-4V-captioned dataset, same RunDiffusion Photo refinement — but distilled for **few-step inference**. Ideal for batch generation, real-time iteration, and live demos where latency matters more than the last 1% of detail.

## Inherited from X v10

- 🧠 Trained from the ground up — not a fine-tune of v9 or earlier
- 👁️ GPT-4 Vision captioning for sharper prompt adherence
- 🔤 Native text rendering for signage, labels, and short phrases
- 📸 Cleaner Full Body / Mid Shot / Portrait shot-type classification
- 💬 Two prompting styles: Natural language *or* Tagging

## What Hyper Adds

- ⚡ **Few-step inference** — generate in a fraction of standard SDXL inference time
- 💻 **Lower VRAM pressure** during inference
- 🔁 **Real-time iteration** — preview prompt changes in seconds

If you need the very best per-image quality and don't mind the wait, use the [**standard X v10**](https://huggingface.co/RunDiffusion/Juggernaut-X-v10) instead.

---

<div align="center">

<h3>⚡ Want speed without setup? Run Hyper on RunDiffusion.</h3>

<p>Pre-loaded with the right scheduler and CFG. <b>No installs. No GPU rental.</b></p>

<a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=midpage_cta"><img alt="Launch on RunDiffusion" src="https://img.shields.io/badge/%E2%96%B6%20Launch%20on%20RunDiffusion-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>

<p><i>Pre-loaded into ComfyUI · Forge · Automatic1111 · Fooocus · InvokeAI · SwarmUI</i></p>

</div>

---

## Recommended Settings

> Hyper-SD distillation needs different settings from standard Juggernaut. Steps low, CFG very low.

| Parameter | Value |
| --- | --- |
| Sampler | `DPM++ SDE` or `TCD` |
| Steps | `4 – 8` (start with `6`) |
| CFG scale | `1.0 – 2.0` |
| Resolution | `≥ 1024 × 1024` (use SDXL-native sizes) |
| VAE | **Already baked in** — no external VAE required |

Standard Juggernaut prompts work as-is. Don't adjust your prompt for Hyper — just adjust the sampler, steps, and CFG.

📖 **Want to go deeper on prompting?** See the [**RunDiffusion prompting library**](https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=prompting_inline). Free to read, no signup.

## Two Ways to Run Juggernaut X Hyper

### 🚀 Easiest — Run it on RunDiffusion (recommended)

One-click access inside ComfyUI, Forge, Automatic1111, Fooocus, InvokeAI, or SwarmUI. **Free trial included**, plus access to every other Juggernaut model on the same account.

<div align="center">
  <a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=quickstart_try"><img alt="Get Started Free" src="https://img.shields.io/badge/Get%20Started%20Free%20%E2%86%92-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>
</div>

### 💻 Or run it locally

For ComfyUI / Forge / InvokeAI / SwarmUI: download the single-file checkpoint and drop it into your `models/checkpoints/` directory. Set sampler to `DPM++ SDE`, steps to `6`, and CFG to `~1.5`.

## The Rest of the Family

| Model | Best for |
| --- | --- |
| [**Juggernaut XL v9**](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) | The flagship SDXL — **6M+ downloads** |
| [**Juggernaut X v10**](https://huggingface.co/RunDiffusion/Juggernaut-X-v10) | Standard X — full-step inference, maximum detail |
| [**Juggernaut XL Lightning**](https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning) | v9-base speed variant (5–7 steps) |
| [**Juggernaut Z**](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image) | Lumina-Image-2 — cinematic, presentation-ready |
| [**Juggernaut Pro Flux**](https://www.rundiffusion.com/juggernaut-pro-flux?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=cross_promo_pro_flux) | FLUX.1 — top-tier photo quality |

## Two Versions: SAFE and Standard

Juggernaut X Hyper ships in two flavors:

- **Juggernaut X Hyper** *(this repo)* — the standard creative model.
- **Juggernaut X Hyper SAFE** — safety-tuned for workplace, education, and family-friendly contexts. Available exclusively on **[RunDiffusion via Fooocus](https://www.rundiffusion.com/juggernaut-xl?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=safe_variant)**.

## Commercial Use

This model **may not be deployed behind paid API services** without explicit licensing. For commercial licensing, custom models, business inquiries, or consultation, contact **[juggernaut@rundiffusion.com](mailto:juggernaut@rundiffusion.com)**.

You are free to use this model for personal and creative work under the terms of the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).

## Credits

Juggernaut X Hyper was created by **[KandooAI](https://twitter.com/Juggernaut_AI)** in collaboration with **[RunDiffusion](https://www.rundiffusion.com/?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=credits)**, distilled with **Hyper-SD** for few-step inference. Photographic backbone by the RunDiffusion Photo team.

---

<div align="center">

<h3>Ready to generate at speed?</h3>

<p>Juggernaut X Hyper is one click away — no setup, no GPU rental.</p>

<a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=footer_cta"><img alt="Try Juggernaut X Hyper — Free" src="https://img.shields.io/badge/%E2%96%B6%20Try%20Juggernaut%20X%20Hyper%20%E2%80%94%20Free-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>

<p><sub><a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=footer_lineup">Compare the full Juggernaut lineup</a> · <a href="https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_hyper&utm_content=footer_prompting">Prompting guides</a> · <a href="mailto:juggernaut@rundiffusion.com">Commercial licensing</a></sub></p>

</div>
