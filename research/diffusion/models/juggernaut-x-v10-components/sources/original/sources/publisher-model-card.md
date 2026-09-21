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
- text-to-image
- photorealistic
- photography
- cinematic
- portrait
- juggernaut
- rundiffusion
- kandooai
---

<div align="center">

<a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=header_banner">
  <img src="https://huggingface.co/RunDiffusion/Juggernaut-X-v10/resolve/main/assets/Juggernaut_Banner.webp" alt="Juggernaut — by RunDiffusion" />
</a>

<h1>Juggernaut X v10 by RunDiffusion</h1>

<p><i>The ground-up retrain — GPT-4V captions, sharper prompt adherence, native text rendering.</i></p>

<p>
  <a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=hero_cta"><img alt="Try Juggernaut X — Free" src="https://img.shields.io/badge/%E2%96%B6%20Try%20Juggernaut%20X%20%E2%80%94%20Free-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>
</p>

<p>
  <a href="https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=prompting_resource"><img alt="Prompting Guides" src="https://img.shields.io/badge/%F0%9F%93%96%20Prompting%20Guides-1f1f23?style=for-the-badge"></a>&nbsp;<a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=lineup_compare"><img alt="Compare the Lineup" src="https://img.shields.io/badge/Compare%20the%20Lineup-1971c2?style=for-the-badge&labelColor=1f1f23"></a>&nbsp;<a href="https://civitai.com/models/133005/juggernaut-xl"><img alt="Civitai" src="https://img.shields.io/badge/Civitai-Overwhelmingly%20Positive-EAB308?style=for-the-badge&labelColor=1f1f23"></a>&nbsp;<img alt="License: CreativeML Open RAIL-M" src="https://img.shields.io/badge/License-OpenRAIL--M-2ea44f?style=for-the-badge">
</p>

</div>

> Juggernaut X (v10) is a **ground-up retrain** of the Juggernaut SDXL line. Not a fine-tune of v9 — the entire dataset was re-captioned with **GPT-4 Vision**, image classifications were rebuilt from scratch, and the model was trained fresh against an expanded, higher-quality image set. The result: noticeably better prompt adherence, cleaner shot-type control, and native text-rendering capability.

## What's Different in X / v10

- 🧠 **Retrained from the ground up** — not fine-tuned on top of a previous Juggernaut.
- 👁️ **GPT-4 Vision captioning** — dataset re-captioned with detailed, structured language descriptions.
- 🎯 **Stronger prompt adherence** — does what you ask more often, with less wrestling.
- 📸 **Cleaner shot-type classification** — Full Body, Mid Shot, Portrait, etc., now distinct in the model's understanding.
- 🔤 **Native text rendering** — handles legible signage, labels, and short phrases directly.
- 🖼️ **RunDiffusion Photo** refinement — same photographic backbone that made v9 popular.

---

<div align="center">

<h3>🚀 Skip the setup. Try Juggernaut X in your browser.</h3>

<p>No installs. No GPU rental. No model downloads. <b>Just generate.</b></p>

<a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=midpage_cta"><img alt="Launch on RunDiffusion" src="https://img.shields.io/badge/%E2%96%B6%20Launch%20on%20RunDiffusion-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>

<p><i>Pre-loaded into ComfyUI · Forge · Automatic1111 · Fooocus · InvokeAI · SwarmUI</i></p>

</div>

---

## Two Prompting Styles

X supports two distinct prompt styles. Pick whichever fits your workflow:

| Style | Best for | Example |
| --- | --- | --- |
| **Natural language** | Storytelling, scenes, complex instructions | *"A photographer on a windswept cliff at golden hour, holding a vintage Leica, the ocean churning below"* |
| **Tagging** | Quick iteration, established workflows, LoRA stacking | *`photograph, photographer, cliff, golden hour, vintage Leica camera, ocean, waves, cinematic lighting`* |

📖 **For deeper prompting technique** — see the [**RunDiffusion prompting library**](https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=prompting_inline). Free to read, no signup.

## Two Ways to Run Juggernaut X

### 🚀 Easiest — Run it on RunDiffusion (recommended)

One-click access inside ComfyUI, Forge, Automatic1111, Fooocus, InvokeAI, or SwarmUI. **Free trial included**, plus access to every other Juggernaut model on the same account.

<div align="center">
  <a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=quickstart_try"><img alt="Get Started Free" src="https://img.shields.io/badge/Get%20Started%20Free%20%E2%86%92-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>
</div>

### 💻 Or run it locally — Diffusers

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-X-v10",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

prompt = "A photographer on a windswept cliff at golden hour, holding a vintage Leica, the ocean churning below, cinematic lighting, hyperdetailed photography"

image = pipe(prompt, width=832, height=1216, num_inference_steps=35, guidance_scale=5.0).images[0]
image.save("juggernaut_x_v10.png")
```

For ComfyUI / Forge / InvokeAI / SwarmUI: download the single-file checkpoint and drop it into your `models/checkpoints/` directory.

## Recommended Settings

| Parameter | Value |
| --- | --- |
| Resolution | `832 × 1216` (portrait) · `1216 × 832` (landscape) |
| Sampler | `DPM++ 2M Karras` |
| Steps | `30 – 40` |
| CFG scale | `3 – 7` (lower = more realistic) |
| VAE | **Already baked in** — no external VAE required |

## Two Versions: SAFE and Standard

Juggernaut X ships in two flavors:

- **Juggernaut X** *(this repo)* — the standard creative model.
- **Juggernaut X SAFE** — a safety-tuned variant for workplace, education, and family-friendly contexts. Available exclusively on **[RunDiffusion via Fooocus](https://www.rundiffusion.com/juggernaut-xl?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=safe_variant)**. *SAFE* = **S**uitable **A**i **F**or **E**veryone.

## Looking For Something Different?

| Model | Why pick it |
| --- | --- |
| [**Juggernaut XL v9**](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) | The most popular Juggernaut — proven SDXL photorealism workhorse. **6M+ downloads.** |
| [**Juggernaut XI v11**](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11) | Continuation of the SDXL line, post-X refinement |
| [**Juggernaut Z**](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image) | Lumina-Image-2 architecture — cinematic, presentation-ready |
| [**Juggernaut Pro Flux**](https://www.rundiffusion.com/juggernaut-pro-flux?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=cross_promo_pro_flux) | FLUX.1 backbone — top-tier photo quality with strong consistency |

## Commercial Use

This model **may not be deployed behind paid API services** without explicit licensing. For commercial licensing, custom models, business inquiries, or consultation, contact **[juggernaut@rundiffusion.com](mailto:juggernaut@rundiffusion.com)**.

You are free to use this model for personal and creative work under the terms of the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).

## Credits

Juggernaut X (v10) was created by **[KandooAI](https://twitter.com/Juggernaut_AI)** in collaboration with **[RunDiffusion](https://www.rundiffusion.com/?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=credits)**. Dataset re-captioned with the **GPT-4 Vision Captioning** tool by **[LEOSAM](https://civitai.com/user/LEOSAM)**. Photographic backbone by the RunDiffusion Photo team.

---

<div align="center">

<h3>Ready to generate?</h3>

<p>Juggernaut X is one click away — no setup, no GPU rental, no model downloads.</p>

<a href="https://app.rundiffusion.com/login?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=footer_cta"><img alt="Try Juggernaut X — Free" src="https://img.shields.io/badge/%E2%96%B6%20Try%20Juggernaut%20X%20%E2%80%94%20Free-7C3AED?style=for-the-badge&labelColor=7C3AED"></a>

<p><sub><a href="https://www.rundiffusion.com/juggernaut?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=footer_lineup">Compare the full Juggernaut lineup</a> · <a href="https://www.rundiffusion.com/prompting?utm_source=huggingface&utm_medium=model_card&utm_campaign=juggernaut_x_v10&utm_content=footer_prompting">Prompting guides</a> · <a href="mailto:juggernaut@rundiffusion.com">Commercial licensing</a></sub></p>

</div>
