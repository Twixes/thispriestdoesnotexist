# Photoreal fast SDXL priors worth comparing

Checked2026-09-21 using publisher model cards, pinned configs and the public Hugging Face model API. **Compare RealVisXL V5 Lightning and Juggernaut X Hyper before extending Turbo adaptation.** Both deliberately target photographic output; neither has yet demonstrated better priest faces in our matched test. No weights downloaded, models run or example images independently evaluated for this report.

## Two concrete candidates

| Candidate | Exact revision | Publisher settings | Practical fit |
| --- | --- | --- | --- |
| [SG161222/RealVisXL_V5.0_Lightning](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/tree/f4454158cedaab9f0688c199561d6c92525f3a85) | `f4454158cedaab9f0688c199561d6c92525f3a85` | **5 steps**, DPM++ SDE or DPM++ SDE Karras, CFG1–2 | Easiest integration: complete FP16 Diffusers components. Stronger photographic prior is a hypothesis; four steps is a new experiment, not the author's recommendation. |
| [RunDiffusion/Juggernaut-X-Hyper](https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/tree/42fee7475922d8de7246cdcbe5e3a5e22c0ecf61) | `42fee7475922d8de7246cdcbe5e3a5e22c0ecf61` | **4–8 steps**, start6; DPM++ SDE or TCD, CFG1–2, resolution≥1024 | The directly supported four-step candidate. Start with TCD4/CFG1 for a latency-focused arm, plus a6-step author-default reference. |

RealVis's author describes its photographic aim and gives the five-step settings. Its published Diffusers scheduler is actually **DDIM with leading timesteps**: loading the repository's defaults does not reproduce the recommended sampler. Select and record an explicit DPM++ SDE implementation/config. Its UNet uses the usual SDXL1024 latent-size configuration. [Author card](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/README.md), [scheduler config](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/scheduler/scheduler_config.json).

Juggernaut X Hyper is the publisher's Hyper-SD variant of X, with its photographic refinement. Its card supports four steps but recommends starting at six. The single-file checkpoint includes its VAE; do not substitute Turbo's VAE or assume identical encoders. [Author card](https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/blob/42fee7475922d8de7246cdcbe5e3a5e22c0ecf61/README.md).

Verified public, ungated weight identifiers from the [RealVis API](https://huggingface.co/api/models/SG161222/RealVisXL_V5.0_Lightning?blobs=true) and [Juggernaut API](https://huggingface.co/api/models/RunDiffusion/Juggernaut-X-Hyper?blobs=true):

| Weight | Bytes | SHA256 |
| --- | ---: | --- |
| RealVis `RealVisXL_V5.0_Lightning_fp16.safetensors` | 6,938,065,512 | `fabcadd9330dcc4f9702063428d40b9d4d07168d8acefc819b8d1d9db466b3ec` |
| RealVis `unet/diffusion_pytorch_model.fp16.safetensors` | 5,135,149,760 | `1143cd2aaf65d24af34b5699d090aed724f6c0978c2ec5a5f56821ccb36260ce` |
| Juggernaut `JuggernautXRundiffusion_Hyper.safetensors` | 7,105,348,616 | `010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98` |

Choose RealVis FP16 components **or** its all-in-one file, not both. Four FP16 component weight files total6,938,011,430bytes plus small configs/tokenizers. Juggernaut's component weights are older FP32 `.bin` files; its safetensors single-file checkpoint is the simpler download. Use a pinned local config with `StableDiffusionXLPipeline.from_single_file`, then archive/verify any converted FP16 Diffusers files. Checkpoint conversion/loading remains untested here.

## Licensing and adaptation

RealVis declares **OpenRAIL++** in its card. Retain that model license independently from our MIT application code. Juggernaut X Hyper declares **CreativeML Open RAIL-M**, with an additional publisher notice requiring explicit licensing for paid API services; its card permits personal/creative use. This report records those terms, not a broader legal clearance. [RealVis card](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/README.md), [Juggernaut terms](https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/blob/42fee7475922d8de7246cdcbe5e3a5e22c0ecf61/README.md).

Both pinned UNet configs match Turbo's principal architecture fields: channels, block widths, cross-attention dimensions, transformer depths, attention head dimensions, text-time conditioning and projection size. This supports **structural SDXL attention-LoRA compatibility**, not proven quality transfer of our Turbo-trained priest adapter. Diffusers supports loading UNet SDXL LoRAs, but neither publisher promises that ordinary post-training preserves few-step distillation. First compare these bases without the priest adapter. If one wins, initialize a fresh priest LoRA against that exact base and regenerate caption/VAE caches; current caches are bound to Turbo's different encoder/VAE weights. [RealVis config](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/unet/config.json), [Juggernaut config](https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/blob/42fee7475922d8de7246cdcbe5e3a5e22c0ecf61/unet/config.json), [Diffusers adapter documentation](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters).

I would skip **Juggernaut XI Lightning** for this public adaptation workflow: its current card declares CC BY-NC-ND4.0 and explicitly disallows redistributing fine-tunes/merges. The older Juggernaut XL Lightning has a filename ending `4Steps`, but its current author settings say5–7 steps. A filename is insufficient evidence for the current recommended recipe. [XI card](https://huggingface.co/RunDiffusion/Juggernaut-XI-Lightning/blob/dbb242e5b54f33ce3bde22464f8813c1427bf1d4/README.md), [XL Lightning card](https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/blob/9c35e7ca1112b7e567ae7b24400b83935909916d/README.md).

## Small decisive comparison

Use the existing eight development prompt/seed pairs first, retain every native image, and leave sealed tests unopened. Compare RealVis5/DPM++SDE/CFG1 and JuggernautHyper4/TCD/CFG1 at1024 for photographic quality; include Juggernaut6/CFG1.5 as a publisher-default reference if its4-step output fails. Specify scheduler class/config and TCD eta before generation. Repeat only a promising base at512 to test the actual latency-oriented candidate.512 is a quality/latency tradeoff, not the publisher's native-quality setting. No face restoration, hires fix or selective rerolls. We seek less waxy/etched skin and intact eyes across multiple faces, not merely a nicer contact sheet.

Four-step GPU generation below500ms is **plausible enough to benchmark, unproven**. At CFG1, standard Diffusers SDXL skips the unconditional guidance branch; above1 it adds substantial UNet work. With four actual UNet evaluations and150ms spent in conditioning/VAE/encoding, each evaluation must average below87.5ms. Sampler “steps” need not equal model evaluations, so instrument the real call count. Native1024 has four times512's latent spatial area; do not infer1024 latency from a512 result or assume linear scaling. Benchmark warm/cold complete requests, including encoding and any retries, on the chosen GPU. A good prior can avoid training for photorealism, but it cannot eliminate the hosting budget/latency tradeoff by itself.
