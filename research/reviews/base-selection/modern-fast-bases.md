# Modern fast bases: FLUX.2 klein 4B versus Z-Image-Turbo

Checked 2026-09-21 against official author pages, pinned Hugging Face metadata and maintained implementation sources. This is a research shortlist, not a model-output or latency test. No weights were downloaded and no resources purchased. Earlier SDXL comparisons establish only the best of the tested configurations; they do **not** establish a globally best base.

## Concrete next decision

**Evaluate `black-forest-labs/FLUX.2-klein-4B` next.** It has a compelling combination of current image-quality priors, four-step inference, Apache 2.0 weights, and an author-documented way to train on the undistilled base then reuse that adapter on the fast distilled model. This is stronger evidence for economical post-training than guessing that a generic SDXL-Turbo fine-tune will preserve one-step quality. Whether it generates convincing priest clothing before training is unknown and should be measured directly.

Keep Z-Image-Turbo as the next modern quality comparator. Its public photorealism claims are relevant, but eight transformer evaluations, a larger download, and a less direct endorsed Turbo training route make it second choice for our strict latency goal.

## Reproducible identifiers and storage

Sizes below are exact sums of the repository's Diffusers safetensors, from pinned `?blobs=true` metadata. They exclude configs/tokenizers and alternate top-level single-file duplicates. Decimal GB is shown for readability; bytes, upstream SHA-256 values and file names are in `modern-fast-bases.json`.

| Model | Revision | Transformer | Text encoder | VAE | Total weights | License |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | 7.751 GB | 8.045 GB | 0.168 GB | 15.964 GB | Apache 2.0 |
| `black-forest-labs/FLUX.2-klein-base-4B` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | 7.751 GB | 8.045 GB | 0.168 GB | 15.964 GB | Apache 2.0 |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` | 24.620 GB | 8.045 GB | 0.168 GB | 32.832 GB | Apache 2.0 |

Sources: [FLUX distilled repository](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294), [FLUX training base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d), [Z-Image-Turbo repository](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6).

Both text encoder configs specify a Qwen3 architecture with 36 layers and 2560 hidden width. Similar total sizes do not prove interchangeable weights; only reuse a component after checking its complete upstream hash. On-disk sizes are not runtime VRAM requirements. Z-Image's transformer archive is much larger than its half-precision runtime footprint; converting dtype at load does not reduce the original download. Retain originals and store every >2GB file in <=1GiB LFS parts, with ignored local reconstruction as in the existing diffusion tooling.

## What the speed claims actually establish

**FLUX.2 klein:** BFL's [January 2026 announcement](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence) claims generation/editing below 0.5 seconds and identifies **GB200 in BF16** as the hardware/precision for its latency comparison. It does not give a textual exact millisecond result for the 4B model or establish our complete HTTP response budget. Separate quantization speedup comparisons use RTX 5080/5090 at 1024px; those are relative improvements, not evidence that a rented 4090 meets 500ms. BFL also says the undistilled base is more diverse and better suited to fine-tuning. These are author-reported general results, not priest-portrait measurements.

The [distilled 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) demonstrates 1024px, four steps, guidance 1.0, BF16, and reports approximately 13GB VRAM accessibility. Its sample enables CPU offload, which is a memory tradeoff, not a path to minimum latency. Its broad image-generation/editing ability is a promising prior, not proof of a collar or photographic skin in our prompts.

**Z-Image-Turbo:** the [author card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) claims sub-second latency on **H800**, eight function evaluations, and strong photorealistic generation. Its example sets nine API steps to obtain eight actual transformer forwards, guidance 0, 1024px. The model-family table explicitly rates Turbo diversity low and fine-tunability N/A while rating the undistilled Z-Image suitable for fine-tuning. The [paper's latency footnote](https://arxiv.org/html/2511.22699v1) requires FlashAttention-3 plus `torch.compile` for the sub-second result. No exact latency or strict <500ms bound was verified; “sub-second” alone does not satisfy our requirement.

Neither claim includes a verified cold start, p95/p99, CPU image conversion, WebP encoding, request handling, concurrency or quality-rejection retries for this site. Need actual resident GPU measurements; do not promise the same result on MPS, A100 or 4090 by extrapolating GB200/H800 numbers.

## Post-training support

The official BFL [June 2026 LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora) gives a particularly useful route: train against **klein base-4B**, then load the adapter onto **distilled klein-4B** for four-step inference. It says this usually performs better in their tests. Its example uses 15–40 images, about an hour on an RTX 4090, and a LoRA run below 24GB memory. Those numbers describe their style-adaptation example, not a guaranteed priest-domain budget. We need a multi-identity, domain-focused training set; a single-character identity LoRA would undermine the requirement for genuinely different people. First measure whether the unadapted distilled model already handles the garments and only needs texture/style refinement.

For Z-Image, the maintained [Diffusers training example](https://github.com/huggingface/diffusers/blob/9f1246971270c84dcbe71233edb7a519596a5d02/examples/dreambooth/README_z_image.md) targets `Tongyi-MAI/Z-Image`, not Turbo. It supports LoRA, cached latents, gradient checkpointing and configurable attention/feed-forward modules. This confirms that training tooling exists for the base, but does not prove that its adapter transfers safely to Turbo or preserves eight-step quality. That is a separate experiment or a re-distillation task. Do not convert an N/A fine-tuning statement into a claim that training is impossible; distinguish the unproven fast-model route from available base-model training.

## Local MPS feasibility

Both have maintained Diffusers pipeline classes. The pinned [Flux2KleinPipeline source](https://github.com/huggingface/diffusers/blob/9f1246971270c84dcbe71233edb7a519596a5d02/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L884) explicitly handles an MPS dtype issue. This supports attempting local inference; it is not a successful run on this Mac. Start with one supervised image and a memory/time ceiling, rather than running a cohort before load compatibility is known.

Z-Image's official repository merged [MPS Flash Attention support](https://github.com/Tongyi-MAI/Z-Image/pull/137) on January 30, 2026, merge `5e066db6d9afc0a5f1ffe7fa4227840d01c12f66`. It offers an optional `mps-flash-attn` backend for the author's implementation. That backend is distinct from NVIDIA FlashAttention-3 and from assuming the same integration exists in Diffusers. Neither local BF16 behavior nor inference duration has been verified here.

Caching prompt embeddings once and then unloading the roughly 8GB text encoder is a useful deployment hypothesis for both models. A small bank of varied text prompts is compatible with drawing fresh image noise on every request; it is not a bank of output images or accepted seeds. Cache and model changes must be benchmarked and archived explicitly.

## Bounded next experiment

1. Download only the distilled klein-4B Diffusers package, not the duplicate top-level transformer or training base yet; archive pinned provenance and LFS chunks.
2. Run one supervised local MPS smoke image, then all eight current comparison prompts/seeds at native 1024px and four steps. Review every output before selecting a deployment candidate.
3. If it wins photographic quality and recognizable priest appearance, compare 512/768px only as separate measured configurations. Do not shrink images silently to claim speed.
4. Benchmark the best visually acceptable configuration on a specified GPU, including encoded response bytes. Only after that choose whether a base-4B LoRA training run is worthwhile.

This closes an important shortlist gap. It does not declare FLUX a winner before seeing its outputs, and it does not establish that the 500ms and hosting-cost requirements can simultaneously be met.
