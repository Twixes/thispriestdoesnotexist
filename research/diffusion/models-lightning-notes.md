# Pinned local SDXL-Lightning two-step assembly

`prepare_lightning.py` fetches ByteDance's **full two-step UNet**, not its LoRA, at revision `c9a24f48e1c025556787b0c58dd67a091ece2e44`. The model directory is `models/sdxl-lightning-2step`. Remaining components and the UNet architecture config come from official SDXL base revision `462165984030d82259a11f4367a4eed129e94a7b`.

The assembly copies existing local assets only when their complete SHA-256 matches the pinned upstream LFS object. Copies keep subsequent edits independent. Small non-LFS files are checked against the upstream Git blob SHA-1; all assembled files also receive SHA-256 hashes. `provenance.json` records exact source URLs, upstream metadata, any reuse, assembly settings and parts. Both upstream licenses and model cards are retained under `sources/`.

The 5,135,149,736-byte full UNet has upstream SHA-256 `15a96a3b213b8b4d6ea720b4fa5bf6723cd877170bd1a90f58e34da2968ef854`. It is retained as five parts of at most 1 GiB for Git LFS. The reconstructed `unet/diffusion_pytorch_model.fp16.safetensors` is already covered by the existing diffusion `.gitignore`. The parts match the existing `.gitattributes` LFS rule. Concatenation is byte preserving, not model conversion.

To verify/reconstruct without loading the model:

```sh
python3 research/diffusion/load_lightning.py
```

To load for a separately supervised experiment:

```python
from load_lightning import load_pipeline
pipe = load_pipeline(device="cuda")
image = pipe(
    prompt="A photographic portrait of an adult man wearing a black clerical shirt and white Roman collar, bareheaded",
    num_inference_steps=2,
    guidance_scale=0,
    height=1024,
    width=1024,
    generator=fresh_generator,
).images[0]
```

The loader forces **EulerDiscreteScheduler, trailing timestep spacing, epsilon prediction** and local-only safetensors. Use exactly two inference steps. The one-step Lightning model has different prediction semantics and is not this checkpoint. The default dtype is FP16 for CUDA/MPS and FP32 for CPU; callers can override it. Resource bounds, RNG choice, inference timing and output archival belong to the experiment runner.

[Official usage and licensing](https://huggingface.co/ByteDance/SDXL-Lightning) and [official SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) remain authoritative. No GPU load or inference was performed while preparing this package; successful hashes establish byte integrity, not model output quality or load compatibility with the current environment.
