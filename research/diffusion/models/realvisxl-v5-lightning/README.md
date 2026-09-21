# RealVisXL V5 Lightning: pinned FP16 components

Publisher `SG161222/RealVisXL_V5.0_Lightning`, revision `f4454158cedaab9f0688c199561d6c92525f3a85`. Downloaded only the four FP16 Diffusers component weights and small model/config/tokenizer files. No all-in-one checkpoint or FP32 weight duplicates. Total component weight bytes:6,938,011,430.

`provenance.json` binds20 files to publisher SHA256/Git blob IDs and retained upstream metadata. The UNet is stored as five ordered LFS parts and the second text encoder as two; every part≤1GiB. Reconstructed originals are ignored. `verification.json` records independent full-file and concatenated-part checks. Use `python3 research/diffusion/restore.py research/diffusion/models/realvisxl-v5-lightning` after Git LFS checkout to reconstruct originals and verify all files, without loading a model.

`sources/publisher-model-card.md` is byte-exact. The card declares OpenRAIL++; the repository has no separate license file. `LICENSE.md` preserves the canonical SDXL CreativeML OpenRAIL++-M text from `stabilityai/stable-diffusion-xl-base-1.0` revision `462165984030d82259a11f4367a4eed129e94a7b`, with explicit provenance. Model licensing remains separate from application MIT licensing.

The author recommends five steps, DPM++ SDE Karras or DPM++ SDE, CFG1–2. The baked scheduler config is DDIM and is **not** the evaluator's sampler. See [author card](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/README.md).

Prepared evaluator: `research/diffusion/compare_realvis.py`. It preserves the exact original eight prompts/seeds plus separate warmup, loads only these components, and uses no Turbo cache or LoRA. The weights have been verified; model loading/inference has not been run during preparation. Refer to `research/diffusion/compare_realvis.README.md` for the fixed sampler, tiling option and command.
