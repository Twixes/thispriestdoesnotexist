# Pinned Juggernaut X Hyper research checkpoint

Publisher: `RunDiffusion/Juggernaut-X-Hyper`, revision `42fee7475922d8de7246cdcbe5e3a5e22c0ecf61`.

Checkpoint: `JuggernautXRundiffusion_Hyper.safetensors`, 7,105,348,616 bytes, SHA256 `010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98`.

`prepare.py` downloads this single-file checkpoint and the publisher's small component configs/tokenizers. It verifies the pinned model API entries: upstream SHA256 for LFS files and Git blob SHA1 for ordinary files. It stores the full checkpoint as an ignored local file and splits it into ordered Git LFS parts of at most1GiB. `provenance.json` records every source URL, upstream entry, file/part hash and safetensors header. `verification.json` verifies both the full file and the concatenated parts. No pickle `.bin` weights, duplicate FP32 components or banner images are downloaded.

The original model card is retained byte-exact at `sources/publisher-model-card.md`. `LICENSE.txt` is the CreativeML OpenRAIL-M text at the CompVis license Space linked by that publisher card, pinned to `14d42d09bffd871b1666a084fc954a50cff72ac0`; it is not an invented license file from the model repository. The card also states that paid API deployment requires explicit licensing. Retain both texts with any model derivative/distribution; application MIT licensing does not replace them.

After Git LFS checkout, run `python3 research/diffusion/models/juggernaut-x-hyper/reassemble.py` to recover the ignored full checkpoint. Run `prepare.py --verify-only` to verify all files and the concatenated parts. Neither script loads the model.

For a future offline load, the intended entry point is `StableDiffusionXLPipeline.from_single_file(local_checkpoint, config=local_model_directory, local_files_only=True, torch_dtype=torch.float16, use_safetensors=True, add_watermarker=False)`. The local component directories contain configuration and tokenizer files; their FP32 `.bin` weight files are intentionally absent because weights should come from the complete safetensors checkpoint. **This conversion/load has not been executed or validated here.** Preserve the checkpoint's own text encoders and VAE instead of reusing Turbo components or training caches.

The author recommends DPM++ SDE or TCD,4–8steps (start6), CFG1–2 and native SDXL sizes≥1024. A bounded prospective latency candidate is TCD4/CFG1/native1024, with512 evaluated separately as a non-native quality/latency tradeoff. Freeze the actual TCD scheduler config and eta before generating images; the publisher does not specify eta. Do not silently use the repository's baked scheduler settings as though they were that recipe. No extra Hyper acceleration LoRA is needed for this already distilled checkpoint.

Its SDXL UNet architecture supports the usual attention LoRA structure. Priest-adapter transfer quality and preservation of distilled sampling after post-training are unproven. This is a research download, not a reviewed output model or a production release; it establishes neither photographic quality nor server generation below500ms.
