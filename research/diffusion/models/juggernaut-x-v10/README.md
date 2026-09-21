# Full Juggernaut X v10 quality reference

Publisher `RunDiffusion/Juggernaut-X-v10`, revision `35e4f3ccbe745c3e1221715d04a452aaa1ee5761`. This is the full non-Hyper release. Downloaded only its single safetensors checkpoint and small configuration/tokenizer/license files; no duplicate pickle/FP32 components.

`Juggernaut-X-RunDiffusion-NSFW.safetensors`:7,105,348,672bytes, SHA256 `d91d35736d8f2be038f760a9b0009a771ecf0a417e9b38c244a84ea4cb9c0c45`. All17 source files, the full checkpoint, and concatenated seven LFS parts were verified. Source provenance SHA256 `154d5e4735df58e6af795657e19a436b540dafdd1d6cb34374ab03befce64e70`. Reconstruct after Git LFS checkout with `python3 research/diffusion/restore.py research/diffusion/models/juggernaut-x-v10`.

The current [publisher card](https://huggingface.co/RunDiffusion/Juggernaut-X-v10/blob/35e4f3ccbe745c3e1221715d04a452aaa1ee5761/README.md) recommends DPM++2M Karras,30–40steps and CFG3–7, with lower guidance favoring realism. It lists832×1216 portrait and1216×832 landscape. Our quality reference uses35steps/CFG3 and1024square for the site's framing, explicitly differing from those aspect ratios. The original eight prompts/seeds and warmup remain fixed. No higher-step Hyper run is substituted for this model.

CreativeML OpenRAIL-M and the publisher's additional explicit-licensing requirement for paid API deployment are retained in `LICENSE.txt` and `sources/publisher-model-card.md`. These model terms are separate from application MIT licensing.

`hyper-comparison-configs.json` finds identical architectural configs. `hyper-comparison-vae.json` compares raw tensor ranges: all248 VAE tensors match Hyper exactly, including dtype, shape and value SHA256. This does not imply UNet/text-encoder weight equality. All models' tensor names, shapes and dtypes match structurally. The evaluation nevertheless uses this full release's own exported weights for every component.

The guarded CPU export completed successfully: all2641 component tensors matched their freshly reloaded saved representations. See sibling `juggernaut-x-v10-components/provenance.json`. No inference or training was performed during this preparation task, and neither photographic acceptance nor server latency is established.
