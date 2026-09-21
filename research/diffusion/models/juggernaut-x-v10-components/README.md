# Verified full Juggernaut X v10 components

Exported from the full non-Hyper checkpoint via `research/diffusion/export_juggernaut_full_components.py`, using the existing pinned single-file FP16 loader on CPU. Completed in32.78seconds, peak sampled RSS11.99GiB, no guard failures. All2641 tensors matched dtype, shape and value SHA256 on fresh CPU component reload. No inference/training occurred during export.

Provenance SHA256: `4e18b462e1e07d35fb753ecb89dadabfd2d66213646f30467ee351f2bb36b40a`. Original checkpoint SHA256: `d91d35736d8f2be038f760a9b0009a771ecf0a417e9b38c244a84ea4cb9c0c45`.

The source model's own text encoders, UNet and VAE are preserved; no Hyper/Turbo component is substituted. Publisher scheduler/tokenizer files are byte-exact. Converted configs occupy the standard component paths; original configs and model card remain under `sources/original`. CreativeML OpenRAIL-M and additional paid-API licensing notice remain applicable.

Large original weight files are ignored locally and archived as verified LFS parts of at most1GiB. After checkout run `python3 research/diffusion/restore.py research/diffusion/models/juggernaut-x-v10-components`. Reproducing the export requires a fresh output directory.
