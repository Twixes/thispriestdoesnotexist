# Juggernaut X Hyper verified component export

Export completed successfully in29.26seconds with peak sampled RSS12.25GiB and no guard failure. All2641 tensors matched exactly on fresh CPU reload:196 in text encoder1,517 in text encoder2,1680 in UNet and248 in VAE. The provenance SHA256 is `8e1c607e04bef6fcee8e778e9e4fc5d163ae19f29d67d80149debd61ad78774d`; the verification SHA256 is `636eabe8b0f32e817b058c084d0fead8110a14efc53961d99d7b501001f5385b`. No model inference or training ran.

The completed export used:

```sh
research/.venv-flux-train/bin/python research/diffusion/export_juggernaut_components.py
```

Reproduction must use a fresh `--output` directory; the exporter refuses to overwrite this archive. It uses the same pinned offline single-file FP16 loader as the successful Juggernaut comparisons, entirely on CPU. Each encoder, UNet and original VAE is saved with `save_pretrained`, then freshly reloaded on CPU. Every named tensor must match dtype, shape and SHA256 of contiguous values before the export can be marked complete. The target is exact preservation of the loader's FP16 output; it does not claim the original checkpoint's FP32 tensors avoid the loader's FP16 cast.

The publisher scheduler and tokenizer files remain byte-exact. Original component configs, license and model card are preserved under `sources/original`; component configs at standard paths are those emitted for the converted modules. No TCD inference scheduler is substituted into the training export. Both the original and loader-resolved scheduler are retained. Training needs fresh image/text caches from this base; Turbo caches and adapters are not silently reused.

The script hashes the upstream checkpoint and all pinned source files before loading. Saved weights larger than1GiB are archived as ordered LFS parts of at most1GiB; the full local originals are ignored. After checkout, reconstruct and verify them with `python3 research/diffusion/restore.py research/diffusion/models/juggernaut-x-hyper-components`.

Resource limits:35% memory available before start,20% runtime floor,24GiB process RSS,512MiB swap growth and20minutes total. The process stays on CPU, so it does not allocate MPS model memory. A supervisor enforces guards during loader/export/reload calls. Full original pipeline plus one newly reloaded component can coexist temporarily; no duplicate full pipeline is loaded. A failed attempt is retained and requires a fresh output directory.

CreativeML OpenRAIL-M and the publisher's additional paid-API licensing notice apply to the model, separate from the application's MIT license. No inference, training, production quality or server-latency claim is made by this export.
