# Photographic base comparison preview

Serve `research/` and open `/reviews/juggernaut-preview/`. The preview keeps all eight original prompt/seed cases and excludes warmups. It compares an earlier pretrained base with Juggernaut at native1024 and at512. The native selector switches between TCD4/CFG1 and TCD6/CFG1.5; both use eta0.3. No model here has priest-specific post-training by this project.

Rebuild from retained artifacts:

```sh
research/.venv-flux-train/bin/python research/reviews/juggernaut-preview/build.py
node --check research/reviews/juggernaut-preview/app.js
```

The builder checks image hashes and exact prompt/seed correspondence. Same numeric seeds across different latent shapes/models do not mean matched identities. Missing cases stay visible as pending; complete outputs are never selected or ranked. The viewer uses original native PNGs, with optional display-only grayscale. The1024 VAE uses512px tiles with25% overlap; the earlier failed untiled run is retained separately. TCD6 is an additional reference configuration, not an exact copy of the publisher's DPM example.

`review-first4.json` binds native1024 TCD4 judgments for000–003 to image, record, model, runtime and protocol hashes. It does not review the512 or six-step images. The observations do not establish that tiling caused the visible texture. `write-review.py` materializes those recorded manual judgments; it does not perform visual evaluation or inference.

No production approval or GPU-server latency claim is implied by this preview.

`review-sixstep-first4.json` records the additional six-step/CFG1.5 native review for000–003 and references the exact four-step images. Both settings changed, so their effects are not isolated.

RealVis native reference path: `realvis-v5-lightning-sde5-native1024-tiled-v1`. The native selector lists it as pending until a completed result and eight validated images exist. Both native Juggernaut runs are available in the comparison-base selector, enabling direct RealVis/Juggernaut comparison. Rebuild after a run finishes; no polling or model computation happens in the browser.
