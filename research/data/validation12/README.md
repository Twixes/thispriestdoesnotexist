# Held-out synthetic validation portraits

Twelve new fictional adult priest portraits generated with the built-in OpenAI
image tool. These are evaluation references, **not outputs from our trained
StyleGAN model**, and are excluded from every training dataset.

Original 1254-square PNG files are in `source/`. Exact individual generation
prompts are in `prompts.json`; `provenance.json` records the returned original
paths. `manifest.json` includes all twelve source checksums and prompts.
`build_manifest.py` reproduces the manifest and labeled contact sheet without
changing the originals. `visual-review.json` records acceptance of the originals.

The separately documented `aligned256/` inputs reuse the current training set's
collar-preserving eye-position recipe. They inherit its reflected-border
behavior, with **more padding on average than the training set**. Read
`alignment-review.md` before interpreting discriminator scores. A score gap can
reflect crop artifacts or other distribution shifts as well as memorization.

This small set supports development diagnostics. It is not real-world ground
truth, a reliable standalone measure of image quality, or an untouched final
test set once it has informed model selection. Do not move it into training or
claim that different file hashes establish different generated identities.
