# Bias-family ablation review

Rebuild from the repository root:

```sh
research/.venv/bin/python research/reviews/adam-bias-ablation/build.py
```

Open `http://127.0.0.1:8787/reviews/adam-bias-ablation/` with the existing research preview server. This static, non-blocking page displays all 20 native images from the completed `adam-native1024-bias-ablation100-v1` run: five arms for each of four fixed latents, plus four original FFHQ baselines for context. Click any image for its unchanged 1024 × 1024 PNG. No images are copied, edited, filtered or sorted by quality.

All five arms start from raw G at iteration 100 of the `source_flattened` probing control with all 17 learned additive noise-strength offsets zeroed. The common control preserves learned bias offsets and reproduces the previous noise-reset control exactly; it is not unchanged raw G. Each remaining arm independently zeros one additional learned bias family: 9 ToRGB, 17 synthesis convolution, 26 style-affine or 8 mapping offsets. These interventions are separate, not cumulative. Zero offset restores the pretrained coefficient or bias, not a zero coefficient or bias. Fixed noise maps remain unchanged.

The builder verifies completed result and supervisor records; protocol and archived script hashes; exact 5 × 4 image coverage; matching arm/global image records; duplicate-free, disjoint and complete noise/bias partitions with expected names and 17/9/17/26/8 family counts; declared versus actual reset names for each arm; unchanged other state and restored original state; matching latent archives and per-latent hashes; byte-exact control reproduction against the prior completed noise ablation; parent checkpoint/snapshot bindings; baseline hashes against iteration 0; and every displayed PNG hash, format and dimension. The manifest records source metadata and image hashes. Weight-file integrity is inherited from the completed checkpoint and run load records; this builder does not reread model weights. It ignores incomplete files and replaces generated artifacts atomically after validation.

This is an inference diagnostic, not a quality-approved priest-generation candidate. No new training, Fisher estimation, main adaptation, image repair or server benchmark is performed. Original FFHQ latents can depict children, women and non-clergy. Four fixed latents do not establish generalization or unseen-latent diversity. Independent visual reports remain separate from the builder's required inputs.

Validation performed: successful rebuild of all 24 hash-checked native PNGs, then HTTP 200 for the page and all 24 image URLs using a temporary loopback server; served bytes matched every manifest hash. The temporary server was stopped. Port 8787 was offline at verification time; no browser was launched.
