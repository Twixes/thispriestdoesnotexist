# Noise-offset ablation review

Rebuild from the repository root:

```sh
research/.venv/bin/python research/reviews/adam-noise-ablation/build.py
```

Open `http://127.0.0.1:8787/reviews/adam-noise-ablation/` when the existing research preview server is running. This is a static, non-blocking review of the completed `adam-native1024-noise-ablation100-v1` diagnostic. It displays all four fixed latents in all three arms, plus their original FFHQ baselines. Every image links to its unchanged native 1024 × 1024 PNG; no images are copied, edited, filtered, sorted by quality or replaced.

The source is raw G at checkpoint 100 of the `source_flattened` probing control. The unchanged arm reproduces the archived raw PNGs exactly. The other arms zero either the learned `b1024.conv0` noise-strength offset or all 17 learned noise-strength offsets. Zero offset retains the pretrained noise-strength coefficient and fixed noise map; it does not disable noise. Run completion records assert that all other model state stayed exact and original state was restored.

The builder verifies completed result and supervisor markers; protocol and archived source-script hashes; matching per-arm and global records; exact 3 × 4 image coverage; the declared versus actual 0/1/17 changed-state names; matching per-latent hashes across arms; matching latent archives; parent checkpoint and snapshot bindings; byte-identical unchanged raw images against the parent checkpoint-100 PNGs; baseline hashes against the complete iteration-0 snapshot; and all displayed PNG hashes and native dimensions. It records each input metadata/image hash in `manifest.json`. Checkpoint-weight integrity is inherited from the completed checkpoint and run load records: the review builder does not load or rehash model weights. Final outputs are atomically replaced only after validation succeeds; incomplete `.partial` files are ignored.

This ablation performs inference interventions only: no new training, Fisher estimation, main adaptation, image repair or server benchmark. It is a diagnostic, not an accepted priest-generation candidate. The original FFHQ population can produce children, women, hats and non-clergy clothing. Four fixed latents do not establish generalization. Independent visual reviews can be archived alongside this page without becoming required builder inputs.

Validation performed: successful rebuild of all 16 hash-checked native PNGs, followed by HTTP 200 for the page and all 16 image URLs, with served bytes matching their manifest hashes. The parent performs browser layout review separately.
