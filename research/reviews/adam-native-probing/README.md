# Native importance-probing preview

From the repository root, rebuild with:

```sh
research/.venv/bin/python research/reviews/adam-native-probing/build.py
```

The static index defaults to the latest complete snapshot. Iteration links also expose pending states. The four latent rows are never filtered, sorted by quality, repaired, or replaced. Original native PNGs remain in the source run and open on click.

The builder reads frozen protocol and finalized JSON markers only. It ignores logs, metrics, `.partial` files and checkpoint weights. Iterations 100/250/500 require both an atomic complete snapshot manifest and the completed checkpoint record binding that manifest. All eight image hashes, dimensions, protocol hash and latent-file hash must match. Original baselines are included only after the complete iteration-0 marker, and must match its raw and EMA PNG hashes exactly. A mismatch fails the rebuild, preserving the previous index. Browser refresh after rebuilding loads the new static state; there is no training-status polling.

This is importance probing, not a priest-generation candidate or quality acceptance. No Fisher estimation or main adaptation has occurred. These unfiltered FFHQ latents can produce children, women, hats and non-clergy clothing. Four fixed latents do not establish generalization or unseen-latent diversity, and this preview makes no server-latency claim.

Validation performed: current complete iteration-0 build (12 PNGs), JavaScript syntax, pending `.partial`/image exclusion, synthetic completed-checkpoint marker acceptance, and corrupted image-hash rejection. The temporary synthetic fixture was removed and is not included as research output. No model or browser was launched.
