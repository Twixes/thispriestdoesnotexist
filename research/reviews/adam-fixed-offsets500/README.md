# Fixed-offset 100-to-500 continuation preview

Rebuild from the repository root, then refresh:

```sh
research/.venv/bin/python research/reviews/adam-fixed-offsets500/build.py
```

Review URL: `http://127.0.0.1:8787/reviews/adam-fixed-offsets500/`.

This static review shows all four fixed latents, with the original FFHQ baseline and raw/EMA snapshots at parent iteration 100 and continuation iterations 250/500. The latest finalized snapshot is selected by default; future snapshots are explicitly pending. Every image opens its untouched native 1024 × 1024 PNG. No selection, filtering, image repair or model execution occurs.

The continuation restores all four raw/EMA G/D states, both Adam optimizers, random streams and path mean from the completed parent. All 17 learned noise offsets and 17 synthesis activation-bias offsets remain fixed at zero, with original source coefficients retained. This is a prototype stability experiment and an explicit modification of AdAM, not full AdAM, Fisher estimation, main adaptation or a priest-generation quality candidate. The present source subjects include children and head coverings; these are research diagnostics, not acceptable outputs for the requested adult priest model. No production approval, population-level quality evidence or server-latency claim is implied.

The builder imports the existing parent review's complete-run and native-image validator. It rechecks the completed parent artifacts and then binds parent hashes to the continuation protocol. It validates the archived runner/modulation, latent hash, identical fixed-offset policy, restoration marker, all eight child100 PNGs against both the parent images and protocol reproduction hashes, and each completed continuation snapshot's protocol/RNG/offset/hash/dimension assertions. Parent100 images are displayed from the parent archive. Child250/500 images become visible only after their atomic complete manifests exist. Their full checkpoint completion markers are separately checked when present and required for terminal success.

Only finalized records are read. Logs, metrics, `.partial` files and checkpoint weights are never loaded. Status describes saved artifacts at build time, not process liveness. Missing terminal records are explicitly ambiguous; only successful worker plus supervisor records indicate completion. The page does not poll or control training. A rebuild atomically replaces its index and manifest after validation.

Initial validation has only iteration100 complete: 12 displayed native PNGs, plus all eight resumed reproduction images checked against their parent. The generated JavaScript passes Node syntax checking. The page and all 12 image URLs serve HTTP200 with bytes matching their recorded SHA-256 hashes. No browser or model was launched.
