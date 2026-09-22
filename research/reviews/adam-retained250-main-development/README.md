# Actual main-adaptation development preview

Rebuild from the repository root:

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-development/build.py
```

Local URL: `http://127.0.0.1:8787/reviews/adam-retained250-main-development/`.

This page displays actual finalized main-adaptation outputs: initial run `adam-native1024-retained250-fixed-offsets-main-adaptation10-q50-v1` at steps0/10, and separate continuation `adam-native1024-retained250-main10-to100-v1` at steps50/100 when available. Each view includes all four fixed latents, their original step0 baseline, raw G and EMA. Entire native1024 PNGs open directly. All snapshots remain in the run archive, with no image selection, crops or repairs.

The initial main10 run starts from original source weights; retained250 probing supplies Fisher importance masks, not the rejected raw probing images as a starting model. Main adaptation can train permitted original rows. The q50 threshold is a research choice, not a proven quality setting. The source population still includes apparent children, hats and ordinary non-clergy. No attractiveness assessment of apparent minors, production approval, population-quality claim or latency proof is made.

The builder verifies the exact source protocol names, q50, fixed34-offset policy, retained250/FI1000 metadata, native resolution, latent hash, archived runner/helper hashes and all completed snapshot PNG hashes and dimensions. It verifies raw/EMA equality at step0 and exact duplicate child10 reproduction when the continuation begins. No checkpoint tensors or models are loaded.

Steps50/100 display only when their completed snapshot **and full checkpoint marker** exist; the marker's snapshot/protocol bindings and exact state-restoration assertion are verified. Snapshot10 can be displayed before its checkpoint finishes, but terminal success requires its checkpoint marker and successful result/supervisor records. Missing terminal records are reported as in progress or awaiting a terminal record, not interpreted as live process health. Failed/stopped runs retain all completed snapshots. Incomplete or `.partial` image files do not enter the page. Rebuilds atomically replace the static index/manifest; the page does not poll or control workers.

Initial validation: actual steps0/10,16 distinct archived PNG records. Generated JavaScript passes Node syntax checking. Native10 visual review is stored in `native-review-010-agent.json`, with exact viewed hashes and limits.
