# Fixed-offset stability preview

Rebuild from the repository root, then refresh the page:

```sh
research/.venv/bin/python research/reviews/adam-fixed-offsets100/build.py
```

Review URL: `http://127.0.0.1:8787/reviews/adam-fixed-offsets100/`. The latest finalized snapshot is selected by default. Iteration links expose 0, 10, 25, 50 and 100, including explicit pending states. Every view shows all four fixed latents, each with its original FFHQ baseline, raw G and EMA. Every image opens its untouched native 1024 × 1024 PNG. No image selection, filtering, repair or model execution occurs in this builder.

This run starts from fresh source G and D and uses `output_rank1` modulation. It fixes all 17 learned noise-strength offsets and 17 learned synthesis activation-bias offsets to zero in both raw G and EMA, excluding them from the optimizer. The pretrained coefficients and biases remain unchanged. This is a 100-iteration stability test and a deliberate departure from AdAM, not full AdAM probing, Fisher estimation, main adaptation, a quality candidate, or evidence of server latency. Four fixed latents do not establish generalization; original FFHQ subjects can include children, women and non-clergy.

The builder reads exact finalized manifest paths only: `baseline-manifest.json`, `step-000/manifest.json`, `step-010/manifest.json`, `step-025/manifest.json`, `step-050/manifest.json`, and `step-100/manifest.json`. It verifies complete flags, protocol and latent-file hashes, raw/EMA fixed-offset and unchanged-RNG assertions, exact per-snapshot coverage, native PNG hashes/dimensions, baseline/step-0 equality, and archived runner/modulation source hashes. It ignores `.partial` files, logs, metrics and checkpoint weights. Snapshot 100 can be displayed before its full checkpoint finishes; its separately completed checkpoint marker is checked if present and required before accepting a complete worker result. Iterations 10/25/50 do not have full checkpoints.

Status is recorded at build time. Without a terminal record it says “in progress or awaiting a terminal record,” which is not a live process-health check. A worker result without the supervisor record remains pending final supervision. Supervisor failure or explicit termination is displayed as ended, retaining completed snapshots; success requires complete worker and supervisor records. The page does not poll, start, stop or restart training. Rebuilding updates the static index/manifest atomically after validation.

Initial validation: iteration 0 only, 12 hash-verified native images. Generated JavaScript passes Node syntax checking; the page and all 12 image URLs return HTTP 200 on localhost:8787 with served bytes matching the manifest. No browser or model was launched.
