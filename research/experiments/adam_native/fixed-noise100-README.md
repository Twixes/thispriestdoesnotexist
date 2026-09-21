# Prepared output-rank1 fixed-noise100 stability test

Prepared only; no native training or rendering has been launched. Root controls scheduling after the bias-family inference diagnostic and a fresh resource preflight. This is a100-iteration stability test, not a quality candidate or completed AdAM adaptation.

```sh
research/.venv/bin/python research/experiments/adam_native/probe_output_rank1_fixed_noise100.py --output research/runs/adam-native1024-output-rank1-fixed-noise100-v1
```

This new file derives from frozen `probe_output_rank1.py` SHA `e72acd143ce7d269549e8ed91b75e897a7ca9130b517ff630070aedd6e5cb73e`; the source-layout probe, prepared output-rank1 probe, smoke runners and `modulation.py` are unchanged. The new runner always starts from the verified original FFHQ1024 G_ema export and original D, fresh factors and fresh optimizers. It never loads a probing checkpoint to initialize training and has no resume option.

Changes and boundaries:

- Preserve output-row `outer(v,u)` convolution modulation. This mathematical variant differs from AdAM's released `outer(u,v).reshape(...)` convolution implementation.
- Preserve all17 original pretrained synthesis noise strengths, frozen exactly. All17 additive noise-strength offsets start at zero, remain gradient-disabled and are excluded from the G optimizer. The dedicated G toggle enables only the filtered parameter list, so it never transiently enables these offsets. All other probing adapters and D epilogue behavior remain unchanged. This is an explicit departure from AdAM.
- G EMA also retains the original strengths and zero offsets. Invariants check raw G and G EMA before and after every optimizer update, after EMA updates, around every snapshot and after checkpoint restoration. They reject an offset value/gradient, unexpected optimizer membership or changed original strength. Invariants never repair or project a failed value.
- Training still uses random spatial synthesis noise; only its learned strengths are frozen. Review uses the same constant synthesis-noise buffers as before. No input/output image editing occurs.
- Native1024 CPU1/batch1 adversarial, R1 and path losses, loss weights, regularization schedule, style mixing, data, LR/betas, seeds and EMA decay are preserved. Static tests compare the actual loss block and generation/update helper ASTs against the frozen parent.
- Stop after100 iterations, expected125 G and107 D optimizer steps. Retain original baseline4 plus all four raw/EMA images at steps0,10,25,50,100, with completion/hash manifests. The original four review latents are hash-pinned and tensor-compared to the previous control fixture; no child/woman/hat seed filtering or replacement is introduced.
- Checkpoint100 retains G/D/raw/EMA states, optimizers, both RNG streams, path mean and cumulative counts. Atomic serialization and same-process exact reload checks remain intact; final raw-G folding is checked against all four native outputs. There is no Fisher estimator or main-adaptation selection.
- Guard limits remain35% initial/20% runtime available system memory,12GiB process-tree RSS and512MiB swap growth, with a2400-second supervisor wall limit. The worker uses an owned process group; partial checkpoints are ignored and no automatic restart or fallback occurs.

The causal ablation showed the large final-resolution noise offset contributed to checkpoint100 grain, while resetting all noise offsets left other color/contrast problems. This new run changes both convolution layout and noise-offset policy relative to the completed source-layout control; it cannot by itself isolate which difference causes an observed change. All four outputs remain subject to native review, beginning at step10, with no claim of demographic eligibility, priest quality, convergence or server latency.

Preparation validation: three tiny32px tests passed in0.678seconds, CPU1. They exercised actual Adam and EMA updates, repeated gradient toggles, optimizer exclusion, preservation of deliberately nonzero original strengths, corruption detection, the100-step counts/review schedule and unchanged loss/generation code. No pretrained weights or native1024 models were loaded by these tests.
