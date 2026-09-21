# Prepared paper-layout compatibility experiment

Status: prepared, **not launched**. Wait for the existing source-layout 500-iteration control to become terminal before running this separate heavy worker. Do not restart or change that live job merely because an observation times out.

After checking available memory and absence of another heavy job:

```sh
research/.venv/bin/python research/experiments/adam_native/smoke_output_rank1.py --output research/runs/adam-native1024-output-rank1-smoke10-v1
```

This starts from original native1024 G_ema and D. It does not resume the source-layout probe or transfer its factors. The one training change is convolution factor layout: `outer(v_out,u_flat)` makes each output factor control one weight row. See `paper-layout-decision.md` for the paper/source mismatch and why old importance scores cannot be relabeled.

The runner derives from the successful frozen ten-iteration smoke. Worker AST comparison is exact after removing the two explicit layout arguments. Data, seeds, style mixing, augmentation, loss equations, lazy regularizers, optimizer settings, EMA behavior, checkpoint checks and folded-export checks remain unchanged. New source filename/protocol labels and explicit parent-source hashes preserve provenance. Both `smoke.py` and `modulation.py` remain untouched.

Use the normal supervised entry point, never `--worker` directly. CPU one thread, FP32, batch1, native1024; 35% system-available memory preflight, 20% runtime floor, 12GiB process-group RSS, 512MiB swap-growth and 1200-second limits remain enforced. No automatic restart or reduced-resolution fallback. Preserve any failure and do not stage incomplete image/checkpoint writes as completed artifacts.

The test must actually establish initial native G/D equality, intended gradients, unchanged frozen weights, all 24 scheduled optimizer updates and exact folded native outputs. Syntax/AST/source review is only preparation, not execution evidence. Even a successful smoke establishes no priest quality, diversity, stable importance ranking, or serving latency. A subsequent output-rank1 probing run and its own importance estimates still precede selective adaptation.
