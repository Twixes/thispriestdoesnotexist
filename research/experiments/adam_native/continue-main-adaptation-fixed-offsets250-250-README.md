# Conditional exact main100→250 continuation

Prepared only; no native load or launch. Requires actual successful100 completion and image review from `research/runs/adam-native1024-retained250-main10-to100-v1`, protocol SHA `a21187fa9476db357b144067e084847f394d82730827d82f94f23c4251d915b9`. The future checkpoint100 digest is read from its completed marker, never guessed.

```sh
research/.venv/bin/python research/experiments/adam_native/continue_main_adaptation_fixed_offsets250_250.py \
  --run research/runs/adam-native1024-retained250-main10-to100-v1 \
  --output research/runs/adam-native1024-retained250-main100-to250-v1
```

The gate verifies complete100 result/supervisor, checkpoint50/100 bytes and marker/snapshot/count links, all source pins, all10/50/100 images, restored native10 ancestry and FI/retained250 lineage. The worker rebuilds original protected references, restores the full G/D/GEMA/Adam/RNG/path/mask/controller state from100, and requires all eight parent100 PNGs to reproduce exactly before updating. The engine and restore helper are unchanged from the frozen10→100 implementation.

Absolute iterations100–249 add150 updates; the first includes G path regularization and no R1. Checkpoints/previews150 and250 preserve all four raw and four EMA latents. Cumulative optimizer counts are188G/160D at150 and313G/266D at250, adding188G/159D beyond100. Atomic checkpoints retain full state and verified perturb/restore behavior; raw/EMA folding at250 must remain exact. No reset or objective/thread change enters this continuation.

Guards remain CPU1/native1024/FP32,35% start/20% runtime available memory,12GiB RSS,512MiB swap growth and3600seconds. There is no automatic launch/restart or concurrent native worker. Root reviews100 before scheduling and can inspect150 non-blockingly while the owned worker proceeds. Lack of conversion at100 alone is not a quality failure; photographic regression or loss of diversity still warrants reassessment.

Three bounded tiny-native32 tests compare every recorded update and complete final state of uninterrupted100→250 against serialization/reconstruction at100 and150; exercise correct global R1/path boundaries; verify exact frozen-engine/helper identity; and reject changed envelope/counter information before mutation. They provide no native completion, quality or latency claim.
