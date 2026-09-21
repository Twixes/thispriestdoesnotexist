# Prepared G-only bias-family ablation

Prepared and tested on tiny offset tensors; **native inference has not been launched**. Root coordinates execution before any new training job:

```sh
research/.venv/bin/python research/experiments/adam_native/bias_ablation.py \
  --output research/runs/adam-native1024-bias-ablation100-v1
```

The script derives from the completed noise-ablation worker, keeps the same source-layout raw G at checkpoint100, and renders all four fixed latents across five arms. Every arm resets all17 learned additive noise-strength offsets. The four additional arms independently reset only ToRGB bias offsets (9), synthesis activation-bias offsets (17), style-affine bias offsets (26, including ToRGB affines), or mapping bias offsets (8). Original source noise strengths, original biases, and all multiplicative weight factors remain unchanged.

All77 raw additive offsets are restored between arms, with complete state hashes checked before applying the next intervention. Changed-state support must equal exactly the declared offsets whose saved values were nonzero; every selected offset must become zero. Rendering must leave state and RNG unchanged. After all arms, complete original raw state must be restored exactly. The common noise-reset control must reproduce all four completed `adam-native1024-noise-ablation100-v1/reset_all_noise_offsets` PNGs byte-for-byte and pixel-for-pixel. All20 native outputs are retained without filtering or image editing.

Checkpoint, constructor, modulation and NVIDIA source pins are preserved. The prior completed noise-ablation protocol/result and control images are additionally bound. Only G is instantiated; no D or EMA model is constructed. Guards remain 4GiB RSS, 35% initial and 25% runtime available system memory, 512MiB swap growth, 600-second deadline, CPU1 thread/inter-op1. No optimizer or training occurs.

Validation: three tiny tensor tests passed for exact per-arm support, inter-arm restoration, original-parameter/weight-factor exclusions, family disjointness, and unknown-input rejection. All77 names from the existing native inventory partition into the expected five families. The supervisor AST matches the source apart from worker filename and output count. Native model execution remains pending.

Run the support tests with:

```sh
research/.venv/bin/python research/experiments/adam_native/test_bias_ablation.py
```

This isolates inference contributions of learned offsets at one checkpoint. It does not prove a stable training intervention or grant photographic-quality or production approval. Review all arms before making causal claims about visible defects.

## Launch preflight

The first native launch was refused before output-directory creation because available memory was below the unchanged35% guard. A bounded30-second readiness check remained below the threshold (about31%); no native worker started. Observations are preserved in `bias-ablation-launch-preflight.json`. The runner remains prepared, not executed. Do not lower guards, close unrelated applications, or treat elapsed waiting as permission to start. A future launch must pass a fresh preflight.

## Completed execution after authorized service cleanup

The user authorized stopping expendable high-memory Node development services. Four detached dev-server trees were stopped and verified absent; available memory was21.2GiB afterward. The unchanged native guard then passed. `research/runs/adam-native1024-bias-ablation100-v1` completed in14.281s with3.551GiB peak RSS and38.14% minimum available memory. All20 native PNGs were preserved; the four common controls reproduced prior noise-reset PNGs exactly. Reviews and interpretation are under `research/reviews/adam-bias-ablation/`. Previous refused launches above remain historical evidence; this diagnostic is now executed.
