# Checkpoint100 raw-G noise-offset ablation

Prepared, not executed. This diagnostic tests whether particular learned noise-strength offsets contribute to the grain seen in the source-layout control. Its results need native visual review; preparation and passing helper checks do not establish causality or a better model.

Run only after `adam-native1024-probing500-v1/supervisor.json` records the control's terminal state and fresh memory preflight passes:

```sh
research/.venv/bin/python research/experiments/adam_native/noise_ablation.py --output research/runs/adam-native1024-noise-ablation100-v1
```

The source is pinned to completed checkpoint100 SHA `d406240601ec748f95ad09c2b2799106d04f0510cd9383526bdc0618bd7416ba`, its completion marker, parent protocol and snapshot manifest. It loads checkpoint `G` using `weights_only=True`, `map_location='cpu'`, `mmap=True`. The checkpoint container includes other states, but only raw G is copied into a constructed model; no D or EMA model is instantiated. The original `source_flattened` factor layout, matching native vendor code and modulation SHA `b032f7965f397a5ebdc93fb1081adaeace651d31a614347123ab83fc46ce456a` are required. This is not the separately proposed output-rank1 experiment.

Three arms retain all four original `eval-z.npz` latents:

1. Unchanged raw G. Each generated PNG must match the corresponding archived `step-100/raw-000..003.png` both in decoded pixels and complete file SHA. A mismatch aborts before interpreting reset results.
2. Zero only `synthesis.b1024.conv0.parametrizations.noise_strength.0.b_vector`.
3. Zero all17 discovered synthesis `noise_strength` additive offsets.

Each arm starts from the original17 offset values. The original learned noise strengths, fixed noise buffers, multiplicative u/v factors and all other additive offsets remain untouched. Per-tensor full-state hashes prove that changed state is confined to declared resets; state hashes and RNG are checked again after rendering. All offsets are restored and verified at the end. No image filtering, editing, latent selection, training or model export occurs.

Outputs are12 unfiltered native1024 PNGs, per-arm complete manifests, exact input/source hashes, original and reset offset values, state support assertions, per-image generation timings, runtime versions and a final result. Timings are local CPU diagnostics, not a server latency claim. PNGs and JSON use temporary files followed by rename; incomplete writes are ignored locally by Git.

The supervisor creates its own process group and enforces CPU1,4GiB process-tree RSS,35% available memory at launch,25% while running,512MiB maximum swap growth and600seconds wall time after input validation. It kills only its owned worker group and preserves failure logs/artifacts. It never relaxes a guard or restarts automatically.

Static preparation checked syntax and CLI help. Five tiny scalar helper cases passed: unchanged restore, one reset, all reset, restoration, and nonmutating rejection of an undeclared reset. No native model was loaded or rendered during preparation. Full-state support assertions, exact baseline reproduction, actual runtime/memory and visual interpretation remain execution gates.
