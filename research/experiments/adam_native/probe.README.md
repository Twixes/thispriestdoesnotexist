# Prepared native1024 probing run

`probe.py` is prepared, not launched. It extends the successful ten-iteration `smoke.py` without modifying that source. It performs exactly500 fresh-source probing iterations, with no Fisher estimation, importance selection, main adaptation, resume option, or production approval. Checkpoint existence and finite losses are not visual-quality evidence.

The source audit is `source-audit.md`. Frozen derivation inputs:

- `smoke.py`: SHA256 `a0f58197d99dea71e19f6b1ce0ade2491173bba1576c8e4ba44ed844faf48e63`.
- `modulation.py`: SHA256 `b032f7965f397a5ebdc93fb1081adaeace651d31a614347123ab83fc46ce456a`.
- Original G_ema export, D safetensors and twenty training images keep the smoke's existing weight/dataset hashes. Neither smoke checkpoint nor held-out data is loaded. Every vendor Python file, metadata file, runner and modulation source is pinned again at launch and checked by the worker.

The parent agent owns launch and the resulting process handle. From the repository root, an example command using a fresh output directory is:

```sh
research/.venv/bin/python research/experiments/adam_native/probe.py --output research/runs/adam-native1024-probing500-v1
```

No run has been started by preparing this file. Do not use `--worker` directly: the normal entry point prepares provenance and enforces supervision. The supervisor has a prospective **7200-second wall deadline**, starting immediately before the worker launch and including loading, previews, checkpoint verification and final export. Start requires35% available system memory; runtime requires20%, RSS stays below12GiB, and global swap growth stays below512MiB. A failed guard kills the worker and preserves failure evidence. There is no automatic restart, continuation, lower resolution, reduced objective, or extended deadline. Root should continue observing the original running handle across turns.

## Exact delta from the frozen smoke

The loop becomes500 iterations instead of10. CPU one-thread, FP32, batch1, native1024, training and review seeds, latent draws, style mixing0.9, real-image horizontal flipping, source-layout modulation, D epilogue training, unfused modulation convolution, and all loss/optimizer settings remain the same. R1 occurs at indices0,16,…,496; path regularization at0,4,…,496. Both G and D EMA update after each iteration. Mapping moving-average updates remain disabled. No sampling or objective setting is tuned during the run.

Actual optimizer calls are counted in the live update wrapper and recorded after every iteration. Expected cumulative counts are checked at each checkpoint:

| Completed iterations | G optimizer steps | D optimizer steps |
|---:|---:|---:|
|100|125|107|
|250|313|266|
|500|625|532|

The first four fixed latent vectors are retained in `eval-z.npz`. `step-000`, `step-100`, `step-250` and `step-500` each contain all four raw and four G-EMA native PNGs. No image is selected or discarded. Generation uses constant synthesis noise for previews only; training uses random synthesis noise. Preview RNG equality is asserted so snapshots do not alter the optimization sequence. Initial baseline images and exact pre-training G/D equivalence checks are retained from the smoke.

Each PNG is written to `.partial` and atomically renamed. A snapshot folder receives `manifest.json` with `complete:true` only after all eight final images exist and their SHA256 values have been recorded. An absent complete manifest means the snapshot is unfinished.

`checkpoint-100.pt`, `checkpoint-250.pt` and `checkpoint-500.pt` each contain raw G/D, G/D EMA, complete G/D optimizer states, global Torch RNG, the separate sampling RNG, path-length moving target, cumulative optimizer counts, iteration, full protocol and protocol/runtime/inventory hashes. Files are written to `.partial`, then atomically renamed. Each checkpoint is loaded with `weights_only=True`, its complete nested state digest is checked, a G factor and both RNG streams are temporarily perturbed, and the saved model/optimizer/RNG/path state is restored and checked exactly. This adds no optimizer steps or alternative training trajectory. The returned path target is used for the next iteration.

Only after those checks does `checkpoint-NNN.json` mark that checkpoint complete and bind its SHA256, snapshot-manifest SHA256, state digest, optimizer counts and protocol hash. A `.pt` without this completed record is not a verified checkpoint. The runner writes a run-local `.gitignore` for `*.partial`; interrupted writes and markerless snapshots/checkpoints must not be staged as completed outputs. The restoration check runs inside the same process; it does not implement or test a process-restart API.

The final smoke behavior is retained: fold raw G modulation into ordinary NVIDIA weights on a copy, compare all four fixed native outputs exactly, and save `folded-generator.safetensors` through an atomic temporary file. This is a research artifact, not a serving release. The supervisor requires the final result,500 iterations, expected optimizer counts and all three completed checkpoint hashes before reporting success.

## Preparation validation

`probe-preparation-validation.json` records syntax/CLI checks and tiny tensor checkpoint tests. The tests used four `Linear(2,2)` fixtures with manually initialized Adam state, exercising the actual AST-extracted checkpoint helper at all three cumulative counts. Serialization, restoration, both RNG streams, path state and atomic completion records passed. They performed no model forward, backward, optimizer step, image generation, pretrained-model loading or native training.

These checks do not establish that500 iterations will finish inside the guards or improve images. The already completed native smoke establishes compatibility for its ten iterations only. Runtime outcomes and photographic quality remain to be measured and reviewed after the parent launches this separate experiment.
