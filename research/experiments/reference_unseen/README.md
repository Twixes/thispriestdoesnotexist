# Reference500 unseen sample evaluation

Generate exactly32 predeclared PCG64 latent vectors and retain the same32 outputs for rawG psi1, EMA psi1 and EMA psi0.7:96 native256 RGB PNGs plus three native-tile contact sheets. Every output is retained; no visual filtering, replacement, compositing or restoration. Only the trainer's clamp/uint8 PNG encoding is applied. `latents.npz` stores integer seeds, float32z and saved fixed16z; exact equality with any fixed preview latent causes failure. This does not prove disjointness from all latent draws seen during training.

The evaluator pins the preserved `resume-000500.pt` checksum, config, archived trainer/adapters and NVIDIA Python/native operator sources. It reads the actual `batch_idx`, `G`, `G_ema`, `fixed_z`, `config` and sampler schema. Architecture comes only from `ast.literal_eval` of the pinned Generator constructor, instantiated explicitly as `training.networks.Generator`. Strict state loading and per-variant equality checks protect the model state. Only one Generator object exists; raw and EMA states load sequentially. No discriminator or optimizer is constructed.

The trusted locally produced resume requires `weights_only=False` for its NumPy/Python RNG records; it uses CPU mmap and never loads an external pickle. NVIDIA licensing remains unchanged. No unadapted baseline is added.

Invocation (requires a fresh output directory):

```sh
research/.venv/bin/python research/experiments/reference_unseen/evaluate.py \
  --execute --output research/runs/reference256-paper-b64/unseen32-000500
```

Output must be a new directory below `research/runs`. The parent starts a separate owned worker group, records an absolute600-second deadline, and always kills/reaps the group on exit. Both worker and parent enforce resource checks. At least35% free memory is required before model imports; CPU threads and interop threads are1; image batch is1. Worker peak RSS is checked every50ms and parent current RSS every~100ms, aborting above6GiB. This is a sampled abort threshold, not an OS hard allocation guarantee; transient overshoot remains possible. Partial failure evidence is retained without retry or overwrite.

A successful result requires both `evaluation.json` and `supervisor-result.json` to say complete. Hashes cover all96 individual images, latent NPZ, contact sheets and inputs; pins/script are checked again after generation. The report does not approve image quality, adulthood, masculine appearance, hats or subjective attractiveness. Review the full native outputs separately.

Preparation performed only stdlib syntax compilation, metadata reads and file hashing. This preparation itself did not execute the model. The first execution is recorded below.

## First execution

Root reviewed the source and ran the evaluator on2026-09-21. The complete96-image run finished successfully in10.68s supervisor wall time, with992,444,416 bytes worker peak RSS. All input hashes remained unchanged. Results are in `research/runs/reference256-paper-b64/unseen32-000500`; this is numeric execution evidence, not visual quality approval.
