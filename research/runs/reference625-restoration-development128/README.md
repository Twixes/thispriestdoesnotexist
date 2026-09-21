# Reference625 restoration development128

Completed CPU-only raw-G generation of every seed202609211000–202609211127, with no filtering, restoration or retry. This is development data, not untouched calibration/test data and not a production seed catalog.

Exact invocation from repository root:

```
research/.venv/bin/python research/experiments/restored_selection/sample.py --output research/runs/reference625-restoration-development128 --count 128 --seed-base 202609211000
```

Supervisor terminal exit0,14.434seconds total; peak worker RSS870,318,080bytes (0.811GiB), below6GiB/600second guards. CPU/interop threads1, batch1. Median generation/PNG time0.08417seconds. No MPS/GPU use. All128 output PNG hashes checked after completion and all128 are distinct. Four labeled contact sheets preserve the complete order.

The control PNG reproduced existing reference625 raw-psi1 index000 byte-for-byte before any new image generation, SHA256 `393c8dd7f05c5888d334841756a750456f67533f738c01bd27c8982cb55aebce`.

Checkpoint SHA256 `c04482a462abf4ece7034cdc09be19dc857ee201b11427f6526c49ae378bf6dc`; configuration SHA256 `b330c97f252c2b0f20d041923900676c806cfdac6b47241e062f50cabef09997`. Actual config and sampler source are copied here. Raw G, psi1, constant noise, FP32 and the original trainer clamp/uint8 PNG encoding are unchanged. Latents use the reference evaluator convention: separate PCG64(integer seed), standard_normal float64 then cast float32. This differs from the separate inference service's direct float32 RNG convention and is recorded explicitly.

The128 new z rows are disjoint from685 saved rows across116 pre-existing NPZ archives; numeric seeds are disjoint from32 saved numeric seed values. Training fixed_z is included. The selected numeric range was also absent from prior research JSON/Python/Markdown text in the pre-launch search. This does not prove absence from every unsaved historical stochastic training draw. All128 exact z rows and seeds are preserved in latents.npz; no approved-seed bank is produced.

Sampler source SHA256 `dc0f0a7234bbaa9586f0eb0a765073419a959e3a595348cd09cf2a46db16fb68`.

Evaluation manifest SHA256 `4a84f1522dacc79c1cee5847d4b4e6c5457c79bd2da1d9bf34537c7edefa6320`.

Latents archive SHA256 `1c166426c21d5f9e4d63ed43d82c2bb507c50f25dc3cb605ff70153a0d2a1abb`.

No quality, age, headwear, collar or attractiveness acceptance was performed. The manifest is ready for a separate authorized restoration pass; future calibration and test cohorts must remain disjoint and untouched.
