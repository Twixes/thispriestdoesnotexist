# Seven-pair strict source preflight

Passed for `manifest-proposed-v2.json` (SHA256 `9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6`). All seven saved source PNGs reproduce exactly with maximum uint8 channel error **0**, and all seven saved W tensors reproduce with maximum absolute error **0**. No tolerances were relaxed.

The six training records are calibration-original, 000, 030, b2-055, b2-051 and b2-020; 028 remains validation. Their region annotations were accepted by the isolated region trainer, and no validation collar annotation was required.

The fresh memory guard reported **31% free**, above the required 25%. Verification used one CPU thread and one interop thread. Source verification took **5.82 seconds**; wrapper wall time including startup and final integrity hashing was **7.01 seconds**. Peak process RSS was **3.65 GiB**. The source model state was unchanged; all manifest, latent, source/target image, model and imported trainer hashes matched before/after.

`preflight.json` contains per-pair measurements, file hashes and imported-code provenance. `verification-integrity.json` records the exact command, guard and before/after hashes. The sibling `source-preflight-v2.log` preserves complete output. `memory-before.txt` and `memory-after.txt` retain host readings.

No optimizer, backward pass, training, production approval, input change, or deployment occurred. This validates source correspondence and region mechanics; it does not approve target edits, judge visual quality, or validate a trained model.
