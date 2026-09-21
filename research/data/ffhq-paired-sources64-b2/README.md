# Second unfiltered FFHQ source batch

All 64 exact 1024×1024 RGB PNGs and matching float32 NPZ z/W latents are retained. These are unadapted FFHQ research sources, not priest output or a production catalog. No previous batch was overwritten.

`generate.py` uses the existing exact CPU serving bundle, one PyTorch intra/inter-op thread, psi=1, constant noise, and force-FP32. New seed prefix: `priest-paired-source-v2-batch2:`. `manifest.json` records seeds, source hashes, latent hashes, mapping output, noise hashes, runtime and licensing. NVIDIA source code and weights retain their upstream research/evaluation restrictions; the application MIT license does not replace those terms.

Generation passed a fresh 33% host-memory preflight and completed with exit 0: 48.9001 seconds for the 64 forwards and saves, 50.3710 seconds including model load and contact sheet; peak RSS 3,735,666,688 bytes (3.48 GiB). `run.log` and `host-memory-before.txt` preserve evidence. No MPS, paid service, training, or additional source forward passes were used.

The PNG/NPZ-only `review.py` verified all 64 PNG/NPZ checksums, decoded RGB hashes, finite float32 z[1,512]/w[1,18,512], and unique seeds/RGB hashes. It does not import or load a model.

All 64 portraits were inspected on `contact.png`; 15 candidate/exclusion images were inspected natively. `selection.json` records subjective decisions and hard exclusions. Only two clean priority candidates were found: **055** and **051**. **020** is a conditional candidate with harsh sun and a blurred crowd. **025** is a conditional candidate with too little neck and a partial second person. The batch does not supply 8–12 strong suitable images, and the quality criterion was not relaxed to reach that count. The labeled `shortlist-contact.png` is only a diagnostic; all native sources remain untouched.

Root subsequently reviewed native 055/051/020/025 and authorized clothing-only edits for **055, 051, 020**. Root rejected 025 for composition. This authorization is for prospective research training data, not production approval or certification that all generated outputs are hot. New edits belong in a separate directory and do not change this immutable source manifest or any active training data.
