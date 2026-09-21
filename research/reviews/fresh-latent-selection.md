# Fresh adult-male latent selection

2026-09-21. **Recommended design: fresh random latents, a small checkpoint-specific latent preselector, then independent checks on the actual generated image.** Start by measuring image-only rejection as a correctness baseline; add the latent preselector to reduce expensive 1024px renders. This is design preparation. No selection model exists yet, no compute or downloads were started, and no adult-only/hot-only guarantee has been demonstrated.

## What already exists

| Asset | Usable role | Limitation |
| --- | --- | --- |
| `inference/server.py` | Fresh 256-bit entropy → PCG64 Gaussian z → actual generator forward; serialized inference and bounded admission | Currently accepts every generated face |
| FFHQ1024 safetensors bundle | Exact base for sampling and paired clothing adaptation | Unconditional (`c_dim=0`); includes unwanted age/appearance/headwear categories |
| PyTorch 2.14, NumPy, Pillow | Fit/run small linear or MLP classifiers; preprocessing | No trained attribute heads |
| Existing SqueezeNet1.1 weights in `research/models/metrics/checkpoints/` and research torchvision 0.29 | Lightweight image features for separately trained heads | ImageNet weights are not an age, hat, collar or attractiveness classifier; torchvision is not in the serving image |
| YuNet ONNX and OpenCV 5 in `research/alignment/` | Face count/landmarks/crop validity | Detection only; no adult, male-appearance, hat or collar decision; serving would need the explicit dependency/model |
| Static hot-priest pool and reviewed synthetic training portraits | Preference examples and labeling references | Not a labeled latent dataset; mostly positive/curated examples and a different image domain |
| `research/evaluate.py`, LPIPS, discriminator diagnostics | Offline novelty/collapse comparison | Neither LPIPS nor D logits establish age, attire, attractiveness or photorealism |

The existing SqueezeNet architecture exposes a feature extractor suitable for fitting new heads; its original classification head predicts ImageNet classes. [Torchvision implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/squeezenet.html)

## Minimal training and runtime design

1. **Define separate labels.** For randomly sampled outputs of the exact final student G, record: clearly adult appearance / uncertain-or-young; male-presenting appearance / other-or-uncertain; bare head / hat-or-uncertain; visible correct priest collar / absent-or-malformed; visual integrity / artifacts. These describe fictional appearance, not a real person's identity or verified age. Label the owner’s “would include in the hot-priest calendar” preference **only after** the adult and other mandatory checks pass. A preference score must never override an adult rejection.
2. **Use a conservative adult operating point.** Initially accept only clearly mature-looking adults, roughly the apparent 30+ range, and reject ambiguous youthful faces even when they might be adults. Do not treat an estimated age ≥18 or an arbitrary softmax 0.99 as proof. Select thresholds from independently reviewed calibration data, with uncertainty/disagreement producing rejection. Hats/collars require head-and-shoulders/full-frame input; a face-only crop cannot check them.
3. **Make a new labeled latent corpus.** Begin with a bounded pilot of roughly 1,000–2,000 independently sampled z values, preserving z/W, uncompressed output hash, generator/mapping hash, psi, noise mode and preprocessing. This is a proposed starting size, not a demonstrated sufficient sample count. Include rejected and ambiguous examples; enrich difficult age/headwear/collar cases but retain a genuinely random evaluation stream. Split by seed identity before augmentation, and keep untouched final-test seeds after threshold selection. The 40–50 clothing-edit pairs alone are insufficient for this classifier. Adult decisions need careful independent review; owner-specific pairwise preferences among eligible adults can train a separate taste ranker.
4. **Fit simple models first.** Train a linear/very small MLP on mapped W for cheap mandatory-attribute and preference screening. Train separate image heads on existing SqueezeNet features using face crops plus a full-frame view. These are experimental baselines; if adult or hat false accepts remain unacceptable, they are not deployable just because loss decreases. The latent model predicts appearance for **this generator and sampler**, not universal human attributes. Recalibrate on the final clothing-adapted outputs, especially collars and hats. Frozen mapping makes W stable but does not make output labels stable when synthesis changes.
5. **At each request, draw new z values continuously.** Map and screen a bounded number, render a surviving candidate with the student, then run the mandatory image checks and the preference threshold on that actual output. Return the complete generated image only after acceptance. Use the same z/W for screening and rendering; do not call the mapping twice with different truncation. Cap full renders (initial candidate: three), retain a wall-clock deadline, and return uncached 503 when none passes. Never relax the adult threshold after a timeout or fall back to a saved image/seed. Preserve the accepted seed and model/selector version in diagnostic metadata.

This is rejection sampling from a continuous learned distribution. The finite labeled training set is not a serving catalog: runtime seeds are newly drawn, with no lookup, interpolation between approved seeds, pixel compositing, or nearest-neighbor replacement. Sampling near a handful of approved seeds would risk near-duplicate identities and should not be the shortcut.

InterFaceGAN supports the narrower premise that attribute boundaries can be learned from paired latent codes and generated-image attribute scores. Its supplied boundaries are tied to named models. Do **not** import a “male” or “age” direction from its original StyleGAN, another FFHQ checkpoint, or FFHQ256 into our adapted StyleGAN2-ADA1024 and assume it works. We need an independently evaluated selector for the exact source/student, not latent editing as an unverified substitute. [Official InterFaceGAN code and procedure](https://github.com/genforce/interfacegan)

An optional stronger external adult-appearance model is FairFace, whose official predictor includes broad age bands and binary appearance labels. Its 10–19 band crosses adulthood and 20–29 remains too close to the proposed conservative margin; inspect calibrated mass on mature bands rather than accepting an argmax age label. Its checkpoint/alignment dependency is **not present**, and it is not a hotness, hat, or collar classifier. If later adopted, review provenance/terms, validate on our synthetic grayscale outputs, and ignore unrelated attribute outputs. It is an optional separately evaluated veto, not proof of age. [Official model source](https://github.com/joojs/fairface), [age-band implementation](https://github.com/dchen236/FairFace/blob/master/predict.py)

## Cost and evidence required

For image-only rejection with independent candidate acceptance p, expected full generator work per accepted image is 1/p. With three attempts, request success probability is `1-(1-p)^3`:

| Acceptance p | Renders per accepted image | Success within 3 attempts | Warm render work at the measured ~4.5 s emulated rate |
| ---: | ---: | ---: | ---: |
| 0.8 | 1.25 | 99.2% | ~5.6 s |
| 0.5 | 2 | 87.5% | ~9 s |
| 0.2 | 5 | 48.8% | ~22.5 s |

The work column is amortized over successes **including failed requests**, not the latency of every successful request. Add classifier work and cold starts. The latent preselector is intended to increase the conditional acceptance rate of expensive renders; its success is unmeasured. Keep attempts serial to avoid multiplying activation memory. Three queued requests with three renders each could exceed the current 45-second Worker deadline once startup/classification is included; admission and deadline behavior must be rebenchmarked rather than retaining current limits blindly.

The existing local 1024 HTTP test fit 3 GiB with a 1.62 GiB cgroup peak; it had no selection models. Native-host speed, classifier memory, acceptance rate, traffic and idle policy determine whether the existing low-traffic <$15/month hosting scenario remains plausible. A low acceptance rate can erase that budget advantage. Existing [Containers cost research](../../infra/cloudflare-containers.md) is usage-based, not a bill cap.

Before serving, audit fresh **accepted outputs**, not classifier accuracy averaged over all candidates: false accepts for adult appearance, hats, collars, and malformed faces; preference agreement; diversity/near-duplicates; rejection counts; p50/p95 latency and memory. Keep adult false accepts separate from subjective taste misses. As a rough independent-binomial guide, zero observed failures in 300 accepted outputs only bounds the unknown failure rate near 1% at 95% confidence (about 3/N); tiny clean grids do not establish a rare-error guarantee. Correlated outputs and subjective labels weaken even that interpretation. Do not ship an automated “only adults” claim if the evidence cannot support the required conservative behavior.

Further work is concrete: label a representative generated corpus, train/calibrate both selector stages and the separate preference head, version them with the final generator, and run fresh acceptance/latency/memory review. Paired clothing adaptation by itself solves none of those selection gates.
