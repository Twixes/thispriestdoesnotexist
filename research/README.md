# Priest generator research

Goal: train a real generative model and run fresh server-side inference on every page load. Sampling the static gallery, pixel morphing, or adding noise does not meet this goal. The production site remains on its curated static gallery until a model passes evaluation.

## Current experiment

Transfer learning from NVIDIA's FFHQ StyleGAN2 at 256px. The initial 50-image pilot was deliberately stopped after its durable step-1000 checkpoint: collars appeared, but fixed samples showed overlapping facial features and conspicuously similar faces/compositions. Its source, weights, logs, and rejected visual reviews remain in `runs/pilot256/`.

The now-paused restart, `aligned110-frozen4`, uses those 50 selected portraits plus 60 newly generated and visually reviewed fictional adult portraits. Original generations, prompts, reviews, and hashes are archived in `data/synthetic/`. The training inputs are `alignment/collar-only/eyes42/`: 110 separate images with a collar-preserving crop that reduces the measured framing mismatch with FFHQ. No hats were introduced. Full FFHQ cropping was rejected because it removed most collars; the earlier whole-head constraint was also relaxed because it unnecessarily limited face size. See `alignment/README.md` and `alignment/collar-only/README.md` for evidence and exact transforms.

The reference network code is reused without modifications. `train.py` provides a portable PyTorch training loop with non-saturating logistic losses, lazy R1, style mixing, EMA, and MIT Han Lab's differentiable color/translation/cutout augmentation. Augmentation probability follows the ADA discriminator-sign heuristic. This is a documented adaptation, not a claim to reproduce the entire NVIDIA training recipe: path-length regularization is currently disabled and augmentation differs from their full pipeline.

The paused control freezes the first four discriminator layers using NVIDIA's built-in FreezeD buffers, uses mapping LR .0005 and EMA half-life .5 kimg, and originally allowed up to 6000 batch-8 steps on the local Apple GPU. It was intentionally paused at its verified, durable step-1000 checkpoint to make room for the reference baseline; this is not a claim that it had reached final convergence. See its `pause.json` for the exact checkpoint hash and continuation command. Frozen weights, optimizer restoration, R1, and side-effect-free snapshots passed real CPU smoke checks. Every 250 steps it saves matched raw-G and EMA grids at truncation 1.0, plus the usual EMA grid at .7. Review these before deciding to continue; a long run is not itself evidence of acceptable quality. This combines several evidence-driven changes and is not an isolated causal ablation. See `reviews/few-shot-options.md`.

The 6000-step limit is only 48,000 image presentations and is a feasibility budget. NVIDIA reports around one million presentations as a common transfer-learning reference, roughly 3.5 days at the measured speed of the earlier portable loop at 256px. This estimate does not apply to the slower full reference loop with both regularizers and is not a quality guarantee. The original 1024px StyleGAN used 70,000 photographs and about a week on eight V100 GPUs; our pretrained weights reuse that kind of upstream investment. See [resource comparison](reviews/stylegan-resource-reference.md) for primary sources and the distinction between unique images, presentations, and optimizer steps.

A paired fresh-base run, `aligned110-frozen4-cdc1000`, now uses the same dataset, seed, FreezeD and adversarial settings, with an additional source-correspondence loss at weight 1000. Its independent latent stream keeps the adversarial RNG unaffected. Real MPS G/D/R1 updates and a full checkpoint/optimizer/RNG round trip passed. See [CDC experiment](experiments/cdc/README.md). Both variants remain unapproved. The control step-1000 review shows clearer features but still strongly contracted identity diversity. The CDC run now shares the GPU with the reference run below, so wall times are not isolated performance measurements.

The active `reference256-paper-b64` baseline reuses NVIDIA's unmodified loss and augmentation code with separate main/regularization Adam phases, PL and R1, transfer ADA, 20-kimg EMA, batch 64 / microbatch 8, source minibatch-statistics group 8, and mirrored data (110 originals, 220 virtual entries). It continues the verified first update from `reference256-b64-m8-smoke` and allows 15,625 total updates: one million presentations. Checkpoints/previews occur every 125 updates (8 kimg). CPU exact continuation, snapshot side effects, actual intended MPS update, and safe generator export were tested; quality is still unproven. See [trainer notes](experiments/reference_phases/trainer-notes.md), [batch64 evidence](experiments/reference_phases/b64-m8-report.md), and `runs/reference256-paper-b64/experiment.json`. Compare experiments at equal image presentations, not equal optimizer steps. The independent [trainer audit](reviews/reference-trainer-audit.md) found no current-run-invalidating math error, but future resumes must explicitly check trainer, adapter, network-operation and PyTorch provenance because the recipe gate does not enforce all of it.

Twelve fresh [held-out portraits](data/validation12/README.md) remain excluded from training. The old control scores them lower than training inputs in both matched discriminator contexts, including a lower-padding subset. This is a diagnostic signal, not proof of memorization: rendering, pose, selection and alignment differences remain confounds. See [held-out diagnostic](reviews/heldout-discriminator/summary.md).

## Reproduce

```sh
uv venv --python 3.11 research/.venv
uv pip install --python research/.venv/bin/python -r research/requirements-lock.txt
git lfs pull --include="research/**" --exclude=""
research/.venv/bin/python research/prepare_data.py --synthetic --output research/data/hot110-256
research/.venv/bin/python research/benchmark.py --device cpu
research/.venv/bin/python research/benchmark.py --device mps --training
research/.venv/bin/python -u research/train.py --device mps --data research/alignment/collar-only/eyes42 --freeze-d-layers 4 --run aligned110-frozen4 --steps 6000 --batch 8 --snapshot-every 250
```

Resume the current run with `--resume research/runs/aligned110-frozen4/resume.pt` and the same `--freeze-d-layers 4`. A mismatch is rejected rather than silently corrupting discriminator optimizer state. Use a new run name when changing experimental settings. Reproduce the alignment inputs with the separate instructions in `alignment/`; they are also stored directly in Git LFS. The initial pretrained pickle is trusted NVIDIA code, fetched only from the URL and SHA256 in `sources.json`. Do not load arbitrary pickles.

## Measured feasibility

On the owner's Apple M5 Pro (48 GB), warm 256px inference measured 0.128–0.165 seconds on CPU (8 threads), and 0.023 seconds on MPS. Single-image training forward/backward and R1 second derivatives succeeded on MPS. Raw timings and environment versions are in `runs/benchmark-*.json`. These do not establish latency on a smaller rented CPU or validate priest image quality.

The first benchmark attempt hit a PyTorch optimizer parameter-type error before training; its log is retained and the float parameter was corrected. A successful rerun is recorded separately.

The verified official FFHQ 1024px checkpoint is now archived with provenance in `models/ffhq1024.pkl`. A research-only generator export passed the actual CPU serving path: warm median 0.996 seconds including WebP encoding and process peak RSS 3.42 GiB on this Mac. This exceeds the inactive Cloudflare candidate's 3 GiB setting. A subsequent Linux/amd64 Docker test with a 4 GiB memory limit and one CPU completed four 1024px generations with a 1.50 GiB cgroup peak and no OOM. This demonstrates that native Mac memory usage is not a Linux sizing prediction; final-model HTTP/concurrency and actual hosting checks remain. See [Linux container evidence](runs/inference-cpu/ffhq1024/docker-4g/README.md). Native Mac or emulation timing is not a production-host guarantee. These baseline faces are not a trained priest model and the production gate rejects this unreviewed export. See [1024 benchmark](runs/inference-cpu/ffhq1024/README.md).

## Evaluation required before production

- Inspect fixed and unseen random latent grids: realistic adult male faces, clearly visible clerical collar, no hats, calendar-style appearance, no frequent artifacts.
- Check diversity and training-set nearest neighbors; a distinct byte hash alone is not evidence of a distinct identity.
- Compare fixed latent samples over checkpoints for mode collapse and overfitting.
- Benchmark the actual inference container, cold start and warm request latency.
- Exercise concurrent server requests, fresh randomness, image dimensions, no-store caching, and failure responses.
- Deploy through CI from `master`, then verify the public domain makes real inference requests and produces different decoded images.

## Hosting research

Ordinary Cloudflare Workers have 128 MB memory and are unsuitable for this PyTorch model. Cloudflare Containers can run CPU PyTorch behind the existing Worker and keep everything in the personal account. They require Workers Paid ($5/month) plus metered container usage. The owner prefers local training and hosting below $15/month. No paid upgrade or rented GPU has been purchased. A fixed-price personal CPU server behind the Cloudflare frontend is also being evaluated to keep the bill predictable.

References checked 2026-09-20:
- https://developers.cloudflare.com/workers/platform/limits/
- https://developers.cloudflare.com/containers/platform/pricing/
- https://developers.cloudflare.com/containers/examples/static-frontend-container-backend/
- https://developers.cloudflare.com/workers/ci-cd/builds/build-image/

The repo default branch and CI production branch have been changed to `master` per the new instruction. Git LFS stores all raster images and models. CI explicitly hydrates `public/**` before validating and deploying, avoiding image-pointer files being published.

Default clones hydrate only `public/**` through `.lfsconfig`, so frontend CI does not repeatedly download experimental checkpoints. Research reproduction explicitly fetches `research/**`; inference CI explicitly fetches only `models/production/**`. All artifacts remain versioned in Git LFS.
