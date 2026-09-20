# Priest generator research

Goal: train a real generative model and run fresh server-side inference on every page load. Sampling the static gallery, pixel morphing, or adding noise does not meet this goal. The production site remains on its curated static gallery until a model passes evaluation.

## Current experiment

Transfer learning from NVIDIA's FFHQ StyleGAN2 at 256px. The training set contains the original curated 20 hot-priest portraits and all 30 subsequently commissioned handsome priest portraits (50 total). All are fictional adults, without hats. `data/manifest.json` pins the source commit, image identities, hashes, and preprocessing. Existing 80-image originals stay in the repo; excluded portraits are not used for training.

The reference network code is reused without modifications. `train.py` provides a portable PyTorch training loop with non-saturating logistic losses, lazy R1, style mixing, EMA, and MIT Han Lab's differentiable color/translation/cutout augmentation. Augmentation probability follows the ADA discriminator-sign heuristic. This is a documented adaptation, not a claim to reproduce the entire NVIDIA training recipe: path-length regularization is currently disabled and augmentation differs from their full pipeline.

A 3000-step, batch-8 pilot is used to test adaptation and identify overfitting before choosing a longer/higher-resolution run. Pilot checkpoints are research outputs, not approved production models.

## Reproduce

```sh
uv venv --python 3.11 research/.venv
uv pip install --python research/.venv/bin/python -r research/requirements-lock.txt
git lfs pull
research/.venv/bin/python research/prepare_data.py
research/.venv/bin/python research/benchmark.py --device cpu
research/.venv/bin/python research/benchmark.py --device mps --training
research/.venv/bin/python -u research/train.py --device mps --steps 3000 --batch 8 --snapshot-every 250
```

Resume with `--resume research/runs/pilot256/resume.pt`. The initial pretrained pickle is trusted NVIDIA code, fetched only from the URL and SHA256 in `sources.json`. Do not load arbitrary pickles.

## Measured feasibility

On the owner's Apple M5 Pro (48 GB), warm 256px inference measured 0.128–0.165 seconds on CPU (8 threads), and 0.023 seconds on MPS. Single-image training forward/backward and R1 second derivatives succeeded on MPS. Raw timings and environment versions are in `runs/benchmark-*.json`. These do not establish latency on a smaller rented CPU or validate priest image quality.

The first benchmark attempt hit a PyTorch optimizer parameter-type error before training; its log is retained and the float parameter was corrected. A successful rerun is recorded separately.

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
