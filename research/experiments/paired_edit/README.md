# Paired-edit pixel baseline

This isolated research prototype directly tunes a generator at known latent codes. It reuses the unmodified NVIDIA network implementation; it does not modify either existing trainer or running job. It is a deliberately simpler experiment than JoJoGAN/PTI, as proposed in `../../reviews/paired-edit-adaptation.md`. There is no discriminator, LPIPS, identity network, adversarial loss, PL, R1, EMA, or output compositing. Those are excluded to make the first question measurable: can a small supervised clothing edit be fitted without damaging the source face?

## Method and intentional choices

- Start from the **exact serving bundle** (`model.json` and checksum-matching `generator.safetensors`), not a different FFHQ resolution or raw-versus-EMA source. Only unconditional RGB, psi=1 bundles are accepted.
- Each NPZ supplies fixed float32 z and W. CPU mapping must reproduce W exactly by default; constant-noise synthesis must reproduce the lossless source PNG's quantized uint8 pixels exactly. Explicit `--w-atol`/`--source-max-uint8-error` options are recorded if a future experiment justifies a tolerance. Regenerated float source tensors, not decoded PNGs, supervise protected regions. MPS does not silently relax source checks: verification always happens first on CPU.
- Freeze mapping, all synthesis blocks through 32px, all buffers and every `noise_strength` parameter. Only blocks above 32px train. This limits capacity and naturally removes frozen-prefix activation gradients. It is not a claim that high-resolution blocks cannot alter identity. Models stay in eval mode so mapping averages remain immutable; weight gradients still run. Both source and student use constant noise and forced FP32.
- Convert student, source and target through the same fixed grayscale operation (.299/.587/.114). Clothing-mask L1 compares student to edit; protected-region L1 compares student to source. Each region is independently normalized by its pixel area, then averaged over batch. Default weights are 1/1, not a published optimal setting. Generated values use the model's [-1,1] convention without extra training-time clipping.
- Masks are explicit normalized polygons supplied in the manifest; their union is rasterized at model resolution. No automatic neck/face detector, implicit crop, or feathered boundary. Native square targets are resized with Lanczos to source resolution. Manual boundary error and L1 blur remain known limitations. Preview overlays make that supervision inspectable before training.
- Optional `--fresh-weight` adds preservation against frozen source on one fresh latent per update, restricted to the top `--upper-fraction` rows (default .75). This is a crude geometric approximation, not segmentation. It is off by default for the one-pair feasibility smoke. Its separate CPU RNG and sequential backward keep it reproducible and avoid retaining two backward graphs at once.
- Adam uses LR 1e-4 and betas .9/.999 without lazy-reg compensation because there are no lazy regularizers. Training pairs are sampled with replacement; validation pairs never enter optimization. Fixed unseen-preview latents have an independent generator. Preview outputs are full student images followed only by grayscale conversion and uint8 encoding; source/target pixels are never pasted into them.

The default limit is ten updates; a longer run requires explicit `--total-steps`. This tests feasibility, not unseen priest quality or all-male sampling. A paired edit of adult males does not remove women/children/hats from the rest of FFHQ's latent space.

## Input schema

All paths resolve relative to the manifest directory. NPZ arrays accept z `[z_dim]` or `[1,z_dim]`, W `[num_ws,w_dim]` or `[1,num_ws,w_dim]`. An omitted split defaults to `train` for a smoke only; real experiments should declare all splits and reserve complete seed identities.

```json
{
  "version": 1,
  "source_bundle": "path/to/bundle",
  "production_approved": false,
  "pairs": [{
    "id": "unique-seed-id",
    "split": "train",
    "latent_path": "latents/seed.npz",
    "source_path": "images/source.png",
    "target_path": "edits/target.png",
    "clothing_polygons": [[[0, 0.8], [1, 0.8], [1, 1], [0, 1]]]
  }]
}
```

The example rectangular mask is schema illustration, not an approved real mask. Both clothing and protected areas must be nonempty. Source PNG dimensions must match the generator; targets may have another square resolution. Source-only manifests without target/mask fields cannot train.

## Checkpoints and reproducibility

Atomic `resume.pt` writes use file and directory fsync. The checkpoint contains student state, Adam state, completed steps/image counters, fixed preview latents, CPU/MPS/global NumPy/Python RNG plus both sampling streams, options, pair/source/code hashes and version provenance. Immutable source state is reconstructed from the checksum-pinned bundle; that bundle and dataset must remain available. Source and frozen-student digests are checked at every checkpoint. Previews preserve RNG and are checked for frozen-buffer mutation.

Resume into a **new empty run directory**, with `--resume old-run/resume.pt` and an absolute `--total-steps` greater than saved progress. Math-affecting options, device/thread count, preview configuration, code/version and all data/source hashes must match. Checkpoint cadence and final total may change. No production export is written; `--production-approved` is explicitly rejected, and the research checkpoint has no `G_ema` key compatible with the general production exporter. `production_approved` stays false. Loading uses `weights_only=True`.

## Completed evidence

`cpu-evidence.json` and `check-cpu.log` record tests of a **64px, channel-max-16 actual NVIDIA generator**, with nonzero constant-noise strengths. No FFHQ weights or MPS were used by these tests. Five continuous updates exactly matched a two-update checkpoint plus three resumed updates in student tensors, optimizer tensors/configuration, all RNG states and counters. Source, frozen blocks, mapping and noise stayed unchanged; finite trainable parameters actually changed. Validation exclusion, mask area/batch normalization and gradients, empty-mask rejection, fresh-latent preservation, strict source mismatch rejection, changed-option rejection and production-approval rejection passed. Generated previews matched direct full-student output and were byte-identical after resume. Tiny fixtures, checkpoints and individual logs are retained under this directory. Total test time was 6.02 seconds, not a 1024 training estimate.

The separately authorized **1024 CPU forward-only preflight** passed with zero W discrepancy and zero source uint8 discrepancy at zero allowed tolerance. Source state was unchanged. Source verification took 1.593 seconds; process peak RSS was 3,410,116,608 bytes (3.18 GiB), with 23% system memory free measured while the process was still resident. It exited normally. See `calibration-preflight/preflight.json`, `memory-after.txt`, and `calibration-preflight.log`.

That preflight used the original mask/manifest hash. The parent subsequently adjusted the mask to include narrow white-shirt edges; old evidence and `calibration-mask-overlay.png` remain intact. The revised diagnostic is `calibration-mask-overlay-v2.png` with sidecar hashes. The completed optimizer smoke reverified the revised manifest with zero source/W error. Its mask covers 13.467% of the image.

**The actual 1024 CPU smoke completed ten first-order updates with exit 0**, batch 1, one thread, fresh-preservation weight 1, checkpoint/preview at steps 5 and 10. Whole-process wall time was 64.94 seconds; max RSS was 5,577,064,448 bytes (5.19 GiB), peak memory footprint 5,766,436,072 bytes. Logged update times averaged 5.624 seconds, excluding loading, snapshots and checkpoint writes. Clothing L1 fell from .98914 to .91950; final protected-source L1 was .005873 and fresh upper-region L1 .003440. Both source/frozen-state checkpoint checks passed. These are optimization and resource observations, not image-quality approval.

The parent inspected the step-10 paired output: **the face remains intact, but the original white shirt is mostly unchanged**. One training identity and two unseen previews cannot establish useful clothing transfer or diversity. Captured metrics, exact source/input checks, hashes and this attributed visual review are in `../../runs/paired1024-cpu-smoke/review.json`; full timing is in `../../runs/paired1024-cpu-smoke.log`. An additional read-only checkpoint tensor scan was skipped because fresh memory was 24%, below its >=25% guard; this review does not claim independent finite-optimizer inspection.

The parent then started an exact continuation from step 10 to **200 total steps** in `../../runs/paired1024-cpu-200`, retaining CPU/thread/batch/loss options and two fixed unseen previews, with checkpoint interval 25. It was active at this documentation handoff; no completion or later visual result is claimed. Trainer/input hashes remain unchanged. This is still a single-pair feasibility experiment. No MPS training or production change has occurred.

The first edit originally used an encoded Linux-generated input; its paired lossless PNG was subsequently regenerated on native Mac. That small codec/backend difference is separately documented by the source-data agent; this is a feasibility pair, not a pixel-identical edit-input claim. Future edits should use the exact stored PNG directly.

## Commands and recorded runs

From repository root, first review the diagnostic overlay. The utility only reads inputs and writes a fresh diagnostic image/metadata; it neither loads a model nor modifies source/target:

```sh
research/.venv/bin/python research/experiments/paired_edit/mask_preview.py --source research/data/ffhq-paired-sources32/images/reproduction-sample002.png --target research/data/ffhq-clothing-edit1/001.png --mask-json research/data/ffhq-clothing-edit1/clothing-mask.json --output research/experiments/paired_edit/NEW-mask-overlay.png
```

The parent ran the following **CPU, one-thread, ten-update** smoke after code/mask review. The time utility captures whole-process peak RSS; there is no automatic CPU memory cap. These paths already contain evidence and must not be overwritten:

```sh
/usr/bin/time -l research/.venv/bin/python -u research/experiments/paired_edit/trainer.py --manifest research/data/ffhq-clothing-edit1/smoke-pair.json --run research/runs/paired1024-cpu-smoke --device cpu --threads 1 --total-steps 10 --batch 1 --fresh-weight 1 --checkpoint-every 5 --preview-count 2 > research/runs/paired1024-cpu-smoke.log 2>&1
```

The active continuation uses the same command options, changing `--run` to `research/runs/paired1024-cpu-200`, adding `--resume research/runs/paired1024-cpu-smoke/resume.pt`, setting `--total-steps 200` and `--checkpoint-every 25`, with log `research/runs/paired1024-cpu-200.log`. Do not launch a duplicate or change its inputs/code while it runs.

A future MPS invocation requires an explicit `--mps-memory-cap-gib` allocation limit. No MPS test or memory-fit claim is made here. A larger paired dataset should be considered only after actual visual clothing transfer and face preservation are demonstrated; decreasing pixel loss alone is insufficient.

NVIDIA's network/weights retain `../../vendor/stylegan2-ada-pytorch/LICENSE.txt`; MIT application code does not relicense those research/evaluation artifacts. Keep all research outputs unapproved. Tests establish implementation behavior, not acceptable photorealism, identity generalization, or deployment rights.

## Completed 200-step continuation

The exact CPU continuation finished successfully at step 200 (190 new updates) in 1,080.21 seconds, with maximum RSS 5,693,980,672 bytes and zero reported swaps. Frozen/source state checks passed at each saved checkpoint. The root native visual review finds preserved facial detail and a darkened shirt, but **no convincing white clerical tab**. This is not a successful priest generator. See `../../runs/paired1024-cpu-200/completion-review.json`. Disjoint-latent evaluation is recorded separately; the previous active-run description above is historical.
