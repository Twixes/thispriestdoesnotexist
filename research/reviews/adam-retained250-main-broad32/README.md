# Preregistered broad review: retained250-route main adaptation

This folder prepares a broader review after a future successful native main-adaptation 10→100 continuation. “Retained250” names the reviewed probing/Fisher source, not the main-adaptation iteration. The renderer intentionally accepts **completed main100 only**. Preparation, latent hashes and blank review rows are not model output or quality evidence. No native render has been run by this preparation.

## Latents and scope

Exactly 32 new `z ∈ R^512` draws are preregistered using NumPy `default_rng(2026092291).standard_normal((32,512))`, then cast to float32. All draws are retained in their original order. The artifact records NumPy version, complete archive hash and per-row hashes. All 32 rows must be finite, mutually distinct and byte-distinct from the original four evaluation latents. There is no rejection sampling, truncation or selection. The already-created latent archive and preregistration are immutable; a changed experiment requires a new version.

The later render produces raw and EMA for all 32 draws: **64 entire native 1024 × 1024 PNGs**, unfiltered. The source forward is unchanged: `model(z[None], None, truncation_psi=1, noise_mode='const', force_fp32=True, fused_modconv=False)`. Only the frozen runner's RGB pixel conversion and lossless PNG encoding occur. No prompts, text conditioning, crops, facial edits, enhancement or image restoration. Constant synthesis noise holds this review's comparison controlled; these images are not serving-latency measurements.

Source NVIDIA code/weight license terms remain in effect. The repo MIT license does not relicense those assets. Both source code license files and original model metadata are hash-pinned by preregistration; the weight path and SHA are explicit. Output render provenance binds the actual future checkpoint and source protocol hashes.

## Explicit later rendering

Only after the native10→100 main run has actually completed successfully and the primary four-latent review has been examined, run the command below with its real source/output paths. This documentation does not authorize an automatic launch while another heavy worker is running.

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-broad32/render.py \
  --run research/runs/ACTUAL-COMPLETED-MAIN100-RUN \
  --output research/runs/NEW-BROAD32-RENDER-ARCHIVE
```

The source must have the exact prepared continuation protocol name, native1024 CPU configuration, retained250 lineage, fixed-offset policy, source-code pins, complete result/supervisor, successful main100 checkpoint marker, exact state/checkpoint hashes, exact native fold checks, and all eight step100 PNGs. No generic failed, intermediate or unrecognized checkpoint is accepted.

The renderer uses the original native generator metadata and original safetensors as its fixed references, installs the existing output-rank1 modulation and the existing `FixedOffsetsAdaptationMasks`, then loads raw G and Gema strictly from the authenticated checkpoint envelope. It checks protected original rows, buffers and all 34 fixed noise/activation-bias offsets. Only one native G is constructed; D and Adam state are authenticated within the checkpoint envelope but are not instantiated or used for inference. There is no training, optimizer step or repair of invalid state. Before new draws, each arm must reproduce all four saved native step100 PNG hashes exactly. A mismatch aborts; it is never rebaselined. Global torch RNG must be unchanged by the forward passes.

The explicitly launched worker has a six-GiB RSS bound, ten-minute deadline, 35% available-memory start gate, 20% runtime gate and 512-MiB swap-growth bound. Its supervisor only controls its own process group. No automatic retry/resume. All PNGs and complete provenance remain in the new run archive. The final image manifest is written only after all 64 images and invariants pass. If a run aborts, partial produced PNGs stay archived but are not passed off as a complete broad review.

## Build a non-blocking review

Initial pending page (no model imports or rendering):

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-broad32/build.py
```

After a later render:

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-broad32/build.py \
  --run research/runs/NEW-BROAD32-RENDER-ARCHIVE
```

Open `http://127.0.0.1:8787/reviews/adam-retained250-main-broad32/`. This static page does not poll or control training. It shows every latent row, both arms, source/image hashes, full-image links, pending/manual-review states, and all 20 unchanged training-image links for manual resemblance comparison. The builder verifies exact 64-image coverage, latent binding, dimensions, PNG hashes, source/checkpoint bindings and supervision status. Build-time artifact status is not process-health evidence.

`annotation-template.json` contains 64 blank rows, one per image, with fields for apparent adulthood, priest clothing/collar, hats, gross artifacts, photographic coherence, repetition/nearest-training-image concerns and free notes. Each row starts `not_reviewed`; observations are null. Once images exist, the template binds each row to its image hash and checkpoint hash. Copy it to a new file before entering actual observations; rebuilds replace the blank template. Supply completed or partially reviewed rows with `--annotations PATH` to display them. Incorrect image/checkpoint bindings or incomplete/duplicate row inventories are rejected. Unknowns remain null. No approval is inferred from a complete render or filled form, and there is no automated attractiveness or nearest-neighbor score. Apparent minors must never receive attractiveness ratings.

## Preparation verification

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-broad32/test_preparation.py
```

Three small NumPy/JSON fixture tests cover latent completeness/disjointness, corrupt checkpoint envelopes/counts/masks, complete image/latent coverage, and unreviewed annotation defaults. Fixtures are explicitly synthetic and contain no model weights. These checks do not prove that native reconstruction or generation succeeds; only the later actual reproduction and render can establish that. Native rendering remains unexecuted.

## Actual main100 review, 2026-09-22

The main100 broad render completed and all64 images were inspected using eight full-frame contact sheets. Raw/EMA020,026 and031 also received direct native follow-up. The result is **0/32 clear priest portraits in raw and0/32 in EMA**, with systematic raw cooling/etched contrast and localized defects. No model is approved. The report and64 bound observation rows are `agent-review-main100.json` and `annotations-agent-main100.json`; unknown adulthood/headwear fields remain null. Contact sheets are review-only uniformly reduced layouts; all native source PNGs remain untouched. No nearest-neighbor computation or attractiveness assessment was performed.

Reproduce the reviewed page, including the actual outcome at the top, without changing frozen preregistered render inputs:

```sh
research/.venv/bin/python research/reviews/adam-retained250-main-broad32/build_reviewed.py
```

This wrapper first runs the frozen builder with the actual completed render and actual annotation file, checks checkpoint/result binding, then adds the observed outcome. `record_agent_review.py` preserves the observations entered after inspection; it is not an automated image evaluator. The displayed render-status label may still say review pending because no human quality approval is implied by agent inspection.
