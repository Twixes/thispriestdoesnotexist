# Completed six-pair training plus one-holdout evaluation

`evaluate_pairs.py` is a separate research entry point. It renders all six supervised identities and the unchanged validation 028 pair, unlike the training preview helper that shows only the first training identity. Neither trainer, the unseen evaluator, nor the running dataset is changed. Training images are clearly labeled TRAIN and never counted as unseen successes; validation 028 is labeled HOLDOUT.

Run only after the exact training process has exited at its configured final step 600. This script verifies checkpoint/config completion but does not inspect process liveness; the caller must verify terminal status first. It then requires at least 25% fresh free memory before importing PyTorch. Do not run it concurrently with training or the unseen evaluation.

```sh
research/.venv/bin/python research/experiments/paired_regions/evaluate_pairs.py \
  --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle \
  --checkpoint research/runs/paired-regions1024-sixpair-600/resume.pt \
  --manifest research/data/paired7/manifest-proposed-v2.json \
  --expected-step 600 \
  --output research/runs/paired-regions1024-sixpair-600/evaluation-pairs7
```

The completed checkpoint is read with `weights_only=True` and hashed from the same open inode. Existing region-evaluator guards enforce its format, nonapproval, source architecture/checksums, full trainer/helper/vendor inventory, options/config provenance and exact configured completion. Additional guards require exactly six training identities plus validation 028, the original manifest hash, and zero source-verification tolerance. Source/student and frozen-state digests are verified.

The unchanged training `load_pairs` helper rerenders all seven source PNGs with one CPU thread, verifies exact mapped W and uint8 source pixels, reloads the edited targets and original masks, and compares its full resulting provenance with the checkpoint. Stored training and validation Z values must be distinct. The actual student then synthesizes each image at the same W, constant noise, FP32 and native 1024 resolution. No optimizer update, stochastic sampling or pixel compositing occurs.

Outputs are 28 native PNGs: source grayscale, actual student grayscale, edited-target grayscale and unmodified student RGB for each identity. Target images follow the already recorded square Lanczos resize from native 1254 to 1024; no alignment warp or crop is added. A clearly labeled diagnostic contact shows source/student/target side by side. The report records image hashes, model hashes, input/mask provenance, code hashes, timing, peak RSS, memory preflight and per-identity split labels. There is no aggregate mixing training and holdout error.

Per-image metrics use unclipped grayscale error on the nominal [0,1] scale: source error on protected pixels, target error on the full clothing region, and optional separately normalized target errors for collar/rest plus their balanced mean. Validation 028 has no collar trace; its collar/rest metrics remain `null`. Fixed top 75%/bottom 25% source errors are also included with the existing helper. None of these values proves identity, attractiveness, collar recognition or photoreal quality. A good TRAIN fit is reconstruction evidence; even a good single beard HOLDOUT is insufficient evidence of population generalization. Review the separate disjoint 30 evaluation as well.

Preparation-only verification, with no PyTorch import or model execution:

```sh
research/.venv/bin/python -m unittest discover \
  -s research/experiments/paired_regions -p 'test_evaluate*.py' -v
```

The native paired evaluator has not yet been run against the active 600 checkpoint. Its seven-source memory/runtime behavior remains unmeasured. No production approval or export is provided.

Eligibility clarification for the separate disjoint 30 review: source 004 is age-uncertain and includes an additional person. Inspect its preservation and neckline artifacts, but exclude it from eligible-adult or hot-success counts. Sources 017, 024, 025 and 028 are the clearly adult masculine native-review subset; 017 and 028 have beards that limit neck visibility. This supersedes the historical region200 label of all five as adults without rewriting that evaluation. The diagnostic 30 manifest is deliberately unfiltered and unchanged. The seven paired examples in this evaluator do not include 004.
