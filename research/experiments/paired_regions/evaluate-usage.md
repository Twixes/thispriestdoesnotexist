# Completed collar-region model evaluation

This separate entry point supports only `paired-regions-research-v1` checkpoints. The previously executed `paired_edit/evaluate.py` and both trainers remain unchanged. It reuses the original evaluator's lightweight hashing, latent identity, fixed-band error calculations, and command-line parser; the region checkpoint format and full imported-code provenance have separate strict checks.

After the intended region-training process exits, invoke from the repository root with its exact completed run path and step:

```sh
research/.venv/bin/python research/experiments/paired_regions/evaluate.py \
  --source-bundle research/runs/inference-cpu/ffhq1024/baseline-bundle \
  --checkpoint research/runs/paired-regions1024-cpu-200/resume.pt \
  --sources research/data/ffhq-paired-sources32/manifest.json \
  --expected-step 200 \
  --output research/runs/paired-regions1024-cpu-200/evaluation32
```

The command names the currently configured continuation; this document does not imply it has finished. Evaluation requires an empty/new output directory, at least 25% free system memory, and a checkpoint step matching both `--expected-step` and the sibling `config.json` total. The caller must verify process exit; checkpoint completeness alone is not a process-status check.

The evaluator loads the exact hashed checkpoint inode with `weights_only=True`, validates source weights/metadata/architecture and config provenance, and verifies the complete region-trainer source inventory. This includes `regions_trainer`, `paired_edit_helpers`, and every recorded Python/C++/CUDA/header file under the vendor's `training`, `torch_utils`, and `dnnlib` trees. Modified, added, removed, or missing entries are rejected. Frozen source/student state and all finite tensors are checked before rendering.

The fixed manifest's original 32 entries are evaluated sequentially with one CPU thread. The separate original-training-seed reproduction is never included, and exact Z overlaps with any training pair are excluded. Every remaining source PNG and mapped W must match an actual source forward exactly. Student outputs are full synthesis forwards; there is no source-pixel compositing.

Outputs match the prior evaluator: native source/student grayscale PNGs, native student RGB PNGs, hashes/provenance, a labeled source/student diagnostic contact sheet, timings/RSS, and full/top-75%/bottom-25% unclipped grayscale MAE. Those horizontal regions are not face or collar segmentations; their errors do not prove identity, attractiveness, priest eligibility, or quality. No collar ground truth or class predictions are invented. Women, children and hats in the fixed sample remain visible for review.

The report includes both this entry point's hash and the imported original evaluator's hash. Prior native evaluation used about 3.72 GiB; this entry point has not yet had a real-model memory measurement. It makes no claim to fit a 3 GiB server or to be production-approved.

Preparation validation only, without importing PyTorch or running a model:

```sh
research/.venv/bin/python -m unittest discover -s research/experiments/paired_regions -p test_evaluate.py -v
```
