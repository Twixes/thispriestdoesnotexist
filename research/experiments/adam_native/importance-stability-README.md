# Prepared FI1000 ranking stability analysis

`analyze_importance_stability.py` reads completed artifacts from the fixed-offset EMA importance evaluator. It loads accumulator tensors and inventory metadata only: no pretrained model construction, model checkpoint deserialization, inference, training, network access, or credentials. It hashes the source checkpoint bytes to authenticate lineage. No actual FI1000 output existed or was analyzed as part of this preparation.

The analysis reports two comparisons separately:

- **Nested prefix500 versus full1000:** useful descriptive convergence diagnostic, but both estimates share500 samples.
- **Disjoint first500 versus last500:** the second half is `(full raw sums - prefix raw sums)/(full successful count - prefix successful count)`. Both signed first-moment and nonnegative second-moment differences are validated; only second moments enter ranking. It never subtracts normalized means, treats a500-pair suffix as1000 samples, clamps negative differences, or silently repairs invalid data.

For G conv, G affine, and D conv pools, it retains the existing fixed-policy selection formulas, pooled percentile interpolation, strict high comparison, and module/output-row ordering. It reports Spearman correlation using average ranks for exact ties, distinct/tied/zero score counts, tied pairs, all-zero/all-equal status, minima/maxima, and descriptive q50/q75 mask counts, agreements, intersections/unions and high-set Jaccards. Constant ranks produce `null` correlation with a reason; two empty high sets produce `null` Jaccard with a reason, even though all-low mask agreement is100%. Neither is presented as evidence of convergence.

q50 andq75 are both reported for comparison. This analyzer chooses no threshold, defines no pass criterion, and does not automatically declare rankings stable. Disjoint latent/noise samples still reuse the same twenty training images and trained model; disagreement measures sampling sensitivity, while agreement cannot establish photographic quality or validity of the importance heuristic.

## Authentication and validation

The CLI requires a successfully completed FI1000 supervisor/result and verifies:

- Exact fixed-offset protocol identity, EMA keys, native1024/output-rank1 and FP64 accumulator policy.
- Pinned source file hashes, source checkpoint bytes, source protocol lineage, bound G/D modulation inventory, and matching sample-plan bytes/metadata.
- Both500 and1000 completion markers, serialized accumulator hashes, protocol and sample-plan hashes, and the terminal1000 accumulator identity.
- Strict saved accumulator schema, wrapper counts500/1000, matching successful counts in G/D, no missing observations, exact permitted coordinate keys/shapes against inventory, CPU FP64 finite signed first moments and finite nonnegative second moments.
- Exact shared lineage between prefix and full payloads, plus valid disjoint raw-sum differences.

The in-memory API `analyze_accumulators(prefix500_payload, full1000_payload, inventories)` performs tensor/schema/count/lineage validation and diagnostics, but callers must authenticate files and inventory themselves. `load_verified_run(run_dir)` provides that filesystem authentication. Existing accumulator schemas and frozen selection helpers are unchanged.

## Conditional command

After an actual fixed-policy1000-pair evaluation has completed:

```sh
research/.venv/bin/python research/experiments/adam_native/analyze_importance_stability.py \
  --run research/runs/adam-native1024-fixed-offsets-ema-importance1000-v1 \
  --output research/reviews/adam-fixed-offsets-importance-stability-v1
```

The run name is an example, not evidence of existence or completion. Output must be a fresh directory within `research/`. The tool archives a JSON report with input/source hashes and a copy of itself. It writes no selections for the main runner and touches no model/accumulator artifact.

## Tests

`test_importance_stability.py` uses synthetic inventory and raw-sum fixtures only, without even constructing a tiny network. Five tests verify:

1. Identical rankings under positive score scaling; actual-count normalization and unchanged inputs.
2. A deliberately adversarial case where nested prefix/full rankings agree perfectly while disjoint halves have rho−1 and disjoint high sets; this exposes the reason both comparisons are required.
3. Explicit tie ranks, all-zero scores and empty masks, JSON-safe undefined results.
4. Rejection of incompatible lineage, G/D count mismatch, missing names, wrong shapes, missing observations, nonfinite/negative moments and impossible negative suffix second moments.
5. Filesystem marker/payload/source authentication using temporary synthetic files; changed sample-plan lineage and changed opaque checkpoint bytes are rejected. The opaque checkpoint cannot be deserialized as a model.

```sh
research/.venv/bin/python research/experiments/adam_native/test_importance_stability.py
```
