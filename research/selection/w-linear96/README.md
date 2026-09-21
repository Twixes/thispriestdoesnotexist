# Stored-W linear prefilter pilot

There is a useful preliminary adult-male appearance signal in the saved W vectors. It warrants larger held-out research; it does **not** establish an adult-only filter. No attractiveness model, image encoder, generator forward, new image, or production integration was used.

The configuration was written to `plan.json` before fitting: train only on batch2, evaluate only on batch1; exclude the combined adult-male `uncertain` labels; use the first saved W row after verifying all18 rows are identical; train-only mean/std normalization; L2 logistic regression with fixed lambda1, an unpenalized intercept, no class balancing, and fixed score thresholds0.5/0.8. There was no test-set hyperparameter or threshold tuning. The sigmoid scores are uncalibrated, not probabilities of adulthood. The second identical execution corrected a metadata field name, without changing data, parameters or thresholds.

Training uses61 labeled batch2 records (22positive/39negative); evaluation uses29 batch1 records (9positive/20negative). Three uncertain records per batch are excluded from fitting and scoring. `results.json` records every feature NPZ path/hash, original PNG provenance hash, label hash, per-record score, optimizer result, and error IDs. All96 original NPZs were checked for the expected finite shapes and exact repetition across W layers. The 512-dimensional feature count exceeds the61 training examples, making regularization and truly separate evaluation essential.

| Fixed held-out rule | TP / FP / FN / TN | Precision | Recall | Acceptance among29 labeled cases |
| --- | --- | ---: | ---: | ---: |
| Linear score≥0.5 | 7 /0 /2 /20 | 7/7 (100%) | 7/9 (77.8%) | 7/29 (24.1%) |
| Linear score≥0.8 | 2 /0 /7 /20 | 2/2 (100%) | 2/9 (22.2%) | 2/29 (6.9%) |
| Accept all | 9 /20 /0 /0 | 31.0% | 100% | 100% |
| Reject all | 0 /0 /9 /20 | undefined | 0% | 0% |

The lower95% Wilson precision bounds are only64.6% for7/7 and34.2% for2/2. These intervals illustrate the small sample, not the full uncertainty from subjective/correlated labels. At0.5 the two false negatives are v1-017 (score0.49943) and v1-024 (0.37701). The threshold was not moved to recover the near-boundary example. Training accuracy at0.5 is100%, which is not evidence of generalization in this feature-to-sample regime. Neither threshold is selected as a deployable operating point from these results.

The 0.5 rule enriches this *labeled* held-out subset from31% positives to7/7, but that is not measured final-site acceptance or end-to-end speed. It still accepts hats/other people and does not check collars, facial integrity, diversity or taste. In particular, the combination label may mostly learn masculine presentation; the excluded youthful masculine cases (v1-004 and v1-023) are precisely examples needed to evaluate a conservative age boundary. Three obvious children are not sufficient adolescent coverage. Uncertain labels remain unknown rather than negative ground truth.

The reference labels came from one Codex visual reviewer, with mixed contact/native scope and prior familiarity, as documented in `../manual96/README.md`. They are not blinded independent human ground truth. A frozen mapping preserves W but does not preserve output appearance when synthesis is adapted, so labels and thresholds must eventually be evaluated on the exact final generator.

CPU work was bounded and light: one-thread environment limits for OpenMP/BLAS/Accelerate, roughly0.04seconds for reading latents, fitting and scoring after imports, about74MiB peak RSS for the process, and under half a second wall time including imports in this environment. NumPy2.4.6 and existing SciPy1.17.1 were used; no package/model download. `run.log` records exact measured timing and memory. This is not hosting latency. `linear-model-research-only.npz` is a small research artifact, not an approved serving selector.

Next useful measurement: collect a separate random source/final-generator sample with independent conservative age labels and enough youthful masculine hard cases, then retain a separate calibration set and untouched test set. Keep this exact model and both fixed thresholds as preregistered baselines on that new sample rather than tuning against these29 cases. Pair a latent prefilter with checks on the actual final generated image; this pilot cannot replace those checks or certify adult-only output.
