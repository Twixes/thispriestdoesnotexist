# Collar-region checkpoint 200 evaluation

The completed single-pair region-balanced model preserves recognizable faces but does not produce convincing priest collars on the five unseen adult male portraits reviewed natively. On the training portrait, whitening resembles a soft patch on the neck rather than the target's clean rectangular clerical tab. This checkpoint remains unapproved for production.

The existing PID 94814 exited after logging step 200; it was never restarted. Evaluation ran only afterward, with 29% free memory, and exited 0. All 32 held-out source PNGs and mapped W tensors matched exactly. All 96 emitted native PNG hashes were verified. The original training-seed reproduction was excluded. See `invocation.json` for the exact command and limits of process-exit evidence.

CPU with one thread sequential evaluation took 43.12 seconds and peaked at 3.72 GiB native macOS RSS. This is not a Linux container or production hosting benchmark. No model work remains from this evaluation.

`comparison.png` is a diagnostic contact sheet only; each pair is source grayscale then student grayscale. Native 1024 PNGs are saved separately, including unchanged student RGB. `visual-review.json` records all five native comparisons plus the training portrait and exact image hashes. Women, children and headwear remain in unrestricted sampling; eligibility is unsolved.

Mean unclipped luminance MAE: full 0.02135, top75% 0.01566, bottom25% 0.03844. These are fixed image bands, not face/clothing segmentation or identity, quality, attractiveness or collar-recognition scores. Lower image difference is not evidence of successful priest conversion.

No trainer, training input, model weight or production configuration was modified. No deployment or additional training was performed. A separate bounded multi-pair experiment can test whether reviewed collar supervision generalizes; this result does not justify automatic production promotion.
