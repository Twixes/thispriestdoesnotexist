# Step-200 unseen-latent evaluation

The completed one-pair model preserves recognizable faces but does **not** demonstrate priest-clothing generalization. None of the five adult male portraits inspected at native 1024px (000, 004, 025, 028, 030) has a convincing clerical collar. Changes are predominantly darkening and local texture around original shirts or necklines; 004 develops an unwanted dark stippled patch. This checkpoint is not production-approved.

All 32 fixed source latents were evaluated after training PID 90542 exited with a final step-200 record. The original training-seed reproduction was excluded; none of the 32 overlapped a training Z. Source weights, architecture, code/runtime provenance, frozen state, every mapped W, and every regenerated source PNG passed exact checks. The actual evaluator exited zero after 42.35 seconds on one CPU thread. Peak native Mac RSS was 3,996,172,288 bytes (3.72 GiB), so this is not evidence of fitting a 3 GiB deployment. Memory was 30% free before evaluation and 29% after.

Mean unclipped grayscale MAE was .02676 across the image, .01902 in the top 75%, and .04999 in the bottom 25%. These fixed bands include background/hair/face/clothing together; they are not semantic masks or identity scores. They establish that lower-region pixels changed more on average, not that a collar was learned.

`comparison.png` is a labeled diagnostic contact sheet containing all 32 source/student pairs. Each sample also has native source-gray, student-gray, and untouched student-RGB PNGs. `evaluation.json` records hashes, exact checks, per-image metrics/timings, checkpoint provenance, runtime and memory. All 96 PNG hashes were verified after generation. `visual-review.json` records the five native reviews and broader contact-sheet findings; `invocation.json` records the execution and training-completion evidence.

Women, children and hats remain in the broad source distribution. The fixed set shows no obvious wholesale identity collapse, but it cannot establish general diversity or identity preservation. A separately bounded experiment with multiple reviewed clothing pairs is reasonable; automatically extending this single-pair run or deploying it is not supported by these results. Sampling attractive adult bareheaded men remains a separate unresolved requirement.

No output image was composited with source pixels. Grayscale is a fixed display conversion; only the diagnostic contact sheet concatenates different images. No training restart, active-input modification, or deployment occurred.
