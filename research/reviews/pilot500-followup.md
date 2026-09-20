# Step-500 follow-up

The bounded CPU fused/unfused comparison used identical mapped styles for four
new seeds, two CPU threads, and checkpoint
`22d4b19933c8efbcfe3ee648ce382ebaa4794c1c77b499da8c6c6559e3de846c`.

- Maximum absolute floating pixel difference: `6.8694e-05`.
- Mean absolute difference: `8.1183e-07`.
- Maximum difference in 8-bit pixel units: `0.00876`.
- Both outputs are finite and allclose at absolute/relative tolerance `1e-4`.
- The two rendered rows are visually indistinguishable and both malformed.

Artifacts: `pilot500-modconv/comparison.json`, `comparison.png`, individual rows,
and reproducible `compare_modconv.py`. This rules out a large **CPU forward**
fused/nonfused discrepancy for this checkpoint; it does not prove MPS backward
equivalence. No GPU computation was used.

The original requested raw-G/EMA comparison was repeated on the atomically saved
step-500 resume state with the same eight diagnostic latents as step 250. Artifacts
are under `pilot500/`, with exact snapshot hash and seed in `comparison.json`.
Raw G has moved substantially toward adult male clerical portraits, with visible
white collars in all eight examples. Facial features remain visibly duplicated or
misplaced. The eight raw-G faces and compositions now also look conspicuously
similar; this is a warning of diversity contraction, not a formal measured proof
of mode collapse. EMA retains more source facial/hair characteristics and stronger
ghosting. Neither output family is suitable for production.

The next expanded-dataset run should retain matched raw-G and EMA grids and the
existing pairwise/diversity diagnostics. Better domain appearance alone must not
hide reduced identity diversity. The earlier bounded trainer recommendations
remain applicable; a change of fused evaluation path is not supported by this
diagnostic.
