# Prospective 36-portrait training set

The previous 24 whole-image portraits plus 12 selected from the 16 new independent generations (165–180). Root viewed every new original and selected for facial/collar integrity and the requested attractive-calendar aesthetic. IDs 167, 169, 172 and 178 remain archived but are excluded from this target set. Selection is subjective, not an age or demographic rule.

The previous 24 derivatives are copied byte-for-byte. Each new portrait is converted from its original to grayscale RGB, then independently resized with Lanczos to 1024 and 256. No cropping, reflection, padding, retouching or face restoration is applied. The new images have somewhat more headroom and chest visibility; this is a changed-data experiment candidate, not an exact replacement for any existing training inputs.

`manifest.json` records original and derivative hashes, selection provenance and preprocessing source. [Root selection](../../reviews/expansion16-v2-root-observations.json) includes all 16 native-image observations; [builder](../../reviews/build-plain-background36-whole-image.py) reproduces this directory only when it does not already exist. The contact sheet is a diagnostic montage.

These are synthetic training inputs generated with imagegen, not output from our trained GAN. Neither the running 110-image reference experiment nor the prepared 24-image NADA experiment uses this set. A future use must explicitly record the changed dataset. Production remains unapproved.
