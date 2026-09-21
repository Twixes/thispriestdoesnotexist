# Independent review of the proposed capacity × feature experiment

No blocking implementation defect found in `clothing_structure/losses.py`. The fixed 2×2 design can distinguish this particular block-range change from this particular feature objective, provided the actual discriminator smoke passes the stated resource/gradient guards. This is a read-only code/image review: no model loading, inference, backward pass, training, new tests, or edits to implementation/inputs were performed.

## Gradient localization and normalization

The student feature input is `M*gray(Y) + (1-M)*gray(S.detach())`; the target uses the same protected source pixels with edited target inside M. Consequently the feature term's derivative with respect to generated image Y is exactly zero outside M. Source, target, masks and RMS normalization are detached. The frozen prefix stays in eval mode with parameters non-trainable while retaining student-input differentiation. Region means correctly divide by each image's mask mass times channel count before the batch mean; fractional area interpolation avoids discarding small collar support. Target-feature RMS is shared across tab/rest for each layer/image, floored at 1, and detached. Layer averaging is explicit. These match the proposed design.

Two limits matter for the experiment:

- Feature receptive fields cross region boundaries. A tab feature can send image gradients to neighboring rest-clothing pixels, and vice versa. The half-tab/half-rest feature weighting therefore does not promise equal **pixel-gradient** budgets in those image regions. The strict support guarantee is only clothing versus protected pixels. Shared generator parameters can still change protected faces, particularly with b32 opened; keep actual protected/full-image review.
- The RMS floor prevents amplification of small feature scales but does not calibrate generator parameter-gradient strength. The 0.05 multiplier is not equivalent to a 5% update contribution. Log raw and **0.05-weighted** feature norms versus the actual pixel objective on the same shared b64+ parameter set, with b32 norms separate in opened arms. Scalar loss magnitudes alone cannot establish effective strength. A negligible feature gradient makes a null C/A result inconclusive, as the design already states. A single identity smoke is a resource/backward check, not proof of all-six gradient balance.

Neither point blocks the guarded smoke. Preserve the declared coefficient rather than tuning after seeing results. Boundary-loss helper is separate and should remain absent from these four arms.

## Calibration and 030 mask feasibility

Viewed the saved source/target overlays (`paired4/calibration-original-overlay.png`, `030-overlay-v2.png`) and target tab overlays (`paired4-tabs/*-tab-overlay.png`), alongside the previously reviewed native source/target/output portraits. The source overlay shows that both masks deliberately include visible lower-neck skin, not only preexisting shirt. Supervising those pixels toward edited fabric is intentional and necessary for the higher clerical neckline. Calling all neck skin protected would misdescribe the current objective.

There is no same-pixel contradictory protected/target supervision: M and 1−M are disjoint, and both feature inputs replace the edited face outside M with the identical source. Edited mouth/chin differences outside M are therefore excluded from the feature target. This does **not** guarantee an anatomically natural join at M's boundary. Features near the boundary see the artificial source/edited-target join, and may reward its structure. Generated previews must remain the complete student output; only loss inputs may be composited.

The historical exact mask analysis retains 7,198 / 7,923 traced tab pixels for calibration (90.85%) and 4,204 / 4,918 for 030 (85.48%). Therefore exact reconstruction of either **entire edited tab** conflicts with the immutable protected region. Reconstruction of their substantial effective in-mask tabs is not ruled out. The prospective gate should judge a recognizable bounded feasible tab and plausible fabric, not demand the clipped target pixels or a fully visible collar. This is already consistent with the proposed design. A few excluded pixels cannot by themselves explain the broad texture/pitting failure in the remaining supervised neck; that causal claim would be unsupported.

030's marked pitted neck output is a quality failure within the intended edit area, not proof that the protected mask contains that entire area. Its extra person on the left predates training. Calibration's broad white source-like neck patch and remaining folded shirt show that low tab MAE has not established correct fabric structure. Keep both as difficult supervised-fit cases; do not silently loosen their masks or use edited faces as identity ground truth.

## Judgment

Proceed only through the root-owned real-D smoke and its already declared guards before any factorial run. The design is a reasonable bounded diagnostic, not a production candidate; I found no code/mask blocker requiring its cancellation. Keep all four arms fixed and interpret native feasible-mask training fit before attributing unseen failures to dataset size. Do not infer that b32 failure rules out every capacity constraint, or that pretrained face-discriminator features necessarily encode clerical fabric semantics. No additional weighting microprobe is recommended here.

Reviewed hashes:

- Design: `d5481c733ad2bd4bf61443bd3d2ef0fa55b7db54d37993559109aae091a9cf86`.
- Feature helpers: `e3178228116ed60e90fda73365b88046978885b13a9938421c0e2025d038b585`.
- Fixed paired7 v2 manifest: `9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6`.
- Calibration source/target overlay: `96c66e05e94bc828ef8fb46475fcd4f51c7b4fdc4d8c53de41a1e7485c3ba8ec`.
- 030 v2 source/target overlay: `529cd5401b59eeb4802b5f4b2e26960563e135548c614c02365e8fed7f7d76b5`.

This review does not audit the still-being-implemented factorial runner or claim the real-weight smoke has passed. Existing seven tiny helper tests were read as evidence, not rerun.
