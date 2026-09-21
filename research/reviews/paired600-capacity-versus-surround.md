# Independent step-600 decision review

The evidence supports testing collar-surround weighting before opening b32. It does **not** establish that b64-and-higher blocks can only change texture or that collar geometry requires b32. The more directly evidenced problem is misplaced brightness around an already well-fitted tab. Parent has accordingly deferred b32 and selected a matched 50-update baseline versus equal-thirds tab/surround/remainder objective probe, keeping capacity fixed. I agree with that ordering; extra training duration alone is not yet justified.

Reviewed the seven-pair contact sheet, full native 1024 student images for calibration-original, 030, b2-055, b2-020, all six training metrics and heldout 028, and the PNG-only surround script/report. No weights were loaded or model compute run. Checkpoint 600 SHA256: `0f433c7ec50d9e80026681125d7fa7e17e9be6e8e2d31b11976fb51d1b0f3fc4`.

## Evidence

| Train identity | Tab L1 [0,1] | Rest L1 [0,1] | Protected-source L1 | 15px ring MAE /255 | Ring share of rest |
|---|---:|---:|---:|---:|---:|
| calibration-original | .0263 | .2071 | .0084 | 94.8 | 2.67% |
| 000 | .0328 | .0823 | .0081 | 128.6 | .92% |
| 030 | .0349 | .2013 | .0167 | 141.7 | 2.16% |
| b2-055 | .0401 | .0967 | .0082 | 88.6 | 1.67% |
| b2-051 | .0201 | .0938 | .0067 | 77.5 | 1.68% |
| b2-020 | .0349 | .0281 | .0156 | 67.7 | 1.01% |

The tab brightness fits much better than its immediate surroundings. At 15px every ring is too bright in mean; five of six ring MAEs exceed their unadapted source errors. b2-055 improves markedly from its very bright source but still has a broad pale neck band. Native calibration has a skin-like bright patch below the chin with largely original shirt structure;030 has a pale patch and unnatural dark neck texture; b2-055 resembles a dark scarf with smeared trim; b2-020 has a dark shirt but no clean tab silhouette. Faces remain recognizable, but low average protected error does not establish absence of visible local damage. Heldout028 clothing error is .4247 and the contact view retains its original garment; supervised fits are not fresh-latent success.

The current loss gives half the clothing weight to a tiny tab and half to **all** remaining clothing (`paired_regions/trainer.py:148-151`). These 15px rings occupy only 0.92–2.67% of the rest area, hence only approximately 0.46–1.34% of the total clothing L1 output-gradient mass away from exact zero residuals. This is an output-space weighting observation, not a parameter-gradient measurement. Low tab error plus high ring error is consistent with fitting brightness while tolerating a diffuse halo. It does not prove that reweighting alone will solve geometry.

The surround diagnostic is appropriate for this question: square dilation of the raw trace, excluding that trace, intersected with the unchanged clothing mask (`analyze_pair_surrounds.py:48-61`); 5/15/30px rings show the same broad direction. Its clipped PNG errors differ from unclipped tensor training metrics. Boundaries are hand-traced and may include antialiasing; target means must be used, not an assumption that every surrounding pixel is black. The reported protected-pixel count is zero by construction, not an independent verification of mask quality.

## What b32 tests, and its risks

The actual freeze rule opens every synthesis block above 32, including convolution, style-affine and ToRGB parameters, while keeping noise strengths fixed (`paired_edit/trainer.py:109-122`). NVIDIA's vendored primary implementation has spatial convolutions and additive, upsampled RGB contributions at each skip block (`training/networks.py:340-419`). Nothing in that code restricts b64+ to texture. Opening b32 gives a coarser editable feature/RGB path and may make coordinated neckline changes easier; that is a plausible hypothesis, not a demonstrated semantic boundary.

The same path can alter face shape, lighting and background across unseen identities. Protected pixel loss constrains only supervised identities; fresh preservation covers the top 75% rows and is a geometric proxy, not full-face/clothing segmentation. More capacity cannot resolve contradictory targets where desired tab pixels lie outside the editable mask:44.6% of000's raw tab and 55.4% of b2-020's raw tab are excluded (other pairs 9.2–21.9%). Those pixels receive source-preservation supervision. Keep that limitation explicit; do not quietly widen masks or count partial-tab L1 as whole-tab reconstruction.

If b32 is tested later, retain common checkpoint weights, existing Adam moments and step counters, pair-sampling and preservation RNG streams, LR and loss. Add fresh optimizer slots only for newly enabled b32 parameters, and document this unavoidable difference. Alternatively reset the optimizer in **both** arms and label that separate experiment; resetting just one arm confounds the comparison. Restore RNG after any setup that consumes it. Freeze mapping, <=16 blocks, buffers and noise strengths in both; record exact parameter lists and unchanged frozen-state hashes. Fifty stochastic updates across six examples is a short adaptation diagnostic, not a convergence test or a generalization verdict.

Judge against the matched baseline using the existing full-output native images, tab/rest/protected metrics, the same three ring radii, heldout 028 and disjoint identities. A lower aggregate loss without a sharper correctly placed tab/neckline is weak evidence. If b32 improves ring/silhouette reconstruction materially with comparable face preservation, further capacity work is supported. If it mainly changes faces or repeats the halo, the capacity hypothesis is weakened; failure at 50 updates does not prove b32 can never help.

## Selected objective-first probe

Parent selected `one-third*tab_L1 + one-third*exterior_ring_L1 + one-third*remaining_clothing_L1`, with per-image area normalization, versus the unchanged half-tab/half-rest baseline for 50 updates each. Prespecified radius 30 uses the existing broader diagnostic and reduces reliance on the bright antialiased 5px neighborhood. Ring = dilate(raw_trace) minus raw_trace intersected with clothing; remaining = clothing minus effective_tab minus ring. Validate nonempty disjoint regions and their exact union. Match actual target pixels, not a hard-coded dark value; leave protected/source/fresh terms, freeze rule, all weights and existing Adam state, data and RNG streams identical at the branch point. Retain complete generated outputs. Treat radius and equal thirds as experimental, not calibrated.

This preserves the total clothing coefficient but reduces tab weight from one-half to one-third while increasing surround emphasis. Interpret it as testing the complete revised objective, not an isolated ring-only intervention. Report all three component losses, tab quality and actual source/student/target ring means at 5/15/30px so a ring improvement bought by a worse tab remains visible. A tab-weight-preserving half/quarter/quarter variant would be a possible later attribution test, not an extra arm to add now.

This is more directly connected to the measured failure than a frozen discriminator feature loss and cheaper to interpret. It risks overfitting imperfect hand boundaries and does not supply semantic collar generalization. A derivative/boundary loss alone can sharpen the wrong structure and is insensitive to constant brightness offsets inside regions, so regional target L1 is the clearer first test for this particular bright-surround error. Do not combine b32 and a new objective in the initial comparison: a two-factor follow-up is justified only if the initial results leave both explanations plausible.

No production recommendation or additional approval gate is introduced. This review selects what the evidence can test; it does not certify model quality.
