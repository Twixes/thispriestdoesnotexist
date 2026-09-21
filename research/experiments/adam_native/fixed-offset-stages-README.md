# Fixed-offset FI and future main-adaptation support

Prepared separately on 2026-09-22. No native importance estimation or main-adaptation training has been executed by this preparation. No existing frozen sources or running model files changed. This carries the native100 and100→500 stability run's explicit policy into subsequent stages; it is a deviation from upstream AdAM, not a claim of reproducing that method.

## Audit and exact policy

Read `importance-source-audit.md`, `source-audit.md`, `fixed-offsets100-README.md`, original `estimate_importance.py`, `adaptation_masks.py`, `selection.py`, `modulation.py`, and the fixed-offset100 and continuation sources. The source-style helpers enable every modulation coordinate independently of checkpoint requires-grad flags. Saved state_dict does not persist these flags. Consequently simply loading fixed-policy EMA checkpoints into the old evaluator or reusing old adaptation masks would silently re-enable all 34 protected offsets.

The new shared `fixed_offset_policy.py` identifies only synthesis conv0/conv1 additive `noise_strength` and activation `bias` offsets. It checks a complete architecture-derived inventory: 17 of each at native1024, seven of each in native32 test networks. All protected offsets must already be exactly zero. It never repairs or zeros incompatible state. Their original pretrained noise strengths and activation-bias tensors remain registered and must remain bit-exact, frozen, gradient-free, and outside the optimizer. Mapping FC, style-affine, and ToRGB bias offsets are not part of this protection; D has no protected offsets under this G-specific policy.

`estimate_importance_fixed_offsets.py` is a separate pinned derivative of the old supervised CLI. It requires the parent protocol's exact34-offset policy, authenticates the completed checkpoint/protocol/inventory as before, restores Gema/Dema, and checks34 protected names. Its original-tensor references are captured from the authenticated loaded EMA checkpoint; whole-model before/after hashes establish the read-only evaluation claim, rather than those references independently authenticating pretrained provenance. Each phased loss-gradient call validates its exact parameter scope, never enables excluded offsets, and checks fixed state before/after. Whole-EMA state hashes before/after retain the stronger general unchanged-state check. It archives excluded/measured parameter names explicitly. The 4-pair diagnostic and1000-pair estimator retain existing guards and objectives;4 pairs are not reliable rankings. No automatic native launch is performed here.

Importantly, enabling additional parameter gradients for an otherwise identical forward pass does **not** change partial derivatives with respect to retained coordinates. Omitting protected-offset gradients therefore leaves u/v and allowed b statistics equivalent for the same model, samples, and arithmetic. This is a measurement-scope distinction, not evidence that the training trajectory matches unconstrained AdAM. Freezing these offsets during probing changes that training trajectory.

`selection_fixed_offsets.py` preserves all three original score pools and formulas. G synthesis-conv score is `mean(F_u)+F_v`, with **no activation-bias or noise term**. G affine is `(mean(F_u)+F_v+F_b)/2`; D bias-bearing convolutions likewise, with bias-free skip `mean(F_u)+F_v`. Thus every required score remains present. Protected names must be omitted, not supplied with zero or measured scores. The wrapper leaves numeric scores/masks identical while correcting policy metadata to fixed-zero and removing excluded fields from the allowable score schema. Quantile remains a caller decision; this preparation chooses no priest threshold and proves no ranking reliability.

`adaptation_masks_fixed_offsets.py` exports `FixedOffsetsAdaptationMasks`, derived from the frozen original controller. It requires explicit fixed-policy selection metadata and exact name coverage. It removes protected offsets from trainable parameters before building protected snapshots and calling any gradient toggle. Their values appear in the protected map, grad-policy checks, optimizer membership validation and zero/no-gradient checks. G synthesis activation **original** biases were already frozen by the old controller because only affine G biases receive row masks; this version continues that protection. D mask behavior is unchanged. No clamping, gradient repair, optimizer reuse, or checkpoint continuation is added.

## Interfaces

After an eligible500-horizon probing checkpoint is completed and independently accepted for this diagnostic, the next bounded command is:

```sh
research/.venv/bin/python research/experiments/adam_native/estimate_importance_fixed_offsets.py \
  --run research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1 \
  --step 500 --pairs 4 \
  --output research/runs/adam-native1024-fixed-offsets-ema-importance4-v1
```

Use a fresh output directory, recheck current memory/other workers, and never infer completed state from this command example. The evaluator keeps35% available-memory start,20% runtime,12GiB process-tree RSS,512MiB swap-growth and20-minute diagnostic guards. Do not run concurrently with a large native training worker merely because source files are ready.

Future main adaptation starts from original authenticated source G/D with fresh factors and fresh optimizers. It consumes only scores/selection from probing, never probing model/optimizer states:

```python
G.requires_grad_(False)
refs = capture_fixed_reference(G)
selection = select_rows_fixed_offsets(means, describe_modulation(G), quantile=chosen_q)
controller = FixedOffsetsAdaptationMasks(G, selection)
optimizer = torch.optim.Adam(controller.parameters(), ...)
controller.bind_optimizer(optimizer)
# For every adversarial/path/R1 backward, as appropriate:
controller.step(optimizer)
# After each EMA update, using the same authenticated original reference:
verify_fixed_offsets(G, refs, optimizer)
verify_fixed_offsets(Gema, refs)
```

Set all EMA requires-grad flags false before verification. Use controller.set_enabled for G/D alternation; old set_probing_grad would violate the policy and is detected. Gema must start from the same fresh original source. A main runner still needs source authentication, masked-state checkpoint/resume, lineage and sample manifests, regularizers, EMA, complete snapshot/quality checks, and memory supervision. These helpers alone do not provide that runner or prove image quality, adult-priest domain conversion, diversity, or server latency.

## Bounded evidence

`test_fixed_offset_stages.py` uses tiny randomly initialized native32 G/D only. The3 tests cover:

1. Same fake/losses and retained partial derivatives versus old all-coordinate evaluation; exact identical score pools/masks; no model-state change or accumulated `.grad`; protected-scope misuse rejection.
2. Two actual alternating adversarial, D R1, and G path Adam cycles. Nonzero protected originals/zero offsets remain exact in raw and updated EMA models and absent from Adam; permitted mapping/affine/ToRGB biases and low original rows actually change; folded outputs remain exact; explicit protected-offset corruption is detected.
3. Nonzero protected offsets and old selection metadata are rejected before parameter mutation; forbidden protected score fields are rejected.

Run the bounded checks with:

```sh
research/.venv/bin/python research/experiments/adam_native/test_fixed_offset_stages.py
```
