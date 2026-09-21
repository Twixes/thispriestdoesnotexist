# Prepared main-adaptation native10 smoke

Prepared only. No native pretrained model was loaded and no main-adaptation training was launched. Current live probing work must finish first. No completed importance1000 input exists yet, so the prospective native input gate has not been exercised with real successful FI artifacts.

After a completed step500 output-rank1 fixed-offset probing run, successful four-pair FI diagnostic, completed fixed-policy 1000-pair EMA estimate, numerical/stability review and a recorded quantile decision, root may schedule:

```sh
research/.venv/bin/python research/experiments/adam_native/main_adaptation_fixed_offsets10.py \
  --importance-run research/runs/CHOOSE-THE-ACTUAL-COMPLETED-FIXED-POLICY-IMPORTANCE1000-RUN \
  --quantile 50 \
  --output research/runs/adam-native1024-fixed-offsets-main-adaptation10-q50-v1
```

The placeholder is intentional: do not create mock FI artifacts or infer ranking completion from the pending probe. Quantile50 is a first development hypothesis taken from the upstream Babies recipe, not a priest-specific result. The program requires an explicit quantile and rejects degenerate all-equal/empty-high pools. It authenticates the FI supervisor, result, protocol, marker, sample plan, checkpoint lineage, all pinned sources, actual1000-count accumulators and fixed-offset policy; the source checkpoint must be step500. It normalizes the saved squared-gradient sums once by actual count, then saves the generated G/D selections with the protocol. Ranking stability and visual quality remain independent review decisions.

The worker reloads the original G_ema and original D, installs fresh output-rank1 factors, installs masks, and creates new Adam states, RNG and a zero path target. It does not load probing weights or optimizer state. After controller installation, four native G outputs and one D response must match the original source exactly. G EMA starts as a copy of the initialized masked G; main adaptation tracks G EMA only, matching the upstream main script. All34 noise/activation-bias offsets stay zero and outside gradient/Adam state; their original tensors remain exact. Selected low-score original rows can now fine-tune, unlike probing.

Ten iterations retain the original logistic/non-saturating loss, style mixing0.9, horizontal flips, R1 at0 and path at0/4/8, fixed learning rates and source G EMA decay. Actual totals must be13 G and11 D Adam updates. Every adversarial and regularizer backward uses the controller's protected-row/moment checks. The two ordinary networks are alternated through controller gradient flags, preserving D's image derivative for G.

The runner stores all four raw and all four EMA previews at0/10, initial equality, exact parameter/optimizer inventory, training metrics and a terminal atomic checkpoint. The checkpoint contains model/Adam/RNG/path/count/mask states with runtime/protocol identity and a perturb/restore equality check. There is no CLI resume path: a later continuation must authenticate and restore the full state using a separately verified runner. Both raw and EMA models are folded separately and each must exactly match all four native evaluation outputs; neither export is automatically deployed.

Resource guards are35% available memory before launch,20% runtime,12GiB owned-process-tree RSS,512MiB swap growth and1200seconds. The first ten iterations are a new memory test because original-weight gradients and two full Adam moment tensors are now allocated wherever a weight tensor has low rows. Probe memory/timing does not establish this stage's resource use. No concurrent heavy worker, automatic restart, lower-resolution fallback, objective reduction or manual application deployment is part of this runner.

`test_main_adaptation_fixed_offsets10.py` uses the actual `AdaptationEngine` with tiny randomly initialized native32 networks, not downloaded weights. Four tests cover: ten real iterations with exact regularizer/count schedule and moving low original rows; exact raw/EMA folding; serialized state restored into separately initialized controllers with exact next-update equality at global16 (both regularizers), with reset-momentum negative control; changed masks/counts rejected before mutation and corrupted protected Adam state detected; absent FI and implicit quantile rejected. Tests do not prove native input authenticity, memory, photographic quality or server latency.

The implementation audit in `main-adaptation-implementation-audit.md` gives source evidence, outstanding inputs and the conditional10→100→1500 development schedule. No horizon should run automatically after this smoke. A successful mechanics smoke alone does not achieve the user's high-quality, adult male priest, no-hat, unconditional, genuinely new per-load, sub500ms server requirement.
