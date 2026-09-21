# Prepared EMA importance evaluator

Status: prepared and tested on tiny random networks only. **No native importance measurement has run.** First finish the paper-layout native smoke and subsequent output-rank-one probing run. Keep the existing source-layout control separate. Do not run another heavy worker concurrently.

The first actual evaluation should be four pairs from a completed paper-layout checkpoint:

```sh
research/.venv/bin/python research/experiments/adam_native/estimate_importance.py --run research/runs/adam-native1024-output-rank1-probing500-v1 --step 500 --pairs 4 --output research/runs/adam-output-rank1-importance4-v1
```

This prospective input run does not exist yet. After the diagnostic succeeds and its timing/memory/state checks are reviewed, the same entry point permits `--pairs 1000` with a new output directory. Four pairs are a compatibility diagnostic, never a reliable ranking. The full estimate retains the source's nominal 1000-pair count; it still needs stability and quality evaluation before choosing masks. There is no optimization or automatic next-stage launch in this evaluator.

The input must be a hash-bound completed checkpoint at iteration 100, 250 or 500, from the output-rank-one probing protocol. It loads the parameterized **Gema and Dema**, not the raw folded generator, retaining all original EMA values. Every requested derivative is with respect to a modulation coordinate; frozen ordinary weights retain their values and do not alter those partial derivatives. Unused base-weight/epilogue derivatives are omitted from storage. The full EMA parameter and buffer state is checked unchanged at completion.

Each example uses one independent Gaussian latent, no style mixing, stochastic synthesis noise with an explicit seed, and a training-only real image with a recorded flip. A fixed 1000-pair plan cycles through shuffled groups of the twenty training IDs; the diagnostic is its exact first four pairs. This balanced plan differs from continuing the source data loader. No validation/test image or caption is read.

G uses non-saturating softplus loss through the frozen D image Jacobian. D then evaluates the same detached fake and real, differentiating their **combined** logistic loss before squaring. Accumulation is per pair, with explicit float64 sums and actual pair counts. Finite zero gradients are valid; missing/nonfinite gradients fail. The tiny-network test compares this actual phased helper against a joint graph with all parameters enabled and verifies unchanged state.

The evaluator saves the sample plan, first four unfiltered native fake images, per-pair loss/timing, and unnormalized gradient/squared-gradient sums at counts 4, 100, 250, 500, 750 and 1000 when reached. Finalized accumulator files are atomically renamed, independently reloaded and bound to the source checkpoint, protocol and sample plan. The evaluator has no resume API; partial estimates are retained as partial evidence, not labeled a completed 1000-pair estimate. A failure between G/D accumulation does not emit a completed pair/checkpoint. No selectors or thresholds run automatically.

Use the supervised entry point. CPU one thread, FP32 network computation, native1024; 35% available-memory preflight, 20% runtime floor, 12GiB process-group RSS and 512MiB swap-growth limits. Four pairs have a 1200-second cap; 1000 pairs have a four-hour cap. No automatic retry, reduced resolution or deadline extension. These are prospective limits, not timing predictions. Actual native memory/timing, importance stability and photographic quality remain unproven.
