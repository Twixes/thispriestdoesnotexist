# Prepared exact fixed-offsets100-to500 continuation

Prepared only: no native model load, forward, or training was executed. This continues the completed output-rank1 fixed34-offset stability experiment; it is not a fresh adaptation, Fisher estimate or approved quality candidate.

```sh
research/.venv/bin/python research/experiments/adam_native/continue_output_rank1_fixed_offsets500.py --output research/runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1
```

Pinned parent:

- Run: `adam-native1024-output-rank1-fixed-offsets100-v1`.
- Checkpoint100 SHA: `e8d50115da9e3be7e98e4982316d290f7379ad0339699e100c0710e355de14ff`.
- Parent protocol SHA: `fb373e6e729c012a3f0a5a7c033adfde85afd60fefe1a3da7f2963a2cfb20169`.
- Nested checkpoint-state digest: `1ca40b33ba6b1d51b55bdeb3f98f909fe2bf710b6e84edf041782891b6338a26`.
- Parent runner SHA: `1a2dddaf8610def1ea23ceb2de14773e4eaed9d239f4ddb59cc27f2f6e3945dd`.

Preparation requires the successful parent supervisor/result and completed checkpoint100 marker. It verifies checkpoint bytes, source/vendor/data pins, parent snapshot hashes and all four latent fixtures. The worker independently rehashes checkpoint100 before `weights_only=True,mmap=True` loading and verifies its complete nested state digest.

The worker constructs the same original native models solely to obtain authenticated original frozen-parameter/buffer and fixed-offset references. It installs the identical parameterization and verifies the entire modulation inventory and optimizer parameter order against the parent. It then **overwrites all G, D, Gema and Dema states with checkpoint100**, restores both Adam states, both RNG streams, path mean and cumulative optimizer counts. The live restored state must reproduce the parent's full nested digest; raw and EMA frozen originals and all34 offsets must still match their original references. Fresh adapter initialization and empty optimizer state never enter continuation updates.

Before any update, all four raw and all four EMA parent step100 images are rendered into a new `step-100/` folder and must reproduce the archived PNG SHA values exactly. This initial preview must leave global and sampling RNG states untouched. A mismatch aborts the run rather than substituting a regenerated baseline.

The loop then uses global iteration indices100 through499. The full loss/update loop, generation helper, style mixing, real-data selection, augmentation, adversarial/R1/path schedule, EMA update and fixed-offset checks remain identical to the parent. In particular, the first new iteration100 includes the scheduled G path update and does not incorrectly restart D R1 at a local index0.

| Cumulative completed iterations | G optimizer steps | D optimizer steps |
|---:|---:|---:|
|100 parent|125|107|
|101 first continuation iteration|127|108|
|250|313|266|
|500|625|532|

The continuation adds400 iterations,500 G optimizer updates and425 D updates. All four raw/EMA images are retained at250 and500 with complete manifests. Atomic checkpoints250/500 include all model/optimizer/RNG/path states, global counters, child protocol and parent lineage, with exact same-process restoration checks. Final raw-G folding must match all four outputs. Metrics cover only global steps101–500 and state their range, avoiding an implied duplicate100-step history.

All17 original noise strengths and17 original synthesis activation biases remain exact; their34 additive offsets stay zero, gradient-disabled and absent from the G optimizer. Mapping/style-affine/ToRGB offsets remain trainable. Output-rank1 layout, native1024 CPU1/batch1 and the same20 training images are preserved. No latent filtering, parameter repair, resume fallback or objective change is allowed.

Resource guards:35% initial/20% runtime available memory,12GiB process-tree RSS,512MiB swap growth and7200seconds under the supervisor's owned process group. Execution remains root-scheduled and must wait for no other heavy worker plus fresh memory preflight. Source-state restore and initial PNG reproduction still require actual native validation; the tests below cannot prove future image quality or latency.

Three tests passed in0.697seconds with tiny linear/tanh/softplus toy G/D models, CPU1. The primary test trained100 toy iterations with two Adam optimizers, EMA models, two RNG streams, a moving path target and the global lazy-regularizer schedule. It serialized the checkpoint, advanced an uninterrupted reference by one iteration, restored unrelated models using the **same restore helper used by the native runner**, and proved exact equality of all four model states, both Adam states, both RNG states, path mean and counters after the next iteration. A deliberately reset optimizer/changed path target diverged. Other tests reject bad cumulative counts before model mutation and compare the entire native loop body and generation/loss helpers by AST to the frozen parent. No native/pretrained model was loaded by these tests.
