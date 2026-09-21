# Native unconditional GAN adaptation

## Current experiment path — 2026-09-22

The completed [fixed-offset100 run](../../runs/adam-native1024-output-rank1-fixed-offsets100-v1/result.json) preserved coherent faces without the earlier severe grain. It did **not** produce an accepted adult priest model. Its [full visual review](../../reviews/adam-fixed-offsets100/index.html) retains every fixed sample.

An exact continuation from100 to500 has [verified launch/restoration evidence](../../runs/adam-native1024-output-rank1-fixed-offsets100-to500-v1/launch-observation.json). This launch record is historical, not proof that the worker is currently live. Check its actual process and terminal supervisor/result records before taking action. The [progressive review](../../reviews/adam-fixed-offsets500/index.html) displays finalized100/250/500 snapshots after its builder is rerun. Do not restart the completed100 run or the earlier source-layout run stopped after128 iterations.

The prepared next stages are [fixed-offset EMA importance measurement and masks](fixed-offset-stages-README.md), followed by a [fresh-source native10 main-adaptation smoke](main-adaptation-fixed-offsets10-README.md). Neither stage has run on native pretrained models. Main adaptation requires actual completed1000-pair importance evidence and a separate ranking review; the four-pair diagnostic alone cannot supply trustworthy selection. The [implementation audit](main-adaptation-implementation-audit.md) distinguishes these stages and their reset/resume rules.

All17 synthesis noise and17 synthesis activation-bias additive offsets remain zero; original source tensors stay exact. This is an explicit change from upstream AdAM. The [training-geometry audit](../../reviews/adam-native-training-geometry-audit.md) records a separate framing and grayscale mismatch hypothesis; no dataset transformation or new training arm has been performed.

The public site still uses its image catalog. No trained candidate has passed quality review or been deployed, and sub500ms generation on the actual server remains unproven. The remainder of this document records the earlier initial compatibility work, not the latest training status.

This experiment ports AdAM-style importance-probing mechanics to the original NVIDIA StyleGAN2 FFHQ1024 networks. The generator takes random noise only. Its static modulation can be folded into ordinary generator weights, without a prompt, text encoder or extra rendering model.

The actual native compatibility run completed in `research/runs/adam-native1024-probing-smoke10-v2`: ten iterations,13 G optimizer updates and11 D optimizer updates including R1 and path regularization, in162.19 seconds total on CPU1. Peak monitored worker RSS was9.93GiB; available system memory stayed above34.76%. Every targeted G/D rank-factor and offset tensor had finite, nonzero gradients on every adversarial update. Frozen original parameters and buffers remained exact. Four original native G outputs and one D output matched exactly before training; checkpoint model states restored exactly; four folded native outputs matched exactly after training.

These are mechanics and resource results, not a successful priest generator. Ten probing iterations are insufficient for domain adaptation. Original FFHQ subjects, including women and children, remain in this unfiltered diagnostic. No Fisher importance estimate or main adaptation has run. No model is approved for production, and no server latency has been measured. The [checkpoint comparison](../../reviews/adam-native-smoke/index.html) retains the four fixed latents without selection.

## Reproduction

```sh
research/.venv/bin/python research/experiments/adam_native/test_modulation.py
research/.venv/bin/python research/experiments/adam_native/smoke.py --output research/runs/adam-native1024-new-smoke
```

Use an unused directory. The supervisor retains failures and guards12GiB process-group RSS,20% system availability,512MiB swap growth and20minutes, after a35% availability preflight. It kills only its own process group. Source/model/data hashes and source snapshots are saved before execution. Source CPU reference operations and FP32 are used; `fused_modconv=False` is explicit. No user apps are closed and no rented compute is used.

The training images are the existing20 native1024 portraits, selected solely by the frozen train split. Captions are ignored; validation/test images are not loaded. Batch1, original G_ema initialization and native NVIDIA forward/scaling are documented differences from the published CUDA256/batch4 implementation. G/D modulation and both EMAs follow the audited probing schedule. This is not a claim of exact reproduction of the published full method.

## Corrected attempts and limitations

`smoke10-v1` reproduced native source outputs exactly, then stopped before training because current Torch requires both Adam beta values to be floats. Changing0 to0.0 corrected the API type without changing the numerical value. Its complete failure is retained.

A stronger small-network regression found that folding a deepcopy could remove parameter properties from the original network's shared dynamic class. The corrected fold isolates that class before removing parametrizations. Seven small-network tests passed, including explicit original-model rendering after folding, second-order gradients and standard-state reload. The successful native run additionally verified four full-resolution folded outputs. Original failed source is retained with v1; [mechanics-tests.json](mechanics-tests.json) records the regression and correction.

The current checkpoint check verifies exact model-state restoration and actually reloads optimizer/RNG state. It does not claim interrupted-versus-uninterrupted training equivalence. Four fixed latents do not establish broad identity diversity, eligibility or photographic quality. Published research/code, original model licenses and every output remain archived separately from the application's MIT license.

Next: a separately frozen500-iteration importance-probing run, followed by correctly computed per-example Fisher statistics and independently audited main-adaptation masks. Main adaptation starts again from original source weights; the probing model itself is not the final generator. See [source-audit.md](source-audit.md) before implementation.
