# Paper256-sized batch and original minibatch-statistics smoke

The fresh **batch64 / microbatch8 / PL shrink2 / mirror** MPS test passed exactly one full four-phase update with finite-update assertions. No long run was launched. `trainer.py` was unchanged at hash `00b5266875dbe07169025eb7b38b22a412230019c52d1bae77eeb8daf0045b13`.

The choice restores paper256's global batch 64 and the loaded discriminator's saved **mbstd_group_size=8**, while retaining eight accumulation rounds. Mirroring matches the source's mirrored training setting. This is a source-configuration fidelity check, not an unconstrained hyperparameter sweep. Other reference defaults remain unchanged: gamma1, base LR .0025 with lazy compensation, PL weight2, G/D intervals4/16, full bgc, initial p0, EMA20 kimg, and no FreezeD.

There are **110 original PNG portraits**. `--mirror` exposes **220 virtual dataset entries**, one original and one horizontal reflection per portrait. These are not 220 independent identities. The completed update presented **64 real examples** to the training loop; all four phases used eight microbatch rounds with the unchanged reference gain rule.

## Result and resource measurements

Fresh preflight reported **54% memory free**, exceeding the required 25% gate. The former no-CDC control was already paused by the parent; CDC continued. A mid-run check reported 36% free. No OOM occurred.

| Phase | Seconds |
| --- | --- |
| Gmain | 10.382 |
| Greg | 42.487 |
| Dmain | 11.387 |
| Dreg | 128.862 |
| Entire heavy regularized update | **193.180** |
| Setup, initial/final previews, checkpoint and reload checks | **198.947** |

These are measurements of one **heavy initial update with both regularizers**, verification and synchronization while another training job ran. They are not a steady throughput estimate. The prior batch32 test had a different concurrent workload, so a speedup claim from comparing those runs would be unsupported.

Maximum observed allocator samples were **991,016,704 bytes current** and **9,115,729,920 bytes driver**. Process peak RSS was **1,939,849,216 bytes**, including checkpoint reload. Memory was sampled after each backward, optimizer step and snapshot; these are not true instantaneous device peaks, and process RSS does not count all device allocation.

All four optimizer steps passed finite-gradient/parameter and actual-update assertions. Both full `resume.pt` (685,989,929 bytes) and the safe `generator-000001.pt` loaded successfully. The lightweight snapshot was loaded with `weights_only=True`; its EMA tensors matched the full checkpoint exactly. Six initial/final raw, untruncated EMA and psi=.7 EMA grids validated as RGB 1024×1024 images. No quality approval is implied by those format checks.

## Artifacts and continuation

- Harness: `check_b64_m8_mps.py`.
- Full evidence: `b64-m8-result.json`, `b64-m8.log`, `b64-m8-preflight.txt`.
- Actual checkpoint, config, metrics and previews: `research/runs/reference256-b64-m8-smoke/`.

The step-1 checkpoint is a valid **technical** starting point for a continued reference run; select it deliberately rather than restarting the heavy initial update. A matching continuation must preserve `--device mps --batch 64 --microbatch 8 --pl-batch-shrink 2 --mirror`, the same base/data and reference recipe. The current seed and consumed virtual-sampler count are checkpointed. `--steps` means total updates, including this completed update. Exact MPS uninterrupted-versus-resume equivalence is still not established by this reload check; exact CPU continuation was independently verified earlier.

No trainer, vendor source, existing run, production model or deployment was changed by this test. Model selection still requires photorealistic faces, distinct adult identities, clerical collars and no hats; finite optimization and valid exports do not satisfy that review.
