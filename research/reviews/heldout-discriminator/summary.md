# Step-1000 held-out discriminator result

The paused discriminator assigned lower raw scores to this twelve-image held-out set than to its training portraits in both matched evaluation contexts. This is evidence of separation between these particular sets, not proof of memorization, a calibrated probability, or a generator-quality measurement.

| Context | Split | N | Mean logit | Median | Positive fraction | Sign mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Batch one | training | 110 | 0.019 | 0.078 | 50.9% | 0.018 |
| Batch one | heldout | 12 | -2.555 | -2.692 | 0.0% | -1.000 |
| Three common anchors + target | training | 107 | 1.644 | 1.673 | 95.3% | 0.907 |
| Three common anchors + target | heldout | 12 | -0.435 | -0.486 | 33.3% | -0.333 |

The training-minus-held-out mean difference is **2.574 logits** for batch one and **2.079 logits** with common anchors. The large absolute shift between contexts confirms that the minibatch-statistics feature materially affects these scores. Neither context is numerically interchangeable with training logs, which used augmented random minibatches (saved augmentation probability 0.57424). Batch one has zero sample variance and the implementation’s 0.0001 epsilon floor; the anchored context excludes its three anchor portraits from the scored training targets.

## Padding strata

| Context | Reflected area | Training N / mean | Held-out N / mean |
| --- | --- | ---: | ---: |
| Batch one | ≤5% | 97 / 0.028 | 8 / -2.449 |
| Batch one | >5% | 13 / -0.043 | 4 / -2.767 |
| Common anchors | ≤5% | 94 / 1.596 | 8 / -0.384 |
| Common anchors | >5% | 13 / 1.989 | 4 / -0.537 |

The mean gap remains in the lower-padding subset: about **2.477 logits** for batch one and **1.980 logits** with common anchors. Thus removing only the four >5%-padding held-out cases would not erase this observed separation. These coarse strata are not a causal adjustment: pose, framing, prompt, generation style, source selection, and subtler padding differences remain. All twelve held-out images were retained in the primary aggregate.

## Every held-out image

| ID | Reflected area | Batch-one logit | Common-anchor logit |
| --- | ---: | ---: | ---: |
| 001 | 0.09% | -1.214 | 0.108 |
| 002 | 11.27% | -4.122 | -1.125 |
| 003 | 4.95% | -3.806 | -1.237 |
| 004 | 5.33% | -0.554 | -0.098 |
| 005 | 16.26% | -4.306 | -1.054 |
| 006 | 9.67% | -2.086 | 0.130 |
| 007 | 2.34% | -0.187 | 0.861 |
| 008 | 1.00% | -3.298 | 0.041 |
| 009 | 0.71% | -1.991 | -0.608 |
| 010 | 3.62% | -1.591 | -0.365 |
| 011 | 0.00% | -3.689 | -1.043 |
| 012 | 0.00% | -3.815 | -0.826 |

## Scope and provenance

This small, nonrandom synthetic hold-out cannot separate memorization from distribution shift by itself. There is no base-discriminator comparison, no generated/fake-image comparison, and no broad validation set in this diagnostic. A further discriminator-training decision should therefore combine this limited signal with actual unseen-latent generator previews and matched held-out data; it should not treat the score gap as an explanation already established.

Checkpoint: `research/runs/aligned110-frozen4/resume.pt`, step 1000, SHA-256 `7ef7338bd97a36cc48e783bb113ca0d2d7e1b7e58186cefe3a245da42f4e09eb`. The hash was unchanged before and after evaluation. The 110 training image hashes match the checkpoint config and accepted alignment manifest. All twelve held-out source/output hashes match their reviewed provenance and differ from training aligned hashes.

CPU only, two threads, no gradients/optimizer/MPS, 98.38 seconds measured evaluation/load time, 2.441 GiB peak process RSS. The preflight had at least 20% free memory; see `memory-before.txt`. Complete per-image training and held-out scores, summaries, padding, source URLs and file hashes are in `results.json`; the method and reproduction command are in `README.md`.
