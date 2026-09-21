# StyleGAN resource reference

Checked 2026-09-21 against primary sources. The relevant original research is
NVIDIA's *A Style-Based Generator Architecture for Generative Adversarial Networks*
(Karras, Laine, Aila; CVPR 2019), rather than a separate website paper.

## Original training from scratch

- [Paper, Appendices A/C](https://arxiv.org/html/1812.04948v3): FFHQ contains
  70,000 aligned 1024×1024 photographs; training took approximately one week on
  an NVIDIA DGX-1 with eight Tesla V100 GPUs. FFHQ training used 25 million real
  image presentations, repeatedly sampling the dataset, not 25 million unique photos.
- [Official implementation](https://github.com/NVlabs/stylegan#training-networks):
  configuration F at 1024 takes 6 days 14 hours with eight V100s, or 41 days 4 hours
  with one. The eight-GPU figure corresponds to 1,264 GPU-hours for a training run;
  it does not include the whole research project's experiments.
- [Dataset release](https://github.com/NVlabs/ffhq-dataset): aligned 1024 PNGs occupy
  89.1 GB. All 70,000 images were used for training in the original paper.

## Transfer-learning reference and this project

[Official StyleGAN2-ADA timing guidance](https://github.com/NVlabs/stylegan2-ada-pytorch#expected-training-time)
says 1,000 kimg is often enough for transfer learning: **one million image
presentations**, not 1,000 optimizer steps. At that duration, its V100 benchmarks
are 6h36m at 256, 21h03m at 512, and 44h at 1024 using one GPU. These are different
hardware/trainer settings and do not guarantee results with our small dataset.

We reuse pretrained FFHQ weights, so we need not repeat scratch training. Our
current target dataset has 110 synthetic portraits at 256×256. The rejected pilot
had 50 portraits and stopped after 1,000 updates × batch 8 = 8,000 presentations.
The aligned110 CDC run was stopped after preserving checkpoint 4,750 (38,000
presentations), because its reviewed faces remained unsuitable. Its planned
6,000 updates would have amounted to only 48,000 presentations, so this was a
feasibility experiment, not a full convergence budget.

At step 580, the earlier custom-loop MPS run logged 1,408.46 seconds for 4,640 image
presentations. A straight-line extrapolation to one million is about 84 hours
(3.5 days), excluding changes in method, contention, interruptions and additional
experiments. This is an estimate for that earlier custom loop at 256 resolution, not a prediction
of visual quality or a benchmark of either the CDC variant or the full reference-phase
trainer. The new reference256-paper-b64 baseline includes both PL and R1 with the
upstream augmentation pipeline; its measured throughput must be reported separately.
Its planned 15,625 batch-64 updates equal one million presentations.

The new loop's first measured complete regularization cycle (steps 17–32)
processed 1,024 presentations in 585.61 seconds, including 16 Gmain, 16 Dmain,
four PL and one R1 phase. A linear extrapolation is **6.62 days per million**
while the CDC job shares the local GPU. This is only one early cycle, excludes
checkpoint/preview overhead, and may change with load and training state.
See `runs/reference256-paper-b64/throughput-step32.json`; do not substitute
the old custom-loop timing or treat either estimate as a quality guarantee.

After the CDC GPU job stopped, two complete reference cycles (steps 385–416)
processed 2,048 presentations in 917.77 seconds while the paired-factorial CPU job
ran alongside it. Both cycles contain 16 Gmain, 16 Dmain, four PL and one R1 phase;
the combined extrapolation is **5.19 days per million**. This is an updated
throughput observation, not a completion promise, and excludes checkpoint and
preview overhead. Exact selected metric rows and the measurement script are
stored in `runs/reference256-paper-b64/throughput-step416.json` and
`measure-throughput-416.py`.

The earlier short-run expectations were too optimistic. Longer training alone
does not establish that facial artifacts or lost diversity will recover. Assess
preservation of pretrained facial quality, target-data diversity, and adaptation
stability before committing to a long run. Final displayed quality also requires
evaluating higher resolution; 256 has sixteen times fewer pixels than 1024.

Training cost and ongoing inference/hosting cost are separate. No paid GPU run
is authorized or launched by this research note.
