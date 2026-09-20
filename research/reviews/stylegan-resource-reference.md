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
The paused aligned110-frozen4 run's planned 6,000 updates amount to only 48,000
presentations, so it is a feasibility experiment, not a full convergence budget.

At step 580, the earlier custom-loop MPS run logged 1,408.46 seconds for 4,640 image
presentations. A straight-line extrapolation to one million is about 84 hours
(3.5 days), excluding changes in method, contention, interruptions and additional
experiments. This is an estimate for that earlier custom loop at 256 resolution, not a prediction
of visual quality or a benchmark of either the CDC variant or the full reference-phase
trainer. The new reference256-paper-b64 baseline includes both PL and R1 with the
upstream augmentation pipeline; its measured throughput must be reported separately.
Its planned 15,625 batch-64 updates equal one million presentations.

The earlier short-run expectations were too optimistic. Longer training alone
does not establish that facial artifacts or lost diversity will recover. Assess
preservation of pretrained facial quality, target-data diversity, and adaptation
stability before committing to a long run. Final displayed quality also requires
evaluating higher resolution; 256 has sixteen times fewer pixels than 1024.

Training cost and ongoing inference/hosting cost are separate. No paid GPU run
is authorized or launched by this research note.
