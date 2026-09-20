# Research licenses and provenance

The root MIT license covers this project's own application and orchestration code, not the third-party works listed here.

- `vendor/stylegan2-ada-pytorch/`: NVIDIA, pinned upstream commit in `sources.json`. Copyright notices and `LICENSE.txt` are retained. The NVIDIA Source Code License permits research/evaluation use only; it is not MIT or a general commercial-use license.
- `models/ffhq256.pkl` and derived checkpoints: NVIDIA FFHQ StyleGAN2 research weights and derivatives, subject to that upstream license. The personal site is a noncommercial generative-model research demonstration. A commercial deployment or redistribution under MIT would require a separately licensed model.
- `vendor/diffaugment/`: MIT Han Lab's DiffAugment implementation, BSD-2-Clause, with its original license retained.
- `data/priests256/`: deterministic resized copies of the project's own synthetic portraits, whose source prompts and original files are in the repository. No photographs of real priests were scraped.

Sources, checkpoint hash, and paper links: `sources.json`.
