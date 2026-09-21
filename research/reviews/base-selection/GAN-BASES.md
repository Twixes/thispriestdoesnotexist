# Pretrained GAN bases for the next bounded comparison

**Keep original StyleGAN2 FFHQ1024 as the easiest adaptation baseline and compare it with the downloaded official StyleGAN3-T FFHQ1024.** Both generate a fresh latent in one forward pass. No candidate has yet been measured against the requested <500 ms complete-server generation target; training-throughput figures are not inference latency. GPU hosting removes the reason to sacrifice native detail by starting from 256px.

This is a source/compatibility review, not a new visual benchmark. No model was executed in this task. Image quality, collar space and identity diversity still require an unfiltered, matched-size sample and direct native-resolution inspection.

| Candidate | Distinct from existing model | Why consider it | Cost/fit caveat |
|---|---|---|---|
| Original NVIDIA StyleGAN2 FFHQ1024, already present | Baseline | Existing loader and NADA/training code work with this architecture; native1024, broad FFHQ face prior | Tight face crop; adaptation must add clothing without damaging faces; mixed age/sex/headwear prior |
| NVIDIA StyleGAN3-T FFHQ1024, downloaded | New alias-free generator architecture and independently trained weights | Direct comparison at same resolution/domain; translated features rather than image-coordinate-locked texture | Requires new layer-selection adapter; reference CPU operators are slower/memory-heavy; no measured quality advantage here |
| NVIDIA StyleGAN3-R FFHQ-U1024 | Rotation-equivariant generator plus unaligned dataset | More native orientation/position variation | Not inherently a wider chest crop; added integration/compute is hard to justify before T evaluation |
| Official StyleGAN-XL FFHQ512/1024 | Progressive/projected-discriminator training and different network | Strong published FFHQ results; real third architecture comparison | New dependencies and staged architecture; existing NADA layer assumptions do not apply; published FID does not predict priest adaptation |
| NVIDIA EG3D FFHQ512 | Tri-plane rendering and camera conditioning | Multi-view-consistent face/pose variation | Camera-conditioned adaptation and neural rendering complicate a single still-portrait task; no clear need for 3D |
| NVIDIA StyleGAN2 CelebA-HQ256 | Different pretrained dataset, same model family | Useful domain-prior ablation | 256px ceiling and celebrity headshot cropping do not solve native1024 clothing detail; lower priority |

## Framing and diversity

FFHQ covers ages, ethnicity, backgrounds and accessories, including hats. Those features are part of the useful pretrained diversity and also require controlling the requested adult, male-appearing, bareheaded domain. Switching between FFHQ-trained architectures does not itself supply priest collars or remove children/hats. [Dataset source](https://github.com/NVlabs/ffhq-dataset/blob/4826aa6ea77aa7f1a7802b938ed7c40afb985cda/README.md).

FFHQ-U is **not a shoulders-and-chest dataset**. Its preparation disables rotation alignment and adds random translation; both branches retain the same eye/mouth-based crop scale. It may move clothing into view on some samples but cannot be assumed to supply more collar space overall. This is an inference from the pinned crop implementation, to be checked on samples. [Crop code](https://github.com/NVlabs/ffhq-dataset/blob/4826aa6ea77aa7f1a7802b938ed7c40afb985cda/download_ffhq.py#L297).

The existing adapted reference625 model's family resemblance is not evidence that the untouched FFHQ base has the same limitation. Compare untouched bases first, then measure how much diversity survives adaptation. Do not infer distinct identities from image hashes or select only flattering seed examples.

## Hugging Face provenance

The [huggan/StyleGAN3 Space](https://huggingface.co/spaces/huggan/StyleGAN3/blob/586431359b6a5363f94c10086435838972538c4a/app.py) loads NVIDIA's NGC checkpoints. It is a convenient discoverability/demo surface, not a separately trained Hugging Face model. Its code clones unpinned main and reloads per request, so copying that serving code would not be a latency design. The model API path `huggan/StyleGAN3` returned HTTP401 during this check; the public Space and pinned source were accessible. Search results were archived, but no random-uploader pickle was downloaded or executed.

## Local compatibility

The current vendored runtime is StyleGAN2-ADA. Our NADA worker explicitly selects `synthesis.b4.const`, `block_resolutions`, and seventeen convolution blocks. StyleGAN3 instead uses a Fourier input plus `layer_names`; passing its checkpoint into the current worker would fail or train the wrong subset. Use the official pinned StyleGAN3 runtime in an isolated import path, then implement and review a model-specific trainable-layer policy.

Official `filtered_lrelu` dispatches to a reference implementation for non-CUDA tensors; upstream describes that path as slow and memory-inefficient. CPU evaluation is possible in principle, but must have its own RSS/time supervisor. MPS compatibility is **unverified**, including gradients and large intermediate convolutions. Native CUDA custom kernels are the intended fast path. These are source findings, not executed compatibility claims. Relevant pinned files are in `gan-sources/stylegan3/`.

## Download and licenses

Downloaded one checkpoint only, from the official NGC version1 endpoint:

- `research/models/stylegan3-t-ffhq1024/stylegan3-t-ffhq-1024x1024.pkl`
- 294,775,112 bytes; SHA256 `efd9fa1f967a11b5390399a8ed512dc32e341c245fa2d4dafa6d94e96222085b`
- Source revision `c233a919a6faee6e36a316ddd4eddababad1adf9`.
- `download.py` now enforces exact byte count and SHA256 on future retrievals. No pickle content was executed.
- Root `*.pkl` Git LFS rule covers the weights. Checkpoint provenance and complete license are adjacent.

NVIDIA's StyleGAN3/EG3D licenses restrict use to **research/evaluation**, and require retained notices; these weights are not MIT merely because the surrounding application is MIT. StyleGAN-XL's top-level repository license is MIT, but inherited NVIDIA files retain their own notices: inspect the actual downloaded source/model terms before assigning blanket permissive licensing. No candidate is marked approved for production. [StyleGAN3 license](https://github.com/NVlabs/stylegan3/blob/c233a919a6faee6e36a316ddd4eddababad1adf9/LICENSE.txt), [StyleGAN-XL license](https://github.com/autonomousvision/stylegan-xl/blob/4241ff9cfeb69d617427107a75d69e9d1c2d92f2/LICENSE.txt).

## Next discriminating experiment

Generate an unfiltered small native sample from SG2 and SG3-T at psi1 and a predeclared moderate truncation level. The same integer seed does not imply the same identity across architectures. Review adult/bareheaded yield, face integrity, visible neck/collar/chest space and variety. Benchmark warmed batch-one CUDA inference, encoding and quality-selection latency separately, then end-to-end. Keep cold-start latency visible. Only then choose the base for a matched short adaptation trial with preservation losses; better source FID alone is insufficient.

The complete candidate URLs, pinned source inventory and download checks are in [gan-candidates.json](gan-candidates.json) and [gan-sources/manifest.json](gan-sources/manifest.json). This task consumed no rented compute and made no production changes.
