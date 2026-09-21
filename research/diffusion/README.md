# Pretrained base comparison

The updated goal requires genuinely fresh server-side priests, high photographic quality, and generation below500ms. GPU hosting is now acceptable. These are **pretrained base evaluations**, not priest post-training or a deployed generator. No rented compute was used.

Six configurations of three pretrained diffusion bases were tested locally with all eight images retained per configuration. Four fixed portrait prompts each get two fresh seeds. Both the prompts and numerical seeds match across configurations; 512/1024 have different latent tensor sizes. Source noise, exact prompts, native PNGs, monochrome WebPs, timings, source snapshots and hashes are in each run. The warmup output is retained separately and excluded from warm statistics. All measurements synchronize MPS and include the PIL image result plus separate WebP encoding; loading, warmup, HTTP, queueing, quality gates and network are excluded. Eight draws are insufficient to prove production latency tails or identity diversity.

| Base/configuration | Output resolution | Local warm median generation + WebP | Maximum of8 |
|---|---:|---:|---:|
| SD-Turbo,1step |512|222ms|254ms|
| SDXL-Turbo,1step |512|430ms|440ms|
| SDXL-Turbo,4steps |512|707ms|723ms|
| SDXL-Lightning,2steps |1024|1986ms|2013ms|
| SDXL-Lightning,2steps |512|517ms|524ms|
| SDXL-Turbo,1step |1024|1682ms|1715ms|

**Leading next-stage candidate: SDXL-Turbo at its intended512px resolution.** Its eight initial outputs have coherent faces/eyes and recognizable clerical clothing, without hats. The one-step images are softer than four-step; the four-step portraits have stronger detail, sometimes overly etched skin. Both retain conspicuous within-prompt family resemblance. A larger fresh cohort must test identity variety before adaptation or serving decisions. Genuine broad clerical collars are acceptable: exact small-tab prompt matching is not a user requirement.

SD-Turbo is a useful smaller speed comparator, with coherent initial faces but less consistent clothing. Lightning's native1024 sample faces are coherent and appealing but look heavily retouched/rendered; only three of eight clearly resemble priests. Its512 outputs also look conspicuously rendered. Turbo sampled at1024 shows severe duplicate/fused face structures: increasing output dimensions is not an effective substitute for a native high-resolution model. Full native reviews are saved beside each completed run.

The separate official StyleGAN3-T baseline reached one native1024 CPU warmup, then exceeded its12GiB RSS cap. It did not complete the8-sample comparison. Its CPU reference implementation's resource cost does not establish CUDA speed or final quality; see `../runs/pretrained-gan-baseline8-v2/`. Existing StyleGAN2 evidence remains in `../runs/inference-cpu/ffhq1024/`.

## Provenance and reproduction

`download.py` pins official Hugging Face model revisions for SD-Turbo, SDXL-Turbo and an SDXL inpainting teacher. `prepare_lightning.py` assembles official SDXL components with ByteDance's full two-step UNet. The inpainting model was downloaded for offline paired-target research but **has not been evaluated or proposed as the sub500ms serving path**. Only safetensors are loaded for diffusion. Model cards and upstream licenses stay with the assets; these weights are not the app's MIT code.

Large UNets are preserved byte-for-byte as at most1GiB LFS parts to avoid individual-file limits. Reconstruct an ignored local UNet using `python3 research/diffusion/restore.py research/diffusion/models/sdxl-turbo`; it checks every part and the full upstream digest. `compare.py` also restores missing reconstructed files before local-only loading. No network calls are needed during inference.

```sh
uv venv --python 3.11 research/diffusion/.venv
uv pip install --python research/diffusion/.venv/bin/python -r research/diffusion/requirements-lock.txt
git lfs pull --include='research/diffusion/**' --exclude=''
research/diffusion/.venv/bin/python research/diffusion/compare.py --model sdxl-turbo --steps 1 --resolution 512 --output research/diffusion/runs/new-name
```

The comparison supervisor caps workerRSS at24GiB, MPS driver allocation at22GiB, wall time at20minutes, and stops below15% available system memory. It preserves all failed runs and does not overwrite output directories. The unchanged historical worker source snapshots define the exact runs above; later additions only add Lightning, resolution selection and offline LFS reassembly.

## Next decision

Compare a larger unfiltered SDXL-Turbo512 cohort and select a post-training method that preserves one/few-step sampling. Ordinary all-timestep LoRA fine-tuning is not proven to retain Turbo's distilled behavior. An alternative is a domain LoRA on ordinary SDXL with a released acceleration adapter; that composition still needs actual quality/latency evaluation. Do not claim post-training has happened just because the base already generates priests.

Benchmark the adapted winner on the actual GPU server including encoding and all acceptance/retry overhead. A local warm430ms result and the authors' A100207ms benchmark are encouraging, but neither proves the site's500ms bound, first-request behavior or the hosting budget. No production model is approved.

The [comparison preview](../reviews/base-comparison-preview/index.html) includes all48 outputs with native/full-size inspection. [Primary-source candidate research](../reviews/base-selection/fast-diffusion-candidates.md) covers model revisions, licenses and adaptation caveats.
