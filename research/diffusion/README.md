# Pretrained base comparison

The updated goal requires genuinely fresh server-side priests, high photographic quality, and generation below500ms. GPU hosting is now acceptable. The base comparison below is followed by a real, bounded SDXL-Turbo priest adaptation. Neither is a deployed or approved generator. No rented compute was used.

## SDXL adaptation (2026-09-21)

The frozen [pilot protocol](sdxl-pilot-protocol.md) uses20 reviewed training portraits, rank16 attention LoRA and LR1e-5. Frozen text conditioning and stochastic VAE posterior parameters were cached in7.05s. A20-update smoke completed in42.90s; a separate fresh100-update run completed in173.02s. Both completed every optimizer update, retained finite values, changed1120 adapter tensors, and left every frozen UNet tensor exactly unchanged. These are local M5 Pro measurements under the current desktop workload, not dedicated GPU training benchmarks.

Each adapter was actually reloaded for inference. The [matched preview](../reviews/sdxl-pilot-preview/index.html) preserves24 development cases with original/trigger prompts for the base and each adapter:144 unique native outputs across the base,20 and100-update runs. Every case reuses exactly the same initial noise and post-noise generator state. The unknown trigger has its own baseline control. Test seeds remain unopened. Source snapshots, dataset/model/cache hashes, resource logs, optimizer states and adapters are retained under `runs/sdxl-*`.

The complete native reviews reject both one-step candidates. On the same-trigger comparison, the joint photographic-priest pass count is12/24 for the base and20-update adapter, then9/24 after100 updates. These are agent judgements, not owner approval. The20-update review found no material regression but no clear improvement;100 introduced cloudier eyes in dev04 and more airbrushed texture in legacy04/05. Ordinary LoRA training stops at100 under the frozen regression rule; no250 run or sealed test has been opened. The first four-step fallback attempt stopped at model loading when available system memory reached19.72%, below the20%reserve. It generated no images and its complete failure evidence is retained. See the [review summary](../reviews/sdxl-pilot-preview/one-step-summary.json).

Successful updates are compatibility evidence, not a photographic-quality pass. The [trainer](vendor/sdxl-train/README.md) preserves ordinary SDXL denoising LoRA rather than Turbo's adversarial distillation objective; preserved one-step quality must be checked empirically. Training seed was amended before any computation to fit NumPy's32bit requirement; the original protocol and exact one-field amendment remain archived.

The [CUDA benchmark](benchmark_cuda_sdxl.README.md) is prepared but has not run on a server. It records actual fresh-noise generation, VAE and WebP encoding, plus separate loading/warmup statistics. The [hosting comparison](../reviews/gpu-hosting-options-2026-09-21.md) finds no checked always-resident GPU below$15/month; a short isolated benchmark is inexpensive, while serverless retains cold-start uncertainty. [Photographic fast SDXL candidates](../reviews/base-selection/photoreal-fast-sdxl.md) identify specialist priors for a further matched comparison rather than assuming more Turbo training will solve realism.

The official [Juggernaut X Hyper checkpoint](models/juggernaut-x-hyper/README.md) is now downloaded:7.105GB, verified against the publisher's full SHA256, with seven at-most1GiB LFS parts and pinned offline configs/tokenizers/license. It has not yet generated a comparison portrait or received priest post-training. Its publisher supports4–8step sampling; neither that statement nor the download proves our quality or server latency target.

`evaluate_sdxl_pilot.py --cached-prompts` subsequently completed the four-step fallback for the base and both adapters:48 outputs each,144 additional native images. It saves FP16 text conditioning before releasing both encoders, then loads the image model. All48 prompt caches/96 tensor-value hashes match exactly across the three runs. This verifies matched conditioning between these runs, not equivalence to historical uncached outputs whose embeddings were not saved. The default evaluation mode stays unchanged. Startup/cache preparation is excluded from image timings and recorded separately. The35% start/20% runtime memory reserves remained unchanged; no extra GPU was used.

The [complete fallback preview](../reviews/sdxl-pilot-4step-preview/index.html) shows the stronger photographic detail obtained with four steps. Same-trigger joint passes are20/24(base),19/24(step20),17/24(step100), below the22/24 frozen threshold. Neither adapter improves aggregate quality; eye/texture issues and framing failures remain, so sealed tests stay unopened. This is a failure on the frozen broad development cohort, not proof that every SDXL prompt or adaptation recipe fails. See the [hash-bound summary](../reviews/sdxl-pilot-4step-preview/four-step-summary.json). Local median generation plus WebP for the trigger arms was673ms(base),736ms(step20),767ms(step100), with conditioning cached and research guard overhead included. These are not GPU server measurements. All288 native one/four-step development outputs, including failures of the quality gate, remain archived.

## Pretrained comparison

Eight configurations of four pretrained diffusion bases were tested locally with all eight images retained per configuration. Four fixed portrait prompts each get two fresh seeds. Both the prompts and numerical seeds match across configurations; 512/1024 have different latent tensor sizes. Source noise, exact prompts, native PNGs, monochrome WebPs, timings, source snapshots and hashes are in each run. The warmup output is retained separately and excluded from warm statistics. All measurements synchronize MPS and include the PIL image result plus separate WebP encoding; loading, warmup, HTTP, queueing, quality gates and network are excluded. Eight draws are insufficient to prove production latency tails or identity diversity.

| Base/configuration | Output resolution | Local warm median generation + WebP | Maximum of8 |
|---|---:|---:|---:|
| SD-Turbo,1step |512|222ms|254ms|
| SDXL-Turbo,1step |512|430ms|440ms|
| SDXL-Turbo,4steps |512|707ms|723ms|
| SDXL-Lightning,2steps |1024|1986ms|2013ms|
| SDXL-Lightning,2steps |512|517ms|524ms|
| SDXL-Turbo,1step |1024|1682ms|1715ms|
| FLUX.2-klein-4B,4steps |1024|10507ms|13392ms|
| FLUX.2-klein-4B,4steps |512|3592ms|3715ms|

**Leading tested speed/appearance candidate: SDXL-Turbo at its intended512px resolution.** Its eight initial outputs have coherent faces/eyes and recognizable clerical clothing, without hats. The one-step images are softer than four-step; the four-step portraits have stronger detail, sometimes overly etched skin. Both retain conspicuous within-prompt family resemblance. A larger fresh cohort must test identity variety before adaptation or serving decisions. Genuine broad clerical collars are acceptable: exact small-tab prompt matching is not a user requirement.

SD-Turbo is a useful smaller speed comparator, with coherent initial faces but less consistent clothing. Lightning's native1024 sample faces are coherent and appealing but look heavily retouched/rendered; only three of eight clearly resemble priests. Its512 outputs also look conspicuously rendered. Turbo sampled at1024 shows severe duplicate/fused face structures: increasing output dimensions is not an effective substitute for a native high-resolution model. Full native reviews are saved beside each completed run.

**FLUX.2 klein4B has the clearest documented post-training route**, but did not win this local speed/appearance comparison. Both resolutions produced eight coherent, bareheaded priests with visible collars. Native1024 has more detail, but all four prompt pairs strongly resemble the same person; the formal parish-staff aesthetic is weaker for our calendar target. The512 outputs are smoother and do not justify their slowdown over SDXL-Turbo. These are subjective agent reviews, not owner approval. Its authors explicitly endorse training a LoRA on base4B then loading it onto distilled4B; see [modern base research](../reviews/base-selection/modern-fast-bases.md). A small adaptation pilot can test this advantage without assuming published style-training budgets guarantee our result.

FLUX timing uses cached embeddings for the same four fixed prompts, with fresh image noise each time. Encoder initialization is saved separately. Other rows include text encoding per request, so these are different serving configurations rather than isolated architecture benchmarks. The1024 FLUX timing excludes CPU noise creation (see its immutable run plus timing-notes.json); the512 runner includes it. Neither has server/CUDA measurements.

The separate official StyleGAN3-T baseline reached one native1024 CPU warmup, then exceeded its12GiB RSS cap. It did not complete the8-sample comparison. Its CPU reference implementation's resource cost does not establish CUDA speed or final quality; see `../runs/pretrained-gan-baseline8-v2/`. Existing StyleGAN2 evidence remains in `../runs/inference-cpu/ffhq1024/`.

## Provenance and reproduction

`prepare_flux2_klein.py` downloads the pinned official distilled4B package. `reassemble_flux2_klein.py` verifies/reassembles its16 LFS parts; `compare_flux.py` performs offline BF16 inference with separately cached fixed-prompt embeddings and retained fresh noise. Its upstream Apache2.0 license stays with the weights.

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

Retain SDXL-Turbo512 for speed/appearance and FLUX.2-klein4B for its explicit base-to-distilled adapter route. Compare identity variety and a bounded adapter pilot before final selection. For SDXL, select a post-training method that preserves one/few-step sampling. Ordinary all-timestep LoRA fine-tuning is not proven to retain Turbo's distilled behavior. An alternative is a domain LoRA on ordinary SDXL with a released acceleration adapter; that composition still needs actual quality/latency evaluation. Do not claim post-training has happened just because the base already generates priests.

Benchmark the adapted winner on the actual GPU server including encoding and all acceptance/retry overhead. A local warm430ms result and the authors' A100207ms benchmark are encouraging, but neither proves the site's500ms bound, first-request behavior or the hosting budget. No production model is approved.

The [comparison preview](../reviews/base-comparison-preview/index.html) includes all64 outputs with native/full-size inspection. [Primary-source candidate research](../reviews/base-selection/fast-diffusion-candidates.md) covers model revisions, licenses and adaptation caveats.

## Local adapter compatibility probe

The FLUX base4B probe is recorded in `flux-adaptation-smoke-report.json`. No optimizer updates completed. The final cached/direct-MPS variant loaded the model and all20 reviewed training inputs, then the first forward pass crossed the20% system-availability floor under the current desktop workload. Memory limits were retained for that final attempt. Earlier attempts and implementation fixes are preserved separately; the staged20-image/text cache completed in11.60s. This does not establish a universal48GB hardware limit or disprove the documented CUDA training route. No rented compute or production generator was launched.

The finalized multi-identity pilot data is [20 train /5 validation /5 test](../data/flux-priest-domain-v1/README.md), with byte-exact teachers and verified per-image captions. A dedicatedGPU pilot may be economical, but local adaptation ease for this base is not yet demonstrated. Continue with SDXL-Turbo as the speed/appearance reference instead of treating documentation as a passed training result.
