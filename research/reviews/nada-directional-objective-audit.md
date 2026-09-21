# NADA directional-objective audit at total100

2026-09-21. Read-only review during the existing continuation. No model imports, inference, training, downloads, or active-file edits. **I found no substantive implementation error in the directional gradient, learning rate, freezing or resume path that explains the absent collars. The stronger current explanation is insufficiently specific supervision, with important untested recipe differences. Step100 does not establish convergence or a directional-loss minimum.**

## Actual evidence

Reviewed the eight-row source/student contact sheet (displayed reduced), plus native1024 step100 gray images003 and006 and native source006. The fedora and ordinary shirt persist on003. On006, the ordinary layered shirt and neck opening remain, while facial spots, brows, facial hair and local contrast intensify; no Roman tab appears. The contact sheet retains children and hats and shows similar texture changes across identities. This is not an exhaustive native audit of all eight, nor evidence of production quality.

The first90 continuation metrics (steps11–100) show mean directional loss0.78455 at11–30 and0.68258 at81–100; edit norms range0.16087–0.63863, averaging0.42159 in the latter window. These are different fresh latents, not a matched convergence metric. Gradients and actual state changes are recorded. The objective is not near its theoretical zero, and the embeddings are not infinitesimally unchanged. Exact step10 student/Adam/RNG and16 native PNG reproductions passed before update11; step100 checkpoint readiness records successful reload/frozen-state checks.

## Comparison with primary code

Our `worker.py:100–117,261–281` keeps frozen source embeddings detached while backpropagating through student synthesis, grayscale, resize and CLIP. The unit target direction and source/student feature difference have the correct sign. The image-centroid construction and normalized directional cosine match upstream's image-guided path; our epsilon guard and one initial global-image bootstrap are explicit deviations. Upstream global loss targets text, and its optional patch term uses random510px patches with text-part directions—not corresponding target-image patches. [Official loss, functions `compute_img2img_direction`, `clip_directional_loss`, `global_clip_loss`, `patch_directional_loss`](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/criteria/clip_loss.py).

Our Adam0.0016 and betas(0,0.99**0.8) correctly apply the default4/5 ratio once. Upstream trains with fresh latents and saves the raw generator under its `g_ema` key; adding an EMA is not a missing required operation. Its example uses batch2, a ViT-B/32+ViT-B/16 ensemble and301 iterations. Our batch1 and single encoder reduce supervision/averaging, so equal update counts are not equal recipes or exposures. [Official loop](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/train.py).

Opening constant plus all17 convolution modules, including their affines/noise strengths, while freezing mapping/ToRGB corresponds to the SG2 manual default in intent. It does not accidentally restrict training to fine texture blocks. Disabling adaptive layer selection is a deliberate difference from its enabled parser default. The official SG2 forward normally randomizes noise; ours uses constant shared noise. Neither changing this nor unfreezing ToRGB is an evidenced collar fix. NVIDIA student `.eval()` still permits gradients; no dropout/BatchNorm training behavior is being silently lost here. [Official wrapper and selector](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/model/ZSSGAN.py).

Our full-square224 bicubic/antialias resize and CLIP means/std are appropriate; a center crop after resizing an already square image adds nothing. Tensor versus PIL interpolation is not bitwise identical, but our generated/source/target branches use one differentiable image path. **Grayscale everywhere is a substantive task variant**, not upstream RGB preprocessing. It discards useful clothing/background color cues and leaves skin/contrast available to optimize. It does not, however, delete black/white collar contrast. [Official CLIP preprocessing](https://github.com/openai/CLIP/blob/main/clip/clip.py#L71-L78).

The options explicitly identify nondirectional losses as unused in the paper and default their weights to zero. A single ViT-B/32 is allowed; image-centroid source-bank default is16 versus our32. Our larger fixed source bank is not itself a defect. [Official options](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/options/train_options.py).

## What the objective does not require

Algebraically `1-cos(delta,d)` is homogeneous in a free nonzero delta: it does not directly require reaching the target centroid or attaining any edit magnitude. Because both image features lie on the unit sphere, not every rescaling of delta is feasible; claiming an arbitrarily tiny edit is an exact achievable minimum would overstate this argument. Here the observed losses and norms do **not** demonstrate such a minimum. The practical problem is that a feature direction can be pursued through eyebrows, facial hair, contrast or background without the visually required shirt/tab geometry.

The24 target mean changes multiple correlated attributes relative to FFHQ: age, male appearance, attire, composition, monochrome texture and background. Their individual contributions have not been measured. The images support facial/texture movement; they do not prove that the centroid is mathematically dominated by gender/background. At224 input, a narrow tab may occupy only a small part of a32px patch; the encoder can represent subpatch information, but there is no spatial collar guarantee. Extending iterations alone may amplify the easiest attributes.

## Strongest next single experiment if500 still lacks collars

**Change only the guidance direction to a clothing-specific text contrast**, restarting from the same successful checkpoint010 with the same first100 fresh continuation latents, Adam/RNG, eval8, LR/layers/preprocessing and no bootstrap. Retain the same directional objective. Fix a small photo-only template set before running, and compute the normalized mean difference between unit text embeddings for `a man wearing an ordinary shirt` and `a man wearing a black clerical shirt with a white Roman collar`. Both descriptions retain the same subject; no hotness/age/background wording is added. This tests whether the unpaired image-centroid semantics are the bottleneck without simultaneously changing losses, encoders or capacity. Text-direction adaptation is the central upstream method; photo-only prompt selection is our controlled domain choice. [Paper](https://arxiv.org/abs/2108.00946), [official template implementation](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/utils/text_templates.py).

Review at50/100 against the existing matched image-guided checkpoints. Success requires new recognizable clerical shirt/tab geometry on several previously ordinary-shirt outputs, preserved facial anatomy/detail, and no worsening texture pattern; a better CLIP score or darker clothes alone fails. Text supervision may still ignore the small tab or produce generic clerical associations. If it fails, do not infer that more global CLIP optimization will solve geometry.

Adding global image-centroid cosine would supply endpoint attraction, but is **not** the upstream global-text term and could intensify the same unwanted facial/texture bias. Adding both global and directional terms is supported as an experimental code option, not established as the paper's successful recipe. A collar-specific image-crop loss would provide more direct spatial supervision but is a new objective requiring consistent neck/chest crops and separate evidence; upstream animal-oriented random-part templates do not justify calling it an official image-patch method. I would not stack either loss before the single direction-isolation experiment. No new implementation or longer run is authorized by this audit.

## Local evidence hashes

```json
{
  "research/experiments/nada_clean24_continue500/worker.py": "2ede6bc1cf632c91d04688e160d95615a5ee32b693bbbb089feebff782080b9d",
  "research/runs/nada-clean24-cpu-continue500/checkpoint-100-review-ready.json": "a8cf0d1531c810b678830246e0995a04639815ff9230bbf33ed881b06f73c370",
  "research/runs/nada-clean24-cpu-continue500/preview-100/source-student-native.png": "d5e00bf128684a7ae136bc88a59d653c82a08aed62747ffb5b7178dba0aef858",
  "research/runs/nada-clean24-cpu-continue500/preview-100/003-student-gray.png": "349485dbe773c89971cbab9fa70f62444db1ca4997f970f436432bd5cf138379",
  "research/runs/nada-clean24-cpu-continue500/preview-100/006-student-gray.png": "bea29e0d5e90ed227f00330f851fae4c420421aba5a40e9a133863a092237039",
  "research/runs/nada-clean24-cpu-smoke10/preview-000/006-source-gray.png": "3e3a70d24e2b282b5a513157ae431d6cbdf0a3e0d8a48db03c3e7086b2ac4f98"
}
```
