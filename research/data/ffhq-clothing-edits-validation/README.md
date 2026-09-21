# Reserved025 validation-only clothing edit

`025.png` is the raw1254px built-in imagegen output from the exact stored v1-025 sourcePNG. It is reserved for edited validation only and was not added to any training proposal. No source-model forward pass, training, or original image mutation was performed.

The full prompt, actual built-in output path, source/target/latent hashes and generation method are in `025-prompt.json`; `requested-prompt.txt` is the exact requested text. Native source and output were visually reviewed. The face, hair, three-quarter pose and slightly open mouth remain recognizable; the black shirt has a clear white Roman tab with an unobscured neck. No hats or additional people appear. Fine face texture and mouth placement are not identical to the source.

**Geometry threshold failed. Root reviewed and accepted only a qualitative held-out clothing reference; exact face/pixel ground truth is rejected.** On matched grayscale inputs, the maximum YuNet landmark displacement is0.01697208 of image size, exceeding the existing0.015 threshold. Mouth-center y shifts−0.01455316; interocular distance changes only+0.0059567%, inside its5% tolerance. The strongest failure is one mouth corner. The threshold was not widened, and no regeneration was attempted. This could remain useful for a strictly clothing-region held-out comparison if explicitly reviewed, with facial preservation measured against the original source; it is not an exact full-image identity target.

`measure.py` reuses the existing geometry helper and local YuNet model, CPU1,256px PillowLANCZOS, matched grayscale controls on both inputs. `geometry.json` includes raw-color and grayscale controls, detector/library/source hashes and all landmarks. `comparison.png` shows RGB source, grayscale source and raw edit; `review.json` records visual findings and unchanged paired7/paired10 manifest checks. No chin landmark exists in YuNet.

Reproduce geometry only with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 research/alignment/.venv/bin/python research/data/ffhq-clothing-edits-validation/measure.py`. This detector pass does not load the GAN or produce new portraits. Third-party FFHQ model/source restrictions remain as documented in the exact source bundle; repository MIT does not relicense upstream weights.

`root-review.json` records the explicit root decision with exact image hashes: recognizable face but changed mouth, nearby texture and chin. No training or mask additions are authorized. `review.json` preserves the earlier pre-root assessment.
