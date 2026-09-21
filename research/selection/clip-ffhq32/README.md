# Fixed 32-image CLIP feasibility experiment

**There is a useful, cheap preliminary image signal here; no adult-only filter is validated.** Official OpenAI CLIP ViT-B/32 correctly separated the three previously labeled children from the four selected adult men, and the independent headwear comparison matched all seven known hat/bare-head labels. This is ten partially labeled synthetic portraits, not an accuracy estimate for the 32 images or unseen seeds. Unknown labels remain unknown. No attractiveness, collar, or image-quality test was performed.

## Method and evidence

`../benchmark_clip.py` scores exactly the existing 32 FFHQ1024 source PNGs, whose hashes are checked against their manifest. No images or latents were generated. Three fixed English prompts per category compare adult/middle-aged/mature man, adult woman, child, headwear, and bare head. Each category averages three raw normalized image/text cosine similarities. The adult-male margin subtracts the larger woman/child category score; the headwear margin subtracts bare-head similarity. Positive/negative signs below describe relative ranking, **not probabilities or calibrated thresholds**. Prompts were fixed before scoring; no prompt search, training, or threshold tuning followed.

The existing source README supplies the partial manual labels. We did not visually relabel the remaining images using model results. `results.json` retains every prompt cosine, category score, margin, known label, file hash, package version, source hash, and per-image duration. `scores.csv` provides a compact table; `run.log` preserves the complete run.

| Known group | IDs | Observed comparison |
| --- | --- | --- |
| Selected adult, bareheaded men | 000, 004, 025, 030 | Adult-male margins +0.0597, +0.0059, +0.0516, +0.0363; headwear margins all negative |
| Children | 008, 019, 026 | Child category wins; adult-male margins −0.0530, −0.0751, −0.0741 |
| Headwear | 021, 023, 027 | Headwear margins +0.0026, +0.0252, +0.0197 |

The small margin on adult 004 and hat 021 exposes weak separation even in these examples. Across all 32, ten have both a positive adult-male margin and a negative headwear margin; this is an unvalidated sign-based diagnostic count, not ten approved people or an estimated production acceptance rate. There are no manually labeled women in the scoped comparison, and no borderline adolescent validation. A model category is a statement about apparent presentation in synthetic pixels, not a real person's age or gender identity.

## CPU and memory

The guarded preflight measured **27% system-wide memory free**, above the required 25%, before imports/loading. Earlier 24% checks did not load the model. CPU and interop threads were both one, image batch one, text batch at most four; no MPS/GPU was used.

- Model load: **1.57 s**; one-time text encoding: **0.21 s**.
- First image including PNG decode/preprocess: **0.342 s**; remaining-image median: **0.0493 s**.
- All 32 image encodes together: **1.07 s**; total imports/load/text/images: **17.92 s**.
- Peak process RSS: **2,369,814,528 bytes (2.21 GiB)**, including cold loading and both image/text encoders. Steady-state RSS was not separately measured.

These are native macOS/arm64 measurements, not Linux VPS or Cloudflare latency/memory guarantees. Warm CPU overhead looks modest versus the existing generator benchmark. The high cold peak means the combined generator + classifier needs a new bounded memory test before choosing a 3 GiB host; adding independent peaks is not a measured combined footprint. No production integration or extra benchmark was performed.

## Provenance and limits

Official source is pinned at [OpenAI CLIP d05afc4](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6). The 353,976,522-byte official ViT-B/32 archive matches the SHA256 embedded in upstream `clip.py`: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. `provenance.json` records URLs and archive hashes. Its official loader rebuilds the official architecture on CPU from that verified archive; no third-party custom model code, unsafe pickle override, credential, or paid service was enabled. Dependencies are isolated in `research/selection/.venv`, with a pinned lock file. The upstream MIT license is retained under `vendor/`; this does not change the FFHQ generator or derived research asset licenses.

The [official model card](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/model-card.md) warns that classification varies with the chosen taxonomy and calls for task-specific testing; its reported age-classification accuracy is far from a reliable adult guarantee. This research-only experiment supports further evaluation, not deployment approval. Next useful work is an independent, conservatively labeled final-generator set with youthful/ambiguous faces and difficult hats, fixed prompts, rejection/coverage measurements, and a separate owner preference measure. If used later, image checks must examine the final adapted output, not just its source latent or unadapted FFHQ image.

To reproduce only with a newly authorized small run and sufficient memory:

```sh
research/selection/.venv/bin/python research/selection/benchmark_clip.py
```

The CLI fails before loading when memory is below 25%; it never changes active training, inference, or deployment configuration.
