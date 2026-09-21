# Clothing-editor oracle gate

No editor training has started. The first gate failed, so no trainable model or trainer was implemented.

`prepare.py` reads the existing paired10 manifest, keeps the original six paired7 training identities and four held-out/diagnostic identities, and materializes grayscale source/teacher/mask/tab/oracle PNGs in `research/data/clothing-editor-oracles-v1/`. It does not load models. The oracle copies original source pixels exactly outside the manual allowed region and resized teacher pixels inside it. This is a supervision diagnostic, never a production model or fulfillment of fresh generation. Images are covered by repository Git LFS rules. Source/target/latent/derived-image hashes are preserved in the manifest.

Native review of all ten images found only two of six training pairs clearly adequate for exploratory garment fitting; the predeclared feasibility memo required at least four. Broad polygonal garment/background mismatches occur in b2-055 and b2-020. Several other neckline joins have obvious cutout contours. These masks were reviewed as **loss support regions** for earlier whole-G training; that never established their suitability as **hard compositing masks**. The 256px small U-Net proposal is therefore not trained on these targets.

The y=397 upper extent of b2-020's mask also falsifies a fixed bottom-half crop without mask clipping. Do not resolve that by silently discarding mask pixels or stretching the crop. A future experiment needs source-aligned teacher edits and protected/transition masks that work for whole composites before fitting a learned clothing renderer.

The updated user requirement is server generation below 500 ms, and GPU hosting is allowed. A frozen GAN plus a learned editor remains one candidate; no sub-500ms pipeline measurement exists here. The older CPU-only hosting preference is not a current requirement. The observed 4.48-second emulated CPU FFHQ1024 forward does not establish GPU latency or disqualify GPU serving. No seed bank, precomputed output reuse, or bitmap garment library should substitute for fresh inference.
