# Reviewed FLUX priest domain pilot

30 native1024 synthetic teachers: **20 training / 5 validation / 5 test**. This supersedes the provisional 24/5/5 plan after conservative family deduplication. These are complete portraits, not oracle composites or edited faces. Every included image is a byte-exact copy of the corresponding image in `plain-background36-whole-image/1024`. The manifest also binds the earlier source image hash, which can differ from the native1024 derivative hash.

All 34 candidates (141–158, 160, 162–166, 168, 170–171, 173–177, 179–180) were viewed individually at native1024. Eyes, facial anatomy, texture, clerical collars, adult male appearance, absence of hats and face-family resemblance were assessed. No additional anatomical/photographic rejection was found. Four additional candidates were excluded as close face-family variants: **150→142, 151→147, 164→156, 176→143**. Prior exclusions **159→149, 161→151**, and original rejected candidates **167, 169, 172, 178**, remain excluded; they are not claimed as newly reviewed here.

The manually chosen split keeps retained representatives of known face families in training, never their near-duplicate relatives in heldout sets. This is conservative visual grouping, not biometric proof of identity independence. Shared photographic style or similar facial hair alone does not establish identity. Five-image heldout sets provide only a small qualitative check and cannot demonstrate generalization by themselves.

| Split | IDs |
| --- | --- |
| Train | 141, 142, 143, 145, 147, 148, 149, 152, 154, 155, 156, 157, 160, 162, 163, 166, 170, 173, 174, 180 |
| Validation | 144, 158, 165, 168, 175 |
| Test | 146, 153, 171, 177, 179 |

Each split is a Hugging Face imagefolder with a `metadata.jsonl` containing `file_name` and `text`. Captions begin with **PR1EST_CAL** and describe observed hair, facial hair, pose, clothing and background. Adult/older appearance is a visual description, not an age claim. Captions omit an explicit monochrome/style label so the trigger can learn the common appearance. The actual images are monochrome, predominantly tight head-and-shoulders framing, dark clerical shirts and white collar inserts; this small dataset does not cover the full diversity of real priests.

Train using **only** `--dataset_name /absolute/path/to/flux-priest-domain-v1/train --image_column image --caption_column text`. Do not pass the parent directory or use `--instance_data_dir`, which bypasses these per-image captions in the selected trainer. Keep validation/test portraits out of training, caching and augmentations. Training code must derive the count from the manifest instead of assuming 24. Training at 512 can resize in memory; original reviewed files remain unchanged.

`annotations.json` records the manual captions and observations. `native-review.json` binds those observations to source-image checksums and split decisions. `manifest.json` binds original sources, native sources, destination images, captions, metadata files, annotations, review and build script. `build.py` materializes the folders and refuses to overwrite existing outputs. Copies deliberately avoid shared-inode mutation risk. No generation, image editing, model training or production approval occurred during this dataset task.

Review acceptance means suitable for a bounded adaptation experiment, not a production-quality model. Evaluate fixed unseen generation seeds before/after adaptation, compare identity diversity, and retain the separate server generation-below-500ms gate. A recognizable legitimate clerical collar is sufficient; an exact small Roman tab is not an extra user requirement.
