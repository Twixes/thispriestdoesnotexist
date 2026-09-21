# Full-frame restored-image CLIP extraction

Prepared extractor; no model execution was performed during preparation. Uses the existing isolated selection environment and locally pinned official OpenAI CLIP ViT-B/32 source and weights. Four metadata-only tests passed; existing completed32 input provenance and eight source/weight hashes were checked without importing Torch.

Run once into a new directory:

```sh
research/selection/.venv/bin/python research/experiments/restored_selection/embed.py \
  --input research/restoration/runs/reference625-development128-v1 \
  --output research/selection/restored-development128
```

Repeat `--input` to concatenate completed cohorts in argument order. Each cohort must have successful completion/supervisor records, consistent attempt/error counts, and matching per-image records. Successful restoration files must retain their recorded SHA256. The extractor decodes the actual final `restored-1024.webp`, converts RGB, and applies official full-square CLIP preprocessing. It does not select, restore, crop faces, or train.

`features.npz` contains normalized float32 `features[N,512]`, bool `valid[N]`, int64 `feature_row[N]`, and the15 normalized text embeddings. Every attempted restoration retains one row, including failures. Invalid rows contain NaNs and have `valid=false`; JSON `embedding_index=null`. For success `embedding_index=feature_row`. `result.json.rows[i]` maps directly to NPZ row i, with original index/cohort, source/WebP/decodedRGB hashes, errors, and exact benchmark prompt cosine scores and margins. These margins are not age probabilities or quality approval. Filter `valid` for fitting; report the full attempted denominator.

CPU1FP32, image batch1, text batches<=4. Supervisor enforces600 seconds from initial preflight and sampled6GiB worker RSS; worker records/checks peak RSS. Initial memory gate25% free. Only its own new process group is killed/reaped on failure/interrupt. Partial logs remain. Run after root schedules available resources. Source/weight/runtime/input hashes are checked before and after execution; no downloads or environment modifications.
