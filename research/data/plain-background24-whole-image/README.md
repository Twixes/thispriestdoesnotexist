# Whole-image grayscale comparison

This is a preprocessing candidate for research, not an accepted training dataset or model output. All24 original portraits141–164 are retained in ascending order, with no selection or removal.

`1024/` and `256/` each contain24 RGB PNGs. Each comes directly from its original via Pillow `convert('L').convert('RGB')` followed by a whole-image LANCZOS resize. There is no affine transform, crop, reflection, padding, inpainting or other restoration. The256 copies do not pass through1024 first. Original SHA256 hashes were identical before and after processing.

`manifest.json` records source/output/script hashes, native dimensions, Python/Pillow/zlib versions and PNG options. All48 outputs were read back to check dimensions, RGB mode and exactly equal grayscale channels. `contact-256.png` is a labeled6×4 diagnostic sheet; its labels are outside the image tiles. `review.json` records framing and diversity observations. All heads and collars remain visible;144 has a tight inherited upper margin. Similar-looking face families are reported without removal.

From the repository root, reproduce into a NEW directory:

```sh
python3 research/reviews/build-plain-background24-whole-image.py --output research/data/plain-background24-whole-image-repeat
```

The script refuses an existing destination. It requires Pillow and never loads a learned model. Generation originals and active training inputs are unchanged. Root must choose preprocessing and approve inclusion separately.
