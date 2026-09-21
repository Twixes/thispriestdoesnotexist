# Native matched500 → 625 review pages

PNG-only diagnostic tooling. It imports NumPy/Pillow and standard-library modules, never Torch, a generator, CLIP or checkpoint tensors. The pages are labeled comparison montages, **not model outputs**. No step625 pages have been created during preparation.

The existing step500 inputs were validated with:

```sh
research/.venv/bin/python research/reviews/reference625-comparison/build.py \
  --validate-baseline > research/reviews/reference625-comparison/baseline-validation.json
```

That real baseline passed: successful evaluation and supervisor,32 ordered seeds, pinned latent NPZ and metadata, all96 individual SHA256 hashes, and every PNG exactly256×256 RGB. It wrote no image or future result. The baseline JSON/NPZ hashes are embedded in the script so a changed baseline fails closed. Source PNG hashes come from that pinned evaluation. Original files are checked again before successful completion.

Only after the actual625 matched evaluation has finished successfully:

```sh
research/.venv/bin/python research/reviews/reference625-comparison/build.py --build \
  --evaluation625 research/runs/reference256-paper-b64-resumed500-to625/matched32-000625 \
  --output research/reviews/reference625-comparison/pages-500-vs625
```

The625 path defaults to the one above; override it only for the actual completed evaluator output. Both runs must report complete evaluations and successful supervisor exit0 with no error, correct checkpoint steps/presentation counts, matching evaluation/launch provenance, and stable inputs. Both must contain all96 unique variant/index images, matching32 seed order, byte-identical saved500 NPZ, and256px RGB PNGs with matching recorded hashes. The625 checkpoint hash must differ from500. No fake625 fixture, substituted500 images or inferred success is used. Missing/incomplete625 inputs fail before any page directory is created. Output must be a new directory below `research/reviews`; no overwrites or retries.

The tool produces12 pages: four each for `raw-psi1`, `ema-psi1` and `ema-psi07`. Every1040×1200 page contains eight native pairs, two pairs per row and four rows. Step500 is on the left of each pair;625 is on the right. Header and label bands carry variant, page, original index and full seed. Portraits occupy their original256×256 pixel rectangles; there is no resizing, filtering, restoration, cropping within a portrait or color conversion.

Every pasted rectangle is compared byte-for-byte against its original pixels, then checked again after saving/reopening the PNG. `manifest.json` records all192 verified native tiles, their boxes, source/page hashes, input preservation and diagnostic status. A failed build can leave partial pages but cannot publish a complete manifest. This is a matched longitudinal review of the same32 seeds, not a fresh unseen evaluation or a quality approval.

Preparation deliberately validates only genuine500 data. Layout/crop-equality assertions will run against actual625 results at execution; no unnecessary fabricated model-output fixtures were made.
