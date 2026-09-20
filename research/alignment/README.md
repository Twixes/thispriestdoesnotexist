# Alignment audit and isolated 110-image training candidate

**Recommendation: restart from the pretrained FFHQ weights with the expanded,
partially aligned 110-image dataset. Do not continue from the visibly collapsed
step-750 pilot, and do not use the full FFHQ crop.** This is a reasoned next
experiment, not a guarantee that alignment alone will solve adaptation.

## Measured geometry

The CPU audit detected one principal face in all 50 original portraits, all 60
accepted new synthetic portraits, and 64 deterministic unadapted FFHQ generator
samples. YuNet provides eye centers, nose, and mouth corners. Values below are
fractions of the image width/height, averaged across each group.

| Dataset | Eye midpoint Y | Interocular distance | Mouth midpoint Y |
| --- | ---: | ---: | ---: |
| Pretrained FFHQ reference, 64 samples | .466 | .252 | .729 |
| Original priest portraits, 50 | .347 | .154 | .497 |
| New synthetic priests, 60 | .338 | .166 | .504 |
| Original portraits, partial alignment | .400 | .180 | .576 |
| New portraits, partial alignment | .400 | .188 | .588 |
| Full FFHQ-like alignment, originals | .476 | .250 | .721 |
| Full FFHQ-like alignment, new portraits | .475 | .249 | .725 |

The source generator's faces are substantially larger and lower than the target
portraits. The mismatch is real, not merely a visual impression from the failed
training grid. `summary.json` also records standard deviations and 10th/90th
percentiles. The 64-sample synthetic reference is a measurement of this actual
pretrained generator at truncation .7, not a claim about every original FFHQ image.

NVIDIA's recipe uses eye and mouth landmarks to orient a square crop with side
length governed by interocular and eye-to-mouth distance. We reproduce its crop
geometry using YuNet's five landmarks instead of the original 68. This is an
explicit approximation; it does not reproduce NVIDIA's padding/blurring pipeline.
The full FFHQ-like crop matches the reference face geometry closely but truncates
**107 of 108 heuristically detected white collars**. The comparison sheet confirms
that many become ordinary face crops or leave only a collar sliver at the bottom.
That would remove the requested priest appearance and is not an acceptable fix.

The partial alternative levels the face and centers the eyes at `(50%, 40%)`,
aiming for interocular distance 24%. It reduces zoom as needed to retain the
estimated top of the head and a detected collar plus four pixels of margin. The
result keeps all 108 detected collars within the frame; the two heuristic misses,
`original-013` and `synthetic-083`, were inspected individually and retain clear
white collars too. All 110 partial candidates were visually inspected on the
contact sheet; bare heads and white collars remain visible. This review accepts
preprocessing candidates, not generated model quality.

This preserves more of the requested composition at the expense of incomplete
alignment: faces remain smaller/higher than FFHQ. Reflected border padding averages
1.5% of pixels for the originals and 3.6% for the new portraits. It affects image
borders, but should still be monitored as a potential training artifact. Collar
localization is a grayscale connected-component heuristic; its counts are not
ground-truth semantic segmentation.

## Ready candidate and reproducibility

- `partial256/`: exactly 110 unique 256×256 RGB PNGs, separate from every existing
  dataset. Raster files and the ONNX detector are covered by Git LFS patterns.
- `partial256-manifest.json`: version, settings, detector and script hashes,
  source hashes, output hashes, exact affine matrices, and visual-review evidence.
- `manifest.json`: original landmarks, face boxes, detection confidence, collar
  heuristic, transforms, and padding fractions for all inputs.
- `original-contact.png`, `ffhq-aligned-contact.png`,
  `partial-aligned-contact.png`, and `comparison-contact.png`: visual evidence.
- `reference/`: deterministic generated FFHQ samples and their seed/model hash.
- `sources.json`, `requirements-lock.txt`, `vendor/`, and `models/`: provenance,
  dependency versions, retained source/license files, and local YuNet weights.

Reproduce from repository root:

```sh
uv venv research/alignment/.venv --python 3.11
uv pip install --python research/alignment/.venv/bin/python \
  -r research/alignment/requirements-lock.txt
research/.venv/bin/python research/alignment/generate_reference.py
research/alignment/.venv/bin/python research/alignment/analyze.py
research/alignment/.venv/bin/python research/alignment/prepare_partial.py
```

Generation uses two CPU threads; landmarks use two CPU threads. No GPU training
or existing-dataset modification was performed by this audit. The last command
asserts the expected 110 inputs and records output hashes. It should be rerun only
after reviewing the contact sheets if the input set or transform changes.

## Restart versus continuation, and remaining research options

I inspected `research/runs/pilot256/samples-000750.png`: its 16 samples share a
conspicuously similar face shape, swept hairstyle, cassock, and architectural
background, while many facial features remain overlaid or displaced. This is
strong visual evidence of diversity contraction, not a biometric identity test.
Restarting from the pretrained checkpoint preserves its visibly broad latent
variation and gives the corrected framing a clean baseline. Resuming a run that
already contracts many latents toward one composition risks carrying that failure
into the improved dataset.

Use the proposed corrected optimizer settings, mapping LR `.0005`, EMA half-life
`.5 kimg`, and the isolated aligned 110-image set. Compare raw G and EMA on the
same fixed seeds every 250 steps. The broader dataset and more consistent geometry
address two observed issues, but source-to-target gender, age, clothing, grayscale,
and background shifts remain. Do not claim success merely because collars appear.

If early contraction recurs, prioritize established few-shot adaptation methods:

1. **FreezeD** freezes the discriminator's lower layers to preserve reusable source
   features during adaptation. A first-three-layer ablation is a modest next
   experiment; the trainer must preserve the freeze mask when toggling gradients.
2. **Cross-domain correspondence** explicitly preserves relationships among
   generated samples during few-shot transfer. This directly targets the observed
   loss of variation. Its official implementation uses a different StyleGAN2
   codebase and CUDA assumptions, so porting its losses requires a bounded CPU/MPS
   check. It is not a ready drop-in pretrained priest model.

Neither method was trained here. Alignment transforms are only dataset
preprocessing; they are not an inference substitute, morphing scheme, or fixed
catalog claimed as a trained generator.

## Primary references and licenses

- [NVIDIA FFHQ alignment recipe](https://github.com/NVlabs/ffhq-dataset/blob/4826aa6ea77aa7f1a7802b938ed7c40afb985cda/download_ffhq.py):
  retained source is CC BY-NC-SA 4.0; its notice is retained in `vendor/`.
- [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet):
  official lightweight face detector and five landmarks; model license is MIT,
  with the complete notice retained locally.
- [Freeze the Discriminator](https://arxiv.org/abs/2002.10964).
- [Few-shot Image Generation via Cross-domain Correspondence](https://arxiv.org/abs/2104.06820)
  and [official implementation](https://github.com/WisconsinAIVision/few-shot-gan-adaptation).

The original FFHQ research weights and generated reference artifacts retain the
upstream model's research/evaluation restrictions; they are not relicensed MIT.
