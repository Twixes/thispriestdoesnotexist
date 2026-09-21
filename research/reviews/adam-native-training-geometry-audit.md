# Native training geometry audit

2026-09-22. Read-only visual/data audit; no training change, image transformation, split change, model loading, example exclusion or quality approval.

**The target/source framing mismatch is substantial enough to test explicitly. Its effect on optimization is a hypothesis, not yet a demonstrated cause of failed priest conversion.** All 20 target portraits place the eyes substantially higher and show considerably more neck/torso than the four fixed source outputs. The targets are also exactly grayscale while the source outputs are colored. A discriminator can therefore distinguish these domains without relying on priest-specific facial or clothing semantics. These shortcuts plausibly increase the work needed from the small adapters, but this audit cannot measure their contribution or prove alignment is the main bottleneck.

## Evidence and method

I opened all 20 training PNGs and all four source `step-000/raw` PNGs individually using the native image-view tool. Observations below are from the actual images, not their captions. Each file is 1024 × 1024 RGB. All 20 target file hashes match their manifest entries. The dataset manifest SHA-256 is `6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b`.

**Geometry numbers are approximate visual estimates, not landmark-detector output or manually clicked coordinates.** Coordinates are fractions of image height measured down from the top; inter-eye distance is a fraction of width. Eye estimates are roughly within ±0.02H and collar estimates within ±0.03H. Hair and beard boundaries make face extent less precise. No face detector, perceptual model, distribution metric or training gradient analysis was run. The four source latents are a diagnostic sample, not a population estimate of FFHQ.

| Quantity | 20 target training portraits, visually estimated | Four source baselines, visually estimated |
|---|---|---|
| Mean eye-line position within a portrait | About 0.32–0.39H; most around 0.34–0.37H | All about 0.47H |
| Inter-eye distance | About 0.13–0.19W | About 0.24–0.27W |
| Face width around cheek/temple region | Approximately 0.32–0.45W, with pose variation | Approximately 0.55–0.68W |
| Lowest chin/beard point | About 0.57–0.71H | About 0.88–0.98H |
| White clerical tab | Clearly visible in every target; bottom around 0.68–0.83H | None; neck is largely at/beyond the lower edge |
| Clothing composition | Full shoulders, substantial upper chest, often pocket/placket | Close face crops; small clothing slivers or scarves |

These rough estimates imply about a tenth to a sixth of an image-height difference in eye location and materially different face scale. Exact values should be annotated before a preprocessing experiment; the discrepancy itself is visible without a detector.

## Every viewed target and local observations

Every target path below is relative to `/Users/twixes/Developer/thispriestdoesnotexist/`. Every row was directly viewed. The final two columns remain rough visual estimates, not computed landmarks.

| Exact viewed path | Composition / photographic observation | Eye line ≈ H | Collar bottom ≈ H |
|---|---|---:|---:|
| `research/data/flux-priest-domain-v1/train/141.png` | Relatively large target head; clear shirt/tab; dense but coherent facial texture | 0.35 | 0.83 |
| `research/data/flux-priest-domain-v1/train/142.png` | Larger head, tilted eyes, blurred foliage; plausible stubble and eyelids | 0.36 | 0.81 |
| `research/data/flux-priest-domain-v1/train/143.png` | Curly hair/beard, clear tab; beard strands look somewhat uniform/etched | 0.38 | 0.83 |
| `research/data/flux-priest-domain-v1/train/145.png` | Broad shoulders and large head; smooth highlights with fine cheek texture | 0.35 | 0.79 |
| `research/data/flux-priest-domain-v1/train/147.png` | Larger head, subtle tilt, clean tab; plausible wrinkles/hair, uniform studio finish | 0.36 | 0.82 |
| `research/data/flux-priest-domain-v1/train/148.png` | Receding hair and beard; clear tab; very fine bright beard strands | 0.36 | 0.83 |
| `research/data/flux-priest-domain-v1/train/149.png` | Three-quarter face/off-camera gaze, broad torso; coherent eyes and tab | 0.33 | 0.75 |
| `research/data/flux-priest-domain-v1/train/152.png` | Beard/gray hair, broad torso; clear tab and sharply delineated beard | 0.37 | 0.74 |
| `research/data/flux-priest-domain-v1/train/154.png` | Smaller head, broad torso; coherent smile and skin texture | 0.34 | 0.75 |
| `research/data/flux-priest-domain-v1/train/155.png` | Three-quarter face, broad torso; high eye highlights and uniform fine texture | 0.37 | 0.81 |
| `research/data/flux-priest-domain-v1/train/156.png` | Smaller head, substantial shirt; heavily detailed freckle/beard texture | 0.34 | 0.74 |
| `research/data/flux-priest-domain-v1/train/157.png` | Smaller tilted head, broad torso, bokeh background; very orderly hair strands | 0.35 | 0.74 |
| `research/data/flux-priest-domain-v1/train/160.png` | Smaller head, long torso/pocket; natural-looking face shape, fine whiskers | 0.34 | 0.72 |
| `research/data/flux-priest-domain-v1/train/162.png` | Turned head, broad torso; distinct tab and coherent facial hair | 0.37 | 0.75 |
| `research/data/flux-priest-domain-v1/train/163.png` | Tilted head and broad torso; coherent pupils, dense fairly even beard texture | 0.36 | 0.78 |
| `research/data/flux-priest-domain-v1/train/166.png` | Small head, large shirt/pocket area; coherent eyelids/skin, clean tab | 0.33 | 0.68 |
| `research/data/flux-priest-domain-v1/train/170.png` | Small turned head, broad chest; coherent anatomy, sharply defined beard | 0.35 | 0.71 |
| `research/data/flux-priest-domain-v1/train/173.png` | Small head, large chest; fine stubble, bright but formed pupils | 0.35 | 0.71 |
| `research/data/flux-priest-domain-v1/train/174.png` | Small head, broad torso; detailed gray hair/beard, clean tab | 0.32 | 0.70 |
| `research/data/flux-priest-domain-v1/train/180.png` | Moderately sized tilted head, broad chest; coherent mustache and age lines | 0.36 | 0.75 |

All viewed target subjects appear adult, all have no headwear, and all have a legible white collar insert. This is an annotation of domain cues, not an attractiveness score, proof of real photography, or approval of any model. The target images have a fairly consistent monochrome portrait aesthetic: soft directional illumination, clean backgrounds, and generally crisp synthetic detail. I did not observe obvious doubled eyes, broken mouths, or detached collars in this viewing. Some hair/beard/skin regions look unusually uniform or etched; image inspection alone cannot establish a physical photographic origin or a reliable artifact rate. The target set also repeats a small number of portrait compositions and grooming patterns; the existing family exclusions are unchanged.

## Every viewed source baseline

| Exact viewed path | Geometry / visible limitations | SHA-256 |
|---|---|---|
| `research/runs/adam-native1024-output-rank1-fixed-offsets100-v1/step-000/raw-000.png` | Apparent child; eyes ≈0.47H, chin ≈0.88H; close crop with malformed neighboring hand/arm-like material on left | `139bd356e72ad296f58229097b7bd5f4e1425902f16491386ed82cd5f65d3884` |
| `research/runs/adam-native1024-output-rank1-fixed-offsets100-v1/step-000/raw-001.png` | Adult-looking face, hat and scarf; eyes ≈0.47H, chin ≈0.97H; hard bright facial exposure and busy lower-right textile | `b6183481bdf66c62ad4d2a8bdc4ac67e06c88e8df225c73369bb8cbeb3c47600` |
| `research/runs/adam-native1024-output-rank1-fixed-offsets100-v1/step-000/raw-002.png` | Adult-looking face with glasses; eyes ≈0.47H, chin ≈0.93H; ordinary shirt mostly cut off, dark eyes behind glasses | `175d7d5a4d1ca45f0c878cd11e4d8bca99eafe6a5eb3eaa2582f8ced18646ff8` |
| `research/runs/adam-native1024-output-rank1-fixed-offsets100-v1/step-000/raw-003.png` | Apparent child in knitted hat; eyes ≈0.47H, chin ≈0.91H; soft face, highly textured hat/clothing | `17e01e3036f8e4d1d0d92f56f84aabcb3cb7ec014965ccb850a3a907477c44b4` |

No attractiveness assessment was made for these subjects. None is evidence of the requested adult priest domain. The baselines have a more varied snapshot-like lighting/background/color distribution than the uniformly styled targets, but also conspicuous source-model artifacts; adaptation should not preserve those artifacts merely to retain source appearance.

## Quantified pixel evidence, separate from visual geometry

Using Pillow and NumPy, each file was decoded as RGB and each pixel's channel spread was computed as `max(R,G,B) - min(R,G,B)` in integer arithmetic. This is a simple chroma diagnostic, not a realism score.

- All 20 target PNGs: **maximum channel spread = 0, mean channel spread = 0.0; mean absolute R−G, R−B and G−B differences are each 0.0**. Therefore all three RGB channels are exactly equal at every pixel.
- Source000: maximum 113, mean 38.401442527770996.
- Source001: maximum 130, mean 33.28232479095459.
- Source002: maximum 129, mean 30.262897491455078.
- Source003: maximum 188, mean 36.833130836486816.

The computation read the 24 viewed files, checked dimensions, and checked all target SHA-256 hashes against the manifest. It did not rewrite any image. There were no quantified landmark or area measurements. Grayscale-vs-color is thus proven for these files; the geometric ranges above are not numerical detector results.

## Hypothesis and controlled next experiment

The geometry mismatch likely makes low-data transfer harder because the target demands a smaller, higher head and a new lower-image neck/collar/torso structure. The current four source latents demonstrate that a recognizable collar cannot simply replace an already present shirt tab at the same location. Grayscale and studio-background differences give the discriminator additional easy cues. This explains why geometry deserves an experiment, but it does not prove current training is too short, the adapters are too restrictive, or that alignment alone will fix conversion. The running continuation remains unchanged. Later main adaptation can update selected original weights, unlike this constrained probing/stability run, so a burden observed during probing need not predict the capacity of main adaptation. This audit is neither evidence that the unfinished 500-step probe failed nor a reason to restart it.

**Do not blindly apply a full FFHQ-style close crop.** Enlarging a target with inter-eye distance ≈0.17W to ≈0.25W requires about 1.47× scale. If its eye line is 0.35H and collar bottom 0.80H, putting its eyes at 0.47H moves the collar bottom to `0.47 + 1.47 × (0.80 − 0.35) ≈ 1.13H`, outside the frame. This illustrative calculation uses rounded visual estimates; it is not a performed transformation. Removing the domain's clearest cue would undermine the actual objective.

A useful next experiment, after the current 500-step stability evidence is assessed:

1. Before changing data, annotate eye centers, chin/beard bottom, collar bounds and head bounds for **all 20 existing training images**. Save coordinates and uncertainty with the existing file hashes. Do not change membership or use validation/test images to choose preprocessing. Calculate prospective transformed collar retention before generating derived files.
2. Prepare one separately versioned deterministic **collar-preserving partial alignment** arm: reduce the eye-position/face-scale discrepancy toward the source while constraining the entire tab to remain within the frame with a margin. Use one documented similarity transform per image from the annotations, no facial warping or image repair, and retain all 20 identities. Record resulting eye/scale errors, crop extent, padding fractions and whether any constraint forces little or no correction. If padding would materially introduce a new discriminator cue, reject that proposed transform policy as infeasible before training; do not silently reflect/synthesize borders. This is a prospective protocol, not permission to claim exact FFHQ alignment.
3. Compare a fresh-source 100-iteration run against the archived fresh-source 100 control with the same seed, 20 identities, grayscale inputs, fixed 34-offset policy, optimizer schedule, losses, regularizers and latent sequence. The only training difference should be the audited geometric preprocessing. Both raw and EMA outputs must be reviewed unfiltered at the same milestones, including the original four latents and an additional preregistered fixed set. Do not compare a fresh 100 arm to continuation 500 as though compute were matched.
4. Primary useful outcome: improved adult priest/collar conversion **while retaining photographic coherence and diverse faces**, with measured output eye/face-scale changes. Record hats, apparent minors, collar failures, gross artifacts and nearest-training-image resemblance; keep all outputs in the review. Lower discriminator loss or closer eye placement alone is not success. If collar-preserving transforms cannot substantially reduce mismatch without padding/cue loss, the next data acquisition should deliberately target closer facial framing with an unusually high, visible collar rather than crop collars out of the current set.

This proposal leaves the monochrome difference constant between arms and does not pretend to resolve every confound. It tests a practical, collar-preserving reduction in geometric mismatch, not the isolated causal effect of a single landmark. No new images, transforms, annotations or training jobs were produced by this audit.
