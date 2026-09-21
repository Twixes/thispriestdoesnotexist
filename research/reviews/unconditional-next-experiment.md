# No-prompt requirement and next GAN hypothesis

The owner explicitly requires a model like thispersondoesnotexist: fresh random noise goes into a priest-specific generator and a new portrait comes out. A hidden fixed prompt, cached text embedding, or prompt bank does not qualify. General-purpose diffusion serving research stops. Completed diffusion outputs remain archived as offline quality references and possible teacher data only.

This is a prospective research recommendation from a read-only review, not an implemented or executed experiment. No production model is approved. The hard visual-quality, genuine variety, no-hat, adult-priest and actual-server generation-under500ms requirements remain.

## What the previous GAN results establish

- NADA500 learned broad clerical bands in32/32 unseen outputs, but all32 retained native texture/detail concerns and some children/hats remained. A small Roman tab is not itself mandatory; native realism and suitable subjects are. The generator has clothing capacity, but the global CLIP objective did not preserve the required quality.
- Paired training failed even supervised collar construction. Small paired coverage, masks and pixel-loss shortcuts are plausible limitations; it is not justified to attribute the result exclusively to a shortage of unseen examples.
- Reference625 damaged faces and contracted diversity. Its256px ceiling, small target distribution and framing shift remain limitations. Existing optimizer/EMA/sign audits and CPU reproduction do not support treating this simply as an MPS rendering bug.
- CDC constrained source relationships and could preserve unwanted source attributes. Our variant was not the complete published recipe, so its failure does not reject every correspondence method.

Do not resume these failed runs unchanged.

## Candidate: target-aware kernel adaptation

AdAM-style target-aware kernel modulation of the untouched native1024 StyleGAN2 FFHQ model is a materially different hypothesis. It probes target-relevant pretrained kernels, modulates important kernels and permits others to adapt, rather than relying on fixed-block freezing, a global CLIP direction or source-only consistency. The resulting serving architecture remains an unconditional generator. Published few-shot results justify investigation, not a prediction that it will solve this dataset.

[NeurIPS2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b122d0a0dcb1a86ffa25ccba154652b-Paper-Conference.pdf) · [implementation pinned at6428e99](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/AdAM_main_adaptation.py) · [probing recipe](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/_bash_importance_probing.sh) · [adaptation recipe](https://github.com/yunqing-me/AdAM/blob/6428e99cfb36bc8bda3506350824f0c7dac9a5ad/_bash_main_adaptation.sh).

The official implementation defaults to256px and hardcodes twelve convolution indices. Merely changing a size flag is unsafe. A native-resolution port, explicit parameter-to-layer mapping and invariant checks must precede any training. Published batch-four CUDA settings do not establish local memory or speed. Any serving-time modulation must be fixed model parameters with no prompt dependency; validate the exported unconditional forward and any weight folding numerically.

## Bounded prospective test

1. Start from original FFHQ1024 G/D, never an already adapted checkpoint. Initially use the existing20 reviewed training portraits and preserve the five validation/five test portraits separately. Twenty images suffice for a compatibility/direction experiment, not proof of broad identity coverage.
2. Verify baseline source-output equality, then run ten actual native-resolution updates to check modulation, frozen-kernel invariants, gradients, checkpoint restoration and resource use. Retain existing safety guards. A port that cannot pass this check does not proceed.
3. If feasible, freeze a separate protocol for500 probing updates,250 Fisher-estimation batches and at most300 adaptation updates, with50th-percentile importance threshold and a two-hour cap. Those are proposed experimental limits, not a claim of convergence or an already launched job. Record any deviations from the official recipe before execution.
4. Inspect32 fixed development latents at baseline and adaptation100/300. Use separate test latents only after selection. Retain every image; require better priest appearance without worse eyes, skin, anatomy, age/headwear or identity variety. No restoration or accepted-seed bank can substitute for a good generator.

Data coverage remains an independent risk: kernel preservation cannot reliably supply target variation absent from twenty examples. The first experiment should determine whether this mechanism improves the direction of adaptation before any bulk teacher generation or extended compute.
