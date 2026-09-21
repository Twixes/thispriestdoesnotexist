# Next experiment: one bounded capacity × feature-objective reconstruction test

**Recommend one predeclared 2×2 experiment, not another50-step reweighting probe or an immediate dataset expansion.** Keep the six known pairs; compare opening b32 and adding frozen-D clothing features as independent factors. The first question is whether this generator can render the supervised collar shapes while preserving their source faces. Generalization cannot be diagnosed cleanly while supervised examples still fail.

The matched600+50 result supports this ordering. Mean training surround L1 fell from0.6792 in the control to0.4053 with the surround objective, but mean tab L1 rose from0.0610 to0.1191 and protected-source L1 rose from0.02183 to0.02409 (all on[-1,1]). Every training tab's error increased. The final contact and native b2-055 show a narrower but still ragged pale band/scarf rather than a clean Roman tab. This establishes that output weighting changes the defect; it does not establish sufficient capacity, convergence or a learned clothing rule.

Primary grounding: [JoJoGAN §3.1](https://arxiv.org/html/2112.11641v4#S3.SS1) uses pretrained discriminator-feature differences because loss choice affects structural detail; its stylization experiments do not establish photorealistic priest clothing. Its paired style-mixing augmentation can intentionally erase source differences, so do not borrow that augmentation for our exact identity pairs. [The official implementation](https://github.com/mchong6/JoJoGAN) documents its discriminator perceptual-loss revision. [GANWarping](https://peterwang512.github.io/GANWarping/) demonstrates that a few paired edits can modify a GAN's geometric rules, but its warp-derived supervision differs from these imagegen edits. Neither source proves that b64+ is texture-only or that six unrelated collar examples suffice.

## Fixed design

All four branches fork original step600 SHA256 `0f433c7ec50d9e80026681125d7fa7e17e9be6e8e2d31b11976fb51d1b0f3fc4`, not a selected650 branch. Use300 updates per arm, an identical predeclared balanced sequence with50 presentations per identity, identical fresh-latent draws and existing Adam moments. This is1,200 updates total, sequential processes, no parameter sweep. Balanced sampling is common to all arms and explicitly changes the sampling policy from the parent; this is an experiment fork, not an exact resume.

| Arm | Trainable synthesis blocks | Objective |
|---|---|---|
|A|64–1024|Original half-tab/half-rest pixel loss|
|B|32–1024|Same pixel loss|
|C|64–1024|Same pixel loss +0.05 frozen-D clothing feature term|
|D|32–1024|Same pixel + feature loss|

Keep mapping, blocks<=16, all noise strengths and buffers fixed. A/C additionally freeze b32. B/D append only b32 non-noise parameters with empty Adam moments and the same1e-4 LR/betas; preserve all existing64+ moments. Keep protected/fresh losses and weights unchanged. Do not combine the surround thirds objective with this test: its tab-weight tradeoff would complicate attribution.

Use the existing reviewed `clothing_structure` helper and guarded extractor: exact original FFHQ1024 D, no D training, taps after b512/b256, same protected source pixels in both **loss-only** composites, fractional region masks, detached target/RMS normalization. The0.05 coefficient is an unvalidated fixed engineering choice, not a paper default. First record feature-versus-pixel gradient norms on shared trainable parameters. A numerically negligible feature gradient makes a null C/A result inconclusive; do not quietly tune the coefficient or claim the objective disproved. Real D extraction/backward has not yet been measured. Guard each process at>=25% free memory; retain an8GiB sampled RSS abort ceiling and4-hour total experiment ceiling. If the actual D smoke fails either resource gate, stop rather than silently changing resolution/taps. Existing pure-pixel50-update runs took279–305 seconds including setup/previews; extrapolating this to D arms would be unjustified.

## Decision evidence, fixed before running

Save native full outputs and region metrics at0/100/200/300 for all six train seeds,028, the original four unfiltered random seeds, and the existing disjoint source30 panel. Also use paired10's three new identities042/058/059 **only as diagnostic held-out edits for this experiment**, after exact source preflight; do not alter the approved manifest or train on them. Their masks are reviewed, but042 retains only37.35% of its tab, so score only feasible in-mask structure and judge the complete image separately.025 remains qualitative-only because its edited face moved.

A useful training-fit gate is at least four of six native outputs visibly showing correctly placed, bounded white tabs with plausible black shirt structure—not just lower mean brightness error—and no visible face degradation. Require tab AND30px-surround improvement over matched A, and no more than0.005 increase in mean protected L1 on[-1,1]. These are prospective engineering gates, not established perceptual thresholds; retain per-identity results so averaging cannot hide damage. Judge original source faces, not edited target faces. Do not demand target pixels outside the immutable clothing masks.

- **B beats A, and D beats C, on supervised collar structure with acceptable faces:** evidence that the restricted trainable block range is a bottleneck. It does not prove a universal semantic boundary at32px.
- **C beats A, and D beats B:** evidence that a structural feature objective helps at fixed capacity. Only D succeeding indicates an interaction: neither single change suffices.
- **An arm fits training collars but fails the three new edited identities and eligible source30 faces:** evidence of a generalization/coverage problem. More data becomes the next justified question, but this does **not prove** insufficient data alone; a later matched6-versus9 data-only comparison would be needed to attribute that cause. Do not spend the three new pairs before learning whether the current method can fit its known targets.
- **All four fail training fit:** do not blame data scarcity from unseen failures or buy more training duration. Revisit target/mask feasibility, optimization scale and a wider representational constraint. Failure of b32 alone does not rule out all capacity limitations. Lower feature/pixel losses without recognizable collars is a failed quality result.

This is one diagnostic experiment with interpretable main effects and interaction, not a production candidate. Hats, age/sex eligibility and perceived attractiveness remain separate sampling problems. No implementation, new asset generation, model loading or training was performed for this assessment.

## Evidence scope

Read the completed comparison mechanics, both final per-pair evaluations, baseline native-review record, prior capacity-versus-surround assessment, paired10 approval, frozen-D design/helper review and current runner. Viewed the final surround contact and native b2-055. Hashes: comparison mechanics `f7f14102a1345a1872942224e7d7cda7ab4cc07281ba39f8df0e8646fdc83f0a`; baseline evaluation `897bb3bbf66681664ae72da0f56ea5fdf9e1e44c7e76814bdcaa0e2fd8570efc`; surround evaluation `7e6739012079d2163bef5580990a6a5cfdbd25c8d6754bb930f97fff8910f319`; surround contact `a7303ea3e442b0bd9fcb71cc4c4732782f6b836cea54b0d21fe1ba98ed91177c`; paired10 manifest `8d0660f1ccf5115a356e9e6b479e4d225c1e9993402fc71575b0fcb9ed8dcff1`. Primary sources accessed2026-09-21; all upstream NVIDIA/license restrictions remain unchanged.
