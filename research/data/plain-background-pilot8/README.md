# Eight synthetic training candidates with simple backgrounds

Generated with the built-in imagegen tool on 2026-09-21. These are **training candidates, not output from our trained StyleGAN and not additions to the production catalog**. Each source is an independent native1254×1254 image, retained unchanged. No source face reference was supplied.

The hypothesis is that broadening backgrounds beyond repeated prominent architecture could help the full portrait model. This has not been established by a training comparison. The source review accepts the eight portraits' face/collar quality for prospective training; it does not establish the quality of any resulting trained model.

- Original images: `141.png` through `148.png`.
- Exact prompts: `prompts-141-144.json` and `prompts-part145-148.json`.
- Tool results, original paths and hashes: `provenance-141-144.json` and `provenance-part145-148.json`.
- Native visual reviews: `review-141-144.json`, `review-part145-148.json`, and `root-review.json`.
- Candidate146 unexpectedly came back in color. Its original remains unchanged; separate preprocessing records monochrome normalization.
- The first eyes42 alignment introduced reflected edge pixels, including a small duplicated hair fragment on141. Those attempts are retained in sibling `plain-background-pilot8-aligned256` and `plain-background-pilot8-aligned256-monochrome` directories as diagnostic evidence. They are not approved training inputs.
- A separate sibling `plain-background-pilot8-zero-reflection` contains native1024 and256 crops. All eight pass the independent border-mode equality check, and141's duplicate hair is gone. The tradeoff is shifted eye positions and tight hair crops on144/145/147; see its `visual-review.json` and `root-review.json`. This is a candidate preprocessing experiment, not an accepted replacement dataset. Failed preprocessing attempts and their logs are retained separately.

The active110-image dataset and all running experiment inputs remain unchanged. Any future expanded-data training is a distinct fork, not an exact continuation under the old dataset digest.
