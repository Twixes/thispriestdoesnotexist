# Expansion to 200 active portraits

This batch adds portraits 81–230 (150 images) to the previous 50 active selections. The active IDs are defined in `public/app.js`; 30 older, excluded portraits remain archived in the repository.

Each portrait was generated separately with the built-in image-generation tool. Its exact model version was not exposed by the tool. Numbered JSON receipts retain the original local PNG path, while `../prompts/<id>.md` stores the prompt. `assets.json` records the resulting production WebP dimensions, sizes, and SHA-256 hashes. Original PNGs are preserved in the generating computer's Codex image directory.

The initial prompts repeated 25 face profiles too closely. Portraits 81–105 were retained; the remaining prompts were revised with more distinct face shapes, ages, hairstyles, facial hair, glasses, and photographic settings. Superseded generation receipts and prompts are kept in `superseded/`; they are not imported or deployed. Receipts 106–230 have `identityRevision: 2`.

The numbered sheets in `review/` show the production images, 15 per sheet. Visual review checked the black-and-white photographic style, adult subjects, visible collars, no hats, obvious image defects, and variety across the batch. Attractiveness and perceived resemblance remain subjective editorial judgments.

`browser-check.json` records the real Chrome shuffle check: two complete 200-image rounds, a partially consumed round restored in a new browser context, and 20 simultaneous tabs without repeats. The browser test uses stand-in pixels to isolate queue behavior; the production asset files are checked separately.
