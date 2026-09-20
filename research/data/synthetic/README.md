# Synthetic training expansion

60 additional fictional adult priest portraits, numbered 081–140, created with the built-in OpenAI `image_gen` tool, one independently specified image per call. No named real person or reference identity was requested. Six generation calls ran concurrently.

The prompt set is in `prompts.json` and individual prompts are in `prompts/`. Each image has a neighboring JSON provenance record containing its prompt, original generated PNG path, time, and deterministic conversion. Original generated files remain in Codex storage; the final high-quality WebP files are committed here through Git LFS. These files are research training data, not new static production gallery entries.

Style constraints: natural black-and-white square calendar portraits, handsome fictional men aged 28–55, visible white Roman clerical collars and black clothing, Roman architecture, bare heads without hats. Prompts vary facial structure, age, hair, skin tone, facial hair, pose, and expression.

Review contact sheets, individual visual-review outcomes, dimensions, and SHA-256 checksums are recorded in `review/` and `manifest.json` after generation completes. The manifest, rather than mere file presence, records which images have passed inspection.

Tool mode: built-in image generation, no API key and no CLI fallback. Postprocessing: `cwebp -q 90`, with no resize, crop, or photometric modification.
