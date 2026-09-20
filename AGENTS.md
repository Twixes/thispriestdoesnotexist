# Personal site workflow

- Work directly on `master` and push to `origin/master`. The owner explicitly does not require pull requests for this repository.
- Deploy only through the connected Cloudflare Workers Builds CI triggered by a push to `master`. Do not run local production deployments or dashboard uploads.
- Run `npm run check` and `node --test` before pushing code changes. Confirm the resulting CI build succeeds.
- Hosting must stay in the personal `michal@matloka.com` account pinned in `wrangler.jsonc`, never a work account.
- Never commit credentials. Keep the site plain HTML, CSS, and JavaScript. The current goal adds a trained server-side generator; do not claim that a fixed catalog, image interpolation, or random image distortions fulfills it.
- Keep images, training datasets, and model weights in Git LFS. Store research code, provenance, logs, and evaluation artifacts under `research/`.
- Do not publish an unreviewed model as the production generator. Verify visual quality, diversity, adult male priest appearance, white collars, and no hats.
- Third-party research code and weights retain their upstream licenses; the website/application code stays MIT.
