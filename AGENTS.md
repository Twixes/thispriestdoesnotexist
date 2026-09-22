# Personal site workflow

- Work directly on `master` and push to `origin/master`. The owner explicitly does not require pull requests for this repository.
- Deploy the static frontend through connected Cloudflare Workers Builds, triggered by a push to `master`. Do not run local production application deployments or dashboard uploads.
- Run `npm run check` and `node --test` before pushing code changes. Confirm the resulting CI build succeeds.
- Hosting must stay in the personal `michal@matloka.com` account pinned in `wrangler.jsonc`, never a work account.
- Reuse saved GitHub/Wrangler authentication and existing personal browser sessions. Do not repeatedly query 1Password from fresh shells. If a new credential is required, fetch only the specific task credentials once into a persistent process, then reuse them without further vault reads. See `infra/auth-session.md` for documented authorization limits; never promise a blanket 12h+ desktop authorization or claim a credential cache exists before creating it.
- Never commit credentials. Keep the site plain HTML, CSS, and JavaScript. The owner cancelled model training and per-request inference; production uses only pre-generated images. Do not resume model work or deploy an inference service unless explicitly requested again.
- The active catalog contains 200 curated fictional adult priest portraits. Preserve the black-and-white photographic style, visible white collars, and no hats. Review new images visually for quality and variety before adding them.
- Show every portrait once in a shuffled round before repeating. Persist the queue in the browser and coordinate tabs with IndexedDB transactions; no backend or IP tracking is needed. Keep the picture-only layout and existing authorship credit.
- Keep images, training datasets, and model weights in Git LFS. Store research code, provenance, logs, and evaluation artifacts under `research/`.
- Third-party research code and weights retain their upstream licenses; the website/application code stays MIT.
- Use `git -c core.fsmonitor=false status --short` before archiving running research logs: the filesystem monitor has missed appended log/metric changes in this checkout. Stage complete checkpoint files only, never temporary writes.
