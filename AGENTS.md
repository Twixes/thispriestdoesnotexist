# Personal site workflow

- Work directly on `main` and push to `origin/main`. The owner explicitly does not require pull requests for this repository.
- Deploy only through the connected Cloudflare Workers Builds CI triggered by a push to `main`. Do not run local production deployments or dashboard uploads.
- Run `npm run check` and `node --test` before pushing code changes. Confirm the resulting CI build succeeds.
- Hosting must stay in the personal `michal@matloka.com` account pinned in `wrangler.jsonc`, never a work account.
- Never commit credentials. Keep the site plain HTML, CSS, JavaScript, and pre-generated images.
