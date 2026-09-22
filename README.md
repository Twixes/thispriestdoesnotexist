# This Priest Does Not Exist

A personal static website: fifty selected fictional adult priest portraits from an eighty-image archive, inspired by the black-and-white Calendario Romano souvenir calendars. Each reload randomly chooses a portrait and avoids immediately repeating the previous one in that tab.

All portraits share one collection. The former `/hot` route redirects to `/`; there is no attractiveness filter.

Plain HTML, CSS, JavaScript, and pre-generated WebP images. No framework or analytics. Production uses only the curated, pre-generated catalog. Model-training and per-request inference experiments under [`research/`](research/) are archived and are no longer being pursued; no trained model is deployed.

## Development

- `git lfs install && git lfs pull`
- `npm ci`
- `npm run dev`
- `npm run check`
- `node --test`

## Deployment

Site: https://thispriestdoesnotexist.com

Cloudflare fallback: https://thispriestdoesnotexist.matloka.workers.dev

Hosting belongs to the personal `michal@matloka.com` Cloudflare account. Production deploys automatically through **Cloudflare Workers Builds** when commits are pushed to `master` in `Twixes/thispriestdoesnotexist`.

The connected build runs `git lfs pull --include="public/**" && npm run check && node --test`, then `npm run deploy` only after the checks pass. Cloudflare installs dependencies from the committed lockfile. Builds for non-production branches are disabled. Build history and logs are available in the Worker’s **Deployments** tab, and configuration is under **Settings → Builds**.

Push changes directly to `master`; pull requests are not required for this personal repository. Do not deploy from the local CLI or upload through the dashboard: the deployed site must come from CI and the committed `master` branch.

`wrangler.jsonc` pins the personal account, custom domain, and static asset directory. Cloudflare manages the CI build token; local Wrangler login is separate, and no credentials are committed. Only `public/` is deployed.

The original image prompts are in `docs/image-prompts.md`; prompts for portraits 13–80 are in `docs/prompts/`. Images were generated with the built-in image-generation tool; no calendar photographs were copied.

## License

The application code is [MIT](LICENSE). Vendored research dependencies and pretrained/derived models retain their upstream licenses; see [`research/THIRD_PARTY.md`](research/THIRD_PARTY.md).
