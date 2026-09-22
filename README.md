# This Priest Does Not Exist

A personal static website: 200 selected fictional adult priest portraits from a 230-image archive, inspired by the black-and-white Calendario Romano souvenir calendars. Each reload takes the next portrait from a shuffled collection. Every portrait appears once before the next round starts, with no immediate repeat between rounds.

All portraits share one collection. The former `/hot` route redirects to `/`; there is no attractiveness filter.

Plain HTML, CSS, JavaScript, and pre-generated WebP images. No framework or analytics. Production uses only the curated, pre-generated catalog. Model-training and per-request inference experiments under [`research/`](research/) are archived and are no longer being pursued; no trained model is deployed.

## Shuffle behavior

The shuffled queue is saved in the browser using IndexedDB, so it survives reloads and browser restarts. Atomic read/write transactions coordinate reservations across tabs on the same origin. No IP addresses, accounts, cookies, backend, or visitor database are needed. Each visit downloads only its selected image.

Different browsers, devices, private sessions, or clearing site data start independent queues. If IndexedDB is unavailable, the site falls back to local storage and then session storage; if all storage is blocked, portraits still load but history cannot survive a reload. The fallback preserves the queue across sequential visits, but simultaneous tabs can race without IndexedDB. Unavailable images are skipped, and changing the collection starts a new shuffle.

## Development

- `git lfs install && git lfs pull`
- `npm ci`
- `npm run dev`
- `npm run check`
- `node --test`

With Playwright and Google Chrome available, `node scripts/browser-shuffle-smoke.cjs` also verifies two full shuffle rounds, persistence, and simultaneous tabs in a real browser. Set `PLAYWRIGHT_MODULE` to an external Playwright module path if needed.

## Deployment

Site: https://thispriestdoesnotexist.com

Cloudflare fallback: https://thispriestdoesnotexist.matloka.workers.dev

Hosting belongs to the personal `michal@matloka.com` Cloudflare account. Production deploys automatically through **Cloudflare Workers Builds** when commits are pushed to `master` in `Twixes/thispriestdoesnotexist`.

The connected build runs `git lfs pull --include="public/**" && npm run check && node --test`, then `npm run deploy` only after the checks pass. Cloudflare installs dependencies from the committed lockfile. Builds for non-production branches are disabled. Build history and logs are available in the Worker’s **Deployments** tab, and configuration is under **Settings → Builds**.

Push changes directly to `master`; pull requests are not required for this personal repository. Do not deploy from the local CLI or upload through the dashboard: the deployed site must come from CI and the committed `master` branch.

`wrangler.jsonc` pins the personal account, custom domain, and static asset directory. Cloudflare manages the CI build token; local Wrangler login is separate, and no credentials are committed. Only `public/` is deployed.

The original image prompts are in `docs/image-prompts.md`; prompts for portraits 13–230 are in `docs/prompts/`. The expansion manifest and review sheets are in `docs/generation-200/`; `scripts/import-portraits.py` converts saved generation results to WebP. Images were generated with the built-in image-generation tool; no calendar photographs were copied.

## License

The application code is [MIT](LICENSE). Vendored research dependencies and pretrained/derived models retain their upstream licenses; see [`research/THIRD_PARTY.md`](research/THIRD_PARTY.md).
