# This Priest Does Not Exist

A personal static website: fifty fictional adult priest portraits, inspired by the black-and-white Calendario Romano souvenir calendars. Each reload randomly chooses a portrait and avoids immediately repeating the previous one in that tab.

The `/hot` route uses a hand-picked, subjective edit of twenty portraits from the same catalog: 01, 02, 03, 04, 07, 09, 10, 13, 18, 20, 22, 23, 25, 27, 33, 36, 38, 45, 46, and 47. The selection favors expressive eyes, strong features, and relaxed calendar-model charisma. It uses the same layout and repeat avoidance; the homepage still includes all fifty. Edit `HOT_PORTRAITS` in `public/app.js` to change the selection. Cloudflare serves `public/hot.html` at `/hot`.

Plain HTML, CSS, JavaScript, and pre-generated WebP images. No framework, backend, database, analytics, or image-generation calls at runtime.

## Development

- `npm ci`
- `npm run dev`
- `npm run check`
- `node --test`

## Deployment

Site: https://thispriestdoesnotexist.com

Cloudflare fallback: https://thispriestdoesnotexist.matloka.workers.dev

Hosting belongs to the personal `michal@matloka.com` Cloudflare account. Production deploys automatically through **Cloudflare Workers Builds** when commits are pushed to `main` in `Twixes/thispriestdoesnotexist`.

The connected build runs `npm run check && node --test`, then `npm run deploy` only after the checks pass. Cloudflare installs dependencies from the committed lockfile. Builds for non-production branches are disabled. Build history and logs are available in the Worker’s **Deployments** tab, and configuration is under **Settings → Builds**.

Push changes directly to `main`; pull requests are not required for this personal repository. Do not deploy from the local CLI or upload through the dashboard: the deployed site must come from CI and the committed `main` branch.

`wrangler.jsonc` pins the personal account, custom domain, and static asset directory. Cloudflare manages the CI build token; local Wrangler login is separate, and no credentials are committed. Only `public/` is deployed.

The original image prompts are in `docs/image-prompts.md`; prompts for portraits 13–50 are in `docs/prompts/`. Images were generated with the built-in image-generation tool; no calendar photographs were copied.

## License

[MIT](LICENSE).
