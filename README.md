# This Priest Does Not Exist

A personal static website: fifty fictional adult priest portraits, inspired by the black-and-white Calendario Romano souvenir calendars. Each reload randomly chooses a portrait and avoids immediately repeating the previous one in that tab.

Plain HTML, CSS, JavaScript, and pre-generated WebP images. No framework, backend, database, analytics, or image-generation calls at runtime.

## Development

- `npm ci`
- `npm run dev`
- `npm run check`
- `node --test`

## Deployment

Site: https://thispriestdoesnotexist.com

Cloudflare fallback: https://thispriestdoesnotexist.matloka.workers.dev

The first deployment was uploaded directly through the Cloudflare dashboard under the personal `michal@matloka.com` account. To update without CLI authorization, open the Worker, select **New deployment**, and upload the `public/` folder.

Alternatively, after authorizing Wrangler, `npm run deploy` uses Cloudflare Workers Static Assets under the same account. The account ID is pinned in `wrangler.jsonc` to prevent accidental deployment to another account. Only `public/` is deployed. Authentication is managed by Wrangler and is never committed.

The original image prompts are in `docs/image-prompts.md`; prompts for portraits 13–50 are in `docs/prompts/`. Images were generated with the built-in image-generation tool; no calendar photographs were copied.

## License

[MIT](LICENSE).
