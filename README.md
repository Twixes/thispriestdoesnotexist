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

Hosting belongs to the personal `michal@matloka.com` account. Use Wrangler for deployments; dashboard uploads are also supported by selecting **New deployment** and uploading `public/`.

`npm run deploy` uses Cloudflare Workers Static Assets under the same account. The account ID is pinned in `wrangler.jsonc` to prevent accidental deployment to another account. Only `public/` is deployed. Authentication is managed by Wrangler and is never committed.

The original image prompts are in `docs/image-prompts.md`; prompts for portraits 13–50 are in `docs/prompts/`. Images were generated with the built-in image-generation tool; no calendar photographs were copied.

## License

[MIT](LICENSE).
