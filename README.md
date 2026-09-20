# This Priest Does Not Exist

A personal static website: twelve fictional adult priest portraits, inspired by the black-and-white Calendario Romano souvenir calendars. Each reload randomly chooses a portrait and avoids immediately repeating the previous one in that tab.

Plain HTML, CSS, JavaScript, and pre-generated WebP images. No framework, backend, database, analytics, or image-generation calls at runtime.

## Development

- `npm ci`
- `npm run dev`
- `npm run check`
- `node --test`

## Deployment

`npm run deploy` uses Cloudflare Workers Static Assets under the personal `michal@matloka.com` account. The account ID is pinned in `wrangler.jsonc` to prevent accidental deployment to another account. Only `public/` is deployed. Authentication is managed by Wrangler and is never committed.

The image prompts are in `docs/image-prompts.md`. Images were generated with the built-in image-generation tool; no calendar photographs were copied.
