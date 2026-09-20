# Prepared inference routing — inactive until model promotion

`worker.js` and `inference/client-app.js` implement the future runtime flow. They
are tested but deliberately not connected to the current production entry point.
The existing gallery remains active until a trained model passes visual review
and is deployed through CI to the personal origin.

The worker accepts only `GET /generate`, ignores the browser's query and headers,
and requests the fixed `https://<configured-origin>/generate` with its own bearer
secret. The configured origin must be an HTTPS origin without credentials, query,
fragment, or path. Redirects are forbidden. Missing configuration, upstream
failure, invalid media, an empty body, or the 15-second timeout returns an empty,
uncached 503 with `Retry-After: 3`. There is no static portrait fallback. Other
paths delegate unchanged to the `ASSETS` binding.

Success responses preserve the WebP bytes and only validated model SHA-256,
256-bit generation seed, and inference timing headers. Origin cookies, server
details, arbitrary headers, and error text are never relayed. Both origin fetch
and browser response disable caching.

The prepared client requests `/generate?request=<fresh crypto.randomUUID()>` on
every page load using `cache: no-store` and a 20-second deadline. It decodes a Blob
URL and assigns that same Blob URL to `#portrait`. This avoids a subtle double
generation: preloading an uncached HTTP image and then assigning its HTTP URL to
the visible element can perform a second request. Failed attempts use new nonces,
wait 3 then 6 seconds, and stop after three attempts. Failed Blob URLs are revoked;
the single displayed Blob is released when the page is destroyed. No controls or
text are added, and there is no session-storage/cached-catalog path.

## Promotion checklist for the repository owner

1. Confirm the exact trained bundle has passed visual and diversity review and is
   running on the personal origin; verify its health hash and actual new images.
2. Configure `INFERENCE_ORIGIN` and secret `INFERENCE_TOKEN` in the personal
   Cloudflare worker. The secret must match the origin's token and must not enter
   Git or browser-visible assets.
3. Set Wrangler's `main` to `worker.js`, bind assets as `ASSETS`, and route
   `/generate` through the worker before static assets (for example,
   `assets.run_worker_first: ["/generate"]`). Preserve the custom domain and the
   existing asset redirects.
4. Promote `inference/client-app.js` to `public/app.js`; update the noscript image
   to `/generate` so it also generates per load. Preserve the current minimal
   layout and credit. If a CSP is introduced, allow `blob:` image sources.
5. Push directly to `master` and let CI deploy. Verify the rendered site, successful
   fresh image hashes/seeds across reloads, no-store headers, token isolation,
   overload behavior, and the actual origin model hash. Do not deploy locally.

Tests: `node --test tests/worker.test.js tests/generated-gallery.test.js`.
The tests cover fixed-origin proxy requests, token/cookie isolation, cache
semantics, static delegation, missing/unsafe configuration, unsupported methods,
upstream failure/media errors, timeout cancellation, metadata filtering, unique
client nonces, decode failures, bounded retries, and no catalog fallback.
