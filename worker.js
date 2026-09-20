// Inactive until wrangler's entry point and the generated-image client are promoted.
const RETRY_AFTER = '3';

function unavailable() {
  return new Response(null, {
    status: 503,
    headers: { 'Cache-Control': 'no-store, max-age=0', 'Retry-After': RETRY_AFTER },
  });
}

export function createHandler({ fetchImpl = globalThis.fetch, timeoutMs = 15_000,
  schedule = setTimeout, cancel = clearTimeout } = {}) {
  return async function handle(request, env) {
    if (new URL(request.url).pathname !== '/generate') return env.ASSETS.fetch(request);
    if (request.method !== 'GET') {
      return new Response(null, { status: 405,
        headers: { Allow: 'GET', 'Cache-Control': 'no-store, max-age=0' } });
    }

    let origin;
    try {
      origin = new URL(env.INFERENCE_ORIGIN);
      if (origin.protocol !== 'https:' || origin.username || origin.password
          || origin.pathname !== '/' || origin.search || origin.hash
          || typeof env.INFERENCE_TOKEN !== 'string' || !env.INFERENCE_TOKEN.trim()) {
        return unavailable();
      }
    } catch {
      return unavailable();
    }

    const controller = new AbortController();
    const timer = schedule(() => controller.abort(), timeoutMs);
    try {
      const upstream = await fetchImpl(new URL('/generate', origin), {
        method: 'GET',
        headers: { Authorization: `Bearer ${env.INFERENCE_TOKEN}`, Accept: 'image/webp' },
        redirect: 'error',
        cache: 'no-store',
        signal: controller.signal,
      });
      if (upstream.status !== 200
          || upstream.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'image/webp') {
        await upstream.body?.cancel();
        return unavailable();
      }
      const body = await upstream.arrayBuffer();
      if (!body.byteLength) return unavailable();
      const headers = new Headers({
        'Content-Type': 'image/webp',
        'Cache-Control': 'no-store, max-age=0',
        'X-Content-Type-Options': 'nosniff',
      });
      for (const name of ['X-Model-SHA256', 'X-Generation-Seed']) {
        const value = upstream.headers.get(name);
        if (value && /^[a-f0-9]{64}$/.test(value)) headers.set(name, value);
      }
      const timing = upstream.headers.get('Server-Timing');
      if (timing && /^inference;dur=\d+(?:\.\d+)?$/.test(timing)) headers.set('Server-Timing', timing);
      return new Response(body, { status: 200, headers });
    } catch {
      return unavailable();
    } finally {
      cancel(timer);
    }
  };
}

export default { fetch: createHandler() };
