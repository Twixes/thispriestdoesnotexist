export const INSTANCE_NAME = 'priest-production';

function unavailable() {
  return new Response(null, { status: 503,
    headers: { 'Cache-Control': 'no-store, max-age=0', 'Retry-After': '3' } });
}

function methodNotAllowed() {
  return new Response(null, { status: 405,
    headers: { Allow: 'GET', 'Cache-Control': 'no-store, max-age=0' } });
}

export async function routeRequest(request, env) {
  const path = new URL(request.url).pathname;
  if (path !== '/generate' && path !== '/health') return env.ASSETS.fetch(request);
  if (request.method !== 'GET') return methodNotAllowed();
  try {
    // A constant Durable Object name means user paths/nonces cannot create instances.
    const container = env.PRIEST_CONTAINER.getByName(INSTANCE_NAME);
    return await container.fetch(new Request(`https://container.internal${path}`, {
      method: 'GET', signal: request.signal,
    }));
  } catch {
    return unavailable();
  }
}

export async function serveContainer(container, request, expectedHash) {
  const path = new URL(request.url).pathname;
  if (request.method !== 'GET') return methodNotAllowed();
  if (!['/generate', '/health'].includes(path)) return new Response(null, { status: 404 });
  if (typeof expectedHash !== 'string' || !/^[a-f0-9]{64}$/.test(expectedHash)) return unavailable();
  const signal = AbortSignal.any([request.signal, AbortSignal.timeout(45_000)]);
  try {
    // Port readiness alone is insufficient. Read the model's real health report
    // on every request, also preventing old-image responses during a rollout.
    const healthResponse = await container.containerFetch('http://localhost/health', {
      method: 'GET', redirect: 'error', signal,
    });
    if (healthResponse.status !== 200) {
      await healthResponse.body?.cancel();
      return unavailable();
    }
    const health = await healthResponse.json();
    if (health.status !== 'ready' || health.reviewed !== true || !(health.training_step > 0)
        || health.device !== 'cpu' || health.model_sha256 !== expectedHash) return unavailable();
    if (path === '/health') {
      return Response.json({ status: 'ready', model_sha256: health.model_sha256,
        training_step: health.training_step, reviewed: true, resolution: health.resolution }, {
        headers: { 'Cache-Control': 'no-store, max-age=0' },
      });
    }
    const generated = await container.containerFetch('http://localhost/generate', {
      method: 'GET', headers: { Accept: 'image/webp' }, redirect: 'error', signal,
    });
    const seed = generated.headers.get('X-Generation-Seed');
    if (generated.status !== 200 || generated.headers.get('Content-Type') !== 'image/webp'
        || generated.headers.get('X-Model-SHA256') !== expectedHash
        || !seed || !/^[a-f0-9]{64}$/.test(seed)) {
      await generated.body?.cancel();
      return unavailable();
    }
    const body = await generated.arrayBuffer();
    if (!body.byteLength) return unavailable();
    const headers = new Headers({ 'Content-Type': 'image/webp',
      'Cache-Control': 'no-store, max-age=0', 'X-Content-Type-Options': 'nosniff',
      'X-Model-SHA256': expectedHash, 'X-Generation-Seed': seed });
    const timing = generated.headers.get('Server-Timing');
    if (timing && /^inference;dur=\d+(?:\.\d+)?$/.test(timing)) headers.set('Server-Timing', timing);
    return new Response(body, { headers });
  } catch {
    return unavailable();
  }
}
