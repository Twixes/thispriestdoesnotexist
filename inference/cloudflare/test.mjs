import assert from 'node:assert/strict';
import test from 'node:test';
import { INSTANCE_NAME, routeRequest, serveContainer } from './routing.js';

const hash = 'a'.repeat(64);
const seed = 'b'.repeat(64);
const health = { status: 'ready', reviewed: true, training_step: 1000,
  model_sha256: hash, device: 'cpu', resolution: 256, private_detail: 'not public' };
const bytes = new Uint8Array([82, 73, 70, 70, 4, 0, 0, 0, 87, 69, 66, 80]);
const request = (path = '/generate') => new Request(`https://site.example${path}`);
const generated = (overrides = {}) => new Response(bytes, { headers: {
  'Content-Type': 'image/webp', 'X-Model-SHA256': hash, 'X-Generation-Seed': seed,
  'Server-Timing': 'inference;dur=143.5', 'Set-Cookie': 'private=value', ...overrides,
} });
function stub(healthReply = () => Response.json(health), imageReply = generated) {
  const calls = [];
  return { calls, async containerFetch(url, options) {
    calls.push({ url, options });
    return url.endsWith('/health') ? healthReply() : imageReply();
  } };
}
function assertUnavailable(response) {
  assert.equal(response.status, 503);
  assert.equal(response.headers.get('Retry-After'), '3');
  assert.match(response.headers.get('Cache-Control'), /no-store/);
}

test('routing uses one instance and strips caller query, headers, and cookies', async () => {
  const names = [], forwarded = [];
  const env = { PRIEST_CONTAINER: { getByName(name) {
    names.push(name);
    return { fetch: async (req) => { forwarded.push(req); return new Response('ok'); } };
  } } };
  for (const nonce of ['first', 'second']) {
    const response = await routeRequest(new Request(`https://site.example/generate?request=${nonce}`, {
      headers: { Cookie: 'session=private', Authorization: 'Bearer caller', 'X-Other': 'ignored' },
    }), env);
    assert.equal(response.status, 200);
  }
  assert.deepEqual(names, [INSTANCE_NAME, INSTANCE_NAME]);
  for (const req of forwarded) {
    assert.equal(req.url, 'https://container.internal/generate');
    assert.equal(req.method, 'GET');
    assert.deepEqual([...req.headers], []);
  }
});

test('assets delegate unchanged; invalid methods never contact the container', async () => {
  const original = request('/hot');
  const env = { ASSETS: { fetch(req) { assert.equal(req, original); return new Response('asset'); } } };
  assert.equal(await (await routeRequest(original, env)).text(), 'asset');
  for (const path of ['/generate', '/health']) {
    const response = await routeRequest(new Request(`https://site.example${path}`, { method: 'POST' }), {});
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('Allow'), 'GET');
  }
  assertUnavailable(await routeRequest(request(), {}));
});

test('every generation checks actual model health then produces new bytes privately', async () => {
  let sequence = 0;
  const container = stub(undefined, () => generated({ 'X-Generation-Seed': (++sequence).toString(16).padStart(64, '0') }));
  const responses = [];
  for (let i = 0; i < 2; i++) {
    const response = await serveContainer(container, request(`/generate?ignored=${i}`), hash);
    responses.push(response);
    assert.equal(response.status, 200);
    assert.deepEqual(new Uint8Array(await response.arrayBuffer()), bytes);
    assert.equal(response.headers.get('Content-Type'), 'image/webp');
    assert.equal(response.headers.get('X-Model-SHA256'), hash);
    assert.equal(response.headers.get('Server-Timing'), 'inference;dur=143.5');
    assert.equal(response.headers.get('Set-Cookie'), null);
    assert.match(response.headers.get('Cache-Control'), /no-store/);
  }
  assert.notEqual(responses[0].headers.get('X-Generation-Seed'), responses[1].headers.get('X-Generation-Seed'));
  assert.deepEqual(container.calls.map(c => c.url), ['http://localhost/health', 'http://localhost/generate',
    'http://localhost/health', 'http://localhost/generate']);
  for (const call of container.calls) {
    assert.equal(call.options.redirect, 'error');
    assert.ok(call.options.signal instanceof AbortSignal);
    assert.deepEqual(call.options.headers ?? {}, call.url.endsWith('/generate') ? { Accept: 'image/webp' } : {});
  }
});

test('health exposes only the approved model status without generating an image', async () => {
  const container = stub();
  const response = await serveContainer(container, request('/health'), hash);
  assert.deepEqual(await response.json(), { status: 'ready', reviewed: true, training_step: 1000,
    model_sha256: hash, resolution: 256 });
  assert.match(response.headers.get('Cache-Control'), /no-store/);
  assert.equal(container.calls.length, 1);
});

test('missing approval, different model, and startup failures never generate', async () => {
  for (const override of [{ reviewed: false }, { training_step: 0 }, { model_sha256: seed },
    { status: 'loading' }, { device: 'cuda' }]) {
    const container = stub(() => Response.json({ ...health, ...override }));
    assertUnavailable(await serveContainer(container, request(), hash));
    assert.equal(container.calls.length, 1);
  }
  for (const reply of [() => new Response('not ready', { status: 503 }),
    () => new Response('invalid json'), () => { throw new Error('private internal detail'); }]) {
    const container = stub(reply);
    assertUnavailable(await serveContainer(container, request(), hash));
    assert.equal(container.calls.length, 1);
  }
  const container = stub();
  assertUnavailable(await serveContainer(container, request(), 'invalid'));
  assert.equal(container.calls.length, 0);
});

test('bad generation responses fail closed without leaking upstream details', async () => {
  for (const reply of [() => new Response('overloaded', { status: 429 }),
    () => generated({ 'Content-Type': 'text/html' }), () => generated({ 'X-Model-SHA256': seed }),
    () => generated({ 'X-Generation-Seed': 'short' }), () => new Response(null, { headers: generated().headers }),
    () => { throw new Error('internal failure'); }]) {
    const response = await serveContainer(stub(undefined, reply), request(), hash);
    assertUnavailable(response);
    assert.equal(await response.text(), '');
  }
});
