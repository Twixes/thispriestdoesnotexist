import test from 'node:test';
import assert from 'node:assert/strict';
import { createHandler } from '../worker.js';

const modelHash = 'a'.repeat(64);
const generationSeed = 'b'.repeat(64);
const image = new Uint8Array([82, 73, 70, 70, 4, 0, 0, 0, 87, 69, 66, 80]);
function environment() {
  return { INFERENCE_ORIGIN: 'https://personal-origin.example', INFERENCE_TOKEN: 'origin-secret',
    ASSETS: { fetch() { throw new Error('Unexpected static fallback'); } } };
}

test('proxy performs an authenticated fixed-path request without forwarding user data', async () => {
  const requests = [];
  const handle = createHandler({ fetchImpl: async (url, options) => {
    requests.push({ url: String(url), options });
    return new Response(image, { headers: {
      'Content-Type': 'image/webp', 'X-Model-SHA256': modelHash,
      'X-Generation-Seed': generationSeed, 'Server-Timing': 'inference;dur=142.5',
      'Cache-Control': 'public, max-age=86400', 'Set-Cookie': 'private=value',
      'X-Private-Origin': 'hidden', Server: 'private-software',
    } });
  } });
  const response = await handle(new Request('https://thispriestdoesnotexist.com/generate?secret=ignored', {
    headers: { Cookie: 'browser-session=private', Authorization: 'Bearer user-input',
      'X-Forwarded-For': 'untrusted', 'If-None-Match': 'cached' },
  }), environment());
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, 'https://personal-origin.example/generate');
  assert.deepEqual(requests[0].options.headers, { Authorization: 'Bearer origin-secret', Accept: 'image/webp' });
  assert.equal(requests[0].options.redirect, 'error');
  assert.equal(requests[0].options.cache, 'no-store');
  assert.ok(requests[0].options.signal instanceof AbortSignal);
  assert.equal(response.status, 200);
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), image);
  assert.equal(response.headers.get('Content-Type'), 'image/webp');
  assert.match(response.headers.get('Cache-Control'), /no-store/);
  assert.equal(response.headers.get('X-Model-SHA256'), modelHash);
  assert.equal(response.headers.get('X-Generation-Seed'), generationSeed);
  assert.equal(response.headers.get('Server-Timing'), 'inference;dur=142.5');
  for (const name of ['Set-Cookie', 'X-Private-Origin', 'Server', 'ETag']) assert.equal(response.headers.get(name), null);
});

test('all other paths delegate unchanged to the static assets binding', async () => {
  const request = new Request('https://thispriestdoesnotexist.com/style.css?version=2');
  const expected = new Response('body{}');
  const env = environment();
  env.ASSETS.fetch = actual => { assert.equal(actual, request); return expected; };
  const handle = createHandler({ fetchImpl() { throw new Error('Unexpected inference'); } });
  assert.equal(await handle(request, env), expected);
});

test('missing credentials and unsafe origins fail closed without fetching', async () => {
  const handle = createHandler({ fetchImpl() { throw new Error('Should not fetch'); } });
  const invalid = [
    { INFERENCE_ORIGIN: undefined }, { INFERENCE_TOKEN: undefined }, { INFERENCE_TOKEN: '' },
    { INFERENCE_ORIGIN: 'http://personal-origin.example' },
    { INFERENCE_ORIGIN: 'https://user:password@personal-origin.example' },
    { INFERENCE_ORIGIN: 'https://personal-origin.example/path' },
    { INFERENCE_ORIGIN: 'https://personal-origin.example/?query=1' },
    { INFERENCE_ORIGIN: 'https://personal-origin.example/#hash' },
  ];
  for (const change of invalid) {
    const response = await handle(new Request('https://site.example/generate'), { ...environment(), ...change });
    assert.equal(response.status, 503);
    assert.equal(response.headers.get('Retry-After'), '3');
    assert.match(response.headers.get('Cache-Control'), /no-store/);
  }
});

test('wrong HTTP methods do not perform inference', async () => {
  const handle = createHandler({ fetchImpl() { throw new Error('Should not fetch'); } });
  for (const method of ['POST', 'HEAD', 'DELETE']) {
    const response = await handle(new Request('https://site.example/generate', { method }), environment());
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('Allow'), 'GET');
  }
});

test('upstream errors, redirects, invalid media, empty bodies and exceptions become uncached 503s', async () => {
  const outcomes = [() => new Response('error', { status: 500 }),
    () => new Response('busy', { status: 503 }),
    () => new Response(null, { status: 302, headers: { Location: 'https://evil.example' } }),
    () => new Response('not an image', { headers: { 'Content-Type': 'text/html' } }),
    () => new Response(null, { headers: { 'Content-Type': 'image/webp' } }),
    () => { throw new Error('private origin connection detail'); }];
  for (const outcome of outcomes) {
    const handle = createHandler({ fetchImpl: outcome });
    const response = await handle(new Request('https://site.example/generate'), environment());
    assert.equal(response.status, 503);
    assert.equal(response.headers.get('Retry-After'), '3');
    assert.match(response.headers.get('Cache-Control'), /no-store/);
    assert.equal(await response.text(), '');
  }
});

test('deadline aborts upstream and clears timer without leaking failure details', async () => {
  let expire;
  let cleared;
  const handle = createHandler({
    fetchImpl: async (_url, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('aborted private origin')), { once: true });
    }),
    schedule(callback, ms) { assert.equal(ms, 15_000); expire = callback; return 'timer'; },
    cancel(timer) { cleared = timer; },
  });
  const pending = handle(new Request('https://site.example/generate'), environment());
  expire();
  const response = await pending;
  assert.equal(response.status, 503);
  assert.equal(cleared, 'timer');
  assert.equal(await response.text(), '');
});

test('arbitrary upstream metadata is not copied into response headers', async () => {
  const handle = createHandler({ fetchImpl: async () => new Response(image, { headers: {
    'Content-Type': 'image/webp', 'X-Model-SHA256': 'internal filename',
    'X-Generation-Seed': 'not-a-seed', 'Server-Timing': 'db;desc="private origin"',
  } }) });
  const response = await handle(new Request('https://site.example/generate'), environment());
  assert.equal(response.status, 200);
  for (const name of ['X-Model-SHA256', 'X-Generation-Seed', 'Server-Timing']) assert.equal(response.headers.get(name), null);
});
