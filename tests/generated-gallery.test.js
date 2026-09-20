import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../inference/client-app.js', import.meta.url), 'utf8');

const flush = () => new Promise(resolve => setImmediate(resolve));

function loadPage({ nonceStart = 0, failures = 0, upstreamFailures = 0 } = {}) {
  const portrait = { src: '' };
  const requested = [];
  const timers = [];
  const revoked = [];
  const decoded = [];
  let nonce = nonceStart;
  let attempts = 0;
  const context = vm.createContext({
    document: { querySelector(selector) { assert.equal(selector, '#portrait'); return portrait; } },
    crypto: { randomUUID() { return `uuid-${++nonce}`; } },
    AbortSignal: { timeout(ms) { assert.equal(ms, 20_000); return 'timeout-signal'; } },
    async fetch(url, options) {
      requested.push(url);
      assert.equal(options.cache, 'no-store');
      assert.equal(options.signal, 'timeout-signal');
      return { ok: requested.length > upstreamFailures,
        headers: { get() { return 'image/webp'; } },
        async blob() { return { nonce }; } };
    },
    URL: { createObjectURL(blob) { return `blob:${blob.nonce}`; },
      revokeObjectURL(url) { revoked.push(url); } },
    Image: class {
      set src(value) { this.url = value; decoded.push(value); }
      get src() { return this.url; }
      async decode() { if (attempts++ < failures) throw new Error('decode failed'); }
    },
    setTimeout(callback, delay) { timers.push({ callback, delay }); },
  });
  const done = vm.runInContext(source, context);
  return { done, portrait, requested, timers, revoked, decoded };
}

test('each page load requests fresh inference with a new nonce', async () => {
  const first = loadPage();
  await first.done;
  const second = loadPage({ nonceStart: 1 });
  await second.done;
  assert.deepEqual(first.requested, ['/generate?request=uuid-1']);
  assert.deepEqual(second.requested, ['/generate?request=uuid-2']);
  assert.equal(first.portrait.src, 'blob:1');
  assert.equal(second.portrait.src, 'blob:2');
  assert.deepEqual(first.decoded, ['blob:1']);
  assert.equal(first.timers.length, 0);
});

test('decode failures retry fresh inference after bounded backoff and then display success', async () => {
  const page = loadPage({ failures: 2 });
  await flush();
  assert.equal(page.portrait.src, '');
  assert.equal(page.timers[0].delay, 3000);
  page.timers[0].callback();
  await flush();
  assert.equal(page.timers[1].delay, 6000);
  page.timers[1].callback();
  await page.done;
  assert.deepEqual(page.requested, ['/generate?request=uuid-1', '/generate?request=uuid-2', '/generate?request=uuid-3']);
  assert.equal(page.portrait.src, 'blob:3');
  assert.deepEqual(page.revoked, ['blob:1', 'blob:2']);
});

test('persistent failure stops after three attempts without a catalog fallback', async () => {
  const page = loadPage({ failures: 100 });
  await flush();
  page.timers[0].callback();
  await flush();
  page.timers[1].callback();
  await page.done;
  assert.equal(page.requested.length, 3);
  assert.equal(page.timers.length, 2);
  assert.equal(page.portrait.src, '');
  assert.ok(page.requested.every(url => url.startsWith('/generate?request=')));
  assert.deepEqual(page.revoked, ['blob:1', 'blob:2', 'blob:3']);
});

test('upstream failure retries without decoding an error body', async () => {
  const page = loadPage({ upstreamFailures: 1 });
  await flush();
  assert.deepEqual(page.decoded, []);
  page.timers[0].callback();
  await page.done;
  assert.equal(page.requested.length, 2);
  assert.deepEqual(page.decoded, ['blob:2']);
  assert.equal(page.portrait.src, 'blob:2');
});
