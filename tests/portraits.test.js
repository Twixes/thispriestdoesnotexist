import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const source = readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
function page(storage, fail = () => false) {
  const elements = { '#portrait': {} };
  vm.runInNewContext(source, {
    document: { querySelector: selector => elements[selector] },
    sessionStorage: storage,
    Image: class { async decode() { if (fail(this.src)) throw new Error('Image unavailable'); } },
    Math,
  });
  return elements;
}
function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test('reloads choose existing portraits without immediate repeats', async () => {
  const saved = storage();
  const seen = new Set();
  let previous;
  for (let i = 0; i < 200; i++) {
    const elements = page(saved);
    await tick();
    const image = elements['#portrait'];
    assert.match(image.src, /^\/portraits\/(0[1-9]|1[0-2])\.webp$/);
    assert.notEqual(image.src, previous);
    seen.add(image.src);
    previous = image.src;
  }
  assert.equal(seen.size, 12);
});

test('blocked storage does not stop portraits loading', async () => {
  const elements = page({ getItem() { throw new Error(); }, setItem() { throw new Error(); } });
  await tick();
  assert.ok(elements['#portrait'].src);
});

test('failed assets fall back to another image', async () => {
  const elements = page(storage(), src => !src.endsWith('/07.webp'));
  await tick();
  assert.equal(elements['#portrait'].src, '/portraits/07.webp');
});

test('total asset failure stops after trying each portrait', async () => {
  let attempts = 0;
  const elements = page(storage(), () => { attempts++; return true; });
  await tick();
  assert.equal(attempts, 12);
  assert.equal(elements['#portrait'].src, undefined);
});
