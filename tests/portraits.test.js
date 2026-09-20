import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const source = readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
function page(storage, fail = () => false, random = Math.random, collection) {
  const elements = { '#portrait': {} };
  vm.runInNewContext(source, {
    document: { body: { dataset: { collection } }, querySelector: selector => elements[selector] },
    sessionStorage: storage,
    Image: class { async decode() { if (fail(this.src)) throw new Error('Image unavailable'); } },
    Math: { floor: Math.floor, random },
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
  for (let i = 0; i < 1000; i++) {
    const elements = page(saved, () => false, () => ((i * 37) % 997) / 997);
    await tick();
    const image = elements['#portrait'];
    assert.match(image.src, /^\/portraits\/(0[1-9]|[1-4][0-9]|50)\.webp$/);
    assert.notEqual(image.src, previous);
    seen.add(image.src);
    previous = image.src;
  }
  assert.equal(seen.size, 50);
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
  assert.equal(attempts, 50);
  assert.equal(elements['#portrait'].src, undefined);
});

test('all fifty numbered portraits are included in the deployment', () => {
  const files = readdirSync(new URL('../public/portraits/', import.meta.url)).filter(name => name.endsWith('.webp')).sort();
  assert.deepEqual(files, Array.from({ length: 50 }, (_, i) => `${String(i + 1).padStart(2, '0')}.webp`));
});


test('hot reloads stay within the curated selection and avoid repeats', async () => {
  const expected = [1, 2, 3, 4, 7, 9, 10, 13, 18, 20, 22, 23, 25, 27, 33, 36, 38, 45, 46, 47]
    .map(number => `/portraits/${String(number).padStart(2, '0')}.webp`);
  const saved = storage();
  const seen = new Set();
  let previous;
  for (let i = 0; i < 1000; i++) {
    const elements = page(saved, () => false, () => ((i * 37) % 997) / 997, 'hot');
    await tick();
    const src = elements['#portrait'].src;
    assert.ok(expected.includes(src));
    assert.notEqual(src, previous);
    seen.add(src);
    previous = src;
  }
  assert.deepEqual([...seen].sort(), expected.sort());
});

test('hot asset failures never fall back outside the curated selection', async () => {
  const attempted = [];
  const elements = page(storage(), src => { attempted.push(src); return true; }, () => 0, 'hot');
  await tick();
  assert.equal(attempted.length, 20);
  assert.equal(new Set(attempted).size, 20);
  assert.equal(elements['#portrait'].src, undefined);
});
