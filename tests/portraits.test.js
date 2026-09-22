import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const source = readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const KEY = 'priest-shuffle-v1';
const expected = [1,2,3,4,7,9,10,13,18,20,22,23,25,27,33,36,38,45,46,47,
  ...Array.from({length:180}, (_,i) => i+51)];
function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
}
function locks() {
  let tail = Promise.resolve();
  return { request: (_key, callback) => {
    const next = tail.then(callback);
    tail = next.catch(() => {});
    return next;
  } };
}
function page(local, { session = storage(), lockManager, fail = () => false, random = Math.random } = {}) {
  const portrait = {};
  const ready = vm.runInNewContext(source, {
    document: { querySelector: () => portrait }, localStorage: local, sessionStorage: session,
    navigator: lockManager ? { locks: lockManager } : {},
    Image: class { async decode() { if (fail(this.src)) throw new Error('Image unavailable'); } },
    Math: { floor: Math.floor, random },
  });
  return { portrait, ready };
}
async function draw(local, options) {
  const p = page(local, options); await p.ready; return p.portrait.src;
}
const id = src => Number(src.match(/([0-9]+)\.webp$/)[1]);

test('two complete rounds cover all 200 exactly once and avoid a boundary repeat', async () => {
  const saved = storage();
  let previous;
  for (let round = 0; round < 2; round++) {
    const seen = [];
    for (let i = 0; i < 200; i++) {
      const src = await draw(saved, { random: () => 0 });
      assert.notEqual(src, previous); previous = src; seen.push(id(src));
    }
    assert.deepEqual(seen.sort((a,b) => a-b), expected);
  }
});
test('fresh page contexts resume a partially consumed persistent queue', async () => {
  const saved = storage(); const seen = [];
  for (let i = 0; i < 27; i++) seen.push(await draw(saved));
  assert.equal(JSON.parse(saved.getItem(KEY)).remaining.length, 173);
  const next = await draw(saved, { session: storage() });
  assert.ok(!seen.includes(next));
  assert.equal(JSON.parse(saved.getItem(KEY)).remaining.length, 172);
});
test('fallback queue serializes 200 reservations under a shared lock', async () => {
  const saved = storage(); const lockManager = locks();
  const pages = Array.from({length:200}, () => page(saved, {lockManager}));
  await Promise.all(pages.map(p => p.ready));
  assert.deepEqual(pages.map(p => id(p.portrait.src)).sort((a,b) => a-b), expected);
});
test('malformed and stale queues recover without accepting invalid portrait IDs', async () => {
  for (const raw of ['{', 'null', JSON.stringify({catalog:'old',remaining:[9999],last:1}),
    JSON.stringify({catalog:expected.join(','),remaining:[51,51],last:2}),
    JSON.stringify({catalog:expected.join(','),remaining:[9999],last:2})]) {
    const saved = storage(); saved.setItem(KEY, raw);
    assert.ok(expected.includes(id(await draw(saved))));
  }
});
test('blocked persistent storage falls back to a complete session queue', async () => {
  const blocked = { getItem(){throw Error();}, setItem(){throw Error();} };
  const session = storage(); const seen = [];
  for(let i=0;i<200;i++) seen.push(id(await draw(blocked,{session})));
  assert.deepEqual(seen.sort((a,b)=>a-b), expected);
  assert.ok(await draw(blocked,{session:blocked}));
});
test('storage becoming unwritable and failed assets still allow a portrait', async () => {
  const saved=storage(); const setter=saved.setItem;
  saved.setItem=(key,value)=>{if(key===KEY)throw Error('quota'); setter(key,value);};
  assert.equal(await draw(saved,{fail:src=>!src.endsWith('/07.webp')}), '/portraits/07.webp');
});
test('asset failure across a cycle boundary tries every portrait at most once', async () => {
  const saved=storage();
  saved.setItem(KEY,JSON.stringify({catalog:expected.join(','),remaining:[1,2],last:3}));
  const attempted=[];
  assert.equal(await draw(saved,{fail:src=>{attempted.push(src);return true;}}),undefined);
  assert.equal(attempted.length,200);
  assert.equal(new Set(attempted).size,200);
});
test('all 230 archived portraits are deployed, including the 200 active selections', () => {
  const files = readdirSync(new URL('../public/portraits/', import.meta.url)).filter(n=>n.endsWith('.webp'));
  assert.deepEqual(files.sort(), Array.from({length:230},(_,i)=>String(i+1).padStart(2,'0')+'.webp').sort());
});
