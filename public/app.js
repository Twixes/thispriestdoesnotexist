// The curated calendar collection: original favorites plus the expanded catalog.
const pool = [1, 2, 3, 4, 7, 9, 10, 13, 18, 20, 22, 23, 25, 27, 33, 36, 38, 45, 46, 47,
  ...Array.from({ length: 180 }, (_, i) => i + 51)];
const catalog = pool.join(',');
const STORAGE_KEY = 'priest-shuffle-v1';
const portrait = document.querySelector('#portrait');
let memoryState = null;

function chooseStorage() {
  for (const name of ['localStorage', 'sessionStorage']) {
    try {
      const storage = globalThis[name];
      const probe = `${STORAGE_KEY}-probe`;
      storage.setItem(probe, '1');
      storage.removeItem(probe);
      return storage;
    } catch { /* Try storage limited to this tab if persistent storage is unavailable. */ }
  }
  return null;
}
let storage = chooseStorage();

function shuffle() {
  const remaining = [...pool];
  for (let i = remaining.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
  }
  return remaining;
}

function advance(state) {
  const last = pool.includes(state?.last) ? state.last : null;
  const valid = state?.catalog === catalog && Array.isArray(state.remaining)
    && state.remaining.length <= pool.length
    && state.remaining.every(id => pool.includes(id))
    && new Set(state.remaining).size === state.remaining.length
    && !state.remaining.includes(last);

  if (!valid || state.remaining.length === 0) {
    state = { catalog, remaining: shuffle(), last };
    // Also avoid an immediate repeat where two shuffled rounds meet.
    const end = state.remaining.length - 1;
    if (state.remaining[end] === last) {
      const j = Math.floor(Math.random() * end);
      [state.remaining[end], state.remaining[j]] = [state.remaining[j], state.remaining[end]];
    }
  }
  const next = state.remaining.pop();
  state.last = next;
  return state;
}

function reserveFallback() {
  let state = memoryState;
  try { if (storage) state = JSON.parse(storage.getItem(STORAGE_KEY)); }
  catch { state = null; }
  memoryState = advance(state);
  try { storage?.setItem(STORAGE_KEY, JSON.stringify(memoryState)); }
  catch { storage = null; }
  return memoryState.last;
}

// A read/write transaction makes reading and advancing the queue one atomic
// operation across tabs. Local storage alone can expose stale per-tab copies.
const database = globalThis.indexedDB ? new Promise(resolve => {
  let settled = false;
  const finish = db => {
    if (settled) { db?.close(); return; }
    settled = true;
    clearTimeout(timeout);
    resolve(db);
  };
  const timeout = setTimeout(() => finish(null), 3000);
  try {
    const request = indexedDB.open(STORAGE_KEY, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('queue');
    request.onsuccess = () => finish(request.result);
    request.onerror = request.onblocked = () => finish(null);
  } catch { finish(null); }
}) : Promise.resolve(null);

async function reservePortrait() {
  const db = await database;
  if (db) {
    try {
      return await new Promise((resolve, reject) => {
        const transaction = db.transaction('queue', 'readwrite');
        const store = transaction.objectStore('queue');
        const request = store.get('current');
        let next;
        request.onsuccess = () => {
          const state = advance(request.result);
          next = state.last;
          store.put(state, 'current');
        };
        transaction.oncomplete = () => resolve(next);
        transaction.onerror = transaction.onabort = () => reject(transaction.error);
      });
    } catch { /* Storage restrictions must not prevent an image from loading. */ }
  }
  return globalThis.navigator?.locks?.request
    ? navigator.locks.request(STORAGE_KEY, reserveFallback)
    : reserveFallback();
}

async function showAnother() {
  const attempted = new Set();
  // A partial round plus a full round visits every candidate, even after load failures.
  for (let draw = 0; draw < pool.length * 2 && attempted.size < pool.length; draw++) {
    const next = await reservePortrait();
    if (attempted.has(next)) continue;
    attempted.add(next);
    const image = new Image();
    image.src = `/portraits/${String(next).padStart(2, '0')}.webp`;
    try {
      await image.decode();
      portrait.src = image.src;
      return;
    } catch { /* Skip an unavailable image and reserve another. */ }
  }
}
showAnother();
