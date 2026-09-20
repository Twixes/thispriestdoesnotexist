const COUNT = 80;
// Calendar-style selection; also the initial training set for the generator.
const pool = [1, 2, 3, 4, 7, 9, 10, 13, 18, 20, 22, 23, 25, 27, 33, 36, 38, 45, 46, 47,
  ...Array.from({ length: 30 }, (_, i) => i + 51)].map(number => number - 1);
const STORAGE_KEY = 'last-priest';
const portrait = document.querySelector('#portrait');
let current = -1;
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved !== null && /^\d+$/.test(saved) && Number(saved) < COUNT) current = Number(saved);
} catch { /* Portraits also work when browser storage is disabled. */ }

async function showAnother() {
  const candidates = pool.filter(i => i !== current);
  while (candidates.length) {
    const position = Math.floor(Math.random() * candidates.length);
    const next = candidates.splice(position, 1)[0];
    const image = new Image();
    image.src = `/portraits/${String(next + 1).padStart(2, '0')}.webp`;
    try {
      await image.decode();
      portrait.src = image.src;
      current = next;
      try { sessionStorage.setItem(STORAGE_KEY, String(current)); } catch { /* Optional storage. */ }
      return;
    } catch { /* Try another portrait if this asset failed to load. */ }
  }
}
showAnother();
