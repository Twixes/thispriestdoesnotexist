const COUNT = 12;
const STORAGE_KEY = 'last-priest';
const portrait = document.querySelector('#portrait');
const button = document.querySelector('#another');
const status = document.querySelector('#status');
let current = -1;
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved !== null && /^\d+$/.test(saved) && Number(saved) < COUNT) current = Number(saved);
} catch { /* Portraits also work when browser storage is disabled. */ }

async function showAnother() {
  button.disabled = true;
  const candidates = Array.from({ length: COUNT }, (_, i) => i).filter(i => i !== current);
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
      status.textContent = 'A new fictional priest portrait is ready.';
      button.disabled = false;
      return;
    } catch { /* Try another portrait if this asset failed to load. */ }
  }
  status.textContent = 'The portraits could not load. Please try again.';
  button.disabled = false;
}
button.addEventListener('click', showAnother);
showAnother();
