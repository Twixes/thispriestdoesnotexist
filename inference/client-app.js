// Promote to public/app.js only with a reviewed, deployed model and active proxy.
const portrait = document.querySelector('#portrait');
const ATTEMPTS = 3;

async function showGeneratedPriest() {
  for (let attempt = 0; attempt < ATTEMPTS; attempt += 1) {
    if (attempt) await new Promise(resolve => setTimeout(resolve, 3000 * attempt));
    let objectUrl;
    try {
      const response = await fetch(`/generate?request=${crypto.randomUUID()}`, {
        cache: 'no-store', signal: AbortSignal.timeout(20_000),
      });
      if (!response.ok || response.headers.get('Content-Type')?.split(';')[0] !== 'image/webp') {
        throw new Error('Generation unavailable');
      }
      objectUrl = URL.createObjectURL(await response.blob());
      const image = new Image();
      image.src = objectUrl;
      await image.decode();
      portrait.src = image.src;
      return;
    } catch {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      // A failed request retries fresh inference, never a stored portrait.
    }
  }
}

showGeneratedPriest();
