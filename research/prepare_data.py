"""Reproducible training set: the original curated 20 plus 30 commissioned hot priests."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
IDS = [1,2,3,4,7,9,10,13,18,20,22,23,25,27,33,36,38,45,46,47] + list(range(51,81))
manifest = {'selection': 'Original subjective hot-priest selection plus all thirty explicitly commissioned handsome portraits. No ordinary-priest expansion.',
            'source_commit': '6ded5a6ee247fc7b67c27989ba4536da8a7c8e89', 'resolution': 256, 'images': []}
out = ROOT / 'data/priests256'
out.mkdir(parents=True, exist_ok=True)
for id in IDS:
    source = ROOT.parent / f'public/portraits/{id:02}.webp'
    target = out / f'{id:02}.png'
    im = Image.open(source).convert('RGB')
    # Preserve the original square composition and collar; do not crop to a face-only image.
    ImageOps.fit(im, (256,256), method=Image.Resampling.LANCZOS).save(target)
    manifest['images'].append({'id':id, 'source':str(source.relative_to(ROOT.parent)),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'training_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Prepared {len(IDS)} training images')
