"""Convert saved built-in image-generation results to web assets and QA sheets.

Run with Python + Pillow: python scripts/import-portraits.py
Safe to rerun as more docs/generation-200/<id>.json receipts arrive.
"""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
receipts = root / 'docs/generation-200'
review = receipts / 'review'
review.mkdir(exist_ok=True)
rows = []
for path in sorted(receipts.glob('[0-9]*.json'), key=lambda p: int(p.stem)):
    receipt = json.loads(path.read_text())
    number = receipt['id']
    destination = root / f'public/portraits/{number:02}.webp'
    if not destination.exists() or path.stat().st_mtime > destination.stat().st_mtime:
        with Image.open(receipt['source']) as image:
            assert image.width == image.height and image.width >= 1024, (number, image.size)
            image.convert('RGB').save(destination, 'WEBP', quality=90, method=6)
    with Image.open(destination) as image:
        dimensions = list(image.size)
    rows.append({'id': number, 'asset': str(destination.relative_to(root)),
                 'dimensions': dimensions, 'bytes': destination.stat().st_size,
                 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                 'prompt': f'docs/prompts/{number}.md', 'tool': receipt['tool']})

# Small labeled full-frame copies are review artifacts, never production portraits.
for start in range(81, 231, 15):
    group = [row for row in rows if start <= row['id'] < start + 15]
    if not group:
        continue
    sheet = Image.new('RGB', (1500, 972), '#161616')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(group):
        x, y = (i % 5) * 300, (i // 5) * 324
        with Image.open(root / row['asset']) as image:
            image.thumbnail((296, 296))
            sheet.paste(image, (x + 2, y + 24))
        draw.text((x + 10, y + 6), str(row['id']), fill='white')
    sheet.save(review / f'{start}-{start+14}.jpg', quality=92)
(receipts / 'assets.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps({'imported': len(rows), 'total_new': 150,
                  'webp_megabytes': round(sum(row['bytes'] for row in rows) / 1e6, 2)}))
