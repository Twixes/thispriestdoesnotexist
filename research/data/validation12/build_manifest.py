"""Archive provenance for held-out image-tool references; never training inputs."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
prompts = json.loads((ROOT / 'prompts.json').read_text())['prompts']
provenance = {row['id']: row for row in json.loads((ROOT / 'provenance.json').read_text())['entries']}
entries = []
sheet = Image.new('RGB', (4 * 320, 3 * 350), 'black')
draw = ImageDraw.Draw(sheet)
for index, item in enumerate(prompts):
    path = ROOT / 'source' / (item['id'] + '.png')
    with Image.open(path) as original:
        original.load()
        assert original.size == (1254, 1254), (path, original.size)
        assert original.mode == 'RGB', (path, original.mode)
        thumbnail = original.resize((320, 320), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, ((index % 4) * 320, (index // 4) * 350))
    draw.text(((index % 4) * 320 + 10, (index // 4) * 350 + 327),
              'validation ' + item['id'] + ' (image tool)', fill='white')
    entries.append({'id': item['id'], 'image': str(path.relative_to(ROOT)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'width': 1254, 'height': 1254, 'bytes': path.stat().st_size,
                    'prompt': item['prompt'], 'source_png': provenance[item['id']]['source_png'],
                    'use': 'held-out validation only', 'training_eligible': False})
assert len(entries) == 12 and len({row['sha256'] for row in entries}) == 12
sheet.save(ROOT / 'contact-sheet.jpg', quality=95)
(ROOT / 'manifest.json').write_text(json.dumps({
    'schema_version': 1, 'generator': 'OpenAI built-in image_gen',
    'purpose': 'Fresh synthetic validation references excluded from training',
    'count': 12, 'training_eligible': False, 'entries': entries,
    'limitations': 'Small development validation set from the same image tool; not real-world ground truth or an untouched final test set.'
}, indent=2) + '\n')
print('Archived 12 unique held-out originals, exact prompts and provenance.')
