"""Prepare a new 36-image candidate set without changing either training run."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'research/data/plain-background24-whole-image'
REVIEW = ROOT / 'research/reviews/expansion16-v2-root-observations.json'
OUT = ROOT / 'research/data/plain-background36-whole-image'
ACCEPTED = [165, 166, 168, 170, 171, 173, 174, 175, 176, 177, 179, 180]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'New output only; never overwrite research data'
    assert sha(OLD / 'manifest.json') == '334ad5bfa22c72baab1d7a24298616483d125a9fccbc3a7816b1b3b2ef59067d'
    previous = json.loads((OLD / 'manifest.json').read_text())
    review_hash = sha(REVIEW)
    review = json.loads(REVIEW.read_text())
    assert review['review_complete'] and review['accepted_target_ids'] == ACCEPTED
    selected = [r for r in review['reviewed_ids'] if r['id'] in ACCEPTED]
    assert len(previous['entries']) == 24 and len(selected) == 12
    for row in selected:
        assert sha(ROOT / row['path']) == row['sha256']
    for row in previous['entries']:
        for output in row['outputs'].values():
            assert sha(OLD / output['path']) == output['sha256']
    OUT.mkdir()
    for size in (1024, 256):
        (OUT / str(size)).mkdir()
    entries = []
    for row in previous['entries']:
        entry = {'id': row['id'], 'origin': 'previous24-byte-identical',
                 'source_path': row['source_path'], 'source_sha256': row['source_sha256'], 'outputs': {}}
        for size in (1024, 256):
            old = row['outputs'][str(size)]
            dest = OUT / str(size) / f"{row['id']}.png"
            shutil.copyfile(OLD / old['path'], dest)
            assert sha(dest) == old['sha256']
            entry['outputs'][str(size)] = {'path': str(dest.relative_to(OUT)), 'sha256': sha(dest)}
        entries.append(entry)
    for row in selected:
        source = ROOT / row['path']
        with Image.open(source) as original:
            assert original.size == (1254, 1254)
            gray = original.convert('L').convert('RGB')
        entry = {'id': row['id'], 'origin': 'new12-root-selected',
                 'source_path': row['path'], 'source_sha256': row['sha256'], 'outputs': {}}
        for size in (1024, 256):
            expected = gray.resize((size, size), Image.Resampling.LANCZOS)
            dest = OUT / str(size) / f"{row['id']}.png"
            expected.save(dest, format='PNG', optimize=False, compress_level=6)
            with Image.open(dest) as saved:
                assert saved.mode == 'RGB' and saved.size == (size, size)
                assert saved.tobytes() == expected.tobytes()
                channels = saved.split()
                assert channels[0].tobytes() == channels[1].tobytes() == channels[2].tobytes()
            entry['outputs'][str(size)] = {'path': str(dest.relative_to(OUT)), 'sha256': sha(dest)}
        assert sha(source) == row['sha256']
        entries.append(entry)
    assert len(entries) == 36 and len({e['id'] for e in entries}) == 36
    sheet = Image.new('RGB', (1536, 1680), '#181818')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(entries):
        x, y = i % 6 * 256, i // 6 * 280
        with Image.open(OUT / row['outputs']['256']['path']) as image:
            sheet.paste(image, (x, y))
        draw.text((x + 8, y + 260), str(row['id']), fill='white')
    sheet.save(OUT / 'contact-256.png')
    assert sha(REVIEW) == review_hash
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'count': 36,
        'purpose': 'Prospective changed-data experiment only; not used by active reference110 or prepared NADA24.',
        'new_accepted_ids': ACCEPTED, 'reserved_excluded_ids': review['reserved_ids'],
        'selection_review': str(REVIEW.relative_to(ROOT)), 'selection_review_sha256': review_hash,
        'previous_manifest_sha256': sha(OLD / 'manifest.json'),
        'method': 'Previous24 outputs copied byte-for-byte; new12 independently converted original L to RGB and full-frame LANCZOS resized to1024 and256. No crop, padding, reflection or restoration.',
        'pillow_version': PIL.__version__, 'script_sha256': sha(Path(__file__)),
        'contact_sha256': sha(OUT / 'contact-256.png'), 'entries': entries,
        'model_output': False, 'production_approved': False}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'count': 36, 'new_count': 12, 'output': str(OUT), 'manifest_sha256': sha(OUT / 'manifest.json')}))


if __name__ == '__main__':
    main()
