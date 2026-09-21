"""PNG/NPZ-only integrity review and labeled diagnostic; no model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((HERE / 'manifest.json').read_text())
assert len(manifest['entries']) == 64
for entry in manifest['entries']:
    png, npz = HERE / entry['source_path'], HERE / entry['latent_path']
    assert sha(png) == entry['image_sha256']
    assert sha(npz) == entry['latents_sha256']
    with Image.open(png) as im:
        assert im.mode == 'RGB' and im.size == (1024, 1024)
        assert hashlib.sha256(im.tobytes()).hexdigest() == entry['rgb_sha256']
    with np.load(npz) as latent:
        for key, shape, checksum in [('z', (1, 512), 'z_raw_sha256'), ('w', (1, 18, 512), 'ws_raw_sha256')]:
            value = latent[key]
            assert value.shape == shape and value.dtype == np.float32
            assert np.isfinite(value).all()
            assert hashlib.sha256(value.tobytes()).hexdigest() == entry[checksum]
assert len({entry['seed_hex'] for entry in manifest['entries']}) == 64
assert len({entry['rgb_sha256'] for entry in manifest['entries']}) == 64

priority = [
    {'id': '055', 'reason': 'Clearly adult appearance, attractive direct gaze, red hair, glasses, clean background, substantial clothing below chin.', 'caveat': 'High neckline hides most neck; existing glasses and facial geometry must remain protected.'},
    {'id': '051', 'reason': 'Clearly mature adult appearance, angular face, red hair, subtle smile, no glasses, plain wall, usable neck.', 'caveat': 'Subjective attractiveness is less conventional calendar-model styling; root should review native image before editing.'},
]
conditional = [
    {'id': '020', 'reason': 'Clearly adult appearance, strong smile, short hair, usable neck and shirt area.', 'caveat': 'Harsh sunlight and blurred crowd; source is not an uncluttered studio portrait.'},
    {'id': '025', 'reason': 'Clearly adult appearance, attractive smile and stubble.', 'caveat': 'Very tight chin/lower-neck crop and partial second-person fragment at left; prefer not to spend an edit on this unless root accepts the composition.'},
]
reviewed = {
    '001': 'Exclude: unmistakable second face at left.',
    '007': 'Exclude: youthful/ambiguous age appearance for this adult-only shortlist, squinting and harsh light.',
    '012': 'Exclude: partial person at right, glasses reflections and tight collar space.',
    '015': 'Exclude despite attractive adult face: unmistakable second face at left.',
    '040': 'Reserve only: appealing mature adult face, glasses, clean background, but chin touches bottom and collar space is inadequate.',
    '042': 'Reserve only: friendly mature adult, bareheaded and usable neck; not a strong match for the requested hot-only calendar selection.',
    '050': 'Do not select: intact mature adult portrait but not a strong match for the requested hot-only calendar selection.',
    '058': 'Do not select: intact older adult portrait, formal jacket, but not a strong match for requested hot-only calendar selection.',
    '059': 'Reserve only: intact mature adult smiling portrait and usable neck, but weaker subjective match than priority candidates.',
    '060': 'Reserve only: intact mature adult portrait with usable neck, but weaker subjective match than priority candidates.',
    '061': 'Hard exclusion: ornate headwear, crowded background and another person; fails no-hats criterion.',
}
result = {
    'status': 'Subjective prospective edit shortlist only; no targets, training eligibility, or production approval.',
    'manifest_sha256': sha(HERE / 'manifest.json'),
    'all_sources_preserved': 64,
    'contact_sheet_all_64_inspected': True,
    'native_1024_images_inspected': sorted(list(reviewed) + [x['id'] for x in priority + conditional]),
    'priority_count': 2,
    'priority_candidates': priority,
    'conditional_candidates': conditional,
    'requested_8_to_12_strong_candidates_available': False,
    'shortfall_explanation': 'This unfiltered FFHQ batch does not contain 8-12 clearly adult, attractive, bareheaded male-presenting portraits with acceptable composition. Do not lower the visual gate to fill a quota.',
    'native_review_exclusions_and_reserves': reviewed,
    'contact_level_hard_exclusions': {
        'headcovering_or_head_ornament': ['003', '004', '006', '018', '023', '028', '029', '033', '036', '038', '041', '046', '047', '049', '061'],
        'child_or_age_ambiguous': ['007', '009', '011', '014', '030', '033', '039', '043', '044', '045', '048', '049', '054', '057', '063'],
        'obvious_partial_second_person': ['001', '002', '012', '015', '031', '039', '044', '053', '061'],
        'face_obscuring_accessory': ['013', '021', '038'],
    },
    'selection_limits': 'All identities are synthetic. Adult appearance and attractiveness are subjective visual judgments, not measured age or a demographic classifier. Unlisted contact images were not shortlisted for the requested masculine calendar aesthetic; no factual gender identity is assigned.',
    'verification': {'png_and_npz_sha256_passed': 64, 'rgb_sha256_passed': 64, 'finite_float32_z_1_512_and_w_1_18_512_passed': 64, 'unique_seed_and_rgb_counts': 64, 'no_model_loaded_by_review': True},
}
(HERE / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
canvas = Image.new('RGB', (1024, 2 * 550), '#171717')
draw = ImageDraw.Draw(canvas)
for i, row in enumerate(priority + conditional):
    x, y = (i % 2) * 512, (i // 2) * 550
    with Image.open(HERE / 'images' / f"{row['id']}.png") as im:
        canvas.paste(im.resize((512, 512), Image.Resampling.LANCZOS), (x, y))
    label = 'priority' if i < 2 else 'conditional: review caveats'
    draw.text((x + 10, y + 521), f"{row['id']} - {label}", fill='white')
canvas.save(HERE / 'shortlist-contact.png')
print(json.dumps({'verified': 64, 'priority': ['055', '051'], 'conditional': ['020', '025'], 'selection_sha256': sha(HERE / 'selection.json')}, indent=2))
