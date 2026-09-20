"""Promote only the visually reviewed alignment previews to an isolated dataset."""
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'manifest.json').read_text())
output=HERE/'partial256'
output.mkdir(exist_ok=True)
entries=[]
for entry in manifest['entries']:
    if entry['group']=='ffhq':continue
    if not entry['detected']:raise ValueError(f'Missing landmark detection: {entry["id"]}')
    source=HERE/'partial-aligned'/f'{entry["id"]}.png'
    if Image.open(source).size!=(256,256):raise ValueError('Unexpected size')
    target=output/source.name
    shutil.copyfile(source,target)
    entries.append({'id':entry['id'],'image':target.name,'source':entry['source'],
                    'source_sha256':entry['source_sha256'],
                    'training_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                    'matrix_source256_to_output256':entry['partial']['matrix_source256_to_output256'],
                    'collar_box_heuristic':entry['collar_box_heuristic'],
                    'reflected_padding_fraction':entry['partial']['reflected_padding_fraction']})
if len(entries)!=110:raise ValueError(f'Expected 110 inputs, got {len(entries)}')
result={'version':'partial-align-v1-2026-09-20','count':110,'resolution':256,
        'source_manifest_sha256':hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),
        'analysis_script_sha256':hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest(),
        'detector_sha256':manifest['detector_sha256'],
        'settings':{'target_eye_y':.4,'target_eye_x':.5,'desired_interocular':.24,
                    'top_guard':.02,'collar_bottom_guard':.96,'collar_padding_source_pixels':4,
                    'border':'OpenCV BORDER_REFLECT_101','resize':'Pillow Lanczos to256square',
                    'warp':'OpenCV INTER_LANCZOS4','scale_clamped_to_preserve_head_and_collar':True},
        'visual_review':{'status':'accepted_as_training_candidates','reviewer':'Codex',
                        'evidence':['partial-aligned-contact.png','comparison-contact.png',
                                    'partial-aligned/original-013.png','partial-aligned/synthetic-083.png'],
                        'finding':'All110 displayed candidates retain clearly visible white collars and bare heads. The two heuristic collar misses were individually inspected. This is preprocessing review, not a generated-model approval.'},
        'entries':entries}
(HERE/'partial256-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Prepared {len(entries)} isolated, reviewed 256x256 training candidates')
