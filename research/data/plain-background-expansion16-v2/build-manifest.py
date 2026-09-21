"""Read original portraits, verify byte-exact provenance and create diagnostic contact."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import PIL
from PIL import Image, ImageChops, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
records=[]
for path in sorted(HERE.glob('tool-results-*.json')):
 records.extend(json.loads(path.read_text()))
records.sort(key=lambda item:item['id'])
assert [x['id'] for x in records]==list(range(165,181))
contact=Image.new('RGB',(4*256,4*280),'#181818'); draw=ImageDraw.Draw(contact)
entries=[]
for ordinal,x in enumerate(records):
 path=HERE/f"{x['id']}.png"; source=Path(x['source_path'])
 before=sha(path); assert before==sha(source)
 with Image.open(path) as image:
  image.load(); dimensions=list(image.size); mode=image.mode; rgb=image.convert('RGB')
  r,g,b=rgb.split()
  delta=max(ImageChops.difference(r,g).getextrema()[1],ImageChops.difference(r,b).getextrema()[1],ImageChops.difference(g,b).getextrema()[1])
  preview=rgb.resize((256,256),Image.Resampling.LANCZOS)
  left,top=(ordinal%4)*256,(ordinal//4)*280
  contact.paste(preview,(left,top));draw.text((left+8,top+260),str(x['id']),fill='white')
 assert before==sha(path)==sha(source)
 entries.append({'id':x['id'],'path':str(path.relative_to(ROOT)),'absolute_path':str(path),'sha256':before,'source_path':str(source),'source_sha256':sha(source),'dimensions':dimensions,'mode':mode,'max_rgb_channel_difference':delta,'exact_grayscale_pixels':delta==0,'copied_byte_exact':True,'call_count':1,'references':[],'prompt':x['prompt']})
contact_path=HERE/'contact-256.png';contact.save(contact_path)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tool':'built-in image_gen.imagegen','calls':16,'retry_count':0,'candidate_count':16,'model_output':False,'production_approved':False,'active_dataset_modified':False,'original_pixel_modifications':False,'all_original_source_and_copy_hashes_equal_before_after':True,'prompt_intent_note':'Demographic, age and handsome descriptors specify fictional generation intent; they do not establish actual ethnicity, age or identity from pixels. Anatomy/collar quality does not automatically establish hot-target acceptance.','versions':{'python':sys.version,'pillow':PIL.__version__,'platform':platform.platform()},'script_sha256':sha(Path(__file__)),'prompts_sha256':sha(HERE/'prompts.json'),'contact':{'path':'contact-256.png','sha256':sha(contact_path),'method':'4x4 whole-original256 LANCZOS diagnostic thumbnails with24px label strip; no image crops, recoloring or original changes.'},'entries':entries}
selection_path=ROOT/'research/reviews/expansion16-v2-root-observations.json'
selection=json.loads(selection_path.read_text())
assert selection['review_complete']
assert set(selection['accepted_target_ids'])|set(selection['reserved_ids'])==set(range(165,181))
assert not set(selection['accepted_target_ids'])&set(selection['reserved_ids'])
by_id={item['id']:item for item in selection['reviewed_ids']}
for entry in entries:
 assert entry['sha256']==by_id[entry['id']]['sha256']
 entry['target_selection']='accepted_hot_training_candidate' if entry['id'] in selection['accepted_target_ids'] else 'reserved_excluded_from_hot_target_subset'
manifest['target_selection']={'source_path':str(selection_path.relative_to(ROOT)),'source_sha256':sha(selection_path),'accepted_target_ids':selection['accepted_target_ids'],'reserved_ids':selection['reserved_ids'],'basis':'Explicit root native review; subjective calendar aesthetic only, not anatomy, age or demographic rule. All16 originals preserved. Training inclusion and production model quality are separate.'}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'count':len(entries),'dimensions':sorted({tuple(e['dimensions'])for e in entries}),'strict_grayscale_count':sum(e['exact_grayscale_pixels']for e in entries),'channel_difference_range':[min(e['max_rgb_channel_difference']for e in entries),max(e['max_rgb_channel_difference']for e in entries)],'manifest_sha256':sha(HERE/'manifest.json')},indent=2))
