"""Lossless review-only contact sheets; entire image uniformly reduced, no crops/repairs."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'research/runs/adam-native1024-retained250-main100-broad32-v1'
manifest=json.loads((RUN/'manifest.json').read_text());supervisor=json.loads((RUN/'supervisor.json').read_text())
assert manifest['complete'] and supervisor['complete'] and supervisor['failure'] is None
assert len(manifest['images'])==64
records={(im['arm'],im['index']):im for im in manifest['images']}
outputs=[]
for start in range(0,32,4):
 canvas=Image.new('RGB',(1024,4*540),(18,18,18));draw=ImageDraw.Draw(canvas)
 for row,index in enumerate(range(start,start+4)):
  for column,arm in enumerate(('raw','ema')):
   record=records[arm,index];path=RUN/record['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
   with Image.open(path) as im:
    assert im.size==(1024,1024)
    thumb=im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
   x=column*512;y=row*540;draw.text((x+8,y+6),f'{index:03} {arm.upper()} - complete frame at 50% display scale',fill=(240,240,240));canvas.paste(thumb,(x,y+28))
 path=HERE/f'contact-{start:03}-{start+3:03}.png';canvas.save(path)
 outputs.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'indices':list(range(start,start+4))})
(HERE/'contact-sheets-manifest.json').write_text(json.dumps({'source_manifest_sha256':hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),'purpose':'Review-only layout; native originals remain untouched and linked in page','method':'All64 images in original order,512px uniform LANCZOS thumbnails,no cropping,color changes,repairs or selection','outputs':outputs},indent=2)+'\n')
print({'contact_sheets':len(outputs),'images':64})
