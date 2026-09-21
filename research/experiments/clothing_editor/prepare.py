"""Materialize diagnostic oracle composites; no model loading or training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'research/data/paired10/manifest-proposed.json'
OUT = ROOT / 'research/data/clothing-editor-oracles-v1'
TRAIN = ['calibration-original', '000', '030', 'b2-055', 'b2-051', 'b2-020']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mask(polygons):
    image = Image.new('L', (1024, 1024))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([tuple(round(x*1023) for x in point) for point in polygon], fill=255)
    return np.asarray(image)>0

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records=[]
    for pair in json.loads(SOURCE.read_text())['pairs']:
        ident=pair['id']; directory=OUT/ident; directory.mkdir()
        paths={key:(SOURCE.parent/pair[key]).resolve() for key in ['source_path','target_path','latent_path']}
        source=np.asarray(Image.open(paths['source_path']).convert('RGB'), dtype=np.float32)@np.array([.299,.587,.114],np.float32)
        target=np.asarray(Image.open(paths['target_path']).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32)@np.array([.299,.587,.114],np.float32)
        assert source.shape==(1024,1024)
        allowed=mask(pair['clothing_polygons']); tab=mask(pair.get('collar_polygons',[])) & allowed
        source=np.rint(source).clip(0,255).astype('uint8'); target=np.rint(target).clip(0,255).astype('uint8')
        composite=np.where(allowed,target,source)
        for name,value in [('source',source),('target',target),('oracle',composite),('mask',allowed.astype('uint8')*255),('tab',tab.astype('uint8')*255)]:
            Image.fromarray(value).save(directory/f'{name}.png')
        ys,xs=np.where(allowed)
        # Fixed bottom-half proposal is audited, never silently cropped.
        record={'id':ident,'split':'train' if ident in TRAIN else 'heldout','source_files':{key:{'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for key,path in paths.items()},'files':{name:{'path':str((directory/f'{name}.png').relative_to(ROOT)),'sha256':sha(directory/f'{name}.png')} for name in ['source','target','oracle','mask','tab']},'allowed_pixels':int(allowed.sum()),'tab_pixels':int(tab.sum()),'allowed_bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'protected_exact':bool(np.array_equal(composite[~allowed],source[~allowed]))}
        records.append(record)
    (OUT/'manifest.json').write_text(json.dumps({'schema':1,'source_manifest':str(SOURCE.relative_to(ROOT)),'source_manifest_sha256':sha(SOURCE),'purpose':'Oracle-mask clothing compositor diagnostic; no learned inference, no production approval','train_ids':TRAIN,'heldout_ids':[r['id'] for r in records if r['split']=='heldout'],'records':records},indent=2)+'\n')
    sheet=Image.new('RGB',(5*320,2*350),(30,30,30));draw=ImageDraw.Draw(sheet)
    for i,record in enumerate(records):
        x=(i%5)*320;y=(i//5)*350
        image=Image.open(OUT/record['id']/'oracle.png').resize((320,320),Image.Resampling.LANCZOS)
        sheet.paste(image,(x,y));draw.text((x+5,y+322),record['id']+' '+record['split'],fill='white')
    sheet.save(OUT/'contact.png')
    print(json.dumps({'output':str(OUT),'count':len(records),'bounds':{r['id']:r['allowed_bbox'] for r in records}},indent=2))
if __name__=='__main__':main()
