"""Compare collar-preserving crops without the optional whole-head constraint."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
OUT=HERE/'collar-only'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)
source=json.loads((HERE/'manifest.json').read_text())
entries=[]
for entry in source['entries']:
    if entry['group']=='ffhq' or not entry['detected']:continue
    rgb=np.array(Image.open(HERE/'original'/f'{entry["id"]}.png'))
    points=np.array(entry['landmarks'])
    left,right=sorted(points[:2],key=lambda p:p[0]);eyes=(left+right)/2
    mouth=points[3:5].mean(axis=0);eye_to_eye=right-left;eye_to_mouth=mouth-eyes
    axis=eye_to_eye-np.array([-eye_to_mouth[1],eye_to_mouth[0]])
    axis/=np.linalg.norm(axis);vertical=np.array([-axis[1],axis[0]])
    basis=np.stack([axis,vertical])
    collar=entry['collar_box_heuristic']
    if collar:
        x,y,w,h=collar
        guard=[np.array([x,y+h+4]),np.array([x+w,y+h+4])]
    else:guard=[mouth+vertical*np.linalg.norm(eye_to_mouth)*1.8]
    depth=max(vertical@(p-eyes) for p in guard)
    result={'id':entry['id'],'source':entry['source'],'source_sha256':entry['source_sha256'],
            'collar_box_heuristic':collar}
    for eye_y in [.4,.42]:
        name=f'eyes{int(eye_y*100)}';folder=OUT/name;folder.mkdir(exist_ok=True)
        scale=.24*256/np.linalg.norm(eye_to_eye)
        if depth>0:scale=min(scale,(.96-eye_y)*256/depth)
        offset=np.array([128,eye_y*256])-basis@eyes*scale
        matrix=np.column_stack([basis*scale,offset])
        transformed=cv2.warpAffine(rgb,matrix,(256,256),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
        target=folder/f'{entry["id"]}.png';Image.fromarray(transformed).save(target)
        coverage=cv2.warpAffine(np.ones((256,256),dtype=np.uint8),matrix,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        aligned=np.column_stack([points,np.ones(5)])@matrix.T
        result[name]={'matrix_source256_to_output256':matrix.tolist(),
                      'training_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'eye_y':eye_y,'interocular':float(np.linalg.norm(aligned[0]-aligned[1])/256),
                      'mouth_y':float(aligned[3:5].mean(axis=0)[1]/256),
                      'reflected_padding_fraction':float(1-coverage.mean())}
    entries.append(result)
summary={}
for variant in ['eyes40','eyes42']:
    summary[variant]={key:{'mean':float(np.mean([e[variant][key] for e in entries])),
                          'std':float(np.std([e[variant][key] for e in entries]))} for key in ['eye_y','interocular','mouth_y','reflected_padding_fraction']}
    columns=10;thumb=128;rows=(len(entries)+columns-1)//columns
    canvas=Image.new('RGB',(columns*thumb,rows*(thumb+16)),(25,25,25));draw=ImageDraw.Draw(canvas)
    for index,entry in enumerate(entries):
        x=index%columns*thumb;y=index//columns*(thumb+16)
        canvas.paste(Image.open(OUT/variant/f'{entry["id"]}.png').resize((thumb,thumb)),(x,y))
        draw.text((x+2,y+thumb+1),entry['id'],fill='white')
    canvas.save(OUT/f'{variant}-contact.png')
selected=entries[::10]
canvas=Image.new('RGB',(256*4,len(selected)*272),(25,25,25));draw=ImageDraw.Draw(canvas)
for row,entry in enumerate(selected):
    paths=[HERE/'original'/f'{entry["id"]}.png',HERE/'partial-aligned'/f'{entry["id"]}.png',
           OUT/'eyes40'/f'{entry["id"]}.png',OUT/'eyes42'/f'{entry["id"]}.png']
    for column,(path,label) in enumerate(zip(paths,['original','whole-head partial','collar-only40','collar-only42'])):
        canvas.paste(Image.open(path),(column*256,row*272))
        draw.text((column*256+2,row*272+257),entry['id']+' '+label,fill='white')
canvas.save(OUT/'comparison-contact.png')
(OUT/'manifest.json').write_text(json.dumps({'version':'collar-only-v1','source_manifest_sha256':hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'settings':{'desired_interocular':.24,'collar_bottom_guard':.96,'collar_padding_source_pixels':4,'whole_head_constraint':False},
    'summary':summary,'entries':entries},indent=2)+'\n')
print(json.dumps(summary,indent=2))
