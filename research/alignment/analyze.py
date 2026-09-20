"""Measure YuNet landmarks and preview collar-preserving alignment; never edits training data.

FFHQ geometry follows NVIDIA's CC BY-NC-SA 4.0 recipe, retained in vendor/.
YuNet supplies five landmarks rather than FFHQ's original 68: an approximation.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
cv2.setNumThreads(2)
detector=cv2.FaceDetectorYN.create(str(HERE/'models/face_detection_yunet_2023mar.onnx'),'',(256,256),.75,.3,5000)


def collar_box(rgb, face, eyes, mouth):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    distance=np.linalg.norm(mouth-eyes)
    mask=(gray>170).astype(np.uint8)
    yy,xx=np.mgrid[:256,:256]
    mask*=((xx>mouth[0]-face[2]*.5)&(xx<mouth[0]+face[2]*.5)&
           (yy>mouth[1]+distance*.55)&(yy<min(251,mouth[1]+distance*2.2)))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<15 or w<face[2]*.08 or h>face[3]*.4 or not .35<w/max(h,1)<8:
            continue
        region=gray[max(0,y-7):min(256,y+h+7),max(0,x-7):min(256,x+w+7)]
        dark=float((region<110).mean())
        if dark<.3:continue
        expected=mouth+np.array([0,distance*1.4])
        score=dark*100-np.linalg.norm(centroids[i]-expected)
        candidates.append((score,[int(x),int(y),int(w),int(h)]))
    return max(candidates)[1] if candidates else None


def transform_matrix(face, kind, collar):
    points=face[4:14].reshape(5,2).astype(np.float64)
    left,right=sorted(points[:2],key=lambda p:p[0])
    eyes=(left+right)/2
    mouth=points[3:5].mean(axis=0)
    eye_to_eye=right-left
    eye_to_mouth=mouth-eyes
    axis=eye_to_eye-np.array([-eye_to_mouth[1],eye_to_mouth[0]])
    axis/=np.linalg.norm(axis)
    vertical=np.array([-axis[1],axis[0]])
    basis=np.stack([axis,vertical])
    if kind=='ffhq':
        half_size=max(np.linalg.norm(eye_to_eye)*2,np.linalg.norm(eye_to_mouth)*1.8)
        scale=256/(2*half_size)
        center=eyes+eye_to_mouth*.1
        offset=np.array([128,128])-basis@center*scale
    else:
        scale=.24*256/np.linalg.norm(eye_to_eye)
        head_top=max(0,float(face[1]-face[3]*.4))
        guard=[np.array([eyes[0],head_top])]
        if collar:
            x,y,w,h=collar
            guard.extend([np.array([x,y+h+4]),np.array([x+w,y+h+4])])
        else:
            guard.append(mouth+vertical*np.linalg.norm(eye_to_mouth)*1.8)
        projected=np.array([vertical@(p-eyes) for p in guard])
        if projected.min()<0:scale=min(scale,(.4-.02)*256/-projected.min())
        if projected.max()>0:scale=min(scale,(.96-.4)*256/projected.max())
        offset=np.array([128,.4*256])-basis@eyes*scale
    return np.column_stack([basis*scale,offset])


def metrics(face, matrix=None):
    points=face[4:14].reshape(5,2).astype(float)
    if matrix is not None:points=np.column_stack([points,np.ones(5)])@matrix.T
    eyes=points[:2].mean(axis=0);mouth=points[3:5].mean(axis=0)
    return {'eye_y':float(eyes[1]/256),'eye_x':float(eyes[0]/256),
            'mouth_y':float(mouth[1]/256),'interocular':float(np.linalg.norm(points[0]-points[1])/256),
            'eye_mouth_distance':float(np.linalg.norm(mouth-eyes)/256)}


def contains_collar(matrix, box):
    if box is None:return None
    x,y,w,h=box
    corners=np.array([[x,y,1],[x+w,y,1],[x,y+h,1],[x+w,y+h,1]])@matrix.T
    return bool((corners>=0).all() and (corners<=255).all())


def sheet(paths, target, columns=10, thumb=128):
    rows=(len(paths)+columns-1)//columns
    canvas=Image.new('RGB',(columns*thumb,rows*(thumb+16)),(25,25,25));draw=ImageDraw.Draw(canvas)
    for i,(path,label) in enumerate(paths):
        x=i%columns*thumb;y=i//columns*(thumb+16)
        canvas.paste(Image.open(path).convert('RGB').resize((thumb,thumb)),(x,y))
        draw.text((x+2,y+thumb+1),label,fill='white')
    canvas.save(target)


entries=[]
original=json.loads((ROOT/'data/manifest.json').read_text())
sources=[('original',f'original-{row["id"]:03}',ROOT.parent/row['source']) for row in original['images']]
synthetic=json.loads((ROOT/'data/synthetic/manifest.json').read_text())
sources += [('synthetic',f'synthetic-{row["id"]:03}',ROOT/'data/synthetic'/row['image']) for row in synthetic['entries'] if row.get('visual_review',{}).get('status')=='accepted']
sources += [('ffhq',f'ffhq-{path.stem}',path) for path in sorted((HERE/'reference').glob('*.png'))]
for folder in ['original','ffhq-aligned','partial-aligned','marked']:(HERE/folder).mkdir(exist_ok=True)
for group,name,path in sources:
    rgb=np.array(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS))
    _,faces=detector.detect(rgb[:,:,::-1].copy())
    entry={'group':group,'id':name,'source':str(path.relative_to(ROOT.parent)),
           'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if faces is None:
        entry['detected']=False;entries.append(entry);continue
    face=max(faces,key=lambda value:value[2]*value[3]*value[-1])
    entry.update({'detected':True,'confidence':float(face[-1]),'face_box':face[:4].tolist(),
                  'landmarks':face[4:14].reshape(5,2).tolist(),'metrics':metrics(face)})
    if group!='ffhq':
        collar=collar_box(rgb,face,face[4:8].reshape(2,2).mean(axis=0),face[10:14].reshape(2,2).mean(axis=0))
        entry['collar_box_heuristic']=collar
        Image.fromarray(rgb).save(HERE/'original'/f'{name}.png')
        marked=Image.fromarray(rgb);draw=ImageDraw.Draw(marked)
        for x,y in face[4:14].reshape(5,2):draw.ellipse((x-2,y-2,x+2,y+2),fill='red')
        if collar:
            x,y,w,h=collar;draw.rectangle((x,y,x+w,y+h),outline='lime',width=2)
        marked.save(HERE/'marked'/f'{name}.png')
        for kind,folder in [('ffhq','ffhq-aligned'),('partial','partial-aligned')]:
            matrix=transform_matrix(face,kind,collar)
            transformed=cv2.warpAffine(rgb,matrix,(256,256),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
            Image.fromarray(transformed).save(HERE/folder/f'{name}.png')
            coverage=cv2.warpAffine(np.ones((256,256),dtype=np.uint8),matrix,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            entry[kind]={'matrix_source256_to_output256':matrix.tolist(),'metrics':metrics(face,matrix),
                         'detected_collar_fully_visible':contains_collar(matrix,collar),
                         'reflected_padding_fraction':float(1-coverage.mean())}
    entries.append(entry)

summary={}
for group in ['original','synthetic','ffhq']:
    selected=[e for e in entries if e['group']==group]
    detected=[e for e in selected if e['detected']]
    result={'count':len(selected),'detected':len(detected)}
    for kind in ['metrics','ffhq','partial']:
        if group=='ffhq' and kind!='metrics':continue
        values=[e['metrics'] if kind=='metrics' else e[kind]['metrics'] for e in detected]
        result[kind]={key:{'mean':float(np.mean([v[key] for v in values])),
                          'std':float(np.std([v[key] for v in values])),
                          'p10':float(np.quantile([v[key] for v in values],.1)),
                          'p90':float(np.quantile([v[key] for v in values],.9))} for key in values[0]}
    if group!='ffhq':
        result['collar_detected']=sum(e['collar_box_heuristic'] is not None for e in detected)
        for kind in ['ffhq','partial']:
            result[kind]['collar_cropped']=sum(e[kind]['detected_collar_fully_visible'] is False for e in detected)
            result[kind]['mean_padding_fraction']=float(np.mean([e[kind]['reflected_padding_fraction'] for e in detected]))
    summary[group]=result
(HERE/'manifest.json').write_text(json.dumps({'detector_sha256':hashlib.sha256((HERE/'models/face_detection_yunet_2023mar.onnx').read_bytes()).hexdigest(),
    'detector':'OpenCV YuNet five landmarks, confidence>=.75, largest confident face',
    'limitations':'FFHQ approximation uses five instead of68 landmarks. Collar detection is heuristic and requires visual review. Padding is reflected; outputs are research previews, not adopted training data.',
    'entries':entries},indent=2)+'\n')
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for folder in ['original','ffhq-aligned','partial-aligned','marked']:
    sheet([(path,path.stem) for path in sorted((HERE/folder).glob('*.png'))],HERE/f'{folder}-contact.png')
selection=[e for e in entries if e['group']!='ffhq' and e['detected']][::10]
paths=[]
for entry in selection:
    for folder in ['original','ffhq-aligned','partial-aligned']:
        paths.append((HERE/folder/f'{entry["id"]}.png',entry['id']+' '+folder))
sheet(paths,HERE/'comparison-contact.png',columns=3,thumb=256)
print(json.dumps(summary,indent=2))
