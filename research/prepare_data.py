"""Reproducible hot-priest training data, optionally including reviewed expansion."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parent
IDS=[1,2,3,4,7,9,10,13,18,20,22,23,25,27,33,36,38,45,46,47]+list(range(51,81))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--synthetic',action='store_true',help='Require all sixty expansion images to be visually accepted')
    p.add_argument('--resolution',type=int,default=256)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.resolution not in [256,512,1024]:p.error('Supported resolutions: 256, 512, 1024')
    out=args.output or ROOT/f'data/priests{args.resolution}'
    out.mkdir(parents=True,exist_ok=True)
    sources=[(id,ROOT.parent/f'public/portraits/{id:02}.webp') for id in IDS]
    if args.synthetic:
        expansion=ROOT/'data/synthetic'
        metadata=json.loads((expansion/'manifest.json').read_text())
        if len(metadata['entries'])!=60 or metadata['reviewed_accepted_count']!=60:
            raise ValueError('Expansion is not yet complete and visually accepted')
        for entry in metadata['entries']:
            if entry['visual_review']['status']!='accepted':raise ValueError(f"Image {entry['id']} was not accepted")
            image=expansion/entry['image']
            if hashlib.sha256(image.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(f'Changed source image: {image}')
            sources.append((entry['id'],image))
    manifest={'selection':'Original curated hot twenty plus commissioned handsome thirty; optional sixty visually reviewed synthetic portraits.',
              'source_commit':'6ded5a6ee247fc7b67c27989ba4536da8a7c8e89','resolution':args.resolution,'images':[]}
    for id,source in sources:
        target=out/f'{id:03}.png' if args.synthetic else out/f'{id:02}.png'
        im=Image.open(source).convert('RGB')
        ImageOps.fit(im,(args.resolution,args.resolution),method=Image.Resampling.LANCZOS).save(target)
        manifest['images'].append({'id':id,'source':str(source.relative_to(ROOT.parent)),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'training_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    if len(list(out.glob('*.png')))!=len(sources):raise ValueError('Output contains stale images; use a fresh output directory')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(sources)} verified training images at {args.resolution}px in {out}')

if __name__=='__main__':main()
