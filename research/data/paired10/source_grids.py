"""Coordinate guides only; original sources and targets remain unchanged."""
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
for ident in ['042','058','059']:
 canvas=Image.new('RGB',(2048,1024))
 for column,path in enumerate([HERE.parent/f'ffhq-paired-sources64-b2/images/{ident}.png',HERE.parent/f'ffhq-clothing-edits-b3/{ident}.png']):
  im=Image.open(path).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS)
  d=ImageDraw.Draw(im)
  for v in range(0,1024,40):
   d.line((v,0,v,1023),fill=(180,80,80),width=1)
   d.text((v+2,610),str(v),fill='yellow')
  for v in range(640,1024,40):
   d.line((0,v,1023,v),fill=(180,80,80),width=1)
   d.text((0,v+2),str(v),fill='yellow')
  canvas.paste(im,(column*1024,0))
 canvas.save(HERE/f'{ident}-coordinate-grid.png')
