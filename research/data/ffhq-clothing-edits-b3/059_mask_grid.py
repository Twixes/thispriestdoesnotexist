"""Diagnostic coordinate grid only; source and raw edited images stay unchanged."""
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
paths = [ROOT / 'research/data/ffhq-paired-sources64-b2/images/059.png', HERE / '059.png']
canvas = Image.new('RGB', (2048, 1024))
for index, path in enumerate(paths):
    image = Image.open(path).convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for step in range(21):
        position = round(step / 20 * 1023)
        draw.line((position, 0, position, 1023), fill=(255, 80, 80), width=1)
        draw.line((0, position, 1023, position), fill=(255, 80, 80), width=1)
        draw.text((position + 2, 620), f'{step / 20:.2f}', fill=(255, 255, 0))
        if step >= 12:
            draw.text((2, position + 2), f'{step / 20:.2f}', fill=(255, 255, 0))
    canvas.paste(image, (index * 1024, 0))
canvas.save(HERE / '059-mask-grid.png')
