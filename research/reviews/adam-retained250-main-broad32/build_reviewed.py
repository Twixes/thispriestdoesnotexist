"""Present the completed main100 broad review without changing frozen render inputs."""
import json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
run=ROOT/'research/runs/adam-native1024-retained250-main100-broad32-v1'
subprocess.run([sys.executable,str(HERE/'build.py'),'--run',str(run),'--annotations',str(HERE/'annotations-agent-main100.json')],check=True)
report=json.loads((HERE/'agent-review-main100.json').read_text());data=json.loads((HERE/'manifest.json').read_text())
assert report['source_checkpoint_sha256']==data['checkpoint_sha256'] and data['available_images']==64
assert report['domain']['clear_priest_clothing_and_collar_raw']==report['domain']['clear_priest_clothing_and_collar_ema']==0
page=(HERE/'index.html').read_text()
lead='<p class="scope"><strong>0/32 clear priest portraits in raw and 0/32 in EMA.</strong> All64 preregistered outputs were reviewed. Raw images show photographic drift: cooler color, harsher texture and several local distortions. No model is approved. See the complete unfiltered images and bounded review observations below.</p>'
page=re.sub(r'<p class="scope">.*?</p>',lead,page,count=1)
page=page.replace('<nav><a href="manifest.json">','<nav><a href="agent-review-main100.json">Actual review and limitations</a><a href="manifest.json">')
path=HERE/'index.html.partial';path.write_text(page);path.replace(HERE/'index.html')
