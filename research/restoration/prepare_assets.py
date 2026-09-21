"""Download only documented official source/release bytes; never imports ML code."""
import hashlib, json, tarfile, urllib.request, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
ASSETS = [
 ('GFPGAN-source.tar.gz','https://codeload.github.com/TencentARC/GFPGAN/tar.gz/7552a7791caad982045a7bbe5634bbf1cd5c8679'),
 ('GFPGANv1.4.pth','https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'),
 ('detection_Resnet50_Final.pth','https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'),
 ('parsing_parsenet.pth','https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'),
]
def main():
 records=[]
 for name,url in ([] if "--extract-only" in sys.argv else ASSETS):
  folder=BASE/('weights' if name.endswith('.pth') else 'sources');folder.mkdir(exist_ok=True)
  path=folder/name
  if path.exists(): raise FileExistsError(path)
  digest=hashlib.sha256();n=0
  with urllib.request.urlopen(url,timeout=60) as src, path.open('xb') as dst:
   while chunk:=src.read(1024*1024):dst.write(chunk);digest.update(chunk);n+=len(chunk)
  record={'path':str(path.relative_to(BASE)),'url':url,'bytes':n,'sha256':digest.hexdigest()};records.append(record)
  print(json.dumps(record),flush=True)
  (BASE/'asset-downloads.json').write_text(json.dumps(records,indent=2)+'\n')
 vendor=BASE/'vendor';vendor.mkdir(exist_ok=True)
 with tarfile.open(BASE/'sources/GFPGAN-source.tar.gz') as archive:
  archive.extractall(vendor,filter='data')
 source=vendor/'GFPGAN-7552a7791caad982045a7bbe5634bbf1cd5c8679'
 (BASE/'GFPGAN-LICENSE').write_bytes((source/'LICENSE').read_bytes())
if __name__=='__main__':main()
