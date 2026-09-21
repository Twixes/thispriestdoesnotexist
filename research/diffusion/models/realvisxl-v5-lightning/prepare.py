"""Pinned FP16 RealVisXL V5 Lightning components; no model loading or execution."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

DEST=Path(__file__).resolve().parent
MODEL='SG161222/RealVisXL_V5.0_Lightning'
REVISION='f4454158cedaab9f0688c199561d6c92525f3a85'
BASE='stabilityai/stable-diffusion-xl-base-1.0'
BASE_REVISION='462165984030d82259a11f4367a4eed129e94a7b'
PART=1024**3
BUF=8*1024**2
EXPECTED={
 'text_encoder/model.fp16.safetensors':'0a873801e35a008b17e228189a83a8b41e068ffe7bde7c1105e8c6e947758bc5',
 'text_encoder_2/model.fp16.safetensors':'dfa429b268cd3f6297927f1fd3eacac04781b17ab4caa97892a461ff41400a5e',
 'unet/diffusion_pytorch_model.fp16.safetensors':'1143cd2aaf65d24af34b5699d090aed724f6c0978c2ec5a5f56821ccb36260ce',
 'vae/diffusion_pytorch_model.fp16.safetensors':'6353737672c94b96174cb590f711eac6edf2fcce5b6e91aa9d73c5adc589ee48'}


def sha(path,git=False):
 h=hashlib.sha1() if git else hashlib.sha256()
 if git:h.update(f'blob {path.stat().st_size}\0'.encode())
 with path.open('rb') as f:
  for b in iter(lambda:f.read(BUF),b''):h.update(b)
 return h.hexdigest()


def write(path,data):
 t=path.with_name(path.name+'.partial');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)


def verify(path,e):
 if path.stat().st_size!=e['size']:raise ValueError('Wrong size: '+str(path))
 digest=sha(path)
 if 'lfs' in e:
  if digest!=e['lfs']['sha256']:raise ValueError('Wrong SHA256: '+str(path))
 elif sha(path,True)!=e['blobId']:raise ValueError('Wrong Git blob: '+str(path))
 return digest


def fetch(spec):
 e,url,name=spec;p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  t=p.with_name(p.name+'.partial')
  with urllib.request.urlopen(url,timeout=120) as response,t.open('wb') as f:
   for b in iter(lambda:response.read(BUF),b''):f.write(b)
  verify(t,e);t.replace(p)
 digest=verify(p,e);chunks=[]
 if p.stat().st_size>PART:
  with p.open('rb') as f:
   while f.tell()<p.stat().st_size:
    size=min(PART,p.stat().st_size-f.tell());remaining=size
    chunk=p.with_name(p.name+f'.weights-part-{len(chunks):03}');t=chunk.with_name(chunk.name+'.partial');h=hashlib.sha256()
    with t.open('wb') as target:
     while remaining:
      b=f.read(min(BUF,remaining))
      if not b:raise ValueError('Unexpected EOF')
      target.write(b);h.update(b);remaining-=len(b)
    t.replace(chunk);chunks.append({'path':str(chunk.relative_to(DEST)),'bytes':size,'sha256':h.hexdigest()})
 print(f'Verified {name}: {p.stat().st_size} bytes, {len(chunks)} parts',flush=True)
 return {'path':name,'bytes':p.stat().st_size,'sha256':digest,'source_url':url,'upstream_entry':e,'chunks':chunks}


def main():
 url=f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
 with urllib.request.urlopen(url,timeout=30) as r:metadata=json.load(r)
 if metadata['sha']!=REVISION or metadata.get('gated'):raise ValueError('Wrong revision/access')
 write(DEST/'upstream-metadata.json',metadata)
 entries=[e for e in metadata['siblings'] if e['rfilename'] in EXPECTED or (e['rfilename'].endswith(('.json','.txt','.md')) and not e['rfilename'].endswith('.safetensors.index.json'))]
 for name,expected in EXPECTED.items():
  if next(e for e in entries if e['rfilename']==name)['lfs']['sha256']!=expected:raise ValueError('Weight metadata changed')
 specs=[(e,f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{e["rfilename"]}','sources/publisher-model-card.md' if e['rfilename']=='README.md' else e['rfilename']) for e in entries]
 with urllib.request.urlopen(f'https://huggingface.co/api/models/{BASE}/revision/{BASE_REVISION}?blobs=true',timeout=30) as r:base=json.load(r)
 if base['sha']!=BASE_REVISION:raise ValueError('Base license revision changed')
 write(DEST/'license-upstream-metadata.json',base)
 license_entry=next(e for e in base['siblings'] if e['rfilename']=='LICENSE.md')
 specs.append((license_entry,f'https://huggingface.co/{BASE}/resolve/{BASE_REVISION}/LICENSE.md','LICENSE.md'))
 total=sum(e[0]['size'] for e in specs);parts=sum(e[0]['size'] for e in specs if e[0]['size']>PART)
 free=shutil.disk_usage(DEST).free
 if free<total+parts+20*PART:raise RuntimeError('Insufficient disk headroom')
 (DEST/'.gitignore').write_text('# Reassembled originals; <=1GiB parts tracked in LFS.\n'+'\n'.join('/'+e['rfilename'] for e in entries if e['size']>PART)+'\n*.partial\n')
 (DEST/'.gitattributes').write_text('*.weights-part-* filter=lfs diff=lfs merge=lfs -text\n')
 write(DEST/'download-plan.json',{'model':MODEL,'revision':REVISION,'download_bytes':total,'additional_part_bytes':parts,'free_bytes_before':free,'selected_paths':[s[2] for s in specs],'max_network_threads':3})
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(fetch,specs))
 manifest={'complete':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':MODEL,'revision':REVISION,'license':'openrail++','license_source':{'model':BASE,'revision':BASE_REVISION,'path':'LICENSE.md','reason':'Canonical SDXL OpenRAIL++ license text matching publisher card tag; RealVis repo has no standalone license file'},'checkpoint_kind':'FP16 Diffusers components only','files':files,'prepare_source_sha256':sha(Path(__file__)),'upstream_metadata_sha256':sha(DEST/'upstream-metadata.json'),'license_metadata_sha256':sha(DEST/'license-upstream-metadata.json'),'inference_reference':{'steps':5,'author_sampler':'DPM++ SDE Karras / DPM++ SDE','guidance_range':[1,2],'quality_comparison_resolution':1024,'exact_diffusers_sampler':'to be frozen in RealVis evaluator, not baked DDIM defaults'},'model_load_or_inference_performed':False,'production_approved':False}
 write(DEST/'provenance.json',manifest)
 for e in files:
  p=DEST/e['path']
  if sha(p)!=e['sha256']:raise ValueError('Final file verification failed')
  if e['chunks']:
   h=hashlib.sha256();total=0
   for chunk in e['chunks']:
    part=DEST/chunk['path']
    if part.stat().st_size!=chunk['bytes'] or sha(part)!=chunk['sha256'] or chunk['bytes']>PART:raise ValueError('Bad LFS chunk')
    with part.open('rb') as f:
     for b in iter(lambda:f.read(BUF),b''):h.update(b);total+=len(b)
   if total!=e['bytes'] or h.hexdigest()!=e['sha256']:raise ValueError('Parts do not reproduce original')
 result={'complete':True,'verified_files':len(files),'all_parts_concatenate_to_upstream_hash':True,'provenance_sha256':sha(DEST/'provenance.json'),'model_load_or_inference_performed':False}
 write(DEST/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
