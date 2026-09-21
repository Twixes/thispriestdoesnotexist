"""Immutable full-generator native previews; no image compositing or filtering."""
import hashlib
import math
import time
from pathlib import Path
from research.experiments.paired_surround.runner import sha, write_json, equal_tree


def save_images(rgb, directory, ident, trainer):
    from PIL import Image
    mono=trainer.grayscale(rgb)
    images={'rgb':Image.fromarray(trainer.quantized(rgb)),
            'gray':Image.fromarray(trainer.quantized(mono)[:,:,0])}
    files={}
    for key,image in images.items():
        path=directory/f'{ident}-{key}.png';image.save(path)
        files[key]={'path':path.name,'sha256':sha(path),'mode':image.mode}
    return files,images['gray']


def contact_sheet(thumbs,path):
    from PIL import Image,ImageDraw
    contact=Image.new('RGB',(4*256,math.ceil(len(thumbs)/4)*280),(24,24,24));draw=ImageDraw.Draw(contact)
    for i,(ident,split,image) in enumerate(thumbs):
        x=i%4*256;y=i//4*280
        contact.paste(image.resize((256,256),Image.Resampling.LANCZOS),(x,y));draw.text((x+3,y+258),f'{ident} {split}',fill='white')
    contact.save(path)


def snapshot(student,pairs,fixed_z,output,step,trainer,torch,sampling,preservation):
    rng=trainer.capture_rng('cpu',sampling,preservation);before=trainer.state_digest(student.state_dict())
    directory=output/f'preview-{step:03}';directory.mkdir();rows=[];thumbs=[];started=time.monotonic()
    try:
        with torch.no_grad():
            requests=[(p['id'],p['split'],p['w'],p) for p in pairs]
            requests += [(f'unseen-{i:03}','unseen',student.mapping(z[None],None,truncation_psi=1,skip_w_avg_update=True),None) for i,z in enumerate(fixed_z)]
            for ident,split,w,pair in requests:
                rgb=student.synthesis(w,noise_mode='const',force_fp32=True)
                if not bool(torch.isfinite(rgb).all()):raise ValueError('Nonfinite preview')
                files,image=save_images(rgb,directory,ident,trainer);thumbs.append((ident,split,image));metrics={}
                if pair is not None:
                    mono=trainer.grayscale(rgb);target=trainer.grayscale(pair['target'])
                    metrics['protected_l1']=float(trainer.masked_l1(mono,trainer.grayscale(pair['original']),1-pair['mask']))
                    metrics['clothing_l1']=float(trainer.masked_l1(mono,target,pair['mask']))
                    if pair['collar'] is not None:
                        metrics['tab_l1']=float(trainer.masked_l1(mono,target,pair['collar']))
                        metrics['rest_l1']=float(trainer.masked_l1(mono,target,pair['rest']))
                        for region in ['surround','remainder']:
                            if region in pair:metrics[region+'_l1']=float(trainer.masked_l1(mono,target,pair[region]))
                rows.append({'id':ident,'split':split,'outputs':files,'metrics_minus1_to1':metrics})
    finally:trainer.restore_rng(rng,'cpu',sampling,preservation)
    if trainer.state_digest(student.state_dict())!=before or not equal_tree(rng,trainer.capture_rng('cpu',sampling,preservation),torch):raise ValueError('Snapshot mutated model/RNG')
    contact_sheet(thumbs,directory/'contact.png')
    result={'fork_step':step,'parent_step':600,'count':len(rows),'source_compositing':False,'rows':rows,'state_sha256':before,'seconds':time.monotonic()-started,'rng_restored':True}
    write_json(directory/'evaluation.json',result);return result


def source30(student,source,manifest_path,train_pairs,output,trainer,torch,sampling,preservation):
    """Stream saved held-out latents; exact source checks, never source-image-as-GT substitution."""
    import json
    import numpy as np
    started=time.monotonic();directory=output/'source30-final';directory.mkdir()
    manifest=json.loads(Path(manifest_path).read_text());rng=trainer.capture_rng('cpu',sampling,preservation)
    before=[trainer.state_digest(m.state_dict()) for m in [student,source]]
    train_hashes={hashlib.sha256(p['z'].numpy().tobytes()).hexdigest() for p in train_pairs}
    rows=[];thumbs=[];excluded=[]
    try:
        with torch.no_grad():
            for entry in manifest['entries']:
                ident=entry['id'];latent_path=Path(manifest_path).parent/entry['latent_path'];image_path=Path(manifest_path).parent/entry['source_path']
                if sha(latent_path)!=entry['latents_sha256'] or sha(image_path)!=entry['image_sha256']:raise ValueError('Source30 input hash changed')
                with np.load(latent_path,allow_pickle=False) as latent:z=torch.from_numpy(latent['z'].copy());w=torch.from_numpy(latent['w'].copy())
                zh=hashlib.sha256(z.numpy().tobytes()).hexdigest()
                if zh!=entry['z_raw_sha256'] or hashlib.sha256(w.numpy().tobytes()).hexdigest()!=entry['ws_raw_sha256']:raise ValueError('Source30 latent content changed')
                if zh in train_hashes:excluded.append(ident);continue
                if z.shape!=(1,source.z_dim) or w.shape!=(1,source.num_ws,source.w_dim) or z.dtype!=torch.float32 or w.dtype!=torch.float32:raise ValueError('Source30 latent shape/dtype')
                if not torch.equal(source.mapping(z,None,truncation_psi=1,skip_w_avg_update=True),w):raise ValueError('Source30 W mismatch')
                original=source.synthesis(w,noise_mode='const',force_fp32=True)
                _,encoded,_=trainer.rgb_tensor(image_path,source.img_resolution,resize=False)
                if not np.array_equal(trainer.quantized(original),encoded):raise ValueError('Source30 PNG regeneration mismatch')
                rgb=student.synthesis(w,noise_mode='const',force_fp32=True)
                if not bool(torch.isfinite(rgb).all()):raise ValueError('Nonfinite source30 output')
                files,image=save_images(rgb,directory,ident,trainer);thumbs.append((ident,'held-out',image))
                sourcefiles,_=save_images(original,directory,ident+'-source',trainer)
                error=(trainer.grayscale(rgb)-trainer.grayscale(original)).abs();height=round(source.img_resolution*.75)
                rows.append({'id':ident,'split':'held-out','z_sha256':zh,'source_uint8_max_error':0,'w_max_error':0,
                             'outputs':files,'source_outputs':sourcefiles,'source_mae_minus1_to1':float(error.mean()),'source_top75_mae_minus1_to1':float(error[:,:,:height].mean())})
    finally:trainer.restore_rng(rng,'cpu',sampling,preservation)
    if len(rows)!=30 or sorted(excluded)!=['000','030']:raise ValueError('Expected exact original 30 held-out source seeds')
    if before!=[trainer.state_digest(m.state_dict()) for m in [student,source]] or not equal_tree(rng,trainer.capture_rng('cpu',sampling,preservation),torch):raise ValueError('Source30 evaluation mutated models/RNG')
    contact_sheet(thumbs,directory/'contact.png')
    result={'count':30,'excluded_training_ids':excluded,'rows':rows,'source_compositing':False,'rng_restored':True,'seconds':time.monotonic()-started,'manifest_sha256':sha(manifest_path)}
    write_json(directory/'evaluation.json',result);return result
