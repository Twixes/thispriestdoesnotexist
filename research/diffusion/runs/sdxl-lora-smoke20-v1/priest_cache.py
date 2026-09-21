"""Hash-bound SDXL inputs; no encoder, tokenizer or VAE model loading."""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_contract(args):
    cache = Path(args.priest_cached_data)
    if not cache.is_absolute():
        raise ValueError('Cache directory must be absolute')
    cache = cache.resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    train = Path(args.dataset_name).resolve()
    dataset_path = train.parent/'manifest.json'
    dataset = json.loads(dataset_path.read_text())
    model = Path(args.pretrained_model_name_or_path).resolve()
    if not manifest['complete'] or manifest['dataset_manifest_sha256'] != sha(dataset_path):
        raise ValueError('Incomplete cache or changed reviewed dataset')
    if manifest['model_provenance_sha256'] != sha(model/'provenance.json'):
        raise ValueError('Cache model provenance differs')
    if manifest['resolution'] != args.resolution or args.resolution != 512:
        raise ValueError('Only native 512 training is supported')
    expected_transform = args.image_interpolation_mode.capitalize() + ' Resize512,CenterCrop512,normalize[-1,1],no flip'
    if manifest['transform'] != expected_transform:
        raise ValueError('Cache transform differs from explicit training interpolation')
    vae = json.loads((model/'vae/config.json').read_text())
    scaling = manifest['vae_scaling_factor']
    if not math.isfinite(scaling) or scaling != vae['scaling_factor']:
        raise ValueError('VAE scaling differs from frozen model')
    if sha(train/'metadata.jsonl') != dataset['metadata_sha256']['train']:
        raise ValueError('Reviewed training metadata changed')
    expected = {Path(e['destination']).name:e for e in dataset['entries'] if e['split']=='train'}
    entries = {e['file_name']:e for e in manifest['entries']}
    if len(entries) != len(manifest['entries']) or set(entries) != set(expected):
        raise ValueError('Cache image set differs from reviewed train split')
    for name, entry in entries.items():
        if entry['caption'] != expected[name]['caption']:
            raise ValueError('Caption differs: '+name)
        if sha(train/name) != expected[name]['destination_sha256']:
            raise ValueError('Reviewed image changed: '+name)
        path = (cache/entry['path']).resolve()
        if not path.is_relative_to(cache) or sha(path) != entry['sha256']:
            raise ValueError('Cache path/hash differs: '+name)
        if len(entry['original_size'])!=2 or len(entry['crop_top_left'])!=2:
            raise ValueError('Invalid size/crop pair: '+name)
        if any(type(v) is not int or v <= 0 for v in entry['original_size']):
            raise ValueError('Invalid original dimensions: '+name)
        if any(type(v) is not int or v < 0 for v in entry['crop_top_left']):
            raise ValueError('Invalid crop coordinates: '+name)
    return cache, manifest


def load_cache(contract, smoke, state):
    cache, manifest = contract
    expected = {'prompt_embeds':((1,77,2048),torch.float16),
                'pooled_prompt_embeds':((1,1280),torch.float16),
                'latent_mean':((1,4,64,64),torch.float32),
                'latent_std':((1,4,64,64),torch.float32)}
    rows=[]
    for entry in manifest['entries']:
        values=load_file(str(cache/entry['path']),device='cpu')
        if set(values)!=set(expected):
            raise ValueError('Unexpected cached tensor keys')
        for name,(shape,dtype) in expected.items():
            value=values[name]
            if tuple(value.shape)!=shape or value.dtype!=dtype:
                raise ValueError('Unexpected tensor shape/dtype: '+entry['file_name']+':'+name)
            smoke.finite(state,value,'cache:'+entry['file_name']+':'+name)
        if bool((values['latent_std']<0).any()):
            raise ValueError('Negative posterior standard deviation')
        rows.append({**values,'original_sizes':tuple(entry['original_size']),
                     'crop_top_lefts':tuple(entry['crop_top_left']),'filenames':entry['file_name']})
    smoke.write_json(state['output']/'external-cache-evidence.json',{
        'manifest_sha256':sha(cache/'manifest.json'),'entries':manifest['entries'],
        'vae_scaling_factor':manifest['vae_scaling_factor'],
        'tensor_contract':{k:{'shape':list(v[0]),'dtype':str(v[1])} for k,v in expected.items()},
        'posterior_sampling':'Fresh mean + std * randn each update; then scaling and FP16 cast.'})
    smoke.resources(state,'external_cache_ready',0)
    return rows


def collate_fn(examples):
    return {**{k:torch.cat([e[k] for e in examples],dim=0) for k in
               ('prompt_embeds','pooled_prompt_embeds','latent_mean','latent_std')},
            **{k:[e[k] for e in examples] for k in ('original_sizes','crop_top_lefts','filenames')}}
