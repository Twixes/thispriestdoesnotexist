"""Validate separately computed raw VAE/text caches without loading encoders."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_contract(args):
    cache = Path(args.priest_cached_data)
    if not cache.is_absolute():
        raise ValueError('priest_cached_data must be absolute')
    cache = cache.resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    dataset = Path(args.dataset_name).resolve()
    dataset_manifest_path = dataset.parent/'manifest.json'
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    if not manifest['complete'] or manifest['dataset_manifest_sha256'] != sha(dataset_manifest_path):
        raise ValueError('Incomplete cache or changed reviewed dataset manifest')
    if args.repeats != 1 or args.caption_dropout != 0 or args.random_flip or not args.center_crop:
        raise ValueError('External cache requires repeats1, no caption dropout/flip, center crop')
    if args.resolution != manifest['resolution'] or args.max_sequence_length != manifest['max_sequence_length']:
        raise ValueError('Cache resolution or prompt length differs from trainer')
    if list(args.text_encoder_out_layers) != manifest['text_encoder_out_layers']:
        raise ValueError('Cached text hidden-state layers differ')
    if sha(dataset/'metadata.jsonl') != dataset_manifest['metadata_sha256']['train']:
        raise ValueError('Training metadata differs from reviewed manifest')
    expected = {Path(e['destination']).name: e for e in dataset_manifest['entries'] if e['split']=='train'}
    actual = {e['file_name']: e for e in manifest['entries']}
    if len(actual) != len(manifest['entries']) or set(actual) != set(expected):
        raise ValueError('Duplicate, missing or extra cached training filenames')
    for name, entry in actual.items():
        if entry['caption'] != expected[name]['caption']:
            raise ValueError('Cached caption differs: '+name)
        if sha(dataset/name) != expected[name]['destination_sha256']:
            raise ValueError('Training image differs: '+name)
        path=(cache/entry['path']).resolve()
        if not path.is_relative_to(cache) or sha(path) != entry['sha256']:
            raise ValueError('Cache path/hash mismatch: '+name)
    return cache, manifest, actual


def load_training_cache(args, dataset, device, smoke, state):
    cache, manifest, entries = validate_contract(args)
    filenames = dataset.instance_file_names
    captions = dataset.custom_instance_prompts
    if len(filenames) != dataset.num_instance_images or len(filenames) != len(entries):
        raise ValueError('Loaded dataset/cache count differs')
    if len(set(filenames)) != len(filenames) or set(filenames) != set(entries):
        raise ValueError('Loaded HF dataset filename set differs from cache')
    prompt_cache, id_cache, latent_cache, evidence = [], [], [], []
    for index, (name, caption) in enumerate(zip(filenames, captions, strict=True)):
        entry=entries[name]
        if entry['caption'] != caption:
            raise ValueError('Actual loaded dataset caption differs: '+name)
        values=load_file(str(cache/entry['path']), device='cpu')
        if set(values) != {'prompt_embeds','text_ids','latents'}:
            raise ValueError('Unexpected cache tensor keys: '+name)
        expected_shapes={'prompt_embeds':(1,args.max_sequence_length,7680),
                         'text_ids':(1,args.max_sequence_length,4),
                         'latents':(1,32,args.resolution//8,args.resolution//8)}
        for key, shape in expected_shapes.items():
            value=values[key]
            if tuple(value.shape) != shape:
                raise ValueError('Unexpected raw cache shape: '+name+':'+key)
            if key!='text_ids' and value.dtype!=torch.float16:
                raise ValueError('Cache must retain original FP16 dtype: '+name+':'+key)
            smoke.finite(state,value,'external_cache:'+name+':'+key)
        prompt_cache.append(values['prompt_embeds'].to(device))
        id_cache.append(values['text_ids'].to(device))
        latent_cache.append(values['latents'].to(device))
        evidence.append({'index':index,'file_name':name,'caption':caption,'cache_path':entry['path'],
                         'cache_sha256':entry['sha256'],
                         'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in values.items()}})
        smoke.resources(state,'external_cache_loaded',0,dataset_index=index)
    smoke.write_json(state['output']/'external-cache-evidence.json',{
        'manifest_path':str(cache/'manifest.json'),'manifest_sha256':sha(cache/'manifest.json'),
        'dataset_manifest_sha256':manifest['dataset_manifest_sha256'],
        'mapping':'Actual HF dataset decode=False image basenames, each caption checked exactly; not cache or directory order.',
        'latents':'Unnormalized raw FP16 VAE mode; upstream patchification and BN normalization remain in training loop.',
        'entries':evidence})
    return prompt_cache,id_cache,latent_cache
