"""Derive staged-cache trainer; do not edit the previous failed-run trainer."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
original=(HERE/'run_training.py').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='66785e3bad79b22c03e1554edfb9f0c5ebda4c17b2337e72a87f1c476e77912b'
s=original

def replace(old,new,count=1):
    global s
    assert s.count(old)==count,(old,s.count(old))
    s=s.replace(old,new)

def replace_between(start,end,new):
    global s
    assert s.count(start)==1 and s.count(end)==1
    a=s.index(start);b=s.index(end,a)
    s=s[:a]+new+s[b:]

replace('import smoke_guards as smoke','import smoke_guards_cached as smoke\nfrom priest_cache import validate_contract, load_training_cache')
replace('    parser.add_argument(\n        "--cache_latents",',
        '    parser.add_argument("--priest_cached_data", type=str, required=True,\n        help="Absolute folder with separately verified raw latent/text caches")\n    parser.add_argument(\n        "--cache_latents",')
replace('    smoke_state = smoke.initialize(args, accelerator)\n',
        '    smoke_state = smoke.initialize(args, accelerator)\n    validate_contract(args)\n')
replace_between('    # Load the tokenizers\n','    # For mixed precision training',
                '    # Qwen/tokenizer are intentionally never loaded in this staged-cache worker.\n\n')
replace('''    quantization_config = None
''','''    # VAE is used only for unchanged FP32 BN statistics, never for a forward pass.
    del vae
    free_memory()
    smoke.resources(smoke_state, "vae_bn_only_released", 0)

    quantization_config = None
''')
replace_between('    text_encoder = Qwen3ForCausalLM.from_pretrained(\n','    # We only train the additional adapter LoRA layers',
                '    smoke.resources(smoke_state, "transformer_loaded_direct_mps", 0)\n\n')
replace('''        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
''','''        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
        # Meta initialization and per-tensor placement avoid a complete FP16 CPU model.
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
    )
''')
replace('    vae.requires_grad_(False)\n','')
replace('    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory\n    vae.to(**to_kwargs)\n','')
replace_between('    text_encoder.to(**to_kwargs)\n','    if args.gradient_checkpointing:',
                '    smoke.resources(smoke_state, "transformer_placed", 0)\n\n')
replace('                from datasets import load_dataset\n',
        '                from datasets import load_dataset, Image as DatasetImage\n')
replace('            instance_images = dataset["train"][image_column]\n',
        '            instance_images = dataset["train"][image_column]\n'
        '            encoded_images = dataset["train"].cast_column(image_column, DatasetImage(decode=False))[image_column]\n'
        '            self.instance_file_names = [Path(item["path"]).name for item in encoded_images]\n')
replace_between('    def compute_text_embeddings(prompt, text_encoding_pipeline):\n','    # Scheduler and math around the number of training steps.',
        '    # Encoding was completed separately without co-resident transformer/Qwen weights.\n'
        '    prompt_embeds_cache, text_ids_cache, instance_latents_cache = load_training_cache(\n'
        '        args, train_dataset, accelerator.device, smoke, smoke_state)\n'
        '    validation_embeddings = {}\n'
        '    smoke.resources(smoke_state, "external_cache_ready", 0)\n\n')
ast.parse(s)
(HERE/'run_training_cached.py').write_text(s)
(HERE/'cached.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='run_training.py',tofile='run_training_cached.py')))
guards=(HERE/'smoke_guards.py').read_text()
guards=guards.replace("for name in ('run_training.py', 'smoke_guards.py', 'patch-provenance.json'):",
    "for name in ('run_training_cached.py', 'smoke_guards_cached.py', 'priest_cache.py', 'cached-patch-provenance.json'):")
# Explicit root-reviewed policy revision for staged phases after observed allocation.
# Preserve original smoke_guards.py and its25% threshold unchanged.
assert guards.count('elif vm.available/vm.total < .25:') == 1
guards=guards.replace('elif vm.available/vm.total < .25:', 'elif vm.available/vm.total < .20:')
guards=guards.replace('system available memory below25%', 'system available memory below20%')
ast.parse(guards);(HERE/'smoke_guards_cached.py').write_text(guards)
manifest={'source_training_sha256':hashlib.sha256(original.encode()).hexdigest(),
          'files':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                   ['run_training_cached.py','smoke_guards_cached.py','priest_cache.py','cached.patch']},
          'purpose':'Preserve20step smoke math while removing simultaneous Qwen/transformer residency.',
          'cache_contract':{'manifest':'complete, dataset_manifest_sha256, resolution, max_sequence_length, text_encoder_out_layers, entries',
                            'entries':'file_name, caption, path, sha256',
                            'safetensor_keys':{'prompt_embeds':'FP16[1,512,7680]','text_ids':'[1,512,4], retain originaldtype','latents':'FP16[1,32,64,64] raw VAE mode before patchify/BN at512'}},
          'base_normalization':'Load original VAE inFP32 only to derive unchanged BN mean/std, then delete before transformer load.',
          'transformer_loading':{'device_map':{'':'mps'},'low_cpu_mem_usage':True,'torch_dtype':'existing weight_dtype (FP16 for smoke)',
              'reason':'Avoid complete CPU FP16 transformer allocation before MPS placement; failed v4 source and run retained.',
              'api_review':'Installed Diffusers modeling_utils.py uses meta initialization and per-tensor device placement; Accelerate verify_device_map only flags maps with more than one entry.',
              'remaining_transient':'Diffusers allocator warmup temporarily allocates half the model element count on MPS; unchanged resource guards remain required.'},
          'cached_phase_resource_policy':{'minimum_system_available_fraction':.20,'maximum_rss_gib':24,'maximum_mps_driver_gib':22,'maximum_swap_growth_mib':512,'reason':'Explicit measured-policy revision approved by root after BF16encoder observed8.74GiBdriver/0.49GiBRSS yet psutilavailability22.65%; original25%attempts retained.'},
          'cache_numerical_note':'Root precomputation uses nativeBF16 frozenQwen then finiteFP16cast, VAE FP16 mode; not claimed equivalent to all-FP16Qwen encoding.',
          'tracker_compatibility':'Cached-directory argument remains a primitive string for TensorBoard hparams; validate_contract converts it to Path and retains all path checks. Failed v5 Path-argument run preserved.',
          'training_objective_changed':False,'training_executed':False,'production_approved':False}
(HERE/'cached-patch-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
