"""Exact checkpoint100 tensor diagnostic previously executed from repository root.

Run with research/.venv-flux-train/bin/python. No model construction or forward.
This archived script was not rerun when saved; timestamps naturally differ on
reproduction. Reads only completed checkpoint100, verified source noise scalars,
and small G/Gema modulation tensors; writes parameter-diagnostics-100.json.
"""
import datetime,hashlib,json,math,re
from pathlib import Path
import torch
from safetensors import safe_open
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path('.');run=Path('research/runs/adam-native1024-probing500-v1');p=run/'checkpoint-100.pt';marker=json.loads((run/'checkpoint-100.json').read_text())
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
checkpoint_sha=sha(p);assert checkpoint_sha==marker['checkpoint_sha256'] and marker['complete']
d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);assert d['iterations']==100
base=Path('research/runs/inference-cpu/ffhq1024/baseline-bundle/generator.safetensors');assert sha(base)==d['protocol']['g_weights_sha256']
def stats(t):
 t=t.detach().to(dtype=torch.float64,device='cpu').flatten();a=t.abs();assert torch.isfinite(t).all()
 return {'entries':t.numel(),'mean':float(t.mean()),'abs_mean':float(a.mean()),'abs_max':float(a.max()),'rms':float(t.square().mean().sqrt()),'l2':float(t.square().sum().sqrt())}
rows=[];noise=[];offsets=[];grouped={}
with safe_open(str(base),framework='pt',device='cpu') as original:
 for arm in ('G','Gema'):
  state=d[arm]
  for name,u in sorted(state.items()):
   if not name.endswith('.parametrizations.weight.0.u_vector'):continue
   module=name.removesuffix('.parametrizations.weight.0.u_vector');v=state[module+'.parametrizations.weight.0.v_vector'];us,vs=stats(u),stats(v)
   match=re.search(r'\.b(\d+)\.',module);res=int(match[1])if match else None
   role='mapping_fc' if module.startswith('mapping.') else 'style_affine' if module.endswith('.affine') else 'torgb_conv' if module.endswith('.torgb') else 'synthesis_conv'
   row={'arm':arm,'module':module,'resolution':res,'role':role,'u':us,'v':vs,
    'factor_outer_modulation':{'abs_mean':us['abs_mean']*vs['abs_mean'],'abs_max':us['abs_max']*vs['abs_max'],'rms':us['rms']*vs['rms'],'frobenius_norm':us['l2']*vs['l2']},
    'initial_factor_std':1e-7,'u_rms_over_initial_std':us['rms']/1e-7,'v_rms_over_initial_std':vs['rms']/1e-7}
   rows.append(row);grouped.setdefault((arm,res,role),[]).extend((u,v))
  for name,value in sorted(state.items()):
   if not name.endswith('.0.b_vector'):continue
   match=re.search(r'\.b(\d+)\.',name);res=int(match[1])if match else None
   module,tensor=name.split('.parametrizations.',1);kind=tensor.split('.',1)[0]
   if kind=='noise_strength':
    original_name=module+'.noise_strength';b=float(original.get_tensor(original_name));frozen=float(state[module+'.parametrizations.noise_strength.original']);assert frozen==b
    delta=float(value);effective=frozen+delta
    noise.append({'arm':arm,'module':module,'resolution':res,'original':b,'offset':delta,'effective':effective,
     'absolute_offset_over_abs_original':abs(delta)/abs(b)if b!=0 else None,
     'abs_effective_over_abs_original':abs(effective)/abs(b)if b!=0 else None,
     'sign_flipped':bool(b*effective<0),'original_matches_verified_source':True})
   else:offsets.append({'arm':arm,'module':module,'resolution':res,'parameter':name,'stats':stats(value)})
aggregates=[{'arm':arm,'resolution':res,'role':role,'factor_entries':stats(torch.cat([t.flatten()for t in tensors]))}for(arm,res,role),tensors in grouped.items()]
summary={}
for arm in ('G','Gema'):
 nr=[x for x in noise if x['arm']==arm];wr=[x for x in rows if x['arm']==arm]
 summary[arm]={'weight_modulation_modules':len(wr),'noise_strength_modules':len(nr),
  'largest_outer_abs_max_modules':sorted(wr,key=lambda x:x['factor_outer_modulation']['abs_max'],reverse=True)[:5],
  'largest_abs_noise_offsets':sorted(nr,key=lambda x:abs(x['offset']),reverse=True)[:5],
  'noise_effective_abs_greater_than_original_count':sum(abs(x['effective'])>abs(x['original'])for x in nr),
  'noise_effective_abs_at_least_double_original_count':sum(abs(x['effective'])>=2*abs(x['original'])and x['original']!=0 for x in nr),
  'noise_sign_flips':sum(x['sign_flipped']for x in nr),
  'largest_abs_noise_offset':max(abs(x['offset'])for x in nr),
  'largest_outer_abs_max':max(x['factor_outer_modulation']['abs_max']for x in wr)}
report={'schema_version':1,'status':'tensor_diagnostics_only','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'checkpoint':str(p),'checkpoint_sha256':checkpoint_sha,'checkpoint_marker_sha256':sha(run/'checkpoint-100.json'),
 'protocol_sha256':d['protocol_sha256'],'modulation_inventory_sha256':d['modulation_inventory_sha256'],'iteration':100,'optimizer_steps':d['optimizer_steps'],
 'original_generator':str(base),'original_generator_sha256':d['protocol']['g_weights_sha256'],
 'method':{'torch_version':str(torch.__version__),'threads':1,'checkpoint_load':'weights_only=True,map_location=cpu,mmap=True',
  'model_constructed':False,'model_forward_executed':False,'gradients_computed':False,
  'tensors_accessed':'G/Gema small u/v and additive-offset tensors;17 source noise-strength scalars from verified safetensors. No full-weight statistics or optimizer-state scans.',
  'outer_statistics':'Exact global statistics of outer(u,v), computed from factor statistics without materializing the outer product; invariant to source_flattened reshape, not output-kernel statistics.',
  'resolution_aggregate':'Entry-weighted concatenation of u/v vectors within one arm/resolution/role; mapping has resolution=null.'},
 'summary':summary,'by_resolution_factor_magnitudes':aggregates,'weight_modulation_factors':rows,'noise_strength_offsets':noise,'other_additive_offsets':offsets,
 'interpretation':{'visual_observation_source':'Root reported strong fine grain/weave and contrast/color changes in raw100 portraits, with subtler changes in EMA100. This diagnostic did not independently inspect images.',
  'limitations':['Parameter correlations do not establish which change causes visible artifacts.','Noise-strength coefficients are applied inside a nonlinear multi-layer network; coefficient ratios do not directly predict final-pixel grain.','A noise coefficient sign flip changes a fixed-noise realization but not the variance of an independent symmetric noise distribution.','Outer-factor magnitude is a relative pre-style/pre-demodulation weight modulation, not an output-image error metric.','EMA is parameter averaging; smaller offsets alone do not imply acceptable photographic quality.','No source10 comparison or causal intervention was performed.'],
  'possible_follow_up':'If a visual checkpoint merits diagnosis, a separately pinned inference-only intervention could reset noise offsets alone while preserving other trained state and identical latent/noise. Such an intervention is not executed here and should retain all controls.'},
 'production_approved':False,'quality_accepted':False}
out=Path('research/reviews/adam-native-probing/parameter-diagnostics-100.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('output',out,'sha256',sha(out))
for arm,s in summary.items():
 print(arm,{k:v for k,v in s.items()if not isinstance(v,list)})
 print('top_noise',[(x['module'],round(x['original'],6),round(x['offset'],6),round(x['effective'],6),round(x['abs_effective_over_abs_original'],2)if x['abs_effective_over_abs_original'] is not None else None)for x in s['largest_abs_noise_offsets']])
 print('top_outer',[(x['module'],x['factor_outer_modulation']['abs_max'])for x in s['largest_outer_abs_max_modules']])
