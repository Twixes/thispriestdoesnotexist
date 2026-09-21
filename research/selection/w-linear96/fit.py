"""Fixed held-out-batch W-only linear experiment; no image/model inference."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[name]='1'
import hashlib,json,time,resource,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=HERE/'plan.json';plan=json.loads(plan_path.read_text())
labels_path=HERE.parent/'manual96/labels.json';labels=json.loads(labels_path.read_text())
started=time.monotonic()
assert plan['lambda']==1.0 and plan['thresholds']==[.5,.8]
features=[];rows=[];provenance=[];excluded=[]
for batch,n in [('b2',64),('v1',32)]:
 mp=ROOT/labels['provenance'][batch]['manifest_path']
 assert sha(mp)==labels['provenance'][batch]['manifest_sha256']
 originals={r['id']:r for r in json.loads(mp.read_text())['entries']}
 for row in [r for r in labels['labels'] if r['batch']==batch]:
  source=originals[row['source_id']]
  lp=mp.parent/source['latent_path'];checksum=sha(lp)
  assert checksum==source['latents_sha256']
  with np.load(lp,allow_pickle=False) as archive:
   w=archive['w'];z=archive['z']
   assert w.shape==(1,18,512) and w.dtype==np.float32 and np.isfinite(w).all()
   assert z.shape==(1,512) and np.isfinite(z).all()
   assert np.array_equal(w,np.repeat(w[:,:1,:],18,axis=1))
   feature=w[0,0].astype(np.float64)
  provenance.append({'id':row['id'],'latent_path':str(lp.relative_to(ROOT)),'npz_sha256':checksum,'w_first_row_sha256':hashlib.sha256(w[0,0].tobytes()).hexdigest(),'source_png_sha256':row['source_sha256'],'label':row['apparent_adult_male']})
  if row['apparent_adult_male']=='uncertain':excluded.append(row['id']);continue
  features.append(feature);rows.append(row)
X=np.stack(features);y=np.array([r['apparent_adult_male']=='yes' for r in rows],np.float64)
train=np.array([r['batch']=='b2' for r in rows]);test=~train
mean=X[train].mean(0);scale=np.maximum(X[train].std(0),1e-8)
A=(X-mean)/scale;At=A[train];yt=y[train]
lam=plan['lambda']
def objective(theta):
 logits=At@theta[:-1]+theta[-1]
 value=np.mean(np.logaddexp(0,logits)-yt*logits)+.5*lam*(theta[:-1]@theta[:-1])
 errors=expit(logits)-yt
 gradient=np.r_[At.T@errors/len(yt)+lam*theta[:-1],errors.mean()]
 return value,gradient
initial=np.zeros(513);initial[-1]=np.log(yt.mean()/(1-yt.mean()))
result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':300,'gtol':1e-8,'ftol':1e-12})
assert result.success and np.isfinite(result.x).all()
scores=expit(A@result.x[:-1]+result.x[-1])
def interval(k,n):
 if n==0:return None
 z=1.959963984540054;p=k/n;d=1+z*z/n
 center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [float(center-half),float(center+half)]
def metrics(actual,accepted):
 actual=actual.astype(bool);n=len(actual)
 tp=int((actual&accepted).sum());fp=int((~actual&accepted).sum());fn=int((actual&~accepted).sum());tn=int((~actual&~accepted).sum())
 return {'n':n,'positive_count':int(actual.sum()),'accepted':int(accepted.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'acceptance':float(accepted.mean()),'accuracy':(tp+tn)/n,'precision_wilson95':interval(tp,tp+fp),'recall_wilson95':interval(tp,tp+fn)}
evaluations={}
for subset,selection in [('train_b2',train),('heldout_v1',test)]:
 evaluations[subset]={'logistic_fixed_thresholds':{str(t):metrics(y[selection],scores[selection]>=t) for t in plan['thresholds']},'accept_all':metrics(y[selection],np.ones(selection.sum(),bool)),'reject_all':metrics(y[selection],np.zeros(selection.sum(),bool))}
np.savez(HERE/'linear-model-research-only.npz',weight=result.x[:-1],intercept=result.x[-1:],mean=mean,scale=scale)
predictions=[{'id':r['id'],'split':'train_b2' if train[i] else 'heldout_v1','label':r['apparent_adult_male'],'uncalibrated_logistic_score':float(scores[i]),'accepted_at_0_5':bool(scores[i]>=.5),'accepted_at_0_8':bool(scores[i]>=.8)} for i,r in enumerate(rows)]
heldout_errors={str(t):[{'id':r['id'],'label':r['apparent_adult_male'],'score':float(scores[i]),'manual_note':r['visual_note']} for i,r in enumerate(rows) if test[i] and (scores[i]>=t)!=(y[i]==1)] for t in plan['thresholds']}
output={'production_approved':False,'plan':plan,'plan_sha256':sha(plan_path),'script_sha256':sha(Path(__file__)),'labels_sha256':sha(labels_path),'feature_provenance':provenance,'excluded_uncertain_ids':excluded,'feature_shape':list(X.shape),'training_ids':[r['id'] for i,r in enumerate(rows) if train[i]],'heldout_ids':[r['id'] for i,r in enumerate(rows) if test[i]],'optimizer':{'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'objective':float(result.fun),'gradient_max_abs':float(np.max(np.abs(result.jac)))},'metrics':evaluations,'predictions':predictions,'heldout_errors':heldout_errors,'model_npz_sha256':sha(HERE/'linear-model-research-only.npz'),'runtime':{'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'thread_environment':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}},'generator_or_image_encoder_inference':False,'linear_scoring_of_saved_W':True,'attractiveness_model':False,'limitations':['Single-reviewer subjective appearance labels; uncertain examples excluded.','Only29 heldout evaluable sources expected; confidence bounds broad.','Combined adult-male target may learn masculine presentation rather than robust adult discrimination.','No heldout masculine-adolescent coverage is established; age-ambiguous sources excluded by design.','512 features greatly exceed training examples; a successful heldout batch remains a tiny pilot.','Source labels may change with synthesis adaptation even if mapping stays frozen.','No hats, collars, facial integrity, attractiveness, or production latency guarantee.']}
(HERE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'optimizer':output['optimizer'],'metrics':evaluations,'heldout_errors':heldout_errors,'runtime':output['runtime']},indent=2))
