"""Fixed protocol logistic probe on saved CLIP features; never loads image models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROTOCOL=HERE/'protocol.json'
PIPELINE_KEYS=('clip_weights_sha256','clip_commit','preprocess','image_source','prompts','pins_sha256','source_sha256')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def resolve(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p

def load_features(directory,split,protocol):
    result=read(directory/'result.json');supervisor=read(directory/'supervisor-result.json')
    require(result['complete'] and supervisor['complete'] and supervisor['worker_exit_code']==0 and supervisor.get('error') is None,'Embedding run incomplete')
    require(sha(directory/'features.npz')==result['features_sha256'],'Embedding NPZ changed')
    n=protocol[split]['count'];rows=result['rows']
    require(result['attempted']==n and len(rows)==n and len(result['cohorts'])==1,'Exactly one complete declared cohort required')
    require([r['index'] for r in rows]==list(range(n)) and [r['feature_row'] for r in rows]==list(range(n)),'Index/feature ordering mismatch')
    cohort=result['cohorts'][0];restoration=resolve(cohort['directory'])
    require(cohort['number']==0 and cohort['attempted']==n,'Cohort numbering/count mismatch')
    for name,digest in cohort['file_sha256'].items():require(sha(restoration/name)==digest,'Restoration provenance changed: '+name)
    with np.load(directory/'features.npz',allow_pickle=False) as archive:
        x=archive['features'].copy();valid=archive['valid'].copy();mapping=archive['feature_row'].copy()
    require(x.shape==(n,512) and x.dtype==np.float32 and valid.shape==(n,) and valid.dtype==np.bool_,'Invalid feature schema')
    require(mapping.dtype==np.int64 and np.array_equal(mapping,np.arange(n)),'Invalid NPZ row mapping')
    require(np.isfinite(x[valid]).all() and np.isnan(x[~valid]).all(),'Invalid finite/failed embedding representation')
    require(np.allclose(np.linalg.norm(x[valid],axis=1),1,atol=1e-5),'Features must be normalized CLIP embeddings')
    generation_paths={resolve(r['original_source_path']).parent/'evaluation.json' for r in rows}
    require(len(generation_paths)==1,'Mixed generator cohorts')
    generation_path=generation_paths.pop();generation=read(generation_path)
    require(generation['complete'] and generation['count']==n and generation['seed_base']==protocol[split]['seed_base'],'Wrong protocol split/seed base')
    require(generation['checkpoint_step']==625 and generation['state']=='G' and generation['psi']==1 and generation['resolution']==256 and generation['noise_mode']=='const','Wrong generator recipe')
    require(generation['checkpoint_sha256']=='c04482a462abf4ece7034cdc09be19dc857ee201b11427f6526c49ae378bf6dc' and generation['config_sha256']=='b330c97f252c2b0f20d041923900676c806cfdac6b47241e062f50cabef09997','Unapproved generator/config checksum')
    require(len(generation['entries'])==n,'Incomplete generator manifest')
    for i,row in enumerate(rows):
        require(row['cohort_number']==0 and resolve(row['input_directory'])==restoration,'Row cohort mismatch')
        require(row['feature_valid'] is bool(valid[i]) and row['embedding_index']==(i if valid[i] else None),'JSON/NPZ valid mask mismatch')
        if valid[i]:require(row['restoration_error'] is None and row['embedding_error'] is None,'Errors cannot be accepted')
        require(sha(restoration/f'{i:03}'/'record.json')==row['record_sha256'],'Restoration record changed')
        entry=generation['entries'][i]
        require(entry['index']==i and entry['seed']==protocol[split]['seed_base']+i,'Generator identity order mismatch')
        require(resolve(row['original_source_path'])==generation_path.parent/entry['path'] and row['original_source_sha256']==entry['sha256'],'Generator/restoration mismatch')
        require(sha(resolve(row['original_source_path']))==entry['sha256'],'Original image changed')
        if row['webp_path'] is not None:require(sha(resolve(row['webp_path']))==row['webp_sha256'],'Final WebP changed')
    require(result['encoded']==int(valid.sum()),'Encoded denominator mismatch')
    provenance={'embedding_directory':str(directory),'result_sha256':sha(directory/'result.json'),'supervisor_sha256':sha(directory/'supervisor-result.json'),'features_sha256':result['features_sha256'],'generation_manifest':str(generation_path),'generation_sha256':sha(generation_path),'restoration_directory':str(restoration)}
    pipeline={k:result[k] for k in PIPELINE_KEYS}
    restoration_launch=read(restoration/'launch.json')
    pipeline['restoration']={k:restoration_launch[k] for k in ('source_sha256','pins_sha256','background_upsampler','decoder_randomize_noise','fixed_seed')}
    provenance['restoration_launch_sha256']=sha(restoration/'launch.json')
    return x.astype(np.float64),valid,rows,pipeline,provenance

def load_labels(paths,rows):
    labels={};sources=[]
    for path in paths:
        data=read(path);sources.append({'path':str(path),'sha256':sha(path)})
        for label in data['entries']:
            index=label['index'];require(type(index) is int and index not in labels,'Duplicate/invalid label index')
            require(type(label['acceptable']) is bool,'Acceptable must be boolean')
            if label['acceptable']:
                require(all(label.get(k) is True for k in ('adult','male','bareheaded','single_subject','plausible_collar','no_major_artifacts','photographic','appealing')),'Accepted label contradicts mandatory appearance/quality/appeal checks')
            labels[index]=label
    require(set(labels)==set(range(len(rows))),'Labels must retain every attempted row exactly once')
    for row in rows:
        label=labels[row['index']]
        if 'record_sha256' in label:
            require(label['record_sha256']==row['record_sha256'],'Label binds a different restoration record')
        else:
            record=read(resolve(row['input_directory'])/f"{row['index']:03}"/'record.json')
            require(label.get('sha256')==record['outputs'].get('restored-1024.png'),'Label lacks matching record or final PNG hash')
    return np.asarray([labels[r['index']]['acceptable'] for r in rows],dtype=bool),sources

def objective(theta,x,y,lam=1.):
    logits=x@theta[:-1]+theta[-1];errors=expit(logits)-y
    value=np.mean(np.logaddexp(0,logits)-y*logits)+.5*lam*(theta[:-1]@theta[:-1])
    return value,np.r_[x.T@errors/len(y)+lam*theta[:-1],errors.mean()]

def metrics(y,accepted):
    tp=int((y&accepted).sum());fp=int((~y&accepted).sum());fn=int((y&~accepted).sum());tn=int((~y&~accepted).sum())
    return {'attempted':len(y),'accepted':int(accepted.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'acceptance_all_attempts':float(accepted.mean())}

def main(args):
    started=time.monotonic();protocol=read(PROTOCOL);protocol_sha=sha(PROTOCOL)
    require(protocol['classifier']=={'type':'L2 logistic regression','lambda':1.0,'standardization':'development mean/std only','max_iterations':300,'optimizer':'L-BFGS-B','target':'manual acceptable whole portrait','other_heads':'fixed existing adult/headwear cosine margins recorded as diagnostics, not presumed probabilities'},'Declared classifier changed')
    require(not args.output.exists(),'Refuse existing output; no overwrite/search')
    split='development' if args.mode=='fit' else args.split
    x,valid,rows,pipeline,provenance=load_features(args.embeddings,split,protocol)
    y,label_sources=load_labels(args.labels,rows) if args.labels else (None,[])
    model_source=None;optimizer=None
    if args.mode=='fit':
        require(y is not None and valid.any(),'Full development labels and valid embeddings required')
        xt=x[valid];yt=y[valid].astype(np.float64)
        require(0<yt.sum()<len(yt),'Fitting requires both classes among valid features')
        mean=xt.mean(0);scale=np.maximum(xt.std(0),1e-8);a=(xt-mean)/scale
        initial=np.zeros(513);initial[-1]=np.log(yt.mean()/(1-yt.mean()))
        result=minimize(objective,initial,args=(a,yt,1.),jac=True,method='L-BFGS-B',options={'maxiter':300,'gtol':1e-8,'ftol':1e-12})
        require(result.success and np.isfinite(result.x).all() and np.isfinite(result.fun),'Fixed fit did not converge')
        weight=result.x[:-1];intercept=result.x[-1:]
        optimizer={'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'objective':float(result.fun),'gradient_max_abs':float(np.max(np.abs(result.jac))),'loss_definition':'mean BCE + 0.5*lambda*L2(weight)^2; lambda1, unpenalized intercept, no class balancing'}
    else:
        require(args.model is not None and args.threshold in protocol['calibration']['threshold_grid'],'Frozen model and declared fixed threshold required')
        model=read(args.model/'result.json')
        require(model['complete'] and model['mode']=='fit' and model['protocol_sha256']==protocol_sha and model['pipeline']==pipeline,'Frozen model/protocol/feature pipeline mismatch')
        require(sha(args.model/'weights.npz')==model['weights_sha256'],'Frozen weights changed')
        with np.load(args.model/'weights.npz',allow_pickle=False) as archive:
            weight=archive['weight'].copy();intercept=archive['intercept'].copy();mean=archive['mean'].copy();scale=archive['scale'].copy()
        require(weight.shape==mean.shape==scale.shape==(512,) and intercept.shape==(1,) and all(np.isfinite(v).all() for v in (weight,intercept,mean,scale)) and (scale>0).all(),'Invalid frozen model arrays')
        model_source={'directory':str(args.model),'result_sha256':sha(args.model/'result.json'),'weights_sha256':model['weights_sha256']}
    scores=np.full(len(rows),np.nan);scores[valid]=expit(((x[valid]-mean)/scale)@weight+intercept[0])
    predictions=[{'index':r['index'],'feature_row':r['feature_row'],'valid':bool(valid[i]),'score':float(scores[i]) if valid[i] else None,'label':bool(y[i]) if y is not None else None,'accepted':bool(valid[i] and scores[i]>=args.threshold) if args.mode=='score' else None,'forced_reject_reason':None if valid[i] else r['restoration_error'] or r['embedding_error'] or 'missing embedding','webp_sha256':r['webp_sha256'],'original_source_sha256':r['original_source_sha256'],'adult_male_margin':r.get('adult_male_margin'),'headwear_margin':r.get('headwear_margin')} for i,r in enumerate(rows)]
    require(sha(PROTOCOL)==protocol_sha,'Protocol changed during fit/score')
    require(sha(args.embeddings/'result.json')==provenance['result_sha256'] and sha(args.embeddings/'features.npz')==provenance['features_sha256'],'Embeddings changed during fit/score')
    for item in label_sources:require(sha(Path(item['path']))==item['sha256'],'Labels changed during fit/score')
    args.output.mkdir(parents=True)
    weights_sha=None
    if args.mode=='fit':
        np.savez(args.output/'weights.npz',weight=weight,intercept=intercept,mean=mean,scale=scale);weights_sha=sha(args.output/'weights.npz')
    report={'complete':True,'mode':args.mode,'split':split,'protocol':protocol,'protocol_sha256':protocol_sha,'source_sha256':sha(Path(__file__)),'pipeline':pipeline,'provenance':provenance,'label_sources':label_sources,'attempted':len(rows),'valid_features':int(valid.sum()),'excluded_from_fit_indices':[i for i in range(len(rows)) if not valid[i]],'positive_labels_all_attempts':int(y.sum()) if y is not None else None,'positive_labels_valid':int(y[valid].sum()) if y is not None else None,'weights_sha256':weights_sha,'frozen_model':model_source,'optimizer':optimizer,'fixed_threshold':args.threshold if args.mode=='score' else None,'predictions':predictions,'metrics_all_attempts':metrics(y,np.array([r['accepted'] for r in predictions],dtype=bool)) if args.mode=='score' and y is not None else None,'runtime':{'seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__},'scores_are_uncalibrated':True,'no_generator_restorer_or_clip_execution':True,'production_approved':False}
    write(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('mode','split','attempted','valid_features','optimizer','fixed_threshold','metrics_all_attempts','runtime')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fit','score']);parser.add_argument('--embeddings',type=Path,required=True)
    parser.add_argument('--labels',type=Path,action='append',default=[]);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path);parser.add_argument('--threshold',type=float);parser.add_argument('--split',choices=['calibration','test'])
    args=parser.parse_args()
    require(args.mode!='fit' or (args.labels and args.model is None and args.threshold is None and args.split is None),'Fit only accepts development labels, no threshold/model override')
    require(args.mode!='score' or (args.split is not None and args.model is not None and args.threshold is not None),'Score requires split, frozen model, fixed threshold')
    main(args)
