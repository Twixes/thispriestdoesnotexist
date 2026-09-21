"""PNG/JSON/hash-only audit; never imports Torch or runs a model."""
import hashlib,json,statistics
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(path.read_text())
def main():
 launch=read(ROOT.with_suffix('.launch.json'))
 assert launch['exit_code']==0
 mechanics=read(ROOT/'comparison-mechanics.json')
 assert mechanics['complete'] and mechanics['initial_student_states_exact'] and mechanics['all50_pair_and_fresh_latent_draws_exact']
 branches=['baseline','collar-surround'];checkpoints=[];summaries=[];outputs=[];evaluations={}
 records=[read(ROOT/b/'complete.json') for b in branches]
 assert records[0]['schedule']==records[1]['schedule'] and len(records[0]['schedule'])==50
 assert records[0]['initial_student_state_sha256']==records[1]['initial_student_state_sha256']
 for b,c in zip(branches,records):
  assert c['complete'] and c['updates']==50 and c['full_checkpoint_roundtrip'] and c['source_and_frozen_checks_passed']
  config=read(ROOT/b/'config.json');parent=REPO/'research/runs/paired-regions1024-sixpair-600/resume.pt'
  assert sha(parent)==config['parent_checkpoint_sha256']
  assert config['parent_student_state_sha256']==c['initial_student_state_sha256']
  for m in c['milestones']:
   f=ROOT/b/m['checkpoint'];actual=sha(f);assert actual==m['sha256'];checkpoints.append({'path':str(f.relative_to(ROOT)),'sha256':actual,'bytes':f.stat().st_size,'student_sha256':m['student_sha256']})
   step=m['probe_step'];d=ROOT/b/f'preview-{step:03}';e=read(d/'evaluation.json');evaluations[(b,step)]=e
   assert e['state_sha256']==m['student_sha256'] and e['count']==11
   for r in e['rows']:
    for kind,o in r['outputs'].items():
     f=d/o['path'];assert sha(f)==o['sha256']
     with Image.open(f) as im:assert im.size==(1024,1024) and im.mode=='RGB'
     outputs.append({'path':str(f.relative_to(ROOT)),'sha256':o['sha256']})
   train=[r['metrics_minus1_to1'] for r in e['rows'] if r['split']=='train'];assert len(train)==6
   means={k:statistics.mean(r[k] for r in train) for k in ['tab_l1','surround_l1','remainder_l1','protected_l1','rest_l1','clothing_l1']}
   holdout=[r for r in e['rows'] if r['split']=='validation'];assert len(holdout)==1 and holdout[0]['id']=='028'
   summaries.append({'branch':b,'probe_step':step,'train_count':6,'train_mean_minus1_to1':means,'heldout_028_minus1_to1':holdout[0]['metrics_minus1_to1']})
 a=evaluations[('baseline',0)];v=evaluations[('collar-surround',0)]
 assert a['rows']==v['rows']
 per_image=[]
 for step in [0,25,50]:
  for br,vr in zip(evaluations[('baseline',step)]['rows'],evaluations[('collar-surround',step)]['rows']):
   assert br['id']==vr['id'];bm=br['metrics_minus1_to1'];vm=vr['metrics_minus1_to1']
   per_image.append({'step':step,'id':br['id'],'split':br['split'],'baseline':bm,'collar_surround':vm,'variant_minus_baseline':{k:vm[k]-bm[k] for k in bm}})
 result={'production_approved':False,'model_inference_performed':False,'launch_exit_code':0,'mechanics_sha256':sha(ROOT/'comparison-mechanics.json'),'all_50_draws_equal':True,'same_initial_student':True,'step0_rows_and_images_exact':True,'full_checkpoint_hashes':checkpoints,'verified_native_outputs':outputs,'summaries':summaries,'per_image':per_image,'units':'L1 measured in [-1,1] luminance; multiply by127.5 for uint8-equivalent MAE. Means are equal across six TRAIN images, not pixel-weighted. HOLDOUT kept separate.','limitations':'Metrics describe mask agreement, not photographic quality, identity preservation, collar recognition or unseen generalization. Collar-surround is a single radius30 square exterior ring clipped to clothing; validation028 has no tab annotation; four fixed unseen have no target metrics.'}
 (ROOT/'comparison-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 # Clearly labeled diagnostic contact only; image content itself is unchanged apart from resizing.
 rows=a['rows'];tile=256;canvas=Image.new('RGB',(tile*3,(tile+24)*len(rows)),(15,15,15));draw=ImageDraw.Draw(canvas)
 for i,r in enumerate(rows):
  for j,(b,s,label) in enumerate([('baseline',0,'parent600'),('baseline',50,'baseline+50'),('collar-surround',50,'surround+50')]):
   f=ROOT/b/f'preview-{s:03}'/f"{r['id']}-gray.png"
   with Image.open(f) as im:canvas.paste(im.resize((tile,tile),Image.Resampling.LANCZOS),(j*tile,i*(tile+24)+24))
   draw.text((j*tile+3,i*(tile+24)+5),f"{r['id']} {r['split']} / {label}",fill='white')
 canvas.save(ROOT/'comparison-diagnostic.png')
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
