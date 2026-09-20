"""Research-only extraction/export from actual one-update reference checkpoint."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

import trainer
from check_trainer_resume import equal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'runs/reference256-mps-intended-smoke/resume.pt'
OUT = HERE/'prototype-tests/export'
OUT.mkdir(parents=True,exist_ok=True)
checkpoint = torch.load(SOURCE,map_location='cpu',weights_only=False)
path = trainer.save_generator_snapshot(checkpoint['G_ema'],checkpoint['batch_idx'],checkpoint['images_seen'],
                                       checkpoint['config'],checkpoint['recipe'],OUT)
safe = torch.load(path,map_location='cpu',weights_only=True)
assert safe['step']==1 and safe['images_seen']==32
counts = {'tensors':0,'elements':0}
equal(checkpoint['G_ema'],safe['G_ema'],'G_ema',counts)
json.dumps(safe['config']);json.dumps(safe['recipe'])
assert set(safe)=={'G_ema','step','images_seen','config','recipe'}
del safe, checkpoint
command = [sys.executable,str(ROOT/'export_model.py'),'--checkpoint',str(path),'--output',str(OUT/'bundle'),
           '--version','reference256-intended-smoke-unreviewed','--truncation','1']
export = subprocess.run(command,capture_output=True,text=True,check=True)
(OUT/'export.stdout.log').write_text(export.stdout)
(OUT/'export.stderr.log').write_text(export.stderr)
sys.path.insert(0,str(ROOT.parent))
from inference.server import Model
metadata = json.loads((OUT/'bundle/model.json').read_text())
assert metadata['training_step']==1 and metadata['review']['approved'] is False
rejected = None
try:
    Model(OUT/'bundle')
except ValueError as error:
    rejected = str(error)
assert rejected and 'visually reviewed' in rejected, 'Production gate did not reject unreviewed bundle'
result = {'passed':True,'source_resume':str(SOURCE),'generator_snapshot':str(path),
          'weights_only_load_passed':True,'ema_state_exact':True,'comparison_counts':counts,
          'json_safe_metadata':True,'offline_export_passed':True,'training_step':1,'images_seen':32,
          'production_gate_rejected_unreviewed':True,'production_gate_message':rejected,
          'bundle_weights_sha256':metadata['weights_sha256'],
          'original_training_source_sha256':json.loads((SOURCE.parent/'config.json').read_text())['trainer_sha256'],
          'serialization_trainer_sha256':hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
          'exporter_sha256':hashlib.sha256((ROOT/'export_model.py').read_bytes()).hexdigest()}
(HERE/'export-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
