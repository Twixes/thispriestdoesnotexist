"""Apply the predeclared threshold rule to scored calibration data only."""
import argparse
import hashlib
import json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
a=parser.parse_args();p=a.directory;r=json.loads((p/'result.json').read_text())
assert r['complete'] and r['split']=='calibration' and r['attempted']==32
assert all(type(x['label']) is bool for x in r['predictions'])
rows=[]
for t in r['protocol']['calibration']['threshold_grid']:
    accepted=[x for x in r['predictions'] if x['valid'] and x['score']>=t]
    rows.append({'threshold':t,'accepted':len(accepted),'false_accepts':[x['index'] for x in accepted if not x['label']],'true_accepts':sum(x['label'] for x in accepted)})
possible=[x for x in rows if x['accepted']>=4 and not x['false_accepts']]
result={'protocol_sha256':r['protocol_sha256'],'model':r['frozen_model'],'calibration_scores_sha256':hashlib.sha256((p/'result.json').read_bytes()).hexdigest(),'grid':rows,'selected_threshold':possible[0]['threshold'] if possible else None,'passed_calibration':bool(possible),'test_data_used':False,'production_approved':False}
path=p/'threshold-selection.json'
if path.exists():
    assert json.loads(path.read_text())==result, 'Existing selection differs; never overwrite'
else:
    path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
