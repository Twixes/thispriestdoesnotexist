"""Compare authenticated completed four-pair native CPU-thread diagnostics only."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / 'research/runs/adam-native1024-fixed-offsets250-ema-importance4'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_run(path):
    protocol = json.loads((path/'protocol.json').read_text())
    result = json.loads((path/'result.json').read_text())
    supervisor = json.loads((path/'supervisor.json').read_text())
    marker = json.loads((path/'importance-0004.json').read_text())
    assert result['complete'] and supervisor['complete'] and supervisor['failure'] is None
    assert result['pairs'] == protocol['pairs'] == marker['pairs'] == 4
    assert protocol['source_step'] == 250 and protocol['resolution'] == 1024
    ph = sha(path/'protocol.json')
    assert result['protocol_sha256'] == supervisor['protocol_sha256'] == marker['protocol_sha256'] == ph
    assert marker['complete'] and marker['state_roundtrip_exact']
    assert marker['sha256'] == result['importance_sha256'] == sha(path/'importance-0004.pt')
    assert result['ema_state_unchanged'] and result['fixed_offsets_zero_and_originals_unchanged']
    assert result['sample_plan_sha256'] == marker['sample_plan_sha256'] == sha(path/'sample-plan.npz')
    values = torch.load(path/'importance-0004.pt', weights_only=True, map_location='cpu')
    assert values['protocol_sha256'] == ph and values['sample_plan_sha256'] == result['sample_plan_sha256']
    for key in ('G','D'):
        assert values[key]['sample_count'] == 4
    return protocol, result, supervisor, values


def main():
    torch.set_num_threads(1)
    base_path = Path(str(PREFIX)+'-v1')
    bp, br, bs, bv = read_run(base_path)
    records=[]
    for threads in (1,2,4,8):
        path = base_path if threads == 1 else Path(str(PREFIX)+f'-threads{threads}-v1')
        p,r,s,v = read_run(path)
        runtime=json.loads((path/'runtime.json').read_text())
        assert runtime['threads'] == p['threads'] == threads
        assert p['checkpoint_sha256'] == bp['checkpoint_sha256']
        assert r['ema_state_sha256'] == br['ema_state_sha256']
        assert r['sample_plan_sha256'] == br['sample_plan_sha256']
        moments={}
        for key in ('G','D'):
            moments[key]={}
            for field in ('gradient_sums','squared_gradient_sums'):
                a,b=bv[key][field],v[key][field]
                assert a.keys() == b.keys()
                assert all(a[n].shape == b[n].shape and a[n].dtype == b[n].dtype for n in a)
                aa=torch.cat([a[n].flatten() for n in a]);bb=torch.cat([b[n].flatten() for n in a])
                assert torch.isfinite(aa).all() and torch.isfinite(bb).all()
                norm=float(torch.linalg.vector_norm(aa))
                moments[key][field]={'exact':torch.equal(aa,bb),'max_abs_difference':float((aa-bb).abs().max()),
                    'relative_l2_difference':float(torch.linalg.vector_norm(aa-bb))/norm if norm else None}
        images=[{'path':f'sample-{i:03}.png','sha256':sha(path/f'sample-{i:03}.png'),
                 'exact_png_bytes':(path/f'sample-{i:03}.png').read_bytes()==(base_path/f'sample-{i:03}.png').read_bytes()} for i in range(4)]
        records.append({'threads':threads,'run':str(path.relative_to(ROOT)),'compute_seconds':r['seconds'],
            'wall_seconds':s['seconds'],'peak_rss_gib':s['peak_rss_gib'],'min_available_fraction':s['min_available_fraction'],
            'speedup_vs_observed_one_thread':br['seconds']/r['seconds'],'moments':moments,'images':images,
            'source_state_and_sample_plan_exact':True,'artifact_sha256':{n:sha(path/n) for n in ('protocol.json','runtime.json','result.json','supervisor.json','importance-0004.json','importance-0004.pt','sample-plan.npz')}})
    output=ROOT/'research/experiments/adam_native/importance-thread-comparison.json'
    output.write_text(json.dumps({'diagnostic_only':True,'pairs_per_run':4,'source_step':250,
        'source_checkpoint_sha256':bp['checkpoint_sha256'],'comparison_source_sha256':sha(Path(__file__)),
        'records':records,'limitations':'Single sequential short trials; timing can reflect thermal/cache/background variation. This is not a long-run performance guarantee, reliable FI ranking, or priest-quality evidence.'},indent=2)+'\n')
    print(json.dumps({'report':str(output),'runs':[{'threads':x['threads'],'compute_seconds':x['compute_seconds'],'peak_rss_gib':x['peak_rss_gib'],'all_moments_exact':all(y['exact'] for k in x['moments'].values() for y in k.values()),'all_pngs_exact':all(y['exact_png_bytes'] for y in x['images'])} for x in records]},indent=2))


if __name__=='__main__':main()
