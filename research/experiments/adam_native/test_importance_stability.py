"""Synthetic metadata and raw-sum fixtures only; no network construction/loading."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
import torch
from analyze_importance_stability import analyze_accumulators, average_ranks, load_verified_run, sha, ROOT, POLICY
from test_selection import inventory_fixture
from selection import build_score_inventory
from fixed_offset_policy import is_fixed_offset


def means(inventory, reverse=False, scale=1):
    desc = build_score_inventory(inventory)
    result = {n: torch.zeros(shape,dtype=torch.float64) for n,shape in desc['score_shapes'].items() if not is_fixed_offset(n)}
    pools = {}
    for name, rec in desc['modules'].items(): pools.setdefault(rec['pool'], []).append((name,rec))
    for members in pools.values():
        count = sum(rec['weight_shape'][0] for _,rec in members)
        values = torch.arange(1,count+1,dtype=torch.float64)
        if reverse: values=values.flip(0)
        values=values*scale; cursor=0
        for _, rec in members:
            rows=rec['weight_shape'][0]
            result[rec['v_parameter']] = values[cursor:cursor+rows]*(2 if rec['b_parameter'] else 1)
            cursor+=rows
    return result


def fixtures(second_scale=1, second_reverse=False, first_scale=1):
    inventories={k:inventory_fixture(k) for k in ('G','D')}
    def payload(count):
        return {'pairs':count, 'protocol_sha256':'a'*64, 'sample_plan_sha256':'b'*64}
    prefix=payload(500); full=payload(1000)
    for key, inv in inventories.items():
        first=means(inv,scale=first_scale); second=means(inv,reverse=second_reverse,scale=second_scale)
        for payload,count,totals in ((prefix,500,{n:t*500 for n,t in first.items()}),
                                    (full,1000,{n:(t+second[n])*500 for n,t in first.items()})):
            payload[key]={'version':1,'missing':'error','dtype':'torch.float64','sample_count':count,
                'observed_counts':{n:count for n in totals},'gradient_sums':{n:torch.zeros_like(t) for n,t in totals.items()},
                'squared_gradient_sums':totals}
    return prefix,full,inventories


class StabilityTests(unittest.TestCase):
    def test_scaled_identical_rankings_masks_and_actual_count_normalization(self):
        prefix,full,inv=fixtures(second_scale=3)
        original=copy.deepcopy(prefix)
        report=analyze_accumulators(prefix,full,inv)
        for comparison in report['comparisons'].values():
            for pool in comparison['pools'].values():
                self.assertAlmostEqual(pool['spearman_rho'],1)
                for q in pool['quantiles'].values():
                    self.assertEqual(q['agreement_fraction'],1)
                    self.assertEqual(q['high_jaccard'],1)
        nested=report['comparisons']['nested_prefix500_vs_full1000']['pools']['g_conv']
        disjoint=report['comparisons']['disjoint_first500_vs_last500']['pools']['g_conv']
        self.assertEqual(nested['right']['maximum'],2*nested['left']['maximum'])
        self.assertEqual(disjoint['right']['maximum'],3*disjoint['left']['maximum'])
        self.assertIsNone(report['automatic_threshold_choice']);self.assertIsNone(report['pass_criteria'])
        for key in ('G','D'):
            for name,t in prefix[key]['squared_gradient_sums'].items():self.assertTrue(torch.equal(t,original[key]['squared_gradient_sums'][name]))
        json.dumps(report,allow_nan=False)

    def test_disjoint_reversal_detected_despite_perfect_nested_rank_agreement(self):
        report=analyze_accumulators(*fixtures(first_scale=10,second_scale=1,second_reverse=True))
        for pool in report['comparisons']['nested_prefix500_vs_full1000']['pools'].values():
            self.assertAlmostEqual(pool['spearman_rho'],1)
            self.assertEqual(pool['quantiles']['50']['agreement_fraction'],1)
        for pool in report['comparisons']['disjoint_first500_vs_last500']['pools'].values():
            self.assertAlmostEqual(pool['spearman_rho'],-1)
            self.assertEqual(pool['quantiles']['50']['agreement_fraction'],0)
            self.assertEqual(pool['quantiles']['50']['high_jaccard'],0)

    def test_ties_and_empty_high_sets_are_explicitly_undefined(self):
        report=analyze_accumulators(*fixtures(first_scale=0,second_scale=0))
        for comparison in report['comparisons'].values():
            for pool in comparison['pools'].values():
                self.assertIsNone(pool['spearman_rho']);self.assertIsNotNone(pool['undefined_reason'])
                self.assertTrue(pool['left']['all_zero']);self.assertTrue(pool['right']['all_equal'])
                self.assertEqual(pool['left']['tied_coordinates'],pool['left']['coordinates'])
                for q in pool['quantiles'].values():
                    self.assertEqual(q['agreement_fraction'],1)
                    self.assertEqual(q['left_high'],0)
                    self.assertIsNone(q['high_jaccard'])
                    self.assertIsNotNone(q['jaccard_undefined_reason'])
        import numpy as np
        np.testing.assert_array_equal(average_ranks(np.array([4.,1.,1.,2.])),[4.,1.5,1.5,3.])
        json.dumps(report,allow_nan=False)

    def test_completed_file_lineage_is_authenticated_without_loading_model(self):
        prefix,full,inventories=fixtures()
        with tempfile.TemporaryDirectory(prefix='test-fi-stability-',dir=ROOT/'research/runs') as tmp:
            base=Path(tmp); source=base/'source';run=base/'fi';source.mkdir();run.mkdir()
            def write(path,value):path.write_text(json.dumps(value)+'\n')
            write(source/'modulation-inventory.json',inventories)
            write(source/'protocol.json',{'synthetic_test_only':True})
            # This is deliberately not a torch checkpoint: the analyzer must
            # authenticate its bytes, never deserialize or load a model.
            (source/'checkpoint.pt').write_bytes(b'synthetic opaque checkpoint fixture')
            inv_path=source/'modulation-inventory.json'
            protocol={'name':'adam-output-rank1-fixed-offsets-ema-importance','fixed_offset_policy':POLICY,
                'pairs':1000,'diagnostic_only':False,'state_keys':['Gema','Dema'],'layout':'output_rank1',
                'resolution':1024,'accumulator_dtype':'torch.float64',
                'pins':{str(inv_path.relative_to(ROOT)):sha(inv_path)},
                'source_run':str(source.relative_to(ROOT)),'source_protocol_sha256':sha(source/'protocol.json'),
                'checkpoint':str((source/'checkpoint.pt').relative_to(ROOT)),'checkpoint_sha256':sha(source/'checkpoint.pt')}
            write(run/'protocol.json',protocol); ph=sha(run/'protocol.json')
            import numpy as np
            np.savez(run/'sample-plan.npz',synthetic=np.arange(1000))
            zh=sha(run/'sample-plan.npz')
            write(run/'sample-plan.json',{'pairs':1000,'npz_sha256':zh})
            for count,payload in ((500,prefix),(1000,full)):
                payload.update(protocol_sha256=ph,sample_plan_sha256=zh)
                name=f'importance-{count:04}.pt';torch.save(payload,run/name)
                write(run/f'importance-{count:04}.json',{'complete':True,'pairs':count,'path':name,
                    'state_roundtrip_exact':True,'sha256':sha(run/name),'protocol_sha256':ph,'sample_plan_sha256':zh})
            result={'complete':True,'pairs':1000,'protocol_sha256':ph,'sample_plan_sha256':zh,
                'ema_state_unchanged':True,'fixed_offsets_zero_and_originals_unchanged':True,
                'importance_path':'importance-1000.pt','importance_sha256':sha(run/'importance-1000.pt')}
            write(run/'result.json',result)
            write(run/'supervisor.json',{'complete':True,'failure':None,'protocol_sha256':ph})
            p,f,i,hashes=load_verified_run(run)
            self.assertEqual(analyze_accumulators(p,f,i)['sample_counts']['last500'],500)
            self.assertIn('importance-0500.pt',hashes)
            marker=run/'importance-0500.json';record=json.loads(marker.read_text())
            record['sample_plan_sha256']='c'*64;write(marker,record)
            with self.assertRaisesRegex(ValueError,'Marker lineage mismatch'):load_verified_run(run)
            record['sample_plan_sha256']=zh;write(marker,record)
            (source/'checkpoint.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source checkpoint hash mismatch'):load_verified_run(run)

    def test_incompatible_lineage_counts_shapes_and_invalid_sums_rejected(self):
        mutations=[lambda p,f,i:f.update(protocol_sha256='c'*64),
            lambda p,f,i:f.update(sample_plan_sha256='c'*64),
            lambda p,f,i:f['D'].update(sample_count=999),
            lambda p,f,i:f.update(pairs=True),
            lambda p,f,i:f['G']['squared_gradient_sums'].pop(next(iter(f['G']['squared_gradient_sums']))),
            lambda p,f,i:f['G']['gradient_sums'].update({next(iter(f['G']['gradient_sums'])):torch.ones(77,dtype=torch.float64)}),
            lambda p,f,i:f['D'].update(missing='zero'),
            lambda p,f,i:f['G']['observed_counts'].update({next(iter(f['G']['observed_counts'])):999})]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                p,f,i=fixtures();mutate(p,f,i)
                with self.assertRaises(ValueError):analyze_accumulators(p,f,i)
        for bad in (-1,float('nan'),float('inf')):
            p,f,i=fixtures();name=next(iter(f['G']['squared_gradient_sums']))
            f['G']['squared_gradient_sums'][name].fill_(bad)
            with self.assertRaises(ValueError):analyze_accumulators(p,f,i)
        p,f,i=fixtures()
        name=next(n for n,t in p['G']['squared_gradient_sums'].items() if bool(t.sum()))
        f['G']['squared_gradient_sums'][name]=p['G']['squared_gradient_sums'][name]*.5
        with self.assertRaisesRegex(ValueError,'negative squared-gradient sum'):analyze_accumulators(p,f,i)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
