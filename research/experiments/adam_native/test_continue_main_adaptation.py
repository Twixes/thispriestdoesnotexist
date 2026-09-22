"""Actual tiny-engine stages plus explicitly opaque source-record fixtures."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import continue_main_adaptation as runner
from test_continue_main_adaptation_fixed_offsets250_100 import make_engine,envelope


class SourceFixture:
    """No fixture checkpoint/image/latent file is model- or image-loadable."""
    def __init__(self,directory):
        self.root=Path(directory).resolve();self.research=self.root/'research';self.here=self.research/'experiments/adam_native'
        self.own=self.here/'continue_main_adaptation.py';self.put(self.own,'opaque runner fixture')
        self.legacy=self.here/'continue_main_adaptation_fixed_offsets250_250.py';self.put(self.legacy,'opaque legacy fixture')
        self.bootstrap=self.root/runner.BOOTSTRAP_RUN
        self.old=self.research/'runs/declared-synthetic-main100';self.put(self.old/'protocol.json','opaque source100 protocol')
        self.fields={key:'declared synthetic '+key for key in runner.TRAINING_FIELDS}
        self.fields.update(threads=1,device='cpu',resolution=1024,batch=1,selections={'G':{},'D':{}},training=[])
        self.old_protocol=dict(self.fields,iterations=100)
        self.old_marker={'checkpoint_sha256':'declared-source100-sha','state_sha256':'declared-source100-state'}
        protocol=dict(self.fields,name='adam-native1024-retained250-main-adaptation100-to250',iterations=250,source_main_step=100,
            source_main_run=str(self.old.relative_to(self.root)),source_main_protocol_sha256=runner.sha(self.old/'protocol.json'),
            source_main_checkpoint_sha256=self.old_marker['checkpoint_sha256'],source_main_state_sha256=self.old_marker['state_sha256'],
            checkpoint_steps=[150,250],preview_steps=[100,150,250],expected_optimizer_steps=runner.expected_steps(250),
            guards={'start_available_fraction':.35,'runtime_available_fraction':.20,'rss_gib':12,'swap_growth_mib':512,'seconds':3600},
            pins={str(self.legacy.relative_to(self.root)):runner.sha(self.legacy)})
        self.emit(self.bootstrap,protocol)
        self.anchor_sha=runner.sha(self.bootstrap/'protocol.json')

    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,dict):runner.write(path,value)
        else:path.write_text(value)

    def emit(self,run,protocol):
        self.put(run/'protocol.json',protocol);digest=runner.sha(run/'protocol.json')
        self.put(run/'eval-z.npz','declared non-npz bytes')
        start,end=protocol['source_main_step'],protocol['iterations'];markers=[]
        for step in protocol['preview_steps']:
            folder=run/f'step-{step:03}';images=[]
            for arm in ('raw','ema'):
                for i in range(4):
                    path=folder/f'{arm}-{i:03}.png';self.put(path,'declared non-image bytes')
                    images.append({'path':path.name,'sha256':runner.sha(path)})
            self.put(folder/'manifest.json',{'complete':True,'step':step,'rng_unchanged':True,'protocol_sha256':digest,
                'eval_z_sha256':runner.sha(run/'eval-z.npz'),'parent_pngs_exact':step==start,'images':images})
        for step in protocol['checkpoint_steps']:
            path=run/f'checkpoint-{step:03}.pt';self.put(path,'declared non-model bytes '+str(step))
            record={'complete':True,'step':step,'checkpoint':path.name,'checkpoint_sha256':runner.sha(path),
                'state_sha256':'declared state '+str(step),'protocol_sha256':digest,'optimizer_steps':runner.expected_steps(step),
                'model_optimizer_rng_path_state_restored_exactly':True,'snapshot_manifest_sha256':runner.sha(run/f'step-{step:03}/manifest.json')}
            self.put(run/f'checkpoint-{step:03}.json',record);markers.append(record)
        self.put(run/'result.json',{'complete':True,'iterations':end,'first_completed_step':start+1,'additional_iterations':end-start,
            'optimizer_steps':runner.expected_steps(end),'additional_optimizer_steps':{key:runner.expected_steps(end)[key]-runner.expected_steps(start)[key] for key in ('g','d')},
            'fixed_offsets_and_mask_invariants':True,'main_adaptation_run':True,'development_continuation_only':True,
            'folded_native_max_errors':{'raw':[0.0]*4,'ema':[0.0]*4},'protocol_sha256':digest,'checkpoints':markers,
            'parent_checkpoint_sha256':protocol['source_main_checkpoint_sha256']})
        self.put(run/'supervisor.json',{'complete':True,'failure':None,'protocol_sha256':digest})
        self.put(run/'restoration.json',{'complete':True,'step':start,'full_engine_state_restored_exactly':True,
            'parent_checkpoint_sha256':protocol['source_main_checkpoint_sha256'],'parent_state_sha256':protocol['source_main_state_sha256'],
            'optimizer_steps':runner.expected_steps(start)})

    def next_endpoint(self):
        protocol=runner.read(self.bootstrap/'protocol.json');marker=runner.read(self.bootstrap/'checkpoint-250.json')
        protocol.update(name=runner.NAME,continuation_schema=1,bootstrap_run=runner.BOOTSTRAP_RUN,bootstrap_protocol_sha256=self.anchor_sha,
            source_main_run=runner.BOOTSTRAP_RUN,source_main_step=250,iterations=500,checkpoint_steps=[500],preview_steps=[250,500],
            source_main_protocol_sha256=self.anchor_sha,source_main_checkpoint_sha256=marker['checkpoint_sha256'],source_main_state_sha256=marker['state_sha256'],
            expected_optimizer_steps=runner.expected_steps(500))
        protocol['pins'][str(self.own.relative_to(self.root))]=runner.sha(self.own)
        run=self.research/'runs/declared-synthetic-endpoint500';self.emit(run,protocol);return run

    def validate(self,run):
        def under(relative):
            path=(self.root/relative).resolve();path.relative_to(self.root);return path
        with mock.patch.multiple(runner,ROOT=self.root,RESEARCH=self.research,HERE=self.here,OWN_FILE=self.own,
                BOOTSTRAP_PROTOCOL_SHA=self.anchor_sha,LEGACY_SHA=runner.sha(self.legacy),under_root=under),\
                mock.patch.object(runner.legacy,'validate_parent',return_value=(self.old_protocol,self.old_marker)):
            return runner.validate_source(run)


class GenericContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_actual_engine_stages_equal_uninterrupted_with_regularizers(self):
        self.assertIs(runner.AdaptationEngine,runner.legacy.AdaptationEngine)
        self.assertIs(runner.restore_checkpoint,runner.legacy.restore_checkpoint)
        engine=make_engine();real=[torch.randn(3,32,32) for _ in range(2)]
        for _ in range(10):engine.iteration(real)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'tiny.pt';torch.save(envelope(engine),path)
            expected_records=[];expected_snapshots=[]
            def append(target,record):target.append({key:value for key,value in record.items() if key!='seconds'})
            runner.run_updates(engine,real,48,[16,32,48],lambda record:append(expected_records,record),expected_snapshots.append)
            expected=runner.state_digest(engine.state());actual_records=[];actual_snapshots=[]
            for target in (16,32,48):
                resumed=make_engine();payload=torch.load(path,weights_only=True)
                runner.restore_checkpoint(resumed,payload,expected_digest=runner.state_digest(payload),protocol_sha='declared-tiny-protocol',
                    runtime_sha='declared-tiny-runtime',inventory_sha='declared-tiny-inventory',expected_iterations=payload['engine']['iterations'])
                runner.run_updates(resumed,real,target,[target],lambda record:append(actual_records,record),actual_snapshots.append)
                torch.save(envelope(resumed),path)
            self.assertEqual(runner.state_digest(resumed.state()),expected)
            self.assertEqual(actual_records,expected_records);self.assertEqual(actual_snapshots,expected_snapshots)
            self.assertEqual([r['iteration'] for r in actual_records if 'r1_loss' in r],[16,32])
            corrupted=copy.deepcopy(payload);corrupted['protocol_sha256']='wrong'
            before=runner.state_digest(resumed.state())
            with self.assertRaises(AssertionError):
                runner.restore_checkpoint(resumed,corrupted,expected_digest=runner.state_digest(corrupted),protocol_sha='declared-tiny-protocol',
                    runtime_sha='declared-tiny-runtime',inventory_sha='declared-tiny-inventory',expected_iterations=32)
            self.assertEqual(runner.state_digest(resumed.state()),before)

    def test_schedules_require_progress_and_terminal_snapshot(self):
        self.assertEqual(runner.schedule(250,1500,[500,1000,1500],21600),(500,1000,1500))
        for arguments in [(250,250,[250],10),(250,500,[500,500],10),(250,500,[300],10),
                (250,500,[200,500],10),(250,500,[500],0),(250,500,[500],True)]:
            with self.assertRaises(AssertionError):runner.schedule(*arguments)

    def test_declared_opaque_bootstrap_and_own_endpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=SourceFixture(temporary)
            protocol,marker=fixture.validate(fixture.bootstrap)
            self.assertEqual(protocol['iterations'],250);self.assertEqual(marker['step'],250)
            endpoint=fixture.next_endpoint();protocol,marker=fixture.validate(endpoint)
            self.assertEqual(protocol['iterations'],500);self.assertEqual(marker['step'],500)

    def test_reject_incomplete_changed_recipe_corrupt_checkpoint_or_unknown_format(self):
        for defect in ('incomplete','recipe','checkpoint','format','runner'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temporary:
                fixture=SourceFixture(temporary);endpoint=fixture.next_endpoint()
                if defect=='incomplete':
                    data=runner.read(endpoint/'supervisor.json');data['complete']=False;fixture.put(endpoint/'supervisor.json',data)
                elif defect in ('recipe','format'):
                    data=runner.read(endpoint/'protocol.json')
                    if defect=='recipe':data['g_lr']='changed optimizer recipe'
                    else:data['name']='generic untrusted experiment'
                    fixture.put(endpoint/'protocol.json',data)
                elif defect=='checkpoint':fixture.put(endpoint/'checkpoint-500.pt','changed bytes')
                else:fixture.put(fixture.own,'changed runner source')
                with self.assertRaises(AssertionError):fixture.validate(endpoint)


if __name__=='__main__':unittest.main(verbosity=2)
