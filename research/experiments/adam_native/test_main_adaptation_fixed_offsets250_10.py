"""AST and declared synthetic provenance fixtures only; no FI/model computation."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import main_adaptation_fixed_offsets250_10 as route


class RetainedSourceFixture:
    """Opaque checkpoint/image bytes exercise provenance gates, never model loading."""
    def __init__(self, root):
        self.root=Path(root).resolve();self.relative=route.RETAINED_SOURCE_REL;self.run=self.root/self.relative
        self.run.mkdir(parents=True)
        self.source={'iterations':500,'checkpoint_steps':[250,500]}
        self.put(self.relative+'/protocol.json',self.source)
        self.put(self.relative+'/checkpoint-250.pt','DECLARED_SYNTHETIC_NON_MODEL_BYTES')
        self.put(self.relative+'/step-250/manifest.json',{'declared_synthetic':True})
        self.record={'complete':True,'step':250,'checkpoint':'checkpoint-250.pt',
            'checkpoint_sha256':self.sha(self.relative+'/checkpoint-250.pt'),
            'protocol_sha256':self.sha(self.relative+'/protocol.json'),
            'snapshot_manifest_sha256':self.sha(self.relative+'/step-250/manifest.json'),
            'model_optimizer_rng_path_state_restored_exactly':True,'optimizer_steps':{'g':313,'d':266}}
        self.put(self.relative+'/checkpoint-250.json',self.record)
        self.reviews={}
        for owner,indices in [('root',[0,1]),('agent',[2,3])]:
            images=[]
            for index in indices:
                for arm in ('raw','ema'):
                    relative=f'{self.relative}/step-250/{arm}-{index:03}.png'
                    self.put(relative,'DECLARED_SYNTHETIC_NON_IMAGE_BYTES '+relative)
                    images.append({'path':relative,'sha256':self.sha(relative)})
            review={'reviewer':owner,'step':250,'indices':indices,'viewed_images':images,
                'photographic_regression':True,'adult_priest_domain_achieved':False,'production_approved':False,
                'snapshot_manifest_sha256':self.record['snapshot_manifest_sha256'],
                'overall':{'major_raw_photographic_regression_vs100':True,'ema_close_to_source':True,'quality_accepted':False}}
            relative=f'research/reviews/synthetic/{owner}.json';self.put(relative,review);self.reviews[relative]=self.sha(relative)
        self.termination={'status':'stopped_after_visual_review','last_completed_step':264,'last_complete_checkpoint':250,
            'planned_iterations':500,'optimizer_steps':{'g':330,'d':281},'supervisor_and_worker_absent':True,
            'production_approved':False,'server_latency_proven':False,'review_files':self.reviews}
        self.supervisor={'complete':False,'failure':'declared synthetic KeyboardInterrupt',
            'protocol_sha256':self.record['protocol_sha256']}
        self.put(self.relative+'/termination.json',self.termination);self.put(self.relative+'/supervisor.json',self.supervisor)
        self.pins={relative:self.sha(relative) for relative in [self.relative+'/'+name for name in
            ('protocol.json','checkpoint-250.pt','checkpoint-250.json','step-250/manifest.json','termination.json','supervisor.json')]+list(self.reviews)}

    def put(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value,dict) else value)

    def sha(self,relative):return route.sha(self.root/relative)

    def evaluate(self):
        with mock.patch.object(route,'ROOT',self.root),mock.patch.object(route,'RETAINED250_PINS',self.pins),mock.patch.object(route,'RETAINED250_REVIEWS',self.reviews):
            return route.validate_retained250_source(self.run,self.source,self.record)


class RetainedRouteTests(unittest.TestCase):
    def test_training_core_worker_and_all_fi_authentication_unchanged(self):
        parent=route.HERE/'main_adaptation_fixed_offsets10.py'
        self.assertEqual(route.sha(parent),route.PARENT_RUNNER_SHA)
        def nodes(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=nodes(parent);new=nodes(route.HERE/'main_adaptation_fixed_offsets250_10.py')
        dump=lambda x:ast.dump(x,include_attributes=False)
        for name in ('AdaptationEngine','worker','expected_steps','state_digest','sha','read','write','under_root'):
            self.assertEqual(dump(old[name]),dump(new[name]),name)
        normalized=copy.deepcopy(new['validated_selection']);body=[]
        for node in normalized.body:
            source=ast.unparse(node)
            if isinstance(node,ast.Assert) and "protocol['source_step'] == 250" in source:
                node=next(copy.deepcopy(n) for n in old['validated_selection'].body if isinstance(n,ast.Assert) and "protocol['source_step'] == 500" in ast.unparse(n))
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='validate_retained250_source':
                body.extend(ast.parse("assert read(source_run / 'supervisor.json')['complete'] is True\nassert read(source_run / 'result.json')['complete'] is True").body)
                continue
            if isinstance(node,ast.Assert) and "under_root(protocol['checkpoint']) == source_run / 'checkpoint-250.pt'" in source:continue
            body.append(node)
        normalized.body=body
        self.assertEqual(dump(old['validated_selection']),dump(normalized))
        old_super=copy.deepcopy(old['supervise']);new_super=copy.deepcopy(new['supervise'])
        class Paths(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='main_adaptation_fixed_offsets250_10.py':node.value='main_adaptation_fixed_offsets10.py'
                return node
        self.assertEqual(dump(old_super),dump(Paths().visit(new_super)))

    def test_declared_synthetic_exact_retained_route_is_accepted_as_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=RetainedSourceFixture(temp)
            self.assertEqual(fixture.evaluate(),{'source_run_complete':False,'source_step':250,'stopped_after_step':264,'reviewed_ema_diagnostic_only':True})

    def test_unreviewed_corrupt_or_unrecognized_source_refused(self):
        for corruption in ('termination_hash','checkpoint_bytes','reviewed_image','unknown_path','missing_review'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as temp:
                f=RetainedSourceFixture(temp)
                if corruption=='termination_hash':f.put(f.relative+'/termination.json',{})
                elif corruption=='checkpoint_bytes':f.put(f.relative+'/checkpoint-250.pt','changed')
                elif corruption=='reviewed_image':f.put(f.relative+'/step-250/ema-003.png','changed')
                elif corruption=='unknown_path':f.run=f.root/'research/runs/generic-failed-run'
                else:(f.root/next(iter(f.reviews))).unlink()
                with self.assertRaises((AssertionError,FileNotFoundError)):f.evaluate()

    def test_failed_run_semantics_cannot_be_relabelled_completed_or_generic_failure(self):
        # Re-pin the synthetic fixture to isolate semantic checks from hash checks.
        for key,value in [('status','failed'),('last_completed_step',500),('last_complete_checkpoint',100),('production_approved',True),('supervisor_and_worker_absent',False)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                f=RetainedSourceFixture(temp);f.termination[key]=value
                f.put(f.relative+'/termination.json',f.termination);f.pins[f.relative+'/termination.json']=f.sha(f.relative+'/termination.json')
                with self.assertRaises(AssertionError):f.evaluate()
        for complete,failure in [(True,None),(False,'out of memory')]:
            with tempfile.TemporaryDirectory() as temp:
                f=RetainedSourceFixture(temp);f.supervisor.update(complete=complete,failure=failure)
                f.put(f.relative+'/supervisor.json',f.supervisor);f.pins[f.relative+'/supervisor.json']=f.sha(f.relative+'/supervisor.json')
                with self.assertRaises(AssertionError):f.evaluate()


if __name__=='__main__':unittest.main(verbosity=2)
