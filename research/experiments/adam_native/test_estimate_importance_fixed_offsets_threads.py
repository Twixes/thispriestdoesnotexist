"""Bounded provenance/AST/thread propagation tests; no native model is loaded."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import estimate_importance_fixed_offsets as parent
import estimate_importance_fixed_offsets_threads as variant


def functions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}


def dump(value):
    return ast.dump(value,include_attributes=False)


class ThreadEstimatorTests(unittest.TestCase):
    def test_frozen_parent_and_unchanged_algorithm_ast(self):
        self.assertEqual(variant.sha(variant.HERE/'estimate_importance_fixed_offsets.py'),variant.BASE_ESTIMATOR_SHA)
        old=functions(variant.HERE/'estimate_importance_fixed_offsets.py')
        new=functions(variant.HERE/'estimate_importance_fixed_offsets_threads.py')
        self.assertEqual(set(old),set(new))
        for name in ('make_sample_plan','pair_gradients','validate_input','sha','write'):
            self.assertEqual(dump(old[name]),dump(new[name]),name)
        # The entire checkpoint/model/accumulation/render/save body is identical.
        def computation(fn):
            start=next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='checkpoint' for t in node.targets))
            return ast.Module(body=fn.body[start:],type_ignores=[])
        self.assertEqual(dump(computation(old['worker'])),dump(computation(new['worker'])))
        # Monitoring, process ownership, deadlines, guards and failure handling are identical.
        class NormalizeSupervisor(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='estimate_importance_fixed_offsets_threads.py':node.value='estimate_importance_fixed_offsets.py'
                return node
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='prepare':node.keywords=[]
                return self.generic_visit(node)
        normalized=NormalizeSupervisor().visit(copy.deepcopy(new['supervise']))
        normalized.args=copy.deepcopy(old['supervise'].args)
        self.assertEqual(dump(old['supervise']),dump(normalized))
        source=(variant.HERE/'estimate_importance_fixed_offsets_threads.py').read_text()
        self.assertIn("torch.set_num_threads(protocol['threads'])",source)
        self.assertIn('torch.set_num_interop_threads(1)',source)
        self.assertIn("parser.add_argument('--threads', type=int, choices=THREAD_CHOICES, default=1)",source)

    def test_prepare_explicit_threads_and_default_preserve_every_other_protocol_field(self):
        with tempfile.TemporaryDirectory(prefix='thread-estimator-test-',dir=variant.RESEARCH/'runs') as temp:
            directory=Path(temp);run=directory/'input';run.mkdir()
            for name in ('protocol.json','checkpoint-250.json','modulation-inventory.json','runtime.json'):
                (run/name).write_text('{}\n')
            source={'pins':{},'training':[{'id':1,'path':'fixture','sha256':'fixture'}]}
            record={'checkpoint':'checkpoint-250.pt','checkpoint_sha256':'fixture'}
            memory=SimpleNamespace(available=1,total=1)
            # Opaque fixtures test protocol construction only; no native input is claimed valid.
            with mock.patch.object(parent,'validate_input',return_value=(source,record)),mock.patch.object(variant,'validate_input',return_value=(source,record)),mock.patch('psutil.virtual_memory',return_value=memory):
                original=parent.prepare(run,250,4,directory/'original')
                values=[(None,1),(1,1),(2,2),(4,4),(8,8)]
                for i,(argument,expected) in enumerate(values):
                    output=directory/f'variant-{i}'
                    actual=variant.prepare(run,250,4,output) if argument is None else variant.prepare(run,250,4,output,threads=argument)
                    self.assertEqual(actual['threads'],expected)
                    self.assertEqual(actual['interop_threads'],1)
                    self.assertEqual(actual['thread_variant_parent_sha256'],variant.BASE_ESTIMATOR_SHA)
                    self.assertEqual(json.loads((output/'protocol.json').read_text()),actual)
                    self.assertEqual(variant.sha(output/'estimate_importance_fixed_offsets_threads.py'),variant.sha(variant.HERE/'estimate_importance_fixed_offsets_threads.py'))
                    own=str((variant.HERE/'estimate_importance_fixed_offsets_threads.py').relative_to(variant.ROOT))
                    self.assertEqual(actual['pins'][own],variant.sha(variant.HERE/'estimate_importance_fixed_offsets_threads.py'))
                    normalized=copy.deepcopy(actual)
                    normalized['threads']=1
                    del normalized['interop_threads'],normalized['thread_variant_parent_sha256'],normalized['pins'][own]
                    self.assertEqual(normalized,original)

    def test_invalid_threads_refuse_before_input_read_or_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='thread-estimator-invalid-',dir=variant.RESEARCH/'runs') as temp:
            output=Path(temp)/'output'
            with mock.patch.object(variant,'validate_input') as validate:
                for value in (True,False,0,3,16,2.0,'2',None):
                    with self.assertRaisesRegex(AssertionError,'Threads must'):
                        variant.prepare(Path(temp)/'missing',250,4,output,threads=value)
                validate.assert_not_called()
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
