"""Tiny actual-engine100→250 equality; no native weights or invented results."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import continue_main_adaptation_fixed_offsets250_250 as runner
import continue_main_adaptation_fixed_offsets250_100 as previous
from test_continue_main_adaptation_fixed_offsets250_100 import make_engine,envelope


def restore(engine,payload,step):
    runner.restore_checkpoint(engine,payload,expected_digest=runner.state_digest(payload),
        protocol_sha='declared-tiny-protocol',runtime_sha='declared-tiny-runtime',inventory_sha='declared-tiny-inventory',expected_iterations=step)


class Continue250Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_same_engine_restore_helper_and_counts(self):
        self.assertIs(runner.AdaptationEngine,previous.AdaptationEngine)
        def helper(module):
            tree=ast.parse(Path(module.__file__).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore_checkpoint'),include_attributes=False)
        self.assertEqual(helper(runner),helper(previous))
        self.assertEqual(runner.sha(Path(previous.__file__)),runner.PARENT_RUNNER_SHA)
        self.assertEqual(runner.expected_steps(150),{'g':188,'d':160})
        self.assertEqual(runner.expected_steps(250),{'g':313,'d':266})

    def test_uninterrupted_equals_restore_at100_and150(self):
        engine=make_engine();real=[torch.randn(3,32,32) for _ in range(3)]
        for _ in range(100):engine.iteration(real)
        with tempfile.TemporaryDirectory() as temporary:
            initial=Path(temporary)/'declared-tiny100.pt';torch.save(envelope(engine),initial)
            expected_records=[engine.iteration(real) for _ in range(150)]
            expected=runner.state_digest(engine.state())
            resumed=make_engine();restore(resumed,torch.load(initial,weights_only=True),100)
            records=[]
            while resumed.iterations<250:
                records.append(resumed.iteration(real))
                if resumed.iterations==150:
                    path=Path(temporary)/'declared-tiny150.pt';torch.save(envelope(resumed),path)
                    resumed=make_engine();restore(resumed,torch.load(path,weights_only=True),150)
            self.assertEqual(runner.state_digest(resumed.state()),expected)
            self.assertEqual(records,expected_records)
            self.assertEqual(records[0]['iteration'],100)
            self.assertIn('path_loss',records[0]);self.assertNotIn('r1_loss',records[0])
            self.assertEqual([r['iteration'] for r in records if 'r1_loss' in r],list(range(112,250,16)))
            self.assertEqual([r['iteration'] for r in records if 'path_loss' in r],list(range(100,250,4)))
            self.assertEqual(resumed.counts,{'g':313,'d':266})

    def test_mismatched_envelope_rejected_before_mutation(self):
        engine=make_engine();payload=copy.deepcopy(envelope(engine));before=runner.state_digest(engine.state())
        for field in ('protocol_sha256','runtime_sha256','modulation_inventory_sha256'):
            changed=copy.deepcopy(payload);changed[field]='changed'
            with self.assertRaises(AssertionError):restore(engine,changed,0)
            self.assertEqual(runner.state_digest(engine.state()),before)
        changed=copy.deepcopy(payload);changed['engine']['iterations']=100
        with self.assertRaises(AssertionError):restore(engine,changed,100)
        self.assertEqual(runner.state_digest(engine.state()),before)
        with self.assertRaisesRegex(AssertionError,'Unexpected source'):runner.validate_parent(runner.RESEARCH/'runs/any-other-main100')


if __name__=='__main__':unittest.main(verbosity=2)
