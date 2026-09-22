"""Tiny actual-engine continuation checks; no native weights or FI artifacts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import continue_main_adaptation_fixed_offsets250_100 as continuation
import main_adaptation_fixed_offsets250_10 as parent
import test_main_adaptation_fixed_offsets10 as fixture


def make_engine():
    # Same tiny random architecture fixture; bind it to the actual reused engine.
    with mock.patch.object(fixture,'AdaptationEngine',continuation.AdaptationEngine):
        return fixture.make_engine()[0]


def envelope(engine):
    return {'engine':engine.state(),'protocol_sha256':'declared-tiny-protocol','runtime_sha256':'declared-tiny-runtime',
            'modulation_inventory_sha256':'declared-tiny-inventory'}


def restore(engine,payload,step):
    continuation.restore_checkpoint(engine,payload,expected_digest=continuation.state_digest(payload),
        protocol_sha='declared-tiny-protocol',runtime_sha='declared-tiny-runtime',inventory_sha='declared-tiny-inventory',expected_iterations=step)


class ContinueMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_actual_frozen_engine_reused_and_source_parent_pinned(self):
        self.assertIs(continuation.AdaptationEngine,parent.AdaptationEngine)
        self.assertEqual(continuation.sha(continuation.HERE/'main_adaptation_fixed_offsets250_10.py'),continuation.PARENT_RUNNER_SHA)
        self.assertEqual(continuation.expected_steps(10),{'g':13,'d':11})
        self.assertEqual(continuation.expected_steps(50),{'g':63,'d':54})
        self.assertEqual(continuation.expected_steps(100),{'g':125,'d':107})

    def test_uninterrupted_versus_serialized10_to100_across_regularizers_and_checkpoint50(self):
        engine=make_engine();real=[torch.randn(3,32,32) for _ in range(3)]
        for _ in range(10):engine.iteration(real)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint10=Path(temp)/'tiny10.pt';torch.save(envelope(engine),checkpoint10)
            records=[engine.iteration(real) for _ in range(90)]
            expected=continuation.state_digest(engine.state())
            resumed=make_engine();source=torch.load(checkpoint10,weights_only=True);restore(resumed,source,10)
            resumed_records=[]
            while resumed.iterations<100:
                resumed_records.append(resumed.iteration(real))
                if resumed.iterations==50:
                    path=Path(temp)/'tiny50.pt';torch.save(envelope(resumed),path)
                    payload=torch.load(path,weights_only=True)
                    # Fresh controller construction is seeded from original references; restore every state.
                    resumed=make_engine();restore(resumed,payload,50)
            self.assertEqual(continuation.state_digest(resumed.state()),expected)
            self.assertEqual(records,resumed_records)
            self.assertEqual(records[0]['iteration'],10)
            self.assertNotIn('r1_loss',records[0]);self.assertNotIn('path_loss',records[0])
            both=[r['iteration'] for r in records if 'r1_loss' in r and 'path_loss' in r]
            self.assertEqual(both,[16,32,48,64,80,96])
            self.assertEqual(resumed.counts,{'g':125,'d':107})
            # Resetting path target while retaining all other source10 state is a real divergence.
            wrong=make_engine();restore(wrong,torch.load(checkpoint10,weights_only=True),10)
            wrong.path_mean=torch.zeros(())
            for _ in range(7):wrong.iteration(real)
            correct=make_engine();restore(correct,torch.load(checkpoint10,weights_only=True),10)
            for _ in range(7):correct.iteration(real)
            self.assertNotEqual(continuation.state_digest(wrong.state()),continuation.state_digest(correct.state()))

    def test_changed_envelope_or_mask_rejected_before_state_mutation(self):
        engine=make_engine();real=[torch.randn(3,32,32)]
        for _ in range(10):engine.iteration(real)
        original=copy.deepcopy(envelope(engine));digest=continuation.state_digest(original)
        for field in ('protocol_sha256','runtime_sha256','modulation_inventory_sha256'):
            bad=copy.deepcopy(original);bad[field]='changed'
            before=continuation.state_digest(engine.state())
            with self.assertRaises(AssertionError):
                continuation.restore_checkpoint(engine,bad,expected_digest=digest,protocol_sha='declared-tiny-protocol',
                    runtime_sha='declared-tiny-runtime',inventory_sha='declared-tiny-inventory',expected_iterations=10)
            self.assertEqual(continuation.state_digest(engine.state()),before)
        bad=copy.deepcopy(original);path=next(iter(bad['engine']['selections']['G']['masks']))
        bad['engine']['selections']['G']['masks'][path][0]=not bad['engine']['selections']['G']['masks'][path][0]
        before=continuation.state_digest(engine.state())
        with self.assertRaisesRegex(AssertionError,'Mask/policy'):restore(engine,bad,10)
        self.assertEqual(continuation.state_digest(engine.state()),before)
        with self.assertRaises(FileNotFoundError):continuation.validate_parent(continuation.RESEARCH/'runs/nonexistent-future-main10')


if __name__=='__main__':unittest.main(verbosity=2)
