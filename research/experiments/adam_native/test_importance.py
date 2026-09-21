"""Tiny analytical tests; no model/checkpoint loads or training."""
import io
import unittest

import torch

from importance import EmpiricalSquaredGradients, source_uvb_score


def assert_state_equal(test, left, right):
    test.assertEqual(set(left), set(right))
    for key in left:
        if isinstance(left[key], dict):
            assert_state_equal(test, left[key], right[key])
        elif isinstance(left[key], torch.Tensor):
            test.assertTrue(torch.equal(left[key], right[key]), key)
        else:
            test.assertEqual(left[key], right[key])


class ImportanceTests(unittest.TestCase):
    def test_cancellation_uses_actual_per_example_autograd(self):
        weight = torch.tensor([2., -3.], requires_grad=True)
        acc = EmpiricalSquaredGradients({'w': weight})
        losses = []
        for x in (torch.tensor([1., 2.]), torch.tensor([-1., -2.])):
            loss = (weight * x).sum()
            losses.append(loss)
            grad, = torch.autograd.grad(loss, weight, retain_graph=True)
            acc.add_example({'w': grad})
        batch_mean_grad, = torch.autograd.grad(torch.stack(losses).mean(), weight)
        self.assertTrue(torch.equal(batch_mean_grad.square(), torch.zeros(2)))
        self.assertTrue(torch.equal(acc.mean_gradients()['w'], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(acc.mean_squared_gradients()['w'], torch.tensor([1., 4.], dtype=torch.float64)))
        self.assertEqual(acc.sample_count, 2)

    def test_discriminator_pair_cross_term_and_sample_normalization(self):
        w = torch.tensor(0., requires_grad=True)
        # Analytic stand-in scalar objective, real gradient 2, fake gradient -1.
        combined, = torch.autograd.grad(2 * w - w, w)
        acc = EmpiricalSquaredGradients({'w': w})
        acc.add_example({'w': combined})
        self.assertEqual(acc.mean_squared_gradients()['w'].item(), 1)
        self.assertNotEqual(acc.mean_squared_gradients()['w'].item(), 2**2 + (-1)**2)
        acc.add_example({'w': torch.tensor(3.)})
        self.assertEqual(acc.mean_squared_gradients()['w'].item(), 5)

    def test_failed_add_atomic_even_late_failure(self):
        acc = EmpiricalSquaredGradients({'a': torch.zeros(2), 'z': torch.zeros(2)}, accumulator_dtype=torch.float32)
        acc.add_example({'a': torch.ones(2), 'z': torch.ones(2)})
        before = acc.state_dict()
        cases = [
            {'a': torch.ones(2), 'z': torch.tensor([float('nan'), 0.])},
            {'a': torch.ones(2), 'z': torch.tensor([float('inf'), 0.])},
            {'a': torch.ones(2), 'z': torch.ones(3)},
            {'a': torch.ones(2), 'z': torch.full((2,), 1e30)},
            {'a': torch.ones(2), 'z': None},
            {'a': torch.ones(2)},
            {'a': torch.ones(2), 'z': torch.ones(2), 'extra': None},
        ]
        for case in cases:
            with self.subTest(case=str(case)):
                with self.assertRaises(ValueError):
                    acc.add_example(case)
                assert_state_equal(self, before, acc.state_dict())

    def test_missing_policy_total_denominator_and_counts(self):
        acc = EmpiricalSquaredGradients({'a': torch.zeros(1), 'z': torch.zeros(1)}, missing='zero')
        with self.assertRaises(ValueError):
            acc.mean_squared_gradients()
        acc.add_example({'a': torch.tensor([2.]), 'z': None})
        acc.add_example({'a': None, 'z': None})
        self.assertEqual(acc.sample_count, 2)
        self.assertEqual(acc.observed_counts, {'a': 1, 'z': 0})
        self.assertEqual(acc.mean_squared_gradients()['a'].item(), 2)
        self.assertEqual(acc.mean_squared_gradients()['z'].item(), 0)

    def test_state_serialization_continuation_and_alias_isolation(self):
        acc = EmpiricalSquaredGradients([('z', torch.zeros(2)), ('a', torch.zeros(()))])
        grads = {'a': torch.tensor(2.), 'z': torch.tensor([3., -4.])}
        acc.add_example(grads)
        state = acc.state_dict()
        self.assertEqual(list(state['gradient_sums']), ['a', 'z'])
        buffer = io.BytesIO()
        torch.save(state, buffer); buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        resumed = EmpiricalSquaredGradients.from_state_dict(loaded)
        assert_state_equal(self, acc.state_dict(), resumed.state_dict())
        resumed.add_example(grads); acc.add_example(grads)
        assert_state_equal(self, acc.state_dict(), resumed.state_dict())
        loaded['gradient_sums']['a'].fill_(999)
        state['squared_gradient_sums']['a'].fill_(999)
        self.assertEqual(resumed.mean_squared_gradients()['a'].item(), 4)
        self.assertEqual(acc.mean_squared_gradients()['a'].item(), 4)

    def test_corrupt_state_counts_and_values_rejected(self):
        acc = EmpiricalSquaredGradients({'w': torch.zeros(1)})
        acc.add_example({'w': torch.ones(1)})
        def invalid(change):
            state = acc.state_dict(); change(state)
            with self.assertRaises(ValueError):
                EmpiricalSquaredGradients.from_state_dict(state)
        invalid(lambda s: s.update(sample_count=-1))
        invalid(lambda s: s.update(sample_count=True))
        invalid(lambda s: s['observed_counts'].update(w=2))
        invalid(lambda s: s['observed_counts'].update(w=0))
        invalid(lambda s: s['squared_gradient_sums']['w'].fill_(-1))
        invalid(lambda s: s['squared_gradient_sums']['w'].fill_(float('nan')))
        invalid(lambda s: s['squared_gradient_sums'].update(w=torch.ones(2, dtype=torch.float64)))
        invalid(lambda s: s.update(version=2))
        state = EmpiricalSquaredGradients({'w': torch.zeros(1)}, missing='zero').state_dict()
        state['gradient_sums']['w'].fill_(1)
        with self.assertRaises(ValueError):
            EmpiricalSquaredGradients.from_state_dict(state)

    def test_exact_source_reductions_no_broadcasting_or_thresholds(self):
        u = torch.tensor([2., 6.]); v = torch.tensor([3., 7.]); b = torch.tensor([1., 5.])
        self.assertTrue(torch.equal(source_uvb_score(u, v), torch.tensor([7., 11.])))
        self.assertTrue(torch.equal(source_uvb_score(u, v, b), torch.tensor([4., 8.])))
        for bad in (torch.ones(1), torch.tensor([-1., 1.]), torch.tensor([float('inf'), 1.])):
            with self.assertRaises(ValueError):
                source_uvb_score(u, v, bad)
        with self.assertRaises(ValueError):
            source_uvb_score(u, v[:, None])


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
