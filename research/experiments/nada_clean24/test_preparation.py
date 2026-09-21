"""Lightweight pin/refusal and latent tests; no torch or model imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('nada_supervisor', HERE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Pins(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nada-pins-')
        self.root = Path(self.temporary.name).resolve()
        self.here = self.root / 'new-code'
        self.here.mkdir()
        self.source = self.here / 'worker.py'
        self.source.write_text('fixture source, never executed\n')
        relative = str(self.source.relative_to(self.root))
        self.pins = {'python_version': platform.python_version(), 'machine': platform.machine(),
            'selection_versions': {}, 'files_sha256': {relative: hashlib.sha256(self.source.read_bytes()).hexdigest()},
            'inventories': {'new-code': [relative]}}
        self.save()
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.here_patch = patch.object(runner, 'HERE', self.here)
        self.root_patch.start(); self.here_patch.start()

    def save(self):
        (self.here / 'pins.json').write_text(json.dumps(self.pins))

    def tearDown(self):
        self.here_patch.stop(); self.root_patch.stop(); self.temporary.cleanup()

    def test_clean(self):
        self.assertEqual(runner.verify_pins(), self.pins)

    def test_source_changed(self):
        self.source.write_text('changed\n')
        with self.assertRaisesRegex(RuntimeError, 'Pinned input/source changed'):
            runner.verify_pins()

    def test_new_source_inventory(self):
        (self.here / 'unreviewed.py').write_text('new dependency\n')
        with self.assertRaisesRegex(RuntimeError, 'source inventory changed'):
            runner.verify_pins()

    def test_version_changed(self):
        self.pins['selection_versions'] = {'torch': 'fixture-old-version'}
        self.save()
        with patch.object(runner.importlib.metadata, 'version', return_value='fixture-new-version'):
            with self.assertRaisesRegex(RuntimeError, 'Package version changed'):
                runner.verify_pins()

    def test_memory_34_refused_35_passes(self):
        with patch.object(runner.subprocess, 'check_output', return_value='System-wide memory free percentage: 34%'):
            with self.assertRaisesRegex(RuntimeError, '35%'):
                runner.memory_guard()
        with patch.object(runner.subprocess, 'check_output', return_value='System-wide memory free percentage: 35%'):
            self.assertEqual(runner.memory_guard()['free_percent'], 35)

    def test_other_trainer_refused(self):
        text = '999999 /usr/bin/python /repo/research/experiments/reference_phases/trainer.py --device mps\n'
        with patch.object(runner.subprocess, 'check_output', return_value=text):
            with self.assertRaisesRegex(RuntimeError, 'another known research'):
                runner.concurrency_guard()


class PreparedLatents(unittest.TestCase):
    def test_actual_saved_arrays_and_sampling(self):
        import numpy as np
        manifest = json.loads((HERE / 'inputs.json').read_text())
        with np.load(HERE / 'latents.npz', allow_pickle=False) as data:
            self.assertEqual({k: list(data[k].shape) for k in data.files}, manifest['array_shapes'])
            self.assertEqual(len({r.tobytes() for k in ['source_z', 'train_z', 'eval_z'] for r in data[k]}), 50)
            for key, count, seed_key in [('train_z', 10, 'train_pcg64_seed'), ('eval_z', 8, 'evaluation_pcg64_seed')]:
                expected = np.random.Generator(np.random.PCG64(manifest[seed_key])).standard_normal((count,512)).astype(np.float32)
                self.assertTrue(np.array_equal(data[key], expected))
            for index, item in enumerate(manifest['source32']):
                with np.load(runner.ROOT / item['latent'], allow_pickle=False) as original:
                    self.assertTrue(np.array_equal(data['source_z'][index], original['z'].reshape(512)))
                    self.assertTrue(np.array_equal(data['source_w'][index], original['w'].reshape(18,512)))
        self.assertNotIn('torch', sys.modules)


class WorkerHandshake(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nada-handshake-')
        self.output = Path(self.temporary.name)
        self.deadline = time.time() + 30
        self.nonce = 'fixture-nonce'
        self.launch = {'nonce': self.nonce, 'deadline_unix': self.deadline,
            'runner_sha256': runner.sha(HERE / 'runner.py'), 'supervisor_pid': os.getppid(),
            'pins_sha256': runner.sha(HERE / 'pins.json')}
        self.child = {'pid': os.getpid(), 'supervisor_pid': os.getppid(), 'nonce': self.nonce}
        runner.write_new(self.output / 'launch.json', self.launch)
        (self.output / 'worker.log').touch()

    def tearDown(self):
        self.temporary.cleanup()

    def preflight(self):
        with patch.object(runner, 'verify_pins', return_value={'fixture': True}), \
             patch.object(runner, 'memory_guard', return_value={'fixture_free_percent': 99}), \
             patch.dict(os.environ, runner.NUMERIC_ENV):
            result = runner.worker_preflight(self.output, self.deadline, self.nonce)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('nada_worker', sys.modules)
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'launch.json', 'worker.log', 'child.json'})
        return result

    def test_child_ready_before_worker_preflight(self):
        runner.publish_child(self.output, self.child)
        self.assertEqual(self.preflight(), (self.launch, {'fixture': True}, {'fixture_free_percent': 99}))

    def test_worker_waits_through_partial_temp_write(self):
        partial_written = threading.Event()
        real_write = runner.write_new
        def slow_write(path, value):
            if path.name == 'child.json.tmp':
                with path.open('x') as stream:
                    stream.write('{"pid":'); stream.flush()
                    partial_written.set()
                    time.sleep(.05)
                    stream.seek(0); json.dump(value, stream); stream.truncate()
            else:
                real_write(path, value)
        with patch.object(runner, 'write_new', side_effect=slow_write):
            thread = threading.Thread(target=runner.publish_child, args=(self.output, self.child))
            thread.start()
            try:
                self.assertTrue(partial_written.wait(timeout=1))
                self.assertFalse((self.output / 'child.json').exists())
                self.assertEqual(self.preflight()[0], self.launch)
            finally:
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())

    def test_wrong_child_pid_refused(self):
        self.child['pid'] += 1000000
        runner.publish_child(self.output, self.child)
        with self.assertRaisesRegex(RuntimeError, 'PID/parent/nonce mismatch'):
            self.preflight()

    def test_wrong_child_nonce_refused(self):
        self.child['nonce'] = 'other-run'
        runner.publish_child(self.output, self.child)
        with self.assertRaisesRegex(RuntimeError, 'PID/parent/nonce mismatch'):
            self.preflight()

    def test_wrong_parent_refused(self):
        self.child['supervisor_pid'] += 1000000
        runner.publish_child(self.output, self.child)
        with self.assertRaisesRegex(RuntimeError, 'PID/parent/nonce mismatch'):
            self.preflight()

    def test_missing_child_refused_without_model_import(self):
        with patch.object(runner, 'HANDSHAKE_SECONDS', .02):
            with self.assertRaisesRegex(RuntimeError, 'readiness timed out'):
                self.preflight()

    def test_existing_results_still_refused(self):
        runner.publish_child(self.output, self.child)
        (self.output / 'old-checkpoint.pt').touch()
        with self.assertRaisesRegex(RuntimeError, 'existing research results'):
            self.preflight()


if __name__ == '__main__':
    unittest.main(verbosity=2)
