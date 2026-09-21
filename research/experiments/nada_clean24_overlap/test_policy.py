"""Read-only policy/handshake fixtures; no model imports or real model launch."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('overlap_policy', HERE / 'supervise.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Policy(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((HERE / 'policy.json').read_text())
        self.pid = self.policy['allowed_reference_pid']
        self.reference = {'pid': self.pid, 'ppid': 100, 'pgid': self.pid,
            'start_time': self.policy['allowed_reference_start_time'],
            'command': self.policy['allowed_reference_command']}

    def test_exact_reference_and_owned_child_allowed(self):
        owned = {'pid': 999999, 'ppid': 1, 'pgid': 999999, 'start_time': 'fixture',
                 'command': 'python /repo/research/experiments/nada_clean24_overlap/runner.py --_worker'}
        result = module.validate_processes([self.reference, owned], self.policy, self.pid, owned_pid=owned['pid'])
        self.assertTrue(result['reference_present'])

    def test_wrong_explicit_pid(self):
        with self.assertRaisesRegex(RuntimeError, 'Explicit allowed PID differs'):
            module.validate_processes([self.reference], self.policy, self.pid+1)

    def test_wrong_command_device_or_run(self):
        for before, after in [('mps', 'cpu'), ('resumed500-to625', 'another-run')]:
            reference = copy.deepcopy(self.reference)
            reference['command'] = reference['command'].replace(before, after)
            with self.assertRaisesRegex(RuntimeError, 'command/start identity differs'):
                module.validate_processes([reference], self.policy, self.pid)

    def test_reused_pid_start_time(self):
        self.reference['start_time'] = 'a new process start'
        with self.assertRaisesRegex(RuntimeError, 'command/start identity differs'):
            module.validate_processes([self.reference], self.policy, self.pid)

    def test_additional_model_refused(self):
        extra = {'pid': 999998, 'ppid': 1, 'pgid': 999998, 'start_time': 'fixture',
                 'command': 'python /repo/research/experiments/other/trainer.py --device cpu'}
        with self.assertRaisesRegex(RuntimeError, 'Another known heavy'):
            module.validate_processes([self.reference, extra], self.policy, self.pid)

    def test_validation_and_competing_overlap_coordinators_refused(self):
        for path in ['reference625_validation/validate.py', 'nada_clean24_overlap/supervise.py']:
            with self.subTest(path=path):
                extra = {'pid': 999998, 'ppid': 1, 'pgid': 999998, 'start_time': 'fixture',
                         'command': f'python /repo/research/experiments/{path} --execute'}
                with self.assertRaisesRegex(RuntimeError, 'Another known heavy'):
                    module.validate_processes([self.reference, extra], self.policy, self.pid)
                if 'supervise.py' in path:
                    extra['pid'] = os.getpid()
                    module.validate_processes([self.reference, extra], self.policy, self.pid)

    def test_reference_must_exist_initially_but_may_finish(self):
        with self.assertRaisesRegex(RuntimeError, 'not running'):
            module.validate_processes([], self.policy, self.pid)
        self.assertFalse(module.validate_processes([], self.policy, self.pid, require_reference=False)['reference_present'])

    def test_initial54_refuses_before_output_or_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / 'research/runs/never-created'
            with patch.object(module, 'ROOT', root), patch.object(module, 'load_frozen', return_value=None), \
                 patch.object(module, 'verify_setup', return_value=({}, self.policy)), \
                 patch.object(module, 'processes', return_value=[self.reference]), \
                 patch.object(module, 'pressure', return_value={'free_percent': 54}), \
                 patch.object(module.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'at least55%'):
                    module.supervise(output, self.pid)
                spawn.assert_not_called()
                self.assertFalse(output.exists())

    def test_pressure_parse_failure(self):
        for output in ['not a valid pressure record', 'System-wide memory free percentage: 150%']:
            with patch.object(module.subprocess, 'check_output', return_value=output):
                with self.assertRaisesRegex(RuntimeError, 'could not parse'):
                    module.pressure()

    def test_observation_gap_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            now = time.monotonic()
            with patch.object(module, 'pressure', return_value={'free_percent': 60, 'monotonic': now}):
                with self.assertRaisesRegex(RuntimeError, 'gap exceeded'):
                    module.checked_pressure(Path(temporary)/'observations.jsonl', 35, {'monotonic': now-5.1})

    def test_actual_copy_identity_and_expanded_pins(self):
        base = module.load_frozen()
        pins, policy = module.verify_setup(base)
        for filename, record in policy['original_source_copies'].items():
            relative = str((HERE/filename).relative_to(module.ROOT))
            self.assertEqual(pins['files_sha256'][relative], record['sha256'])
            self.assertEqual((HERE/filename).read_bytes(), (module.ROOT/record['source']).read_bytes())
        self.assertNotIn('torch', sys.modules)

    def test_copied_worker_handshake_checks_its_actual_new_pins(self):
        base = module.load_frozen()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            deadline, nonce = time.time()+30, 'fixture-overlap-nonce'
            launch = {'nonce': nonce, 'deadline_unix': deadline, 'supervisor_pid': os.getppid(),
                'runner_sha256': module.sha(HERE/'runner.py'), 'pins_sha256': module.sha(HERE/'pins.json')}
            base.write_new(output/'launch.json', launch)
            (output/'worker.log').touch()
            base.publish_child(output, {'pid': os.getpid(), 'supervisor_pid': os.getppid(), 'nonce': nonce})
            with patch.object(base, 'memory_guard', return_value={'fixture_free_percent': 60}), \
                 patch.dict(os.environ, base.NUMERIC_ENV):
                checked_launch, pins, _ = base.worker_preflight(output, deadline, nonce)
            self.assertEqual(checked_launch, launch)
            for filename in ['runner.py', 'worker.py', 'inputs.json', 'latents.npz']:
                self.assertIn(str((HERE/filename).relative_to(module.ROOT)), pins['files_sha256'])
            self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
