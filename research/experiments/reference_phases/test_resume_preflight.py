"""Small-file provenance fixtures only; no torch import or model reads."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('resume_preflight', Path(__file__).with_name('resume_preflight.py'))
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ResumePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'research/runs/fixture'
        self.run.mkdir(parents=True)
        self.runtime = {'packages': {'torch': '2.14.0'}, 'device_fixture': 'cpu'}
        contents = {guard.TRAINER: '# trainer\n', guard.ADAPTER: '# adapter\n',
            guard.VENDOR + '/legacy.py': '# legacy\n',
            guard.VENDOR + '/training/networks.py': '# networks\n',
            guard.VENDOR + '/torch_utils/ops/conv2d_gradfix.py': '# conv\n',
            guard.VENDOR + '/torch_utils/ops/grid_sample_gradfix.py': '# grid\n',
            guard.VENDOR + '/dnnlib/__init__.py': '# dnnlib\n'}
        for name, content in contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (self.run / 'train-source.py').write_text(contents[guard.TRAINER])
        (self.run / 'adapters-source.py').write_text(contents[guard.ADAPTER])
        config = {'trainer_sha256': guard.sha256(self.root / guard.TRAINER), 'torch_version': '2.14.0',
            'recipe': {'source_hashes': {'training/networks.py': guard.sha256(
                self.root / guard.VENDOR / 'training/networks.py')}}}
        (self.run / 'config.json').write_text(json.dumps(config))
        self.manifest = self.root / 'baseline.json'
        guard.record(self.root, self.run, self.manifest, self.runtime)

    def verify(self, runtime=None):
        return guard.check(self.root, self.run, self.manifest, runtime or self.runtime)

    def test_clean_pass(self):
        self.assertEqual(self.verify()['status'], 'pass')

    def test_trainer_mismatch_fails(self):
        (self.root / guard.TRAINER).write_text('# changed math\n')
        with self.assertRaisesRegex(ValueError, 'Trainer differs'):
            self.verify()

    def test_torch_version_mismatch_fails(self):
        runtime = copy.deepcopy(self.runtime)
        runtime['packages']['torch'] = '2.15.0'
        with self.assertRaisesRegex(ValueError, 'PyTorch version differs'):
            self.verify(runtime)

    def test_unrecorded_vendor_operation_change_fails(self):
        (self.root / guard.VENDOR / 'torch_utils/ops/conv2d_gradfix.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'changed file'):
            self.verify()

    def test_missing_archive_fails(self):
        (self.run / 'adapters-source.py').unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_added_source_file_fails(self):
        (self.root / guard.VENDOR / 'torch_utils/ops/new_op.py').write_text('# new\n')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            self.verify()

    def test_baseline_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            guard.record(self.root, self.run, self.manifest, self.runtime)


if __name__ == '__main__':
    unittest.main()
