"""Small metadata-only checks; never imports Torch or loads model weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('embed_metadata', HERE/'embed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(2):
            folder = self.root/f'{index:03}'
            folder.mkdir()
            row = dict(index=index, path=f'original-{index}.png', sha256='a'*64,
                       error='No face detected' if index else None, outputs={})
            if index == 0:
                webp = folder/'restored-1024.webp'
                webp.write_bytes(b'fixture: metadata only, not decoded')
                row['outputs']['restored-1024.webp'] = module.sha(webp)
            (folder/'record.json').write_text(json.dumps(row))
            self.rows.append(row)
        self.result = dict(complete=True, cohort='fixture', attempted=2,
                           successful_restorations=1, errors=1, records=self.rows)
        (self.root/'result.json').write_text(json.dumps(self.result))
        (self.root/'supervisor-result.json').write_text(json.dumps(dict(complete=True, worker_exit_code=0, error=None)))
        (self.root/'inputs.json').write_text('{}')
    def tearDown(self):
        self.temp.cleanup()
    def test_failed_row_preserved_without_substitution(self):
        data = module.inventory([self.root])
        self.assertEqual(len(data['rows']), 2)
        self.assertEqual(data['rows'][1]['feature_row'], 1)
        self.assertEqual(data['rows'][1]['restoration_error'], 'No face detected')
        self.assertIsNone(data['rows'][1]['webp_path'])
    def test_changed_final_webp_rejected(self):
        (self.root/'000/restored-1024.webp').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Final WebP changed'):
            module.inventory([self.root])
    def test_incomplete_cohort_rejected(self):
        self.result['complete'] = False
        (self.root/'result.json').write_text(json.dumps(self.result))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            module.inventory([self.root])
    def test_duplicate_inputs_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Duplicate input'):
            module.inventory([self.root, self.root])

if __name__ == '__main__':
    unittest.main()
