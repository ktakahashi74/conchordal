import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'profile_transfer', Path(__file__).resolve().parents[1]
    / 'scripts/evaluate_temporal_profile_transfer.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TransferInputTests(unittest.TestCase):
    def test_whole_cell_ties_unknown_cells_and_horizon(self):
        offsets = [0, 10, 20]
        bindings = [None, 7, 9]
        self.assertEqual(AUDIT.nearest_cell(offsets, 5), 0)
        self.assertIsNone(bindings[AUDIT.nearest_cell(offsets, 5)])
        self.assertEqual(AUDIT.nearest_cell(offsets, 6), 1)
        self.assertEqual(AUDIT.nearest_cell(offsets, 20), 2)
        self.assertIsNone(AUDIT.nearest_cell(offsets, -1))
        self.assertIsNone(AUDIT.nearest_cell(offsets, 21))

    def test_rejects_changed_or_unregistered_inputs_before_reading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.json'
            model.write_text('original')
            registered = root / 'registration.json'
            registered.write_text(json.dumps({'inputs': {str(model): AUDIT.digest(model)}}))
            model.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed registered input'):
                AUDIT.evaluate(registered, model, root / 'profile.bin', root, root / 'result.json')
            model.write_text('original')
            with self.assertRaisesRegex(ValueError, 'unregistered input'):
                AUDIT.evaluate(registered, root / 'other-model.json', root / 'profile.bin', root, root / 'result.json')
            self.assertFalse((root / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
