import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "energy_audit", Path(__file__).parents[1] / "scripts/evaluate_temporal_energy_footprints.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class EnergyFootprintTests(unittest.TestCase):
    def test_constant_energy_has_the_declared_units_and_common_denominator(self):
        own = audit.Curve(7, 10, [2.] * 20)
        other = audit.Curve(7, 10, [6.] * 20)
        for count in (8, 16, 32):
            result = audit.integrate(own, other, 7, 207, count)
            self.assertEqual(result["status"], "supported")
            self.assertEqual(result["own_integral"], 400.)
            self.assertAlmostEqual(result["overlap"], 6 / (8 + 1e-12), places=14)
            self.assertAlmostEqual(result["audibility"], 2 / (8 + 1e-12), places=14)

    def test_window_centers_edges_missing_values_and_fractional_positions(self):
        curve = audit.Curve(7, 10, [5., 6., 7., 8., 9.])
        np.testing.assert_allclose(curve.at([7, 12, 17, 22.5, 56.999]),
                                   [5, 5, 5.5, 6.05, 9], rtol=0, atol=1e-14)
        self.assertTrue(np.isnan(curve.at([6.999, 57])).all())
        missing = audit.Curve(7, 10, [5., np.nan, 7.])
        np.testing.assert_equal(missing.at([16.99, 17, 26.99, 27]), [5., np.nan, np.nan, 7.])

    def test_zero_own_integral_requires_known_silence(self):
        own = audit.Curve(0, 1, np.zeros(400))
        unknown = audit.Curve(0, 1, [np.nan] * 400)
        silent = audit.integrate(own, unknown, 0, 400, 16)
        self.assertEqual(silent["status"], "known_silent")
        self.assertEqual(silent["overlap"], 0.)
        self.assertIsNone(silent["audibility"])
        impulse = np.zeros(400)
        impulse[1] = 1.
        short = audit.Curve(0, 1, impulse)
        other = audit.Curve(0, 1, np.ones(400))
        missed = audit.integrate(short, other, 0, 400, 16)
        self.assertEqual(missed["status"], "missed_own_energy")
        self.assertIsNone(missed["overlap"])
        dense = audit.integrate(short, other, 0, 400, 16, dense=True)
        self.assertEqual(dense["status"], "supported")
        self.assertAlmostEqual(dense["own_integral"], 1., places=14)

    def test_a_gap_between_sampled_points_still_invalidates_support(self):
        own = audit.Curve(0, 1, np.ones(400))
        values = np.ones(400)
        values[1] = np.nan
        other = audit.Curve(0, 1, values)
        self.assertTrue(np.isfinite(other.at((np.arange(16) + 0.5) * 25)).all())
        result = audit.integrate(own, other, 0, 400, 16)
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["overlap"])
        self.assertIsNone(result["audibility"])

    def test_piecewise_reference_matches_an_analytic_log_integral(self):
        own = audit.Curve(0, 1, np.ones(100))
        external = audit.Curve(0, 1, np.arange(100) + 0.5)
        expected = math.log((100.5 + 1e-12) / (1.5 + 1e-12)) / 99
        for order in (8, 16):
            result = audit.integrate(own, external, 0.5, 99.5, order, dense=True)
            self.assertAlmostEqual(result["audibility"], expected, places=14)
            self.assertAlmostEqual(result["overlap"], 1 - (1 + 1e-12) * expected, places=14)

    def test_negative_energy_is_unknown_and_empty_support_has_no_ratio(self):
        own = audit.Curve(0, 1, [1.] * 10)
        invalid = audit.Curve(0, 1, [-1.] * 10)
        result = audit.integrate(own, invalid, 0, 10, 16)
        self.assertEqual(result["status"], "unsupported")
        empty = audit.integrate(own, invalid, 10, 10, 16)
        self.assertEqual(empty["status"], "empty_intersection")
        self.assertIsNone(empty["overlap"])

    def test_changed_corpus_is_rejected_before_any_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verification.json").write_text(json.dumps({"files_sha256": {"pcm": "wrong"}}))
            (root / "pcm").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "verified input changed"):
                audit.evaluate(root, root / "output")
            self.assertFalse((root / "output").exists())

    def test_replacing_the_verification_cannot_bypass_the_frozen_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verification.json").write_text(json.dumps({"files_sha256": {}}))
            with self.assertRaisesRegex(ValueError, "frozen I10 registry"):
                audit.evaluate(root, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
