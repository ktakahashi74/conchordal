"""Check the independent target oracle at known-zero and support boundaries."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "action_features", Path(__file__).resolve().parents[1] / "scripts/collect_temporal_action_features.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ActionFeatureReferenceTests(unittest.TestCase):
    def test_silence_does_not_create_shape_and_warmup_does_not_erase_energy(self):
        values = oracle.raw_reference(np.zeros(6), np.zeros((6, 3)), np.array([6., 7., 8.]),
                                      np.array([False, False, False, True, True, True]))
        self.assertTrue(np.isnan(values[:, [0, 1, 7, 8, 9]]).all())
        np.testing.assert_array_equal(values[:, 2], np.full(6, np.log2(1e-6)))
        np.testing.assert_array_equal(values[:, 6], np.zeros(6))
        self.assertTrue(np.isnan(values[:4, 5]).all())
        np.testing.assert_array_equal(values[4:, 5], [0., 0.])
        self.assertTrue(np.isnan(values[0, 3:5]).all())
        np.testing.assert_array_equal(values[1:, 3:5], np.zeros((5, 2)))

    def test_energy_derivatives_and_band_boundaries(self):
        energy = np.array([1., 4., 1.])
        scan = np.array([[.25, .5, .25], [1., 2., 1.], [.25, .5, .25]])
        values = oracle.raw_reference(energy, scan, np.array([6.5, 7., 7.5]), np.ones(3, dtype=bool))
        np.testing.assert_allclose(values[:, 0], [7., 7., 7.])
        np.testing.assert_allclose(values[:, 1], np.sqrt(.125))
        np.testing.assert_array_equal(values[:, 7:], [[0., 1., 0.]]*3)
        np.testing.assert_array_equal(values[1:, 3:6], [[1., 0., 1.], [0., 1., 0.]])
        missing = oracle.raw_reference(energy, np.zeros((3, 3)), np.array([6.5, 7., 7.5]), np.ones(3, dtype=bool))
        self.assertTrue(np.isnan(missing[:, 5]).all())


if __name__ == "__main__":
    unittest.main()
