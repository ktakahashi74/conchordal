"""Boundary and analytic checks for the standalone I11-2 CDF reference."""

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from i11_stage2_cdf_reference import (  # noqa: E402
    POINTS,
    active_group_window,
    arrival_window_probability,
    cdf_at,
    hazard_cdf,
    hazard_cdf_for_forecast,
    periodic_cdf,
)


class I11Stage2CdfReferenceTest(unittest.TestCase):
    def test_constant_hazard_matches_closed_form(self):
        c0 = math.log(math.expm1(2.0))
        values = hazard_cdf(0.3, 0.125, c0=c0, c10=0.0)
        self.assertEqual(len(values), POINTS)
        for i, value in enumerate(values):
            self.assertAlmostEqual(value, -math.expm1(-2.0 * i * 0.02), places=12)

    def test_trial_hazard_is_monotone_and_has_phase_effect(self):
        early = hazard_cdf(0.0, 0.5)
        late = hazard_cdf(0.25, 0.5)
        self.assertEqual(early[0], 0.0)
        self.assertTrue(all(0.0 <= a <= b <= 1.0 for a, b in zip(early, early[1:])))
        self.assertGreater(early[5], late[5])

    def test_grid_interpolation_and_horizon(self):
        values = tuple(i / 200 for i in range(POINTS))
        self.assertEqual(cdf_at(values, 0), 0.0)
        self.assertAlmostEqual(cdf_at(values, 480), 0.0025)
        self.assertEqual(cdf_at(values, 192_000), 1.0)
        self.assertIsNone(cdf_at(values, 192_001))
        self.assertIsNone(arrival_window_probability(values, 191_999, 2))
        self.assertAlmostEqual(arrival_window_probability(values, 960, 960), 0.01)

    def test_unknown_and_periodic_step(self):
        self.assertIsNone(hazard_cdf(None, 0.5))
        self.assertIsNone(hazard_cdf(0.0, None))
        self.assertIsNone(hazard_cdf_for_forecast((0.1, 0.2), False, 0.5))
        self.assertIsNone(hazard_cdf_for_forecast((0.1, 0.1), True, 0.5))
        self.assertIsNone(hazard_cdf_for_forecast(None, False, 0.5))
        self.assertIsNotNone(hazard_cdf_for_forecast((0.1, 0.1), False, 0.5))
        self.assertIsNone(periodic_cdf(0.0, None))
        values = periodic_cdf(0.3, 0.5)
        self.assertEqual(values[9], 0.0)
        self.assertEqual(values[10], 1.0)
        self.assertEqual(arrival_window_probability(None, 0, 0), None)

    def test_group_deadlines_and_known_zero(self):
        values = tuple(0.0 for _ in range(POINTS))
        self.assertEqual(active_group_window(values, 100, 192_100, 100, 100, 0), 0.0)
        self.assertIsNone(active_group_window(values, 100, 192_100, 99, 100, 0))
        self.assertIsNone(active_group_window(values, 100, 192_100, 192_100, 100, 0))
        self.assertIsNone(active_group_window(values, 100, 192_100, 100, 192_100, 1))


if __name__ == "__main__":
    unittest.main()
