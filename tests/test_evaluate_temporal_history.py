import math
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_temporal_history import TemporalHistory, run


class TemporalHistoryTests(unittest.TestCase):
    def test_step_response_matches_integrated_gamma_kernel(self):
        ages = np.array([.1, .4, 1.6])
        history = TemporalHistory(ages, 3, 2)
        history.advance(1.2, [1., .25])
        x = 3 * 1.2 / ages
        integral = 1 - np.exp(-x) * (1 + x + x*x/2 + x*x*x/6)
        result = history.snapshot()
        np.testing.assert_allclose(result["known_envelope_by_age"], integral[:, None] * [1., .25], atol=1e-15)
        np.testing.assert_allclose(result["known_coverage_by_age"], integral, atol=1e-15)

    def test_rectangular_episode_matches_independent_quadrature(self):
        ages = np.array([.05, .2, .8, 3.2])
        order = 8
        history = TemporalHistory(ages, order, 1)
        history.advance(.1, [1.])
        history.advance(.4, [0.])
        lag = np.linspace(.4, .5, 20_001)
        expected = []
        for age in ages:
            rate = order / age
            kernel = np.exp((order + 1) * np.log(rate) + order * np.log(lag)
                            - rate * lag - math.lgamma(order + 1))
            expected.append(np.trapezoid(kernel, lag))
        np.testing.assert_allclose(np.array(history.snapshot()["known_envelope_by_age"])[:, 0],
                                   expected, rtol=1e-6, atol=1e-12)

    def test_constant_input_partition_does_not_change_state(self):
        ages = [.001, .1, 10., 1000.]
        one, many = [TemporalHistory(ages, 12, 2) for _ in range(2)]
        one.advance(1.07, [.1, 1.])
        for interval in [.2, .31, .0001, .5599]:
            many.advance(interval, [.1, 1.])
        np.testing.assert_allclose(one.state, many.state, rtol=2e-12, atol=1e-15)

    def test_joint_time_scaling_preserves_the_history(self):
        baseline = None
        for scale in [.01, 1., 100.]:
            history = TemporalHistory(np.array([.05, .2, .8]) * scale, 4, 2)
            for interval, value in [(.013, [1., 0.]), (.097, [0., 1.]), (.35, [0., 0.])]:
                history.advance(interval * scale, value)
            if baseline is None:
                baseline = history.state.copy()
            else:
                np.testing.assert_allclose(history.state, baseline, rtol=2e-13, atol=1e-15)

    def test_equal_present_and_totals_do_not_erase_order(self):
        first, second = [TemporalHistory([.05, .1, .2, .4], 12, 2) for _ in range(2)]
        for a, b in [([1., 0.], [0., 1.]), ([0., 1.], [1., 0.])]:
            first.advance(.1, a)
            second.advance(.1, b)
        first.advance(.1, [0., 0.])
        second.advance(.1, [0., 0.])
        a = np.array(first.snapshot()["known_envelope_by_age"])
        b = np.array(second.snapshot()["known_envelope_by_age"])
        np.testing.assert_allclose(a, b[:, ::-1], atol=1e-15)
        self.assertGreater(a[1, 1], a[1, 0])
        self.assertGreater(a[2, 0], a[2, 1])

    def test_unknown_input_retains_old_evidence_but_changes_coverage(self):
        silent, unknown = [TemporalHistory([.1, .4, 1.6], 4, 1) for _ in range(2)]
        for history in [silent, unknown]:
            history.advance(.2, [1.])
        silent.advance(.5, [0.])
        unknown.advance(.5, None)
        a, b = silent.snapshot(), unknown.snapshot()
        np.testing.assert_array_equal(a["known_envelope_by_age"], b["known_envelope_by_age"])
        self.assertTrue(np.all(np.array(a["known_coverage_by_age"]) > b["known_coverage_by_age"]))
        self.assertTrue(np.any(np.array(b["known_envelope_by_age"]) > 0.))
        self.assertEqual(b, unknown.snapshot())

    def test_small_known_coverage_survives_complement_cancellation(self):
        history = TemporalHistory([1., 10., 100.], 12, 1)
        history.advance(.0001, [1.])
        coverage = np.array(history.snapshot()["known_coverage_by_age"])
        self.assertTrue(np.all(coverage > 0.))
        self.assertTrue(np.all(coverage < 1e-30))
        history.advance(1e300, None)
        np.testing.assert_array_equal(history.state, np.zeros_like(history.state))

    def test_invalid_input_is_rejected_without_advancing(self):
        for ages, order, channels in [([], 4, 1), ([0.], 4, 1), ([1., .1], 4, 1),
                                      ([math.inf], 4, 1), ([1.], 0, 1), ([1.], 4, 0)]:
            with self.assertRaises(ValueError):
                TemporalHistory(ages, order, channels)
        history = TemporalHistory([.1], 4, 1)
        for dt, value in [(0., [1.]), (math.inf, [1.]), (.1, [-1.]), (.1, [math.nan]), (.1, [1., 2.])]:
            with self.assertRaises(ValueError):
                history.advance(dt, value)
        np.testing.assert_array_equal(history.state, np.zeros_like(history.state))

    def test_observation_adapter_requires_an_explicit_gap(self):
        parameters = {"ages_sec": [.01, .1], "post_order": 4, "channels": 2,
                      "step_sec": .001, "report_steps": 1}
        rows = [{"kind": "auditory_envelope", "available_sec": .001, "envelope_scan": [1., 0.]},
                {"kind": "input_gap", "available_sec": .010, "missing_start_sec": .001},
                {"kind": "auditory_envelope", "available_sec": .011, "envelope_scan": [0., 1.]}]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.gz", Path(folder) / "output.gz"
            with gzip.open(source, "wt") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in rows))
            result = run(source, parameters, target)
            self.assertEqual(result["observations"], 2)
            self.assertEqual(result["last_available_sec"], .011)
            with gzip.open(target, "rt") as stream:
                stored = [json.loads(line) for line in stream]
            self.assertEqual([r["kind"] for r in stored[1:]],
                             ["temporal_history", "input_gap", "temporal_history"])
            self.assertGreater(stored[2]["known_envelope_by_age"][0][0], 0.)
            with gzip.open(source, "wt") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in (rows[0], rows[2])))
            with self.assertRaisesRegex(ValueError, "unreported input gap"):
                run(source, parameters, Path(folder) / "rejected.gz")


if __name__ == "__main__":
    unittest.main()
