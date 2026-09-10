import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_temporal_associations import TemporalAssociations, run
from evaluate_temporal_history import TemporalHistory


class TemporalAssociationTests(unittest.TestCase):
    def test_interval_mean_matches_independent_time_quadrature(self):
        ages, order = np.array([.025, .1, .4, 1.6]), 4
        history = TemporalHistory(ages, order, 1)
        mean = history.advance(.3, [2.])
        time = np.linspace(0., .3, 40_001)
        x = order * time[:, None] / ages
        step_response = 1 - np.exp(-x) * sum(x**n / math.factorial(n) for n in range(order + 1))
        expected = np.trapezoid(step_response, time, axis=0) / .3
        np.testing.assert_allclose(mean[:, 0], 2 * expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(mean[:, 1], expected, rtol=1e-8, atol=1e-10)

    def test_integrated_history_survives_small_inputs_and_large_gaps(self):
        history = TemporalHistory([.01, 1., 100.], 12, 1)
        mean = history.advance(.0001, [1.])
        self.assertTrue(np.all(mean > 0.))
        self.assertTrue(np.all(mean <= history.state[:, -1, :]))
        before = history.state.copy()
        mean = history.advance(1e6, None)
        expected = np.sum(before, axis=1) / history.rates[:, None] / 1e6
        np.testing.assert_allclose(mean, expected, rtol=1e-14)

    def test_constant_input_partition_preserves_associations_and_exposure(self):
        one, many = [TemporalAssociations([.01, .1, 1., 10.], 12, 2) for _ in range(2)]
        for model in [one, many]:
            model.observe(.05, [1., .2])
        one.observe(.71, [.1, 2.])
        for interval in [.01, .15, .001, .3, .249]:
            many.observe(interval, [.1, 2.])
        np.testing.assert_allclose(one.associations, many.associations, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(one.pair_exposure_sec, many.pair_exposure_sec, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(one.history.state, many.history.state, rtol=1e-12, atol=1e-16)

    def test_age_quadrature_matches_continuous_impulse_overlap(self):
        order, learned_delay, query_delay = 12, .7, 1.1
        expected = (order * math.comb(2 * order, order) * learned_delay**order * query_delay**order
                    / (learned_delay + query_delay)**(2 * order + 1))
        errors = []
        for cells in [65, 129, 257]:
            ages = np.geomspace(.001, 100., cells)
            model = TemporalAssociations(ages, order, 2)
            rate = order / ages
            # Delta-pair limit, independent of the time integration under test.
            learned = np.exp((order + 1) * np.log(rate) + order * np.log(learned_delay)
                             - rate * learned_delay - math.lgamma(order + 1))
            query = np.exp((order + 1) * np.log(rate) + order * np.log(query_delay)
                           - rate * query_delay - math.lgamma(order + 1))
            model.associations[:, 1, 0] = learned
            model.history.state[:, -1, 0] = query
            recall = model.predict()
            actual = recall["cross_band_activation"][1]
            errors.append(abs(actual / expected - 1))
            self.assertEqual(recall["within_band_activation"], [0., 0.])
            self.assertEqual(recall["activation"][0], 0.)
            self.assertAlmostEqual(np.sum(model.age_weights), ages[-1] - ages[0])
        self.assertLess(errors[1], errors[0] / 3)
        self.assertLess(errors[2], errors[1] / 3)
        self.assertLess(errors[2], .0004)

    def test_cue_predicts_learned_successor_at_a_scale_dependent_delay(self):
        peaks, scaled_curves = [], []
        for delay in [.25, 1., 4.]:
            ages = np.geomspace(delay / 100, delay * 100, 129)
            model = TemporalAssociations(ages, 12, 2)
            width = delay * .0001
            model.observe(width, [1 / width, 0.])
            model.observe(delay - width, [0., 0.])
            issued = model.predict()
            self.assertEqual(issued["activation"][1], 0.)
            model.observe(width, [0., 1 / width])
            # A fresh cue queries acquired weights; the probe does not relearn them.
            probe = TemporalAssociations(ages, 12, 2)
            probe.associations[:] = model.associations
            probe.history.advance(width, [1 / width, 0.])
            times = np.linspace(.1, 2., 191) * delay
            curve, previous = [], width
            for time in times:
                probe.history.advance(time - previous, [0., 0.])
                curve.append(probe.predict()["cross_band_activation"][1])
                previous = time
            peaks.append(times[int(np.argmax(curve))] / delay)
            scaled_curves.append(np.array(curve) * delay)
        np.testing.assert_allclose(peaks, np.full(3, 12 / 13), atol=.01)
        np.testing.assert_allclose(scaled_curves[0], scaled_curves[1], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scaled_curves[1], scaled_curves[2], rtol=1e-10, atol=1e-10)

    def test_same_band_and_cross_band_learning_remain_distinct(self):
        repeated, changed = [TemporalAssociations(np.geomspace(.01, 4., 33), 12, 2) for _ in range(2)]
        for model, target in [(repeated, [1., 0.]), (changed, [0., 1.])]:
            model.observe(.01, [1., 0.])
            model.observe(.49, [0., 0.])
            model.observe(.01, target)
        self.assertGreater(repeated.associations[:, 0, 0].max(), changed.associations[:, 0, 0].max())
        self.assertEqual(repeated.associations[:, 1, 0].max(), 0.)
        self.assertGreater(changed.associations[:, 1, 0].max(), 0.)
        changed.history.advance(.2, [1., 0.])
        recall = changed.predict()
        self.assertGreater(recall["cross_band_activation"][1], 0.)
        np.testing.assert_allclose(recall["activation"], np.array(recall["within_band_activation"])
                                   + recall["cross_band_activation"])

    def test_missing_input_does_not_erase_or_invent_associations(self):
        silent = TemporalAssociations([.1, 1., 10.], 4, 2)
        silent.observe(.3, [1., .5])
        unknown = copy.deepcopy(silent)
        before = unknown.associations.copy()
        exposure = unknown.pair_exposure_sec.copy()
        silent.observe(2., [0., 0.])
        unknown.observe(2., None)
        np.testing.assert_array_equal(unknown.associations, before)
        np.testing.assert_array_equal(silent.associations, before)
        np.testing.assert_array_equal(unknown.pair_exposure_sec, exposure)
        np.testing.assert_array_equal(silent.predict()["activation"], unknown.predict()["activation"])
        self.assertTrue(np.all(silent.pair_exposure_sec > unknown.pair_exposure_sec))
        silent.observe(.1, [.5, .25])
        unknown.observe(.1, [.5, .25])
        np.testing.assert_array_equal(silent.associations, unknown.associations)
        self.assertTrue(np.all(silent.pair_exposure_sec > unknown.pair_exposure_sec))

    def test_recall_is_immutable_and_not_a_probability_or_amplitude(self):
        base, louder = [TemporalAssociations([.1, .3, 1.], 4, 2) for _ in range(2)]
        for model, gain in [(base, 1), (louder, 3)]:
            model.observe(.2, np.array([1., .5]) * gain)
            model.observe(.5, np.array([.2, 1.]) * gain)
        np.testing.assert_allclose(louder.associations, base.associations * 9, rtol=1e-14)
        np.testing.assert_allclose(louder.predict()["activation"], np.array(base.predict()["activation"]) * 27,
                                   rtol=1e-14)
        issued = base.predict()
        saved = copy.deepcopy(issued)
        self.assertEqual(issued, base.predict())
        base.observe(.3, [2., 4.])
        self.assertEqual(issued, saved)

    def test_adapter_issues_before_the_target_and_does_not_close_at_eof(self):
        parameters = {"ages_sec": [.01, .1], "post_order": 4, "channels": 2,
                      "step_sec": .01, "report_steps": 1}
        outputs = []
        with tempfile.TemporaryDirectory() as folder:
            for branch, last in enumerate(([1., 0.], [0., 2.])):
                source, target = Path(folder) / f"{branch}.in.gz", Path(folder) / f"{branch}.out.gz"
                rows = [{"kind": "auditory_envelope", "available_sec": (i+1) * .01,
                         "envelope_scan": value} for i, value in enumerate(([1., 0.], [0., 1.], last))]
                with gzip.open(source, "wt") as stream:
                    stream.write("".join(json.dumps(row) + "\n" for row in rows))
                result = run(source, parameters, target)
                self.assertEqual(result["forecasts"], 4)
                self.assertEqual(result["observed_outcomes"], 3)
                self.assertEqual(result["pending_forecasts"], 1)
                with gzip.open(target, "rt") as stream:
                    outputs.append([json.loads(line) for line in stream][1:])
            self.assertEqual([row["kind"] for row in outputs[0]], ["forecast", "outcome"] * 3 + ["forecast"])
            self.assertEqual(outputs[0][-3], outputs[1][-3])
            self.assertNotEqual(outputs[0][-2], outputs[1][-2])
            self.assertEqual(outputs[0][-3]["issued_sec"], .02)
            self.assertEqual(outputs[0][-3]["target_end_sec"], .03)

    def test_adapter_requires_explicit_gaps_and_keeps_learning_coverage(self):
        parameters = {"ages_sec": [.01, .1], "post_order": 4, "channels": 1,
                      "step_sec": .001, "report_steps": 1}
        rows = [{"kind": "auditory_envelope", "available_sec": .001, "envelope_scan": [1.]},
                {"kind": "input_gap", "available_sec": .010},
                {"kind": "auditory_envelope", "available_sec": .011, "envelope_scan": [1.]}]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "in.gz", Path(folder) / "out.gz"
            with gzip.open(source, "wt") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in rows))
            result = run(source, parameters, target)
            self.assertEqual(result["final_known_current_sec"], .002)
            self.assertEqual(result["censored_forecasts"], 1)
            self.assertEqual(result["pending_forecasts"], 1)
            with gzip.open(target, "rt") as stream:
                stored = [json.loads(line) for line in stream]
            self.assertEqual(stored[4]["kind"], "censored")
            self.assertEqual(stored[5]["kind"], "input_gap")
            self.assertGreater(stored[6]["activation"][0], 0.)
            with gzip.open(source, "wt") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in (rows[0], rows[2])))
            with self.assertRaisesRegex(ValueError, "unreported input gap"):
                run(source, parameters, Path(folder) / "rejected.gz")


if __name__ == "__main__":
    unittest.main()
