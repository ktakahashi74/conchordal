import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_temporal_waiting import evaluate, waiting_score


class TemporalWaitingTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {"issued_sec": 0., "means_log2_sec": [math.log2(.3)],
                           "context_log2_sec": [math.log2(.3), math.log2(.3)],
                           "weights": {name: [.95] for name in ("ordered", "one_interval", "marginal")}}
        onset_ticks = {10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310}
        self.rows = [{"time_sec": tick / 100, "observed_band": int(tick in onset_ticks)}
                     for tick in range(401)]

    def test_arrival_and_waiting_partition_conditional_probability(self):
        for start, end in ((0., .1), (.1, .2), (.29, .3), (.3, .31), (.8, 1.)):
            arrival = waiting_score(self.prediction, start, end, True)
            waiting = waiting_score(self.prediction, start, end, False)
            for name in arrival:
                self.assertAlmostEqual(arrival[name]["probability"] + waiting[name]["probability"], 1.)

    def test_waiting_then_arrival_matches_unconditional_interval_mass(self):
        # Independent CDF for the known mixture; densities are in log2 seconds.
        cdf = lambda time: (.05 * .5 * math.erfc(-(math.log2(time) + 1) / (2 * math.sqrt(2)))
                            + .95 * .5 * math.erfc(-(math.log2(time) - math.log2(.3)) / (.07 * math.sqrt(2))))
        losses = [waiting_score(self.prediction, i / 100, (i + 1) / 100, False)["ordered"]["loss_bits"]
                  for i in range(29)]
        losses.append(waiting_score(self.prediction, .29, .30, True)["ordered"]["loss_bits"])
        self.assertAlmostEqual(sum(losses), -math.log2(cdf(.30) - cdf(.29)), places=11)

    def test_future_suffix_does_not_rewrite_earlier_forecasts(self):
        prefix = evaluate(self.rows[:351])["scores"]
        changed = [dict(row) for row in self.rows]
        changed[370]["observed_band"] = 2
        extended = evaluate(changed)["scores"]
        self.assertEqual(prefix, [row for row in extended if row["available_sec"] <= 3.5])

    def test_no_onset_is_scored_before_a_successor_arrives(self):
        result = evaluate(self.rows)
        waiting = [row for row in result["scores"] if row["issued_sec"] == 3.1]
        self.assertTrue(waiting)
        self.assertTrue(all(row["kind"] == "no_detected_onset" for row in waiting))
        self.assertGreater(sum(row["models"]["ordered"]["loss_bits"] for row in waiting), 4.)
        self.assertEqual(result["right_censored"], {"issued_sec": 3.1, "last_valid_sec": 4.})

    def test_missing_rows_censor_without_counting_the_gap_as_waiting(self):
        result = evaluate(self.rows[:351] + [{"time_sec": 8., "observed_band": 1},
                                            {"time_sec": 8.01, "observed_band": 0}])
        self.assertEqual(result["gaps"], [{"available_sec": 8., "last_valid_sec": 3.5,
                                          "censored_origin_sec": 3.1}])
        self.assertTrue(all(row["available_sec"] <= 3.5 for row in result["scores"]))
        self.assertIsNone(result["right_censored"])

    def test_silence_without_learned_forecast_does_not_create_expectation(self):
        result = evaluate([{**row, "observed_band": 0} for row in self.rows])
        self.assertEqual(result["scores"], [])
        self.assertIsNone(result["right_censored"])

    def test_invalid_observations_and_unresolved_numerical_tails(self):
        for row in ({"time_sec": .1, "observed_band": .5},
                    {"time_sec": float("nan"), "observed_band": 0}):
            with self.assertRaises(ValueError):
                evaluate([row])
        with self.assertRaises(ValueError):
            waiting_score(self.prediction, 1., 1., False)
        tails = waiting_score(self.prediction, 1e299, 1e300, True)
        self.assertTrue(all(row["probability"] is None and row["loss_bits"] is None
                            and row["unavailable_reason"] == "numerical_tail_resolution"
                            for row in tails.values()))


if __name__ == "__main__":
    unittest.main()
