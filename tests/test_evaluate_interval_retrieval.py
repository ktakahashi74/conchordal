import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_interval_retrieval import IntervalRetrieval, evaluate, score_waiting


class IntervalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {"max_context": 3, "buffer_items": 8, "buffer_sec": 2.,
                           "post_buffer_weight": .6, "decay_half_life_sec": 1.,
                           "context_width_log2": .1, "target_width_log2": .07,
                           "prior_mean_log2": -1., "prior_width_log2": 2., "prior_mass": 1.}

    def rows(self, intervals, trailing_ticks=0):
        cursor = 10
        ticks = [cursor]
        for interval in intervals:
            cursor += round(interval / .01)
            ticks.append(cursor)
        onsets = set(ticks)
        return [{"time_sec": tick * .01, "observed_band": int(tick in onsets)}
                for tick in range(1, ticks[-1] + trailing_ticks + 1)]

    def trained(self, intervals):
        model = IntervalRetrieval(self.parameters, step_sec=.01)
        for row in self.rows(intervals):
            model.process(row["time_sec"], bool(row["observed_band"]))
        return model

    def test_continuous_order_improves_repeat_prediction(self):
        result = evaluate(self.rows([.3, .3, .6, .6] * 12), self.parameters, step_sec=.01)
        scores = [event["interval_loss_bits"] for event in result["events"] if "interval_loss_bits" in event][-16:]
        gain = sum(score["context_0"] - score["context_2"] for score in scores) / len(scores)
        self.assertGreater(gain, 0.)
        model = self.trained([.3, .3, .6, .6] * 4)
        close = copy.deepcopy(model)
        close.intervals[-1] = math.log2(.601)
        exact = model.retrieve(model.last_observation)["support"]["context_2"]
        similar = close.retrieve(close.last_observation)["support"]["context_2"]
        self.assertGreater(similar, exact * .99)

    def test_quiet_retrieval_loses_support_without_rewriting_issued_forecast(self):
        model = self.trained([.3, .3, .6, .6] * 4)
        issued = copy.deepcopy(model.pending)
        recalled = model.retrieve(model.last_observation + 60)
        self.assertGreater(recalled["weights"]["context_0"][0], .999999)
        self.assertLess(recalled["support"]["context_0"], 1e-10)
        self.assertLess(issued["weights"]["context_0"][0], .5)
        self.assertEqual(model.pending, issued)

    def test_item_interference_and_time_decay_are_separate(self):
        self.parameters["buffer_items"] = 3
        quiet = self.trained([.3])
        occupied = self.trained([.3, .02, .02, .02, .02])
        time = occupied.last_observation
        first = quiet.retrieve(time)["weights"]["context_0"]
        second = occupied.retrieve(time)["weights"]["context_0"]
        self.assertAlmostEqual(first[1] / first[0], 1.)
        self.assertLess(second[1] / second[0], .6)
        after_time_expiry = quiet.retrieve(quiet.last_observation + 3)["weights"]["context_0"]
        self.assertAlmostEqual(after_time_expiry[1] / after_time_expiry[0], .3)

    def test_waiting_and_arrival_likelihood_telescope(self):
        prediction = self.trained([.3, .3, .6, .6] * 4).pending
        totals = {name: 0. for name in prediction["weights"]}
        for tick in range(1, 31):
            scores = score_waiting(prediction, (tick - 1) * .01, tick * .01, tick == 30)
            for name, score in scores.items():
                totals[name] += score["loss_bits"]
        for name, weights in prediction["weights"].items():
            mass = 0.
            for mean, width, weight in zip(prediction["means_log2_sec"], prediction["widths_log2_sec"], weights):
                mass += weight * .5 * (math.erf((math.log2(.3) - mean) / (math.sqrt(2) * width))
                                      - math.erf((math.log2(.29) - mean) / (math.sqrt(2) * width)))
            self.assertAlmostEqual(totals[name], -math.log2(mass), places=10)
            self.assertAlmostEqual(sum(weights), 1.)
            self.assertTrue(all(weight >= 0 for weight in weights))

    def test_future_inputs_do_not_change_past_outputs(self):
        rows = self.rows([.3, .3, .6, .6] * 3)
        prefix = rows[:200]
        first = evaluate(prefix, self.parameters, step_sec=.01)
        second = evaluate(rows, self.parameters, step_sec=.01)
        self.assertEqual(first["events"], second["events"][:len(first["events"])])
        model = self.trained([.3, .3, .6, .6])
        other = copy.deepcopy(model)
        saved = copy.deepcopy(model.pending)
        model.process(model.last_observation + .01, True)
        other.process(other.last_observation + .01, False)
        self.assertEqual(other.pending, saved)

    def test_time_unit_conversion_preserves_forecasts_and_scores(self):
        rows = self.rows([.3, .3, .6, .6] * 2)
        seconds = evaluate(rows, self.parameters, step_sec=.01)
        parameters = dict(self.parameters)
        parameters["buffer_sec"] *= 1000
        parameters["decay_half_life_sec"] *= 1000
        parameters["prior_mean_log2"] += math.log2(1000)
        milliseconds = evaluate([{**row, "time_sec": row["time_sec"] * 1000} for row in rows], parameters, step_sec=10.)
        for first, second in zip(seconds["events"], milliseconds["events"]):
            if "interval_loss_bits" not in first:
                continue
            for name, loss in first["interval_loss_bits"].items():
                self.assertAlmostEqual(loss, second["interval_loss_bits"][name], places=10)
            for name, weights in first["next_forecast"]["weights"].items():
                for a, b in zip(weights, second["next_forecast"]["weights"][name]):
                    self.assertAlmostEqual(a, b, places=10)

    def test_unresolved_tail_is_unavailable_instead_of_closure(self):
        prediction = self.trained([.3, .6]).pending
        for fired in (True, False):
            scores = score_waiting(prediction, 1e290, 1e300, fired)
            for score in scores.values():
                self.assertIsNone(score["loss_bits"])
                self.assertIsNone(score["probability"])
                self.assertEqual(score["unavailable_reason"], "numerical_tail_resolution")

    def test_gap_preserves_scored_evidence_and_censors_unknown_interference(self):
        rows = self.rows([.3, .3, .6, .6] * 2, trailing_ticks=5)
        rows += [{"time_sec": rows[-1]["time_sec"] + .1, "observed_band": 1}]
        result = evaluate(rows, self.parameters, step_sec=.01)
        gap = result["events"][-1]
        self.assertEqual(gap["kind"], "gap")
        self.assertTrue(all(value > 0 for value in gap["censored_loss_bits"].values()))
        self.assertIsNotNone(gap["censored_forecast"])
        self.assertIsNone(result["right_censored"])

    def test_silence_does_not_create_events_or_expectations(self):
        rows = [{"time_sec": tick * .01, "observed_band": 0} for tick in range(1, 100)]
        result = evaluate(rows, self.parameters, step_sec=.01)
        self.assertEqual(result["events"], [])
        self.assertEqual(result["scored_waiting_windows"], 0)
        self.assertIsNone(result["right_censored"])

    def test_invalid_parameters_and_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            IntervalRetrieval({**self.parameters, "human_memory_length": 4}, step_sec=.01)
        with self.assertRaises(ValueError):
            IntervalRetrieval({**self.parameters, "decay_half_life_sec": 0}, step_sec=.01)
        model = self.trained([.3, .6])
        saved = copy.deepcopy(model.__dict__)
        for time, fired in [(model.last_observation, True), (float("nan"), False), (2., 1)]:
            with self.assertRaises(ValueError):
                model.process(time, fired)
            self.assertEqual(model.__dict__, saved)


if __name__ == "__main__":
    unittest.main()
