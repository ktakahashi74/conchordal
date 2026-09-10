import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ppm_decay_reference import PpmDecay

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ppm_decay_reference.json").read_text())


class PpmDecayReferenceTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "buffer_length_time": 1000., "buffer_length_items": 3, "buffer_weight": 1.,
            "only_learn_from_buffer": True, "only_predict_from_buffer": True,
            "stm_weight": 1., "stm_duration": 0., "ltm_weight": .6,
            "ltm_half_life": 2., "ltm_asymptote": 0., "noise": 0.,
        }

    def test_upstream_predictions_with_paired_retrieval_noise(self):
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                model = PpmDecay(case["alphabet_size"], case["order_bound"], case["parameters"], seed=case["seed"])
                for symbol, time, expected in zip(case["symbols"], case["times"], case["expected"]):
                    increments = iter(expected["retrieval_noise"])
                    model.noise_draw = lambda: next(increments)
                    actual = model.observe(symbol, time)
                    self.assertIsNone(next(increments, None))
                    self.assertEqual(actual["order"], expected["order"])
                    self.assertAlmostEqual(actual["loss_bits"], expected["loss_bits"], delta=2e-12)
                    self.assertEqual(len(actual["distribution"]), len(expected["distribution"]))
                    for observed, reference in zip(actual["distribution"], expected["distribution"]):
                        self.assertAlmostEqual(observed, reference, delta=2e-12)

    def test_item_expiry_starts_decay_after_intervening_events(self):
        model = PpmDecay(3, 2, self.parameters, seed=1)
        for symbol, time in [(0, 0.), (1, .1), (2, .2)]:
            model.observe(symbol, time)
        self.assertEqual(model.weight((0, 1), .3, noisy=False), 1.)
        model.observe(2, .3)
        self.assertAlmostEqual(model.weight((0, 1), .3, noisy=False), .6)
        self.assertAlmostEqual(model.weight((0, 1), 2.3, noisy=False), .3)

    def test_time_expiry_starts_decay_without_new_events(self):
        self.parameters["buffer_length_time"] = .1
        model = PpmDecay(3, 2, self.parameters, seed=1)
        model.observe(0, 0.)
        self.assertEqual(model.weight((0,), .099999, noisy=False), 1.)
        self.assertAlmostEqual(model.weight((0,), .1, noisy=False), .6)
        self.assertAlmostEqual(model.weight((0,), 2.1, noisy=False), .3)
        self.assertEqual(model.times, [0.])

    def test_original_exact_boundary_differs_for_prediction_and_learning(self):
        self.parameters["buffer_length_time"] = .1
        model = PpmDecay(3, 2, self.parameters, seed=1)
        model.observe(0, 0.)
        result = model.observe(1, .1)
        self.assertEqual(result["order"], 1)
        self.assertNotIn((0, 1), model.traces)
        self.assertEqual(model.weight((1,), .1, noisy=False), 1.)

    def test_symbol_identity_cannot_change_its_own_forecast(self):
        model = PpmDecay(3, 2, self.parameters, seed=1)
        for index, symbol in enumerate([0, 0, 0, 1, 0]):
            model.observe(symbol, index * .1)
        other = copy.deepcopy(model)
        first = model.observe(0, .5)
        second = other.observe(2, .5)
        self.assertEqual(first["distribution"], second["distribution"])
        self.assertNotEqual(first["loss_bits"], second["loss_bits"])
        saved = copy.deepcopy(first)
        model.observe(1, .6)
        self.assertEqual(first, saved)

    def test_invalid_observations_preserve_history_and_random_state(self):
        model = PpmDecay(3, 2, self.parameters, seed=1)
        model.observe(0, .1)
        snapshot = copy.deepcopy((model.symbols, model.times, model.traces, model.random.bit_generator.state))
        for symbol, time in [(-1, .2), (3, .2), (.5, .2), (0, .1), (0, -.1), (0, float("nan")), (0, float("inf"))]:
            with self.subTest(symbol=symbol, time=time):
                with self.assertRaises(ValueError):
                    model.observe(symbol, time)
                self.assertEqual((model.symbols, model.times, model.traces, model.random.bit_generator.state), snapshot)
        with self.assertRaises(ValueError):
            model.weight((0,), .05, noisy=False)
        self.assertEqual((model.symbols, model.times, model.traces, model.random.bit_generator.state), snapshot)

    def test_published_buffer_does_not_identify_quiet_retention(self):
        reference = FIXTURE["quiet_retention"]
        for case in reference["cases"]:
            with self.subTest(intervening=case["intervening_symbols"]):
                model = PpmDecay(reference["alphabet_size"], reference["order_bound"], reference["parameters"], seed=reference["seed"])
                for symbol, time in zip(case["symbols"], case["times"]):
                    model.observe(symbol, time)
                for time, expected in zip(case["query_times"], case["expected_weights"]):
                    self.assertAlmostEqual(model.weight((0,), time, noisy=False), expected, delta=2e-12)
        self.assertEqual(reference["cases"][0]["expected_weights"], [1., 1., 1.])
        self.assertAlmostEqual(reference["cases"][1]["expected_weights"][1], .3)


if __name__ == "__main__":
    unittest.main()
