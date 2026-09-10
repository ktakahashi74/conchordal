"""Temporal relations, unavailable covariance, and causal streaming boundaries."""

import copy
import math
from pathlib import Path
import pickle
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_component_relations import (CENTERS_LOG2, WINDOW_FRAMES, ComponentRelations,
                                          control_audio, evaluate, summarize)
from evaluate_frequency_events import FrequencyEventObserver


def frames(count=1100):
    return [{"kind": "components", "available_sec": (tick + 1) / 100,
             "initial_observation": tick == 0, "evidence_start_sec": 0,
             "components": [{"frequency_log2": math.log2(f), "power": .02} for f in (310, 480.5)]
             if tick % 50 < 12 else []} for tick in range(count)]


class ComponentRelationTests(unittest.TestCase):
    def test_same_frequency_positions_retain_different_temporal_relations(self):
        for case in ("synchronous", "alternating", "independent", "two_pairs", "weaker_partner"):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 21)
                rows = evaluate(FrequencyEventObserver(24_000).process(0, audio))
                result = summarize(rows, truth)
                self.assertTrue(result["observation_passed"], result)
                self.assertTrue(all(r["observed_frames"] for r in result["pair_checks"]))

    def test_constant_channels_are_observed_but_cannot_supply_covariance(self):
        for case in ("steady_pair", "startup_pair", "silence"):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 21)
                rows = evaluate(FrequencyEventObserver(24_000).process(0, audio))
                result = summarize(rows, truth)
                self.assertTrue(result["observation_passed"], result)
                late = [r for r in rows if 4 <= r["available_sec"] < 10]
                self.assertTrue(all(r["status"] == "no_modulated_pair" for r in late))
                self.assertTrue(all(r["energetic_peaks"] == (0 if case == "silence" else 2) for r in late))

    def test_packet_boundaries_and_unavailable_future_do_not_change_relations(self):
        audio, _ = control_audio("two_pairs", 1)
        audio = audio[:round(6.3 * 24_000)]
        full = evaluate(FrequencyEventObserver(24_000).process(0, audio))
        front, model, chunked = FrequencyEventObserver(24_000), ComponentRelations(), []
        for start in range(0, len(audio), 347):
            for row in front.process(start, audio[start:start + 347]):
                result = model.process(row)
                if result is not None:
                    chunked.append(result)
        self.assertEqual(full, chunked)
        end = round(5.417 * 24_000)
        prefix = evaluate(FrequencyEventObserver(24_000).process(0, audio[:end]))
        self.assertEqual(prefix, [r for r in full if r["available_sec"] <= end / 24_000])
        self.assertTrue(all(r["evidence_start_sec"] <= r["available_sec"] for r in full))

    def test_gap_forgets_relations_and_restarts_with_a_full_window(self):
        model = ComponentRelations()
        before = []
        for row in frames(450):
            result = model.process(row)
            if result is not None:
                before.append(result)
        self.assertTrue(any(r["status"] == "available" for r in before))
        gap = model.process({"kind": "input_gap", "available_sec": 5.0})
        self.assertEqual(gap["kind"], "input_gap")
        after = []
        for tick in range(300):
            result = model.process({"kind": "components", "available_sec": 5.09 + tick / 100,
                                    "initial_observation": tick == 0, "evidence_start_sec": 5.0,
                                    "components": [{"frequency_log2": math.log2(f), "power": .02}
                                                   for f in (310, 480.5)]})
            if result is not None:
                after.append(result)
        self.assertTrue(all(r["kind"] == "relations" for r in after))
        self.assertTrue(all(r["evidence_start_sec"] >= 5 for r in after))
        self.assertTrue(all(not r["pairs"] for r in after))
        calibrated = [r for r in after if r["window_frames"] == WINDOW_FRAMES]
        self.assertGreaterEqual(calibrated[0]["available_sec"], 7.08)
        self.assertTrue(all(r["status"] == "no_modulated_pair" for r in calibrated))

    def test_component_order_and_gain_do_not_invent_a_temporal_relation(self):
        rows = frames()
        altered = copy.deepcopy(rows)
        for row in altered:
            row["components"].reverse()
            for c in row["components"]:
                c["power"] *= .25
        first, second = evaluate(rows), evaluate(altered)
        for a, b in zip(first, second):
            self.assertEqual(a["status"], b["status"])
            self.assertEqual(a["pairs"], b["pairs"])
            self.assertEqual(a["energetic_peaks"], b["energetic_peaks"])

    def test_invalid_input_is_atomic_and_filter_memory_is_bounded(self):
        model = ComponentRelations()
        for row in frames(1300):
            model.process(row)
        self.assertEqual(len(model.history), WINDOW_FRAMES)
        self.assertEqual(model.cross_sum.shape, (len(CENTERS_LOG2), len(CENTERS_LOG2)))
        snapshot = pickle.dumps(model)
        for row in ({"kind": "components", "available_sec": 0, "components": []},
                    {"kind": "components", "available_sec": 14, "components": [
                        {"frequency_log2": math.inf, "power": .1}]},
                    {"kind": "components", "available_sec": 14, "components": [
                        {"frequency_log2": 9, "power": -1}]}):
            with self.assertRaises(ValueError):
                model.process(row)
            self.assertEqual(pickle.dumps(model), snapshot)


if __name__ == "__main__":
    unittest.main()
