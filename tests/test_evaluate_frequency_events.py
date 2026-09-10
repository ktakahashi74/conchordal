"""Audio-only component observation, streaming boundaries, and ambiguity limits."""

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_frequency_events import FrequencyEventObserver, MAX_COMPONENTS, control_audio, summarize


class FrequencyEventTests(unittest.TestCase):
    def test_packet_boundaries_and_unavailable_future_do_not_change_observations(self):
        audio, _ = control_audio("overlapping", 21)
        audio = audio[:round(2.3 * 24_000)]
        full = FrequencyEventObserver(24_000).process(0, audio)
        observer, chunks = FrequencyEventObserver(24_000), []
        for start in range(0, len(audio), 173):
            chunks.extend(observer.process(start, audio[start:start + 173]))
        self.assertEqual(full, chunks)
        end = 32_789
        prefix = FrequencyEventObserver(24_000).process(0, audio[:end])
        self.assertEqual(prefix, [r for r in full if r["available_sec"] <= end / 24_000])
        self.assertTrue(all(0 <= r["evidence_start_sec"] < r["available_sec"] for r in full))

    def test_simultaneous_and_weak_components_have_separate_frequency_positions(self):
        for case in ("overlapping", "simultaneous", "soft_overlay", "harmonic"):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 21)
                rows = FrequencyEventObserver(24_000).process(0, audio)
                result = summarize(rows, truth)
                self.assertTrue(result["observation_passed"], result)
                self.assertGreaterEqual(result["maximum_simultaneous_components"], 2)
        self.assertEqual(result["expected_components"], 72)

    def test_steady_pair_and_vibrato_do_not_become_repeated_rises(self):
        for case in ("steady_pair", "vibrato", "silence"):
            with self.subTest(case=case):
                audio, truth = control_audio(case, 42)
                rows = FrequencyEventObserver(24_000).process(0, audio)
                result = summarize(rows, truth)
                self.assertTrue(result["observation_passed"], result)

    def test_continuous_glide_retains_motion_without_repeated_rises(self):
        audio, truth = control_audio("glide", 1)
        rows = FrequencyEventObserver(24_000).process(0, audio)
        self.assertTrue(summarize(rows, truth)["observation_passed"])
        trajectory = [c["frequency_log2"] for r in rows for c in r["components"]]
        self.assertGreater(max(trajectory) - min(trajectory), .65)
        identities = {c["ridge_id"] for r in rows for c in r["components"]}
        self.assertEqual(len(identities), 1)

    def test_gap_forgets_ridges_without_inventing_a_rise_from_an_existing_tone(self):
        fs = 24_000
        time = np.arange(2 * fs) / fs
        audio = .2 * np.sin(math.tau * 327.3 * time)
        observer = FrequencyEventObserver(fs)
        before = observer.process(0, audio[:fs // 2])
        after = observer.process(fs, audio[fs:])
        self.assertEqual(after[0]["kind"], "input_gap")
        self.assertEqual(after[0]["missing_start_sec"], .5)
        self.assertFalse(any(c["rise"] for r in before + after if r["kind"] == "components" for c in r["components"]))
        old_ids = {c["ridge_id"] for r in before for c in r["components"]}
        new_ids = {c["ridge_id"] for r in after[1:] for c in r["components"]}
        self.assertTrue(old_ids.isdisjoint(new_ids))
        self.assertTrue(all(r["evidence_start_sec"] >= 1 for r in after[1:]))
        self.assertGreaterEqual(after[1]["available_sec"], 1.08)

    def test_same_waveform_does_not_reveal_how_many_sources_produced_it(self):
        fs = 24_000
        t = np.arange(fs) / fs
        envelope = np.clip((t - .2) / .012, 0, 1)
        single = .2 * np.sin(math.tau * 310 * t) * envelope
        two_sources = single * .5 + single * .5
        np.testing.assert_array_equal(single, two_sources)
        a = FrequencyEventObserver(fs).process(0, single)
        b = FrequencyEventObserver(fs).process(0, two_sources)
        self.assertEqual(a, b)
        self.assertEqual(summarize(a)["component_rises"], 1)

    def test_invalid_input_is_atomic_and_internal_buffers_are_bounded(self):
        observer = FrequencyEventObserver(24_000)
        time = np.arange(24_000) / 24_000
        tones = sum(.01 * np.sin(math.tau * frequency * time) for frequency in np.linspace(100, 6800, 40))
        observer.process(0, tones)
        self.assertLessEqual(len(observer.tracks), MAX_COMPONENTS)
        self.assertEqual(len(observer.window), observer.size)
        self.assertTrue(all(len(t["powers"]) == 6 for t in observer.tracks))
        snapshot = copy.deepcopy(observer.__dict__)
        for start, values in ((0, [0.0]), (24_000, [math.nan]), (24_000, [[0.0]]), (-1, [0.0])):
            with self.assertRaises(ValueError):
                observer.process(start, values)
            for key, previous in snapshot.items():
                if isinstance(previous, np.ndarray):
                    np.testing.assert_array_equal(observer.__dict__[key], previous)
                else:
                    self.assertEqual(observer.__dict__[key], previous)

    def test_sample_rates_recover_the_same_component_positions(self):
        for fs in (24_000, 48_000):
            with self.subTest(fs=fs):
                time = np.arange(fs) / fs
                envelope = np.clip((time - .2) / .012, 0, 1)
                audio = .2 * np.sin(math.tau * 327.3 * time) * envelope
                rows = FrequencyEventObserver(fs).process(0, audio)
                truth = {"expected_components": [{"onset_sec": .2, "frequency_log2": math.log2(327.3)}]}
                result = summarize(rows, truth)
                self.assertTrue(result["observation_passed"], result)


if __name__ == "__main__":
    unittest.main()
