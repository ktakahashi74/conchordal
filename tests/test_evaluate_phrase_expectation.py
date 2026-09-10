"""Causality and acoustic observability of the offline contour comparison."""

import copy
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_phrase_expectation import (
    FS, HOP, WINDOW, AcousticObserver, ContourPredictor, observe_audio, periodic_pitch, predict_notes,
    main, stimulus, synthesize, validate_campaign,
)
import evaluate_contour_observability as observability


class PhraseExpectationTests(unittest.TestCase):
    def test_observability_assay_preserves_failed_acceptance_checks(self):
        original = observability.mixture

        def missing_single(case, seed):
            audio, truth = original(case, seed)
            if case == "single":
                audio.fill(0)
            return audio, truth

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "observability"
            with patch.object(observability, "DEV_SEEDS", (1,)), \
                    patch.object(observability, "VALIDATION_SEEDS", ()), \
                    patch.object(observability, "mixture", side_effect=missing_single), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "observability checks failed"):
                    observability.run(root)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed_checks")
            failed = [c["case"] for c in manifest["cases"] if not c["passed"]]
            self.assertIn("single", failed)
            self.assertTrue(manifest["missing"][0]["passed"])
            for case in manifest["cases"]:
                for name, digest in case["sha256"].items():
                    self.assertEqual(observability.sha256(root / case["directory"] / name), digest)

    def test_longer_period_agreement_rejects_beating_without_rejecting_harmonics(self):
        t = np.arange(WINDOW) / FS
        for frequency in (173.7, 261.3, 417.9, 737.2):
            for amplitudes in ((1, 0, 0), (1, .55, .3), (0, 1, .8)):
                sound = sum(a * np.sin(math.tau * frequency * k * t)
                            for k, a in enumerate(amplitudes, 1))
                result = periodic_pitch(sound, coherence_check=True)
                self.assertIsNotNone(result)
                self.assertLess(abs(1200 * math.log2(result["frequency_hz"] / frequency)), 3)
        for frequency in (233, 220 * math.sqrt(2)):
            mixture = .2 * (np.sin(math.tau * 220 * t) + np.sin(math.tau * frequency * t))
            self.assertIsNotNone(periodic_pitch(mixture))
            self.assertIsNone(periodic_pitch(mixture, coherence_check=True))

    def test_identical_waveforms_cannot_reveal_source_count(self):
        t = np.arange(WINDOW) / FS
        fundamental = 137.4
        second = np.sin(math.tau * fundamental * 2 * t)
        third = .8 * np.sin(math.tau * fundamental * 3 * t)
        single_missing_fundamental = second + third
        independent_sources = np.stack((second, third)).sum(axis=0)
        self.assertTrue(np.array_equal(single_missing_fundamental, independent_sources))
        self.assertEqual(periodic_pitch(single_missing_fundamental, coherence_check=True),
                         periodic_pitch(independent_sources, coherence_check=True))

    def test_chunk_sizes_and_unfinished_hops_do_not_change_observations(self):
        spec = stimulus("repeat", 7)
        spec["events"], spec["duration_sec"] = spec["events"][:9], 4.2
        audio = synthesize(spec)
        for coherent in (False, True):
            whole = AcousticObserver(coherence_check=coherent).process(0, audio)
            observer = AcousticObserver(coherence_check=coherent)
            pieces = []
            cursor, index = 0, 0
            while cursor < len(audio):
                size = (1, 127, 511, 240, 819)[index % 5]
                pieces.extend(observer.process(cursor, audio[cursor:cursor + size]))
                cursor += min(size, len(audio) - cursor)
                index += 1
                self.assertLessEqual(len(observer.window), WINDOW)
                self.assertLess(observer.pending_count, HOP)
            self.assertEqual(whole, pieces)
            cutoff = round(2.867 * FS)
            prefix = AcousticObserver(coherence_check=coherent).process(0, audio[:cutoff])
            self.assertEqual(prefix, [e for e in whole if e["available_sec"] <= cutoff / FS])

    def test_missing_samples_are_not_a_rest_or_an_attack_on_reconnection(self):
        spec = stimulus("repeat", 7)
        spec["events"], spec["duration_sec"] = spec["events"][:20], 8.2
        audio = synthesize(spec)
        missing_start, resumed = round(3.51 * FS), round(5.09 * FS)
        observer = AcousticObserver(coherence_check=True)
        events = observer.process(0, audio[:missing_start])
        events += observer.process(resumed, audio[resumed:])
        gaps = [e for e in events if e["kind"] == "input_gap"]
        self.assertEqual(gaps, [{"kind": "input_gap", "available_sec": 5.09,
                                 "missing_start_sec": 3.51}])
        self.assertFalse(any(e["kind"] == "gap" and e["available_sec"] <= 5.3 for e in events))
        notes = [e for e in events if e["kind"] == "note"]
        post = next(e for e in notes if e["available_sec"] > 5.09)
        self.assertEqual(post["onset_sec"], 5.28)
        predictions = predict_notes(notes, input_gaps=gaps)
        post_prediction = next(p for p in predictions if p["onset_sec"] == 5.28)
        self.assertIsNone(post_prediction["delta_log2"])
        self.assertIsNone(post_prediction["gain_bits"])
        self.assertFalse(post_prediction["error_candidate"])
        silence = audio.copy()
        silence[missing_start:resumed] = 0
        self.assertTrue(any(g["available_sec"] < 5.09 for g in observe_audio(silence)["gaps"]))

    def test_invalid_input_does_not_advance_the_observer(self):
        observer = AcousticObserver()
        observer.process(0, np.zeros(100))
        for start, audio in ((99, np.zeros(10)), (100, np.array([math.nan])),
                             (100, np.zeros((2, 2))), (-1, np.zeros(10))):
            with self.assertRaises(ValueError):
                observer.process(start, audio)
            self.assertEqual(observer.next_sample, 100)
            self.assertEqual(observer.pending_count, 100)

    def test_initial_ignorance_does_not_mask_a_later_learned_violation(self):
        pooled, supported = ContourPredictor("pooled"), ContourPredictor("supported")
        contour = (0, .2, .5, .32, .71, .61, .12, .42)
        for index in range(51):
            pitch = 220 * 2 ** contour[index % 8]
            a, b = pooled.step(pitch), supported.step(pitch)
            self.assertEqual(a["loss_bits"], b["loss_bits"])
            self.assertEqual(a["gain_bits"], b["gain_bits"])
            if index >= 24:
                self.assertFalse(b["error_candidate"])
        a, b = pooled.step(220 * 2 ** contour[4]), supported.step(220 * 2 ** contour[4])
        self.assertEqual(a["loss_bits"], b["loss_bits"])
        self.assertEqual(a["gain_bits"], b["gain_bits"])
        self.assertFalse(a["error_candidate"])
        self.assertTrue(b["error_candidate"])
        self.assertTrue(b["prediction"]["calibration_eligible"])

    def test_unseen_context_does_not_calibrate_or_claim_a_known_pattern_violation(self):
        model = ContourPredictor()
        for index in range(55):
            model.step(220 * 2 ** ((0, .19, .43, .11)[index % 4]))
        model.step(997)
        forecast = model.forecast()
        self.assertFalse(forecast["calibration_eligible"])
        self.assertIsNone(forecast["error_threshold_bits"])
        count = len(model.losses)
        result = model.step(101)
        self.assertFalse(result["error_candidate"])
        self.assertEqual(len(model.losses), count)

    def test_current_pitch_cannot_change_its_own_prediction(self):
        a = ContourPredictor()
        for index in range(55):
            a.step(220 * 2 ** (index % 5 * .13))
        b = copy.deepcopy(a)
        left, right = a.step(237), b.step(398)
        self.assertEqual(left["prediction"], right["prediction"])
        self.assertNotEqual(left["gain_bits"], right["gain_bits"])

    def test_forecasts_are_normalized_bounded_and_transposition_invariant(self):
        a, b = ContourPredictor(), ContourPredictor()
        for i in range(350):
            f = 220 * 2 ** ((0, .18, .41, .07, .6)[i % 5])
            left, right = a.step(f), b.step(f * 2 ** .37)
            self.assertAlmostEqual(sum(left["prediction"]["weights"]) + left["prediction"]["prior_weight"], 1)
            if left["gain_bits"] is not None:
                self.assertAlmostEqual(left["gain_bits"], right["gain_bits"], places=10)
            self.assertLessEqual(len(a.history), 128)
            self.assertLessEqual(len(a.losses), 128)

    def test_unknown_pitch_does_not_invent_an_interval_across_it(self):
        model = ContourPredictor()
        model.step(220)
        model.step(330)
        count = len(model.history)
        self.assertIsNone(model.step(None)["gain_bits"])
        self.assertIsNone(model.step(440)["gain_bits"])
        self.assertEqual(len(model.history), count)

    def test_periodicity_observation_handles_harmonics_and_rejects_noise(self):
        t = np.arange(WINDOW) / FS
        for frequency in (173.7, 261.3, 417.9, 737.2):
            for amplitudes in ((1, 0, 0), (1, .55, .3), (0, 1, .8)):
                sound = sum(a * np.sin(math.tau * frequency * k * t)
                            for k, a in enumerate(amplitudes, 1))
                result = periodic_pitch(sound)
                self.assertIsNotNone(result)
                self.assertLess(abs(1200 * math.log2(result["frequency_hz"] / frequency)), 3)
        self.assertIsNone(periodic_pitch(np.zeros(WINDOW)))
        self.assertIsNone(periodic_pitch(np.random.default_rng(21).normal(0, .2, WINDOW)))

    def test_audio_prefix_has_identical_observations_without_future_samples(self):
        spec = stimulus("repeat", 7)
        spec["events"] = spec["events"][:10]
        spec["duration_sec"] = 5.5
        sound = synthesize(spec)
        cutoff = 3 * FS
        before = observe_audio(sound[:cutoff])
        complete = observe_audio(sound)
        for key in ("notes", "gaps"):
            self.assertEqual(before[key], [r for r in complete[key] if r["available_sec"] <= 3])
        self.assertEqual(predict_notes(before["notes"]),
                         predict_notes([r for r in complete["notes"] if r["available_sec"] <= 3]))

    def test_matched_controls_keep_times_levels_and_pitch_multisets(self):
        base = stimulus("repeat", 21)
        for name in ("altered", "shuffled"):
            other = stimulus(name, 21)
            for a, b in zip(base["events"], other["events"]):
                for field in ("onset_sec", "duration_sec", "amplitude", "timbre"):
                    self.assertEqual(a[field], b[field])
                if not a["evaluation"]:
                    self.assertEqual(a, b)
            for start in range(0, 80, 8):
                self.assertEqual(sorted(e["frequency_hz"] for e in base["events"][start:start + 8]),
                                 sorted(e["frequency_hz"] for e in other["events"][start:start + 8]))

    def test_pause_has_same_prefix_as_a_performance_that_ends_there(self):
        spec = stimulus("pause", 7)
        audio = synthesize(spec)
        pause_start = .6 + 7 * 8 * .36
        cutoff = round((pause_start + 1) * FS)
        stopped = observe_audio(audio[:cutoff])
        continued = observe_audio(audio)
        self.assertTrue(stopped["gaps"])
        self.assertEqual(stopped["gaps"], [g for g in continued["gaps"] if g["available_sec"] <= cutoff / FS])

    def test_campaign_validates_real_audio_and_rejects_changed_or_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            args = ["assay", "--output", str(root), "--seeds", "1", "--cases", "repeat", "altered", "shuffled"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(validate_campaign(root)["verified_cases"], 3)
            manifest_path = root / "manifest.json"
            original = manifest_path.read_text()
            manifest = json.loads(original)
            manifest["cases"].pop()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                validate_campaign(root)
            manifest_path.write_text(original)
            audio = root / "seed-1/altered/audio.wav"
            audio.write_bytes(audio.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                validate_campaign(root)

    def test_failed_validation_does_not_publish_an_audition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            args = ["assay", "--output", str(root), "--seeds", "1", "--cases", "repeat"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                with patch("evaluate_phrase_expectation.validate_campaign", side_effect=ValueError("bad audio")):
                    with self.assertRaisesRegex(ValueError, "bad audio"):
                        main()
            self.assertFalse((root / "audition.html").exists())
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "failed_validation")


if __name__ == "__main__":
    unittest.main()
