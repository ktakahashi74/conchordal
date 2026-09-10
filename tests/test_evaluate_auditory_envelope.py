import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_auditory_envelope import AuditoryEnvelopeObserver


FS = 24_000
CENTERS = np.log2([250., 500., 1000., 2000., 4000.])


class AuditoryEnvelopeTests(unittest.TestCase):
    def test_impulse_matches_closed_form_fourth_order_response(self):
        observer = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24)
        signal = np.zeros(2400)
        signal[0] = 1.
        rows = observer.process(0, signal)
        n = np.arange(len(signal))[:, None]
        radius = np.exp(-2 * np.pi * (24.7 + np.exp2(CENTERS) / 9.265) * 16 / (5 * np.pi) / FS)
        envelope = 2 * (1 - radius) ** 4 * ((n + 1) * (n + 2) * (n + 3) / 6) * radius ** n
        expected = np.sqrt(np.mean(envelope.reshape(-1, 24, len(CENTERS)) ** 2, axis=1))
        np.testing.assert_allclose([r["envelope_scan"] for r in rows], expected, rtol=2e-12, atol=1e-15)

    def test_center_gain_and_auditory_band_selectivity_at_two_sample_rates(self):
        for fs in [24_000, 48_000]:
            times = np.arange(round(.3 * fs)) / fs
            for frequency in [1000., 1000. + (24.7 + 1000 / 9.265)]:
                observer = AuditoryEnvelopeObserver(fs, np.log2([1000.]), stride_samples=round(fs / 1000))
                rows = observer.process(0, .1 * np.cos(2 * np.pi * frequency * times))
                measured = np.mean([r["envelope_scan"][0] for r in rows[-100:]])
                if frequency == 1000.:
                    self.assertAlmostEqual(measured, .1, delta=.0001)
                else:
                    self.assertLess(measured, .03)
                    self.assertGreater(measured, .02)

    def test_packets_and_future_input_do_not_change_evidence(self):
        audio = np.random.default_rng(9).normal(0, .01, 1234)
        whole = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24).process(0, audio)
        streamed = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24)
        pieces = []
        for start in range(0, len(audio), 37):
            pieces.extend(streamed.process(start, audio[start:start + 37]))
        self.assertEqual(whole, pieces)
        prefix = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24).process(0, audio[:613])
        self.assertEqual(prefix, [r for r in whole if r["available_sec"] <= 613 / FS])
        self.assertEqual(streamed.pending, len(audio) % 24)

    def test_gap_does_not_splice_history_or_synthesize_silence(self):
        observer = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24)
        observer.process(0, np.ones(101))
        before_empty = observer.next_sample
        self.assertEqual(observer.process(101, []), [])
        self.assertEqual(observer.next_sample, before_empty)
        resumed = observer.process(1000, np.ones(120))
        fresh = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24).process(1000, np.ones(120))
        self.assertEqual(resumed[0]["kind"], "input_gap")
        self.assertEqual(resumed[0]["missing_start_sec"], 101 / FS)
        self.assertEqual(resumed[1:], fresh)
        self.assertEqual(fresh[0]["observation_age_sec"], 24 / FS)

    def test_observed_silence_is_an_envelope_observation(self):
        observer = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24)
        rows = observer.process(0, np.zeros(240))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["kind"] == "auditory_envelope" for r in rows))
        self.assertTrue(all(r["envelope_scan"] == [0.] * len(CENTERS) for r in rows))
        self.assertEqual(rows[-1]["observation_age_sec"], .01)

    def test_level_scaling_preserves_envelope_shape(self):
        audio = np.random.default_rng(7).normal(0, .1, 2400)
        a = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24).process(0, audio)
        b = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24).process(0, audio * .25)
        np.testing.assert_array_equal(np.array([r["envelope_scan"] for r in a]) * .25,
                                      [r["envelope_scan"] for r in b])

    def test_invalid_grid_and_audio_are_rejected(self):
        for centers in [[], [math.nan], [-1.], [math.log2(FS)], [2., 4., 3.], [2., 4., 5.]]:
            with self.assertRaises(ValueError):
                AuditoryEnvelopeObserver(FS, centers, stride_samples=24)
        observer = AuditoryEnvelopeObserver(FS, CENTERS, stride_samples=24)
        for start, audio in [(-1, [0.]), (0, [math.nan]), (0, [[1.]])]:
            with self.assertRaises(ValueError):
                observer.process(start, audio)
        observer.process(0, [0.])
        with self.assertRaises(ValueError):
            observer.process(0, [0.])


if __name__ == "__main__":
    unittest.main()
