from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_modal_continuation import ModalObserver, fit_candidates, select_candidate


def ringing(fs, duration, frequencies=(277., 421., 633., 911.)):
    t = np.arange(round(fs * duration)) / fs
    return sum(.1 * np.exp(-np.log(1000.) * t / decay) * np.cos(2 * np.pi * f * t + phase)
               for f, decay, phase in zip(frequencies, (.4, 1., 2.5, 5.), (.3, 1.2, 2.1, .7)))


class ModalContinuationTests(unittest.TestCase):
    def test_unknown_rank_past_selection_recovers_modes_and_unseen_continuation(self):
        for fs in (24000, 48000):
            audio = ringing(fs, .07)
            past = audio[:round(.05 * fs)]
            model, _, selection = select_candidate(
                past, 0, validation_samples=round(.01 * fs), pencil_size=96,
                ranks=(0, 2, 4, 8, 12, 16))
            self.assertEqual(selection["selected_rank"], 8)
            positive = model.poles[model.poles.imag > 0]
            order = np.argsort(np.angle(positive))
            np.testing.assert_allclose(np.angle(positive[order]) * fs / (2 * np.pi),
                                       [277., 421., 633., 911.], atol=2e-6, rtol=0.)
            np.testing.assert_allclose(-np.log(1000.) / (fs * np.log(np.abs(positive[order]))),
                                       [.4, 1., 2.5, 5.], atol=2e-6, rtol=0.)
            np.testing.assert_allclose(model.predict(len(past), len(audio) - len(past)),
                                       audio[len(past):], atol=1e-8, rtol=0.)

    def test_reexcitation_keeps_old_modes_and_new_frequency_needs_residual(self):
        fs = 24000
        original = ringing(fs, .38)
        for new_frequency in (False, True):
            audio = original.copy()
            change = round(.3 * fs)
            if new_frequency:
                t = np.arange(len(audio) - change) / fs
                audio[change:] += .15 * np.exp(-t * 2.) * np.sin(2 * np.pi * 1337 * t)
            else:
                audio[change:] += .7 * ringing(fs, (len(audio) - change) / fs)
            rows = ModalObserver(fs).process(0, audio[:round(.35 * fs)])
            row = rows[-1]
            self.assertEqual(row["issued_sample"], round(.35 * fs))
            truth = audio[row["issued_sample"]:row["target_end_sample"]]
            errors = np.mean((row["predictions"] - truth) ** 2, axis=1)
            if new_frequency:
                self.assertGreater(errors[1], 1e-4)
                self.assertLess(errors[2], errors[1] * 1e-4)
            else:
                self.assertGreater(errors[0], 1e-4)
                self.assertLess(errors[1], 1e-15)

    def test_chunking_future_and_freezing(self):
        fs = 24000
        audio = ringing(fs, .12)
        whole = ModalObserver(fs).process(0, audio)
        observer = ModalObserver(fs)
        pieces = []
        for start in range(0, len(audio), 173):
            pieces.extend(observer.process(start, audio[start:start + 173]))
        self.assertEqual([r["forecast_sha256"] for r in whole],
                         [r["forecast_sha256"] for r in pieces])
        prefix = ModalObserver(fs).process(0, audio[:2400])
        self.assertEqual([r["forecast_sha256"] for r in prefix],
                         [r["forecast_sha256"] for r in whole])
        snapshot = pieces[0]["predictions"].copy()
        observer.process(len(audio), np.ones(2400))
        np.testing.assert_array_equal(pieces[0]["predictions"], snapshot)
        with self.assertRaises(ValueError):
            pieces[0]["predictions"][3, 0] = 10

    def test_gap_clears_model_and_does_not_create_silence(self):
        fs = 24000
        audio = ringing(fs, .1)
        observer = ModalObserver(fs)
        observer.process(0, audio)
        self.assertEqual(observer.process(2400, []), [])
        resumed = observer.process(5000, audio)
        fresh = ModalObserver(fs).process(5000, audio)
        self.assertEqual(resumed[0], {"kind": "input_gap", "missing_start_sample": 2400,
                                      "available_sample": 5000})
        self.assertEqual([r["forecast_sha256"] for r in resumed[1:]],
                         [r["forecast_sha256"] for r in fresh])
        self.assertEqual(resumed[1]["available"].tolist(), [False, False, False, True])

    def test_silence_is_zero_candidate_and_growing_modes_are_not_clamped(self):
        fs = 24000
        rows = ModalObserver(fs).process(0, np.zeros(2400))
        self.assertEqual(rows[-1]["selection"]["selected_rank"], 0)
        np.testing.assert_array_equal(rows[-1]["predictions"], np.zeros((4, 240)))
        t = np.arange(960) / fs
        growing = .1 * np.exp(4 * t) * np.cos(2 * np.pi * 431 * t)
        model = fit_candidates(growing, 0, pencil_size=96, ranks=(2,))[0]
        np.testing.assert_allclose(np.log(np.abs(model.poles)) * fs, [4., 4.], atol=1e-8)

    def test_validation_noise_does_not_replace_the_validated_fit(self):
        fs = 24000
        t = np.arange(1440) / fs
        clean = .1 * np.exp(-4 * t) * np.cos(2 * np.pi * 431 * t)
        past = clean[:1200].copy()
        past[960:] += np.random.default_rng(21).normal(0, 1e-4, 240)
        model, _, _ = select_candidate(past, 0, validation_samples=240,
                                        pencil_size=96, ranks=(0, 2, 4, 8))
        np.testing.assert_allclose(model.predict(1200, 240), clean[1200:], atol=1e-10, rtol=0.)

    def test_invalid_input_and_out_of_order_rejected(self):
        for kwargs in ({"horizon_sec": 0.}, {"ranks": (2,)}, {"step_sec": .001},
                       {"fit_sec": .001}, {"ranks": (0, -1)}):
            with self.assertRaises(ValueError):
                ModalObserver(24000, **kwargs)
        observer = ModalObserver(24000)
        for start, audio in ((0, [np.nan]), (-1, [0.]), (0, [[0.]])):
            with self.assertRaises(ValueError):
                observer.process(start, audio)
        observer.process(0, [0.])
        with self.assertRaises(ValueError):
            observer.process(0, [0.])


if __name__ == "__main__":
    unittest.main()
