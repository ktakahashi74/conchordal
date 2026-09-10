from pathlib import Path
from dataclasses import replace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from acoustic_trajectory import (TrajectoryFit, TrajectoryObserver,
                                fit_trajectory_candidates, update_trajectory_state)


class AcousticTrajectoryTests(unittest.TestCase):
    def test_passive_fit_recovers_damped_frequencies_and_unseen_decay(self):
        for fs in (24000, 48000):
            t = np.arange(round(.08*fs))/fs
            audio = .1*np.exp(-17*t)*np.cos(2*np.pi*417*t+.3)+.07*np.exp(-53*t)*np.cos(2*np.pi*863*t+1.1)
            n = round(.04*fs)
            fit = fit_trajectory_candidates(audio[:n], 0, fs, counts=(0, 1, 2), glide=False, passive=True)[-1]
            order = np.argsort(fit.frequency_hz)
            np.testing.assert_allclose(fit.frequency_hz[order], [417., 863.], atol=1e-7, rtol=0.)
            np.testing.assert_allclose(fit.log_gain_per_sec[order], [-17., -53.], atol=1e-6, rtol=0.)
            np.testing.assert_allclose(fit.predict(n, len(audio)-n), audio[n:], atol=1e-8, rtol=0.)

    def test_passivity_is_fitted_with_resolved_amplitudes_not_posthoc_pole_clipping(self):
        fs = 24000
        t = np.arange(960)/fs
        audio = .05*np.exp(30*t)*np.cos(2*np.pi*417*t+.3)
        unrestricted = fit_trajectory_candidates(audio, 0, fs, counts=(0, 1), glide=False)[-1]
        constrained = fit_trajectory_candidates(audio, 0, fs, counts=(0, 1), glide=False, passive=True)[-1]
        clipped = replace(unrestricted, log_gain_per_sec=np.minimum(unrestricted.log_gain_per_sec, 0.))
        reference = np.column_stack([np.cos(2*np.pi*417*t), np.sin(2*np.pi*417*t)])
        valid_error = np.mean((audio-reference@np.linalg.lstsq(reference, audio, rcond=None)[0])**2)
        error = np.mean((audio-constrained.predict(0, len(t)))**2)
        self.assertGreater(unrestricted.log_gain_per_sec[0], 29.99)
        self.assertEqual(constrained.log_gain_per_sec[0], 0.)
        self.assertLessEqual(error, valid_error+1e-10)
        self.assertLess(error, np.mean((audio-clipped.predict(0, len(t)))**2))

    def test_prediction_matches_independent_recursive_oscillators(self):
        fs = 24000
        fit = TrajectoryFit(fs, 0., np.array([417., 863.]), np.array([1700., -900.]),
                            np.array([-3., -.7]), np.array([.1+.04j, -.05+.08j]), 0, 960)
        state = fit.amplitude.copy()
        expected = []
        for n in range(1920):
            expected.append(float(state.real.sum()))
            midpoint_frequency = fit.frequency_hz + fit.rate_hz_per_sec * (n+.5)/fs
            state *= np.exp(fit.log_gain_per_sec/fs + 2j*np.pi*midpoint_frequency/fs)
        np.testing.assert_allclose(fit.predict(1200, 720), expected[1200:], atol=2e-14, rtol=0.)

    def test_unknown_frequencies_and_glide_rates_predict_unseen_audio(self):
        for fs in (24000, 48000):
            t = np.arange(round(.07*fs))/fs
            audio = (.1*np.exp(-3*t)*np.cos(2*np.pi*(413*t+850*t*t)+.3)
                     + .07*np.exp(-7*t)*np.cos(2*np.pi*(823*t-450*t*t)+1.1))
            n = round(.04*fs)
            moving = fit_trajectory_candidates(audio[:n], 0, fs, counts=(0, 1, 2))[-1]
            stationary = fit_trajectory_candidates(audio[:n], 0, fs,
                                                    counts=(0, 1, 2), glide=False)[-1]
            order = np.argsort(moving.frequency_hz)
            expected_frequency = np.array([413., 823.]) + (n-1)/(2*fs)*np.array([1700., -900.])
            np.testing.assert_allclose(moving.frequency_hz[order], expected_frequency, atol=1e-7, rtol=0.)
            np.testing.assert_allclose(moving.rate_hz_per_sec[order], [1700., -900.], atol=1e-5, rtol=0.)
            np.testing.assert_allclose(moving.predict(n, len(audio)-n), audio[n:], atol=1e-8, rtol=0.)
            self.assertGreater(np.mean((stationary.predict(n, len(audio)-n)-audio[n:])**2), 1e-5)

    def test_state_update_keeps_trajectory_and_existing_fit_immutable(self):
        fs = 24000
        fit = TrajectoryFit(fs, 0., np.array([417.]), np.array([1700.]),
                            np.array([-3.]), np.array([.1+.04j]), 0, 960)
        changed = TrajectoryFit(fs, 0., fit.frequency_hz, fit.rate_hz_per_sec,
                                fit.log_gain_per_sec, np.array([.06-.05j]), 1200, 2160)
        before = fit.predict(1200, 1200)
        update = update_trajectory_state(changed.predict(1200, 960), 1200, fit)
        np.testing.assert_allclose(update.predict(2160, 240), changed.predict(2160, 240), atol=2e-14)
        np.testing.assert_array_equal(fit.predict(1200, 1200), before)
        np.testing.assert_array_equal(update.rate_hz_per_sec, fit.rate_hz_per_sec)
        self.assertGreater(np.mean((before[:960]-changed.predict(1200, 960))**2), .001)

    def test_nested_glide_fit_keeps_the_stationary_training_solution_available(self):
        fs=24000
        t=np.arange(960)/fs
        audio=(.1*np.cos(2*np.pi*(413*t+850*t*t)+.3)
               +.07*np.cos(2*np.pi*(493*t-450*t*t)+1.1)
               +np.random.default_rng(97).normal(0.,.003,len(t)))
        stationary=fit_trajectory_candidates(audio,0,fs,counts=(0,1,2,4),glide=False)
        moving=fit_trajectory_candidates(audio,0,fs,counts=(0,1,2,4),initial=stationary)
        for base,extension in zip(stationary,moving):
            baseline_error=np.mean((base.predict(0,len(t))-audio)**2)
            extended_error=np.mean((extension.predict(0,len(t))-audio)**2)
            self.assertLessEqual(extended_error,baseline_error+1e-14)
        with self.assertRaises(ValueError):
            fit_trajectory_candidates(audio,2400,fs,counts=(0,1),initial=stationary)

    def test_packetization_prefix_forecast_freeze_and_gap(self):
        fs = 24000
        t = np.arange(6000)/fs
        audio = .1*np.cos(2*np.pi*(413*t+850*t*t)+.3)
        options = dict(counts=(0, 1, 2))
        whole = TrajectoryObserver(fs, **options).process(0, audio)
        observer = TrajectoryObserver(fs, **options)
        split = []
        for start in range(0, len(audio), 179):
            split.extend(observer.process(start, audio[start:start+179]))
        self.assertEqual([r['prediction_sha256'] for r in whole],
                         [r['prediction_sha256'] for r in split])
        prefix = TrajectoryObserver(fs, **options).process(0, audio[:2400])
        self.assertEqual([r['prediction_sha256'] for r in prefix],
                         [r['prediction_sha256'] for r in whole[:2]])
        original = split[0]['predictions'].copy()
        observer.process(len(audio), np.ones(1200))
        np.testing.assert_array_equal(split[0]['predictions'], original)
        resumed = observer.process(9000, audio)
        fresh = TrajectoryObserver(fs, **options).process(9000, audio)
        self.assertEqual(resumed[0]['kind'], 'input_gap')
        self.assertEqual([r['prediction_sha256'] for r in resumed[1:]],
                         [r['prediction_sha256'] for r in fresh])

    def test_zero_noise_invalid_input_and_frequency_domain(self):
        observer = TrajectoryObserver(24000, counts=(0, 1, 2))
        row = observer.process(0, np.zeros(1200))[0]
        np.testing.assert_array_equal(row['predictions'], np.zeros((4, 240)))
        self.assertEqual(row['available'], [False, False, True, True])
        noise = observer.process(1200, np.random.default_rng(47).normal(0, .03, 1200))
        self.assertTrue(np.all(np.isfinite(noise[0]['predictions'])))
        with self.assertRaises(ValueError):
            observer.process(1200, np.ones(1))
        with self.assertRaises(ValueError):
            TrajectoryObserver(24000, counts=(1,))
        crossing = TrajectoryFit(24000, 0., np.array([10.]), np.array([-1000.]),
                                  np.array([0.]), np.array([.1+0j]), 0, 120)
        with self.assertRaises(ArithmeticError):
            crossing.predict(240, 240)


if __name__ == '__main__':
    unittest.main()
