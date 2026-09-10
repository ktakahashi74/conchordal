from pathlib import Path
import copy
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from acoustic_body import DrivenBodyPosterior
from acoustic_mixture import AcousticModelBank
from acoustic_posterior import AcousticPosterior
from acoustic_trajectory import TrajectoryFit
from evaluate_driven_acoustic_state import DrivenFit


def dense_prior(fs, frequencies, gains, intensity, precision, count):
    # Closed-form continuous covariance, independent of the Kalman recurrence.
    time = np.arange(count) / fs
    lag = abs(time[:, None] - time)
    earlier = np.minimum(time[:, None], time)
    covariance = np.eye(count)
    for f, g, d in zip(frequencies, gains, np.broadcast_to(intensity, np.shape(frequencies))):
        drive = earlier if g == 0 else np.expm1(2 * g * earlier) / (2 * g)
        covariance += np.cos(2 * np.pi * f * lag) * (
            np.exp(g * (time[:, None] + time)) / precision + d * np.exp(g * lag) * drive)
    return covariance


class DrivenBodyTests(unittest.TestCase):
    def test_missing_intervals_match_dense_conditioning_on_observed_indices_only(self):
        fs, stop, horizon = 24000, 91, 11
        indices = np.r_[np.arange(13), np.arange(37, 49), np.arange(73, 78)]
        observed = np.random.default_rng(192101).normal(0., .2, len(indices))
        for frequencies, gains, intensity in [([], [], 0.), ([317., 860.], [-17., 0.], [0., 900.]),
                                               ([500.], [-4.], 0.)]:
            precision, a0, b0 = .07, 2.3, .004
            model = DrivenBodyPosterior(fs, frequencies, gains, drive_per_sec=intensity,
                                       initial_precision=precision, variance_prior=(a0, b0))
            for index, value in zip(indices, observed):
                model.advance_to(int(index))
                model.observe(int(index), [value])
            frozen = model.forecast(5)
            digest = frozen.digest()
            retained = model.variance_shape, model.variance_scale, model.log_evidence
            model.advance_to(stop)
            self.assertEqual((model.variance_shape, model.variance_scale, model.log_evidence), retained)
            self.assertEqual(model.observed_samples, len(indices))
            self.assertEqual(model.last_observed_sample, 78)
            self.assertEqual(frozen.digest(), digest)
            prior = dense_prior(fs, frequencies, gains, intensity, precision, stop+horizon)
            train, cross = prior[np.ix_(indices, indices)], prior[stop:, indices]
            solved = np.linalg.solve(train, observed)
            a, b = a0+len(indices)/2, b0+observed@solved/2
            covariance = prior[stop:, stop:]-cross@np.linalg.solve(train, cross.T)
            prediction = model.forecast(horizon)
            np.testing.assert_allclose(prediction.mean, cross@solved, atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(prediction.diagonal_variance(), np.diag(covariance)*b/(a-1), rtol=1e-10)
            evidence = (math.lgamma(a)-math.lgamma(a0)+a0*np.log(b0)-a*np.log(b)
                        -.5*(len(indices)*np.log(2*np.pi)+np.linalg.slogdet(train)[1]))
            self.assertAlmostEqual(model.log_evidence, evidence, places=10)

    def test_missing_time_is_not_zero_evidence_and_preserves_atomic_clock(self):
        model = DrivenBodyPosterior(24000, [317.], [-3.], drive_per_sec=100.,
                                   initial_precision=.1, variance_prior=(2., .004))
        model.observe(0, np.ones(50)*.2)
        split, zero = copy.deepcopy(model), copy.deepcopy(model)
        model.advance_to(24050)
        for end in [81, 2000, 11001, 24050]:
            split.advance_to(end)
        np.testing.assert_allclose(model.state_mean, split.state_mean, atol=1e-12)
        np.testing.assert_allclose(model.state_covariance, split.state_covariance, rtol=1e-11)
        zero.observe(50, np.zeros(20))
        self.assertEqual(model.observed_samples, 50)
        self.assertEqual(zero.observed_samples, 70)
        self.assertNotEqual(model.variance_shape, zero.variance_shape)
        before = model.forecast(8).digest()
        for invalid in [24049, 25000., float('nan'), -1]:
            with self.assertRaises(ValueError):
                model.advance_to(invalid)
            self.assertEqual(model.forecast(8).digest(), before)
        model.advance_to(24050)
        self.assertEqual(model.forecast(8).digest(), before)

    def test_filter_and_student_forecast_match_dense_continuous_covariance(self):
        fs, frequencies, gains = 24000, [417., 860.], [-17., 0.]
        intensity, precision, n, horizon = [400., 1700.], .07, 51, 13
        a0, b0 = 2.3, .004
        model = DrivenBodyPosterior(fs, frequencies, gains, drive_per_sec=intensity,
                                   initial_precision=precision, variance_prior=(a0, b0))
        audio = np.random.default_rng(185181).normal(0., .2, n + horizon)
        model.observe(0, audio[:n])
        prior = dense_prior(fs, frequencies, gains, intensity, precision, n + horizon)
        train, cross = prior[:n, :n], prior[n:, :n]
        a, b = a0 + n / 2, b0 + audio[:n] @ np.linalg.solve(train, audio[:n]) / 2
        mean = cross @ np.linalg.solve(train, audio[:n])
        shape = prior[n:, n:] - cross @ np.linalg.solve(train, cross.T)
        evidence = (math.lgamma(a) - math.lgamma(a0) + a0 * np.log(b0) - a * np.log(b)
                    - .5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(train)[1]))
        self.assertAlmostEqual(model.log_evidence, evidence, places=10)
        prediction = model.forecast(horizon)
        np.testing.assert_allclose(prediction.mean, mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(prediction.diagonal_variance(), np.diag(shape) * b / (a - 1), rtol=1e-11)
        error = audio[n:] - mean
        expected = (math.lgamma(a + horizon / 2) - math.lgamma(a)
                    - .5 * (horizon * np.log(2 * np.pi * b) + np.linalg.slogdet(shape)[1])
                    - (a + horizon / 2) * np.log1p(error @ np.linalg.solve(shape, error) / (2 * b)))
        self.assertAlmostEqual(prediction.log_density(audio[n:]), expected, places=10)

    def test_passive_limit_matches_proper_amplitude_posterior(self):
        fs, start, n = 24000, 500, 80
        for frequencies in ([], [417., 863.], [417., 417.]):
            k = len(frequencies)
            gain = np.full(k, -19.)
            trajectory = TrajectoryFit(fs, float(start), np.array(frequencies), np.zeros(k),
                                       gain, np.zeros(k, complex), start, start + n)
            audio = np.random.default_rng(185187).normal(0., .1, n)
            kwargs = dict(initial_precision=.001, variance_prior=(2., .003))
            body = DrivenBodyPosterior(fs, frequencies, gain, drive_per_sec=0., start_sample=start, **kwargs)
            body.observe(start, audio)
            amplitude = AcousticPosterior(trajectory, DrivenFit(np.array([]), .001, start, start + n),
                                          audio, start, learn_variance=True, amplitude_precision=.001,
                                          variance_prior=kwargs['variance_prior'])
            self.assertAlmostEqual(body.log_evidence, amplitude.log_evidence, places=9)
            p, q = body.forecast(75), amplitude.forecast(75)
            np.testing.assert_allclose(p.mean, q.mean, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(p.diagonal_variance(), q.diagonal_variance(), rtol=1e-9)

    def test_exact_transition_and_drive_accumulation_across_sample_rates(self):
        kwargs = dict(frequency_hz=[317., 6000.], log_gain_per_sec=[-31., 0.],
                      drive_per_sec=[120., 800.], initial_precision=.1, variance_prior=(2., .01))
        slow, fast = [DrivenBodyPosterior(fs, **kwargs) for fs in (24000, 48000)]
        np.testing.assert_allclose(slow.transition, fast.transition @ fast.transition, atol=4e-16)
        np.testing.assert_allclose(slow.drive, fast.transition @ fast.drive @ fast.transition.T + fast.drive,
                                   atol=1e-17)

    def test_indistinguishable_modal_splits_remain_indistinguishable(self):
        one = DrivenBodyPosterior(24000, [417.], [-13.], drive_per_sec=1000.,
                                 initial_precision=.01, variance_prior=(2., .002))
        two = DrivenBodyPosterior(24000, [417., 417.], [-13., -13.], drive_per_sec=500.,
                                 initial_precision=.02, variance_prior=(2., .002))
        audio = np.random.default_rng(185199).normal(0., .1, 110)
        for model in (one, two):
            model.observe(0, audio)
        self.assertAlmostEqual(one.log_evidence, two.log_evidence, places=10)
        np.testing.assert_allclose(one.forecast(80).mean, two.forecast(80).mean, atol=1e-13)
        np.testing.assert_allclose(one.forecast(80).diagonal_variance(), two.forecast(80).diagonal_variance(), rtol=1e-11)

    def test_updates_are_contiguous_chunk_invariant_and_keep_issued_forecasts(self):
        model = DrivenBodyPosterior(24000, [400., 400.0001], [-1., -2.], drive_per_sec=300.,
                                   initial_precision=1e-8, variance_prior=(2., 1e-6))
        audio = np.random.default_rng(185191).normal(0., .2, 250)
        batch = copy.deepcopy(model)
        batch.observe(0, audio)
        model.observe(0, audio[:100])
        issued = model.forecast(150)
        digest = issued.digest()
        expected = issued.log_density(audio[100:])
        actual = model.observe(100, audio[100:170]) + model.observe(170, audio[170:])
        self.assertAlmostEqual(actual, expected, places=9)
        self.assertAlmostEqual(model.log_evidence, batch.log_evidence, places=9)
        np.testing.assert_array_equal(model.state_covariance, batch.state_covariance)
        np.testing.assert_array_equal(model.state_mean, batch.state_mean)
        self.assertEqual(issued.digest(), digest)
        with self.assertRaises(ValueError):
            issued.state_mean[0] = 3.
        with self.assertRaises(ValueError):
            model.observe(249, [0.])
        self.assertEqual(model.observe(250, []), 0.)

    def test_whole_trajectory_samples_and_known_action_translation(self):
        model = DrivenBodyPosterior(24000, [6000.], [-12.], drive_per_sec=5000.,
                                   initial_precision=.2, variance_prior=(8., .04))
        prediction = model.forecast(6)
        covariance = dense_prior(24000, [6000.], [-12.], 5000., .2, 6) * .04 / 7
        draws = prediction.sample(np.random.default_rng(185197), 100000)
        np.testing.assert_allclose(draws.mean(axis=0), prediction.mean, atol=.0017)
        np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=.03, atol=.00035)
        # The unknown common scale also couples orthogonal quadratures' squared errors.
        expected_cross = covariance[0, 0] * covariance[1, 1] * 7 / 6
        self.assertAlmostEqual(np.mean(draws[:, 0] ** 2 * draws[:, 1] ** 2) / expected_cross, 1., delta=.035)
        own = np.arange(6) * .04
        translated = prediction.with_known_waveform(own)
        self.assertAlmostEqual(translated.log_density(draws[0] + own), prediction.log_density(draws[0]), places=12)
        np.testing.assert_allclose(translated.diagonal_variance(), prediction.diagonal_variance())
        np.testing.assert_allclose(translated.sample(np.random.default_rng(9), 3),
                                   prediction.sample(np.random.default_rng(9), 3) + own)
        # Observation creates cross-mode covariance that trajectory draws must retain.
        coupled = DrivenBodyPosterior(24000, [413., 831.], [-5., -19.], drive_per_sec=[200., 1500.],
                                      initial_precision=.2, variance_prior=(8., .04))
        audio = np.array([.1, -.1, .2, .05])
        coupled.observe(0, audio)
        prior = dense_prior(24000, [413., 831.], [-5., -19.], [200., 1500.], .2, 10)
        cross = prior[4:, :4]
        shape = prior[4:, 4:] - cross @ np.linalg.solve(prior[:4, :4], cross.T)
        covariance = shape * coupled.variance_scale / (coupled.variance_shape - 1)
        draws = coupled.forecast(6).sample(np.random.default_rng(185201), 100000)
        np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=.04, atol=.00015)

    def test_model_bank_updates_chosen_outcomes_and_rejects_future_parameters(self):
        models = [DrivenBodyPosterior(24000, [400.], [-12.], drive_per_sec=drive,
                                      initial_precision=.01, variance_prior=(2., .002),
                                      parameters_available_sample=20) for drive in (0., 1000.)]
        with self.assertRaises(ValueError):
            models[0].forecast(1)
        past = np.sin(np.arange(20) * .1)
        for model in models:
            model.observe(0, past)
        bank = AcousticModelBank(models, [.5, .5])
        forecast = bank.forecast(20)
        target = .8 * past
        expected = forecast.log_density(target)
        actual, _ = bank.observe(20, target)
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertFalse(np.allclose(bank.log_weights, np.log(.5)))
        self.assertEqual(bank.next_sample, 40)
        for kwargs in (dict(log_gain_per_sec=[1.]), dict(frequency_hz=[12000.]),
                       dict(drive_per_sec=-1.), dict(initial_precision=0.), dict(variance_prior=(1., .1))):
            settings = dict(fs=24000, frequency_hz=[400.], log_gain_per_sec=[-1.], drive_per_sec=0.,
                            initial_precision=.01, variance_prior=(2., .01))
            settings.update(kwargs)
            with self.assertRaises(ValueError):
                DrivenBodyPosterior(**settings)


if __name__ == '__main__':
    unittest.main()
