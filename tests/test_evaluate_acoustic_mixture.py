import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_mixture import AcousticMixtureForecast, AcousticModelBank
from acoustic_posterior import AcousticForecast


class FixedPrediction:
    def __init__(self, mean, variance, start=0):
        self.value, self.variance, self.next_sample = mean, variance, start

    def forecast(self, count):
        return AcousticForecast(self.next_sample, np.full(count, self.value),
            np.r_[1., np.zeros(count-1)], np.empty((count, 0)), np.array([]), self.variance)

    def observe(self, start, audio):
        assert start == self.next_sample
        result = self.forecast(len(audio)).log_density(audio)
        self.next_sample += len(audio)
        return result


class AcousticMixtureTests(unittest.TestCase):
    def test_joint_density_and_total_variance_preserve_competing_trajectories(self):
        bank = AcousticModelBank([FixedPrediction(-1., .01), FixedPrediction(1., .01)], [.5, .5])
        forecast = bank.forecast(4)
        np.testing.assert_allclose(forecast.mean, np.zeros(4), atol=1e-15)
        np.testing.assert_allclose(forecast.diagonal_variance(), np.full(4, 1.01), atol=1e-15)
        target = np.array([-1., -1., 1., 1.])
        # A joint mixture has one model per trajectory, unlike a product of marginal mixtures.
        expected = np.logaddexp(*[np.log(.5)+c.log_density(target) for c in forecast.components])
        self.assertAlmostEqual(forecast.log_density(target), expected, places=12)
        per_sample = sum(np.logaddexp(*[np.log(.5)+m.forecast(1).log_density([x]) for m in bank.models]) for x in target)
        self.assertGreater(per_sample-expected, 100.)
        draws = forecast.sample(np.random.default_rng(185101), 40000)
        covariance = np.cov(draws, rowvar=False, bias=True)
        np.testing.assert_allclose(covariance, np.ones((4, 4))+.01*np.eye(4), atol=.005)
        shifted = forecast.with_known_waveform(np.arange(4)/10)
        np.testing.assert_allclose(shifted.sample(np.random.default_rng(185101), 40000), draws+np.arange(4)/10, atol=1e-15)

    def test_log_weights_can_revive_and_issued_forecasts_are_immutable(self):
        bank = AcousticModelBank([FixedPrediction(-3., .01), FixedPrediction(3., .01)], [1., 1.])
        prediction = bank.forecast(20)
        original = prediction.digest()
        first = np.full(20, 3.)
        expected = prediction.log_density(first)
        actual, scores = bank.observe(0, first)
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(np.exp(bank.log_weights[0]), 0.)
        self.assertTrue(np.isfinite(bank.log_weights[0]))
        bank.observe(20, np.full(20, -6.))
        self.assertGreater(bank.log_weights[0], bank.log_weights[1])
        self.assertEqual(prediction.digest(), original)
        before = bank.forecast(5).digest()
        with self.assertRaises(ValueError): bank.observe(41, np.zeros(5))
        self.assertEqual(bank.forecast(5).digest(), before)

    def test_splitting_prior_mass_and_chunked_updates_do_not_change_predictions(self):
        bank = AcousticModelBank([FixedPrediction(-.1, .1), FixedPrediction(.1, .4)], [.3, .7])
        duplicate = AcousticModelBank([FixedPrediction(-.1, .1), FixedPrediction(-.1, .1),
                                       FixedPrediction(.1, .4)], [.15, .15, .7])
        chunked = copy.deepcopy(bank)
        audio = np.array([.3, -.2, .4, .1, .7])
        density = bank.observe(0, audio)[0]
        split_density = chunked.observe(0, audio[:2])[0]+chunked.observe(2, audio[2:])[0]
        duplicate_density = duplicate.observe(0, audio)[0]
        self.assertAlmostEqual(density, split_density, places=12)
        self.assertAlmostEqual(density, duplicate_density, places=12)
        for other in (chunked, duplicate):
            np.testing.assert_allclose(bank.forecast(4).mean, other.forecast(4).mean, atol=1e-15)
            np.testing.assert_allclose(bank.forecast(4).diagonal_variance(), other.forecast(4).diagonal_variance(), atol=1e-15)


if __name__ == '__main__':
    unittest.main()
