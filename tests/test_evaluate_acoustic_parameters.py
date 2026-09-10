import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_parameters import chain_diagnostics, parameter_forecast, sample_passive_parameters
from evaluate_driven_acoustic_state import DrivenFit


def quadrature(audio, fs, horizon, resolution):
    # Independent midpoint integration with dense two-coefficient normal equations.
    unit = (np.arange(resolution)+.5)/resolution
    u, v = np.meshgrid(unit, unit)
    f, g = 300*2**(u.ravel()*np.log2(800/300)), -120*v.ravel()
    t = np.arange(len(audio)+horizon)/fs
    basis = np.exp(g[:, None]*t)*np.exp(2j*np.pi*f[:, None]*t)
    basis = np.stack((basis.real, basis.imag), axis=2)
    train, future = basis[:, :len(audio)], basis[:, len(audio):]
    gram = np.einsum('sni,snj->sij', train, train)+.1*np.eye(2)
    inverse = np.linalg.inv(gram)
    rhs = np.einsum('sni,n->si', train, audio)
    coefficients = np.einsum('sij,sj->si', inverse, rhs)
    errors = audio-np.einsum('sni,si->sn', train, coefficients)
    a = 2.+len(audio)/2
    b = .001+(np.sum(errors**2, axis=1)+.1*np.sum(coefficients**2, axis=1))/2
    log_mass = -.5*np.linalg.slogdet(gram)[1]-a*np.log(b)
    weights = np.exp(log_mass-np.logaddexp.reduce(log_mass))
    means = np.einsum('sni,si->sn', future, coefficients)
    variances = b[:, None]/(a-1)*(1+np.einsum('sni,sij,snj->sn', future, inverse, future))
    mean = weights@means
    variance = weights@(variances+means**2)-mean**2
    return weights@np.column_stack((f, g)), mean, variance


class AcousticParameterTests(unittest.TestCase):
    def test_continuous_prediction_matches_independent_quadrature(self):
        fs, n, horizon = 24000, 64, 36
        t = np.arange(n)/fs
        audio = .06*np.exp(-46*t)*np.cos(2*np.pi*513*t+.7)
        audio += np.random.default_rng(1091).normal(0., .027, n)
        initial = np.array([[[340., -15.]], [[470., -40.]], [[620., -80.]], [[760., -110.]]])
        bank, point, report = sample_passive_parameters(audio, 0, fs,
            DrivenFit(np.array([]), .001, 0, n), initial, frequency_range=(300, 800), max_decay=120,
            amplitude_precision=.1, variance_prior=(2., .001), rng=np.random.default_rng(1093),
            warmup=512, draws=512, thin=2)
        expected, mean, variance = quadrature(audio, fs, horizon, 160)
        coarse = quadrature(audio, fs, horizon, 80)
        np.testing.assert_allclose(expected, coarse[0], atol=.02)
        np.testing.assert_allclose(mean, coarse[1], atol=2e-6)
        np.testing.assert_allclose(variance, coarse[2], atol=2e-7)
        np.testing.assert_allclose(report['parameters'].mean(axis=(0, 1))[0], expected, atol=3.)
        prediction = bank.forecast(horizon)
        self.assertEqual(parameter_forecast(bank, horizon).digest(), prediction.digest())
        np.testing.assert_allclose(prediction.mean, mean, atol=.002)
        np.testing.assert_allclose(prediction.diagonal_variance(), variance, rtol=.07, atol=2e-5)
        self.assertTrue(all(d['rhat'] is not None and d['rhat'] < 1.03 for d in report['diagnostics']))
        self.assertTrue(all(d['bulk_ess'] > 100 for d in report['diagnostics']))
        # A point-parameter forecast has the same amplitude/variance prior and target.
        self.assertEqual(prediction.issued_sample, point.forecast(horizon).issued_sample)
        self.assertGreater(np.max(abs(prediction.diagonal_variance()-point.forecast(horizon).diagonal_variance())), 1e-5)

    def test_multiple_modes_are_exchangeable_and_updates_do_not_alias(self):
        fs, n = 24000, 32
        audio = np.random.default_rng(1103).normal(0., .02, n)
        initial = np.array([[[440., -30.], [440., -30.]], [[460., -40.], [510., -10.]]])
        args = dict(frequency_range=(300, 900), max_decay=120, amplitude_precision=.1,
                    variance_prior=(2., .001), warmup=0, draws=12, thin=1)
        bank, point, report = sample_passive_parameters(audio, 0, fs,
            DrivenFit(np.array([-.3]), .001, 0, n), initial, rng=np.random.default_rng(1109), **args)
        self.assertEqual(report['parameters'].shape, (2, 12, 2, 2))
        self.assertEqual(parameter_forecast(bank, 30).digest(), bank.forecast(30).digest())
        self.assertTrue(np.isfinite(report['log_density']).all())
        self.assertTrue(np.all(report['parameters'][:, :, :, 1] <= 0))
        np.testing.assert_array_equal(report['frozen_step_size'], np.full((2, 2, 2), .03))
        control = copy.deepcopy(bank)
        previous = bank.forecast(4).digest()
        update = np.array([.01, -.03, .02, .04])
        joint = bank.observe(n, update)[0]
        chunks = control.observe(n, update[:1])[0]+control.observe(n+1, update[1:])[0]
        self.assertAlmostEqual(joint, chunks, places=10)
        self.assertEqual(bank.next_sample, n+4)
        self.assertEqual(point.next_sample, n)
        np.testing.assert_allclose(bank.forecast(3).mean, control.forecast(3).mean, atol=1e-12)
        self.assertNotEqual(bank.forecast(4).digest(), previous)
        # Permuting identical-prior modes leaves marginal likelihood and forecast invariant.
        model = copy.deepcopy(bank.models[0])
        fit = model.trajectory
        from acoustic_posterior import AcousticPosterior
        from acoustic_trajectory import TrajectoryFit
        permuted = TrajectoryFit(fs, 0., fit.frequency_hz[::-1], np.zeros(2),
                                 fit.log_gain_per_sec[::-1], np.zeros(2, complex), 0, n)
        other = AcousticPosterior(permuted, model.residual, np.r_[audio, update], 0,
                                  learn_variance=True, amplitude_precision=.1, variance_prior=(2., .001))
        self.assertAlmostEqual(other.log_evidence, model.log_evidence, places=10)
        np.testing.assert_allclose(other.forecast(6).mean, model.forecast(6).mean, atol=1e-12)

    def test_diagnostics_detect_stuck_chains_and_scale_disagreement(self):
        rng = np.random.default_rng(1117)
        iid = rng.normal(size=(4, 1000, 1))
        good = chain_diagnostics(iid)[0]
        self.assertLess(good['rhat'], 1.01)
        self.assertGreater(good['bulk_ess'], 2000)
        shifted = iid+np.arange(4)[:, None, None]*3
        self.assertGreater(chain_diagnostics(shifted)[0]['rhat'], 1.5)
        scaled = iid.copy(); scaled[0] *= .01
        self.assertGreater(chain_diagnostics(scaled)[0]['rhat'], 1.1)
        stuck = np.broadcast_to(np.arange(4)[:, None, None], (4, 1000, 1))
        bad = chain_diagnostics(stuck)[0]['rhat']
        self.assertTrue(bad is None or bad > 10)
        with self.assertRaises(ValueError): chain_diagnostics(iid[:1])


if __name__ == '__main__':
    unittest.main()
