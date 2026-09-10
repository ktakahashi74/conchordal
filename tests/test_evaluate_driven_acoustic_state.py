from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_driven_acoustic_state import (DrivenFit, DrivenObserver, analyze_lattice,
                                           auditory_power_moments, fit_burg, synthesize_lattice)


class DrivenAcousticTests(unittest.TestCase):
    def test_matrix_lattice_matches_independent_columns_and_chunked_state(self):
        rng = np.random.default_rng(1049)
        for order in (0, 1, 8):
            reflection = rng.uniform(-.8, .8, order)
            for width in (0, 1, 5):
                audio, state = rng.normal(size=(41, width)), rng.normal(size=(order, width))
                whole, final = analyze_lattice(audio, reflection, state)
                for j in range(width):
                    one, last = analyze_lattice(audio[:, j], reflection, state[:, j])
                    np.testing.assert_array_equal(whole[:, j], one)
                    np.testing.assert_array_equal(final[:, j], last)
                left, middle = analyze_lattice(audio[:13], reflection, state)
                empty, unchanged = analyze_lattice(audio[:0], reflection, middle)
                self.assertEqual(empty.shape, (0, width))
                np.testing.assert_array_equal(middle, unchanged)
                right, last = analyze_lattice(audio[13:], reflection, middle)
                np.testing.assert_array_equal(np.concatenate((left, right)), whole)
                np.testing.assert_array_equal(last, final)
                reconstructed, final_synthesis = synthesize_lattice(whole, reflection, state)
                np.testing.assert_allclose(reconstructed, audio, atol=1e-12)
                for j in range(width):
                    one, last = synthesize_lattice(whole[:, j], reflection, state[:, j])
                    np.testing.assert_array_equal(reconstructed[:, j], one)
                    np.testing.assert_array_equal(final_synthesis[:, j], last)
                left, middle = synthesize_lattice(whole[:13], reflection, state)
                right, last = synthesize_lattice(whole[13:], reflection, middle)
                np.testing.assert_array_equal(np.concatenate((left, right)), reconstructed)
                np.testing.assert_array_equal(last, final_synthesis)
        for audio, state in [(np.zeros((2, 3, 1)), None), (np.zeros((2, 3)), np.zeros(2))]:
            with self.assertRaises(ValueError):
                analyze_lattice(audio, [.1, .2], state)

    def test_lattice_matches_independent_direct_filter_and_streaming_state(self):
        reflection = np.array([-.6, .42, -.27, .18])
        polynomial = np.array([1.])
        for k in reflection:
            polynomial = np.pad(polynomial, (0, 1)) + k * np.pad(polynomial[::-1], (1, 0))
        rng = np.random.default_rng(41)
        audio = rng.normal(0., .1, 1000)
        errors, state = analyze_lattice(audio, reflection)
        expected = np.convolve(audio, polynomial)[:len(audio)]
        np.testing.assert_allclose(errors, expected, atol=2e-16, rtol=1e-12)
        restored, _ = synthesize_lattice(errors, reflection, np.zeros(4))
        np.testing.assert_allclose(restored, audio, atol=5e-16, rtol=1e-12)
        a, split = analyze_lattice(audio[:317], reflection)
        b, final = analyze_lattice(audio[317:], reflection, split)
        np.testing.assert_array_equal(np.r_[a, b], errors)
        np.testing.assert_array_equal(final, state)
        np.testing.assert_array_equal(DrivenFit(reflection, .01, 0, 1000).state(audio), state)

    def test_burg_recovers_driven_second_order_response_without_true_parameters(self):
        reflection = np.array([-.7, .6])
        rng = np.random.default_rng(311)
        audio, _ = synthesize_lattice(rng.normal(0., .03, 30000), reflection, np.zeros(2))
        candidate = fit_burg(audio[1000:], 1000, orders=(0, 2), variance_floor=1e-12)[1]
        np.testing.assert_allclose(candidate.reflection, reflection, atol=.012, rtol=0.)
        self.assertAlmostEqual(candidate.drive_variance, .03 ** 2, delta=.03 ** 2 * .04)

    def test_pcm_covariance_and_density_match_joint_gaussian(self):
        reflection = np.array([-.3, .4])
        fit = DrivenFit(reflection, .02, 0, 100)
        past = np.linspace(-.1, .1, 100)
        mean, impulse = fit.forecast(past, 20)
        transform = np.zeros((20, 20))
        for k in range(20):
            transform[k:, k] = impulse[:20-k]
        covariance = .02 * transform @ transform.T
        truth = mean + transform @ np.random.default_rng(7).normal(0., np.sqrt(.02), 20)
        delta = truth - mean
        independent = -.5 * (20 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
                              + delta @ np.linalg.solve(covariance, delta))
        self.assertAlmostEqual(fit.log_density(past, truth), independent, places=11)

    def test_expected_auditory_power_retains_uncertain_drive(self):
        fs, count = 24000, 240
        observer = AuditoryEnvelopeObserver(fs, np.log2([250., 500., 1000.]), stride_samples=24)
        observed = np.random.default_rng(8).normal(0., .02, 1200)
        observer.process(0, observed)
        fit = DrivenFit(np.array([-.4, .3]), .003, 0, 1200)
        mean, impulse = fit.forecast(observed, count)
        before = observer.state.copy()
        mean_only, expected = auditory_power_moments(observer, mean, impulse, fit.drive_variance)
        np.testing.assert_array_equal(observer.state, before)
        self.assertTrue(np.all(expected >= mean_only))
        # Trace of an independently constructed linear innovation response.
        basis_power = np.zeros_like(expected)
        for index in range(count):
            drive = np.zeros(count)
            drive[index] = 1.
            signal, _ = synthesize_lattice(drive, fit.reflection, np.zeros(2))
            fresh = AuditoryEnvelopeObserver(fs, observer.centers_log2, stride_samples=24)
            rows = fresh.process(0, signal)
            basis_power += np.array([r['envelope_scan'] for r in rows]) ** 2 * fit.drive_variance
        np.testing.assert_allclose(expected, mean_only + basis_power, rtol=3e-14, atol=1e-17)

    def test_chunking_prefix_freeze_gap_and_retained_responses(self):
        fs = 24000
        t = np.arange(6000) / fs
        audio = .1 * np.cos(2 * np.pi * 417 * t)
        kwargs = dict(sample_quantum=1/32768, orders=(0, 4, 8), capacity=3)
        whole = DrivenObserver(fs, **kwargs).process(0, audio)
        observer = DrivenObserver(fs, **kwargs)
        parts = []
        for start in range(0, len(audio), 179):
            parts.extend(observer.process(start, audio[start:start+179]))
        self.assertEqual([r['mean_sha256'] for r in whole], [r['mean_sha256'] for r in parts])
        self.assertEqual([r['impulse_sha256'] for r in whole], [r['impulse_sha256'] for r in parts])
        prefix = DrivenObserver(fs, **kwargs).process(0, audio[:2400])
        self.assertEqual([r['mean_sha256'] for r in prefix], [r['mean_sha256'] for r in whole[:2]])
        self.assertTrue(any('retained' in r['candidate_kinds'] for r in parts[1:]))
        frozen = parts[0]['mean'].copy()
        observer.process(len(audio), np.ones(1200))
        np.testing.assert_array_equal(parts[0]['mean'], frozen)
        resumed = observer.process(9000, audio)
        fresh = DrivenObserver(fs, **kwargs).process(9000, audio)
        self.assertEqual(resumed[0]['kind'], 'input_gap')
        self.assertEqual([r['mean_sha256'] for r in resumed[1:]], [r['mean_sha256'] for r in fresh])

    def test_silence_precision_floor_and_invalid_input(self):
        observer = DrivenObserver(24000, sample_quantum=1/32768)
        row = observer.process(0, np.zeros(1200))[0]
        np.testing.assert_array_equal(row['mean'], np.zeros(240))
        self.assertEqual(row['drive_variance'], (1/32768) ** 2 / 12)
        for start, audio in [(0, [0.]), (1200, [np.nan]), (1200, [[0.]])]:
            with self.assertRaises(ValueError):
                observer.process(start, audio)
        for args in [dict(sample_quantum=0), dict(sample_quantum=.001, orders=(4,)),
                     dict(sample_quantum=.001, capacity=0)]:
            with self.assertRaises(ValueError):
                DrivenObserver(24000, **args)


if __name__ == '__main__':
    unittest.main()
