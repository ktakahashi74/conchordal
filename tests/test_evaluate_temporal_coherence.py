import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_temporal_coherence import TemporalCoherenceObserver
from evaluate_auditory_envelope import AuditoryEnvelopeObserver


def model(**options):
    parameters = dict(centers_log2=np.log2([110., 220., 440.]), pairs=[[0, 1], [0, 2]],
                      rates_hz=[4., 8.], step_sec=.005, report_steps=1,
                      positive_tail_cycles=4., lookahead_cycles=1., hilbert_padding=8)
    return TemporalCoherenceObserver(**(parameters | options))


class TemporalCoherenceTests(unittest.TestCase):
    def test_fixed_peripheral_alignment_retains_intentional_phase_offsets(self):
        centers = np.log2(110.) + np.arange(8) * .75
        front = AuditoryEnvelopeObserver(24000, centers, stride_samples=24)
        radius = np.abs(front.pole)
        step, rate = .001, 32.
        baseband = ((1 - radius) / (1 - radius * np.exp(-2j * np.pi * rate / 24000))) ** 4
        for intentional in (0., .006):
            observer = model(centers_log2=centers, pairs=[[0, 7]], rates_hz=[2., 4., 8., 16., 32.],
                             step_sec=step, report_steps=5, peripheral_sample_rate=24000)
            shifts = np.zeros(8)
            shifts[-1] = intentional
            scores, expected_scores = [], []
            carrier = baseband * np.exp(-2j * np.pi * rate * shifts)
            positive = np.array([np.exp(-2j * np.pi * rate * np.arange(len(h)) * step) @ h
                                 for h in observer.kernels])
            negative = np.array([np.exp(2j * np.pi * rate * np.arange(len(h)) * step) @ h
                                 for h in observer.kernels])
            for i in range(4000):
                time = (i + 1) * step
                value = .6 + .2 * np.abs(baseband) * np.cos(2 * np.pi * rate * (time - shifts) + np.angle(baseband))
                row = observer.observe(time, value)
                if row is not None and row["observation_sec"] >= 3.:
                    scores.append(row["total_normalized_inner_product"][0])
                    expected = .1 * (positive * carrier * np.exp(2j * np.pi * rate * time)
                                     + negative * np.conj(carrier) * np.exp(-2j * np.pi * rate * time))
                    np.testing.assert_allclose(row["response"], expected, rtol=1e-9, atol=1e-15)
                    cross = np.real(np.sum(expected[:, 0] * np.conj(expected[:, -1])))
                    denominator = np.sqrt(np.sum(np.abs(expected[:, 0]) ** 2) * np.sum(np.abs(expected[:, -1]) ** 2))
                    expected_scores.append(cross / denominator)
            np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-11)
            # The finite Hilbert/DC approximation is not an ideal all-pass response.
            self.assertAlmostEqual(float(np.mean(scores)), np.cos(2 * np.pi * rate * intentional), delta=.01)

    def test_calibrated_stream_matches_its_bandwise_convolution(self):
        observer = model(peripheral_sample_rate=24000)
        values = np.random.default_rng(191017).uniform(.1, 1., (300, 3))
        rows = [observer.observe((i + 1) * .005, value) for i, value in enumerate(values)]
        response = np.array([r["response"] for r in rows])
        for rate, kernel in enumerate(observer.kernels):
            valid = np.arange(len(values)) + 1 >= len(kernel)
            for band in range(3):
                expected = np.convolve(values[:, band], kernel[:, band], mode="full")[:len(values)]
                np.testing.assert_allclose(response[valid, rate, band], expected[valid], rtol=1e-11, atol=1e-15)
        observer.observe(3., None)
        for i in range(300):
            row = observer.observe(3. + (i + 1) * .005, [.1, .2, .3])
        np.testing.assert_array_equal(row["modulation_power"], 0.)

    def test_alignment_requires_a_valid_declared_peripheral_rate(self):
        for fs in (True, 0, 12000, 24000.5):
            with self.assertRaises(ValueError):
                model(peripheral_sample_rate=fs)

    def test_alignment_does_not_mix_or_fit_other_channels(self):
        left, right = model(peripheral_sample_rate=24000), model(peripheral_sample_rate=24000)
        for i, value in enumerate(np.random.default_rng(191041).uniform(.1, 1., (300, 3))):
            modified = value.copy()
            modified[1] *= 3.
            a = left.observe((i + 1) * .005, value)
            b = right.observe((i + 1) * .005, modified)
            np.testing.assert_array_equal(a["response"][:, [0, 2]], b["response"][:, [0, 2]])
        self.assertFalse(np.array_equal(a["response"][:, 1], b["response"][:, 1]))

    def test_finite_pair_matches_analytic_gamma_sine_response_in_sampled_passband(self):
        observer = model(rates_hz=[2., 4., 8., 16., 32.], step_sec=.001, report_steps=5)
        ratio = np.linspace(.25, 2., 141)
        expected = 2 / 1j * ((3.5 + 2j * np.pi * (ratio - 1)) ** -3
                            - (3.5 + 2j * np.pi * (ratio + 1)) ** -3)
        peak = np.max(np.abs(expected))
        for rate, kernel in zip(observer.rates_hz, observer.kernels):
            offsets = (np.arange(len(kernel)) - observer.delay_steps) * observer.step_sec
            positive = np.exp(-2j * np.pi * rate * ratio[:, None] * offsets) @ kernel
            negative = np.exp(2j * np.pi * rate * ratio[:, None] * offsets) @ kernel
            # This is a numerical passband budget, not a perceptual discrimination threshold.
            self.assertLess(np.max(np.abs(positive - expected)) / peak, .01)
            self.assertLess(np.max(np.abs(negative)) / peak, .01)

    def test_streaming_matches_direct_convolution_and_six_phase_gram(self):
        observer = model()
        values = np.random.default_rng(17031).uniform(.1, 1., (500, 3))
        rows = [observer.observe((i + 1) * .005, v) for i, v in enumerate(values)]
        response = np.array([r["response"] for r in rows])
        for rate, kernel in enumerate(observer.kernels):
            expected = np.array([np.convolve(values[:, b], kernel, mode="full")[:len(values)]
                                 for b in range(3)]).T
            valid = np.arange(len(values)) + 1 >= len(kernel)
            np.testing.assert_allclose(response[valid, rate], expected[valid], rtol=1e-11, atol=1e-15)
            self.assertTrue(np.all(np.isnan(response[~valid, rate])))
        for row in rows[-10:]:
            z = row["response"]
            phase = np.arange(6) * np.pi / 3
            projected = z.real[..., None] * np.cos(phase) + z.imag[..., None] * np.sin(phase)
            explicit = np.einsum("rfp,rgp->rfg", projected, projected)
            np.testing.assert_allclose(row["cross_power"], explicit[:, [0, 0], [1, 2]], rtol=1e-12, atol=1e-18)
            self.assertAlmostEqual(row["available_sec"] - row["observation_sec"], observer.delay_sec)

    def test_constant_and_silence_have_zero_power_and_undefined_normalization(self):
        for constant in (0., .2):
            observer = model()
            for i in range(300):
                row = observer.observe((i + 1) * .005, np.full(3, constant))
            np.testing.assert_array_equal(row["modulation_power"], 0.)
            np.testing.assert_array_equal(row["cross_power"], 0.)
            self.assertTrue(np.all(np.isnan(row["normalized_inner_product"])))
            self.assertTrue(np.all(np.isnan(row["total_normalized_inner_product"])))

    def test_gain_scaling_keeps_power_and_normalized_view_distinct(self):
        a, b = model(), model()
        for i, value in enumerate(np.random.default_rng(719).uniform(.1, 1., (300, 3))):
            left = a.observe((i + 1) * .005, value)
            right = b.observe((i + 1) * .005, 3 * value)
        np.testing.assert_allclose(right["modulation_power"], 9 * left["modulation_power"], rtol=1e-12)
        np.testing.assert_allclose(right["normalized_inner_product"], left["normalized_inner_product"], atol=1e-12)

    def test_gap_refills_each_rate_without_inventing_prehistory(self):
        observer = model()
        for i in range(300):
            observer.observe((i + 1) * .005, [.1, .2, .3])
        self.assertEqual(observer.observe(3., None)["kind"], "input_gap")
        for i in range(300):
            row = observer.observe(3. + (i + 1) * .005, [.2, .3, .4])
            np.testing.assert_array_equal(row["rate_available"], i + 1 >= observer.support_steps)
        np.testing.assert_array_equal(row["modulation_power"], 0.)
        self.assertEqual(row["segment_start_sec"], 3.)

    def test_bad_input_does_not_mutate_state_or_returned_observation(self):
        observer = model()
        row = observer.observe(.005, [.1, .2, .3])
        before = copy.deepcopy(observer.__dict__)
        for endpoint, value in ((.01, [-1., 0., 0.]), (.01, [np.nan, 0., 0.]), (.02, [.1, .2, .3])):
            with self.assertRaises(ValueError):
                observer.observe(endpoint, value)
        self.assertEqual(observer.time_sec, before["time_sec"])
        np.testing.assert_array_equal(observer.differences, before["differences"])
        frozen = copy.deepcopy(row)
        observer.observe(.01, [.9, .1, .6])
        for key in frozen:
            np.testing.assert_array_equal(row[key], frozen[key])

    def test_parameters_reject_aliasing_and_ambiguous_grids(self):
        for options in (dict(report_steps=20), dict(step_sec=.1), dict(rates_hz=[8., 4.]),
                        dict(pairs=[[0, 0]]), dict(centers_log2=[1., 2., 4.]),
                        dict(hilbert_padding=1), dict(lookahead_cycles=0.)):
            with self.assertRaises(ValueError):
                model(**options)


if __name__ == "__main__":
    unittest.main()
