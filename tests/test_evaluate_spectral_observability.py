import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_spectral_observability import (
    period_pair, sequence, write_pcm, read_pcm, observe_pcm,
)


def recurrence_windows(audio, sample_rate):
    low = mid = 0.
    alpha_low = 1 - np.exp(-2 * np.pi * 200 / sample_rate)
    alpha_mid = 1 - np.exp(-2 * np.pi * 3000 / sample_rate)
    windows, states = [], []
    for chunk in np.asarray(audio).reshape(-1, round(sample_rate / 100)):
        energy = np.zeros(3)
        for value in chunk:
            low += alpha_low * (value - low)
            mid += alpha_mid * (value - mid)
            energy += np.array([low * low, (mid-low)**2, (value-mid)**2])
        windows.append(energy / len(chunk))
        states.append([low, mid])
    return np.array(windows), np.array(states)


class SpectralObservabilityTests(unittest.TestCase):
    def test_transfer_construction_matches_independent_filter_recurrence(self):
        for fs in (24000, 48000):
            for seed in (4301, 4313, 4327):
                pair = period_pair(fs, seed)
                measured, states = recurrence_windows(pair["periods"].reshape(-1), fs)
                np.testing.assert_allclose(measured, pair["expected_energy"][:, :3],
                                           rtol=2e-10, atol=1e-15)
                np.testing.assert_allclose(states, 0, atol=2e-14)
                np.testing.assert_allclose(pair["periods"][:, [0, -1]], 0, atol=2e-14)
                np.testing.assert_allclose(pair["expected_energy"][0], pair["expected_energy"][1],
                                           rtol=2e-10, atol=1e-15)
                self.assertTrue(np.all(pair["group_power"][0] * pair["group_power"][1] == 0))

    def test_arbitrary_order_keeps_ideal_energy_and_zero_boundary_states(self):
        for fs in (24000, 48000):
            pair = period_pair(fs, 4313)
            choices = np.random.default_rng(33).integers(0, 2, size=70)
            measured, states = recurrence_windows(pair["periods"][choices].reshape(-1), fs)
            expected = np.broadcast_to(pair["expected_energy"][0, :3], measured.shape)
            np.testing.assert_allclose(measured, expected, rtol=2e-10, atol=1e-15)
            np.testing.assert_allclose(states, 0, atol=2e-14)

    def test_equal_energy_signals_have_disjoint_spectra_and_different_orders(self):
        pair = period_pair(24000, 4301)
        power = np.abs(np.fft.rfft(pair["periods"], axis=1)) ** 2
        active = power > np.max(power) * 1e-12
        self.assertFalse(np.any(active[0] & active[1]))
        self.assertGreater(np.count_nonzero(active[0]), 0)
        self.assertGreater(np.count_nonzero(active[1]), 0)
        a, order_a = sequence(pair, "alternating")
        b, order_b = sequence(pair, "paired")
        self.assertEqual(np.bincount(order_a).tolist(), np.bincount(order_b).tolist())
        self.assertFalse(np.array_equal(order_a, order_b))
        self.assertGreater(np.mean((a-b)**2), 1e-5)

    def test_saved_pcm_preserves_rate_and_quantization_is_measured_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            for fs in (24000, 48000):
                audio, _ = sequence(period_pair(fs, 4327), "alternating", .8)
                path = Path(directory) / f"{fs}.wav"
                write_pcm(path, audio, fs)
                read_rate, decoded = read_pcm(path)
                self.assertEqual(read_rate, fs)
                np.testing.assert_array_equal(decoded, np.rint(audio * 32768) / 32768)
                self.assertLessEqual(np.max(np.abs(decoded - audio)), .5 / 32768)
                self.assertTrue(np.any(decoded != audio))

    def test_fine_observer_preserves_spectral_difference_and_causal_prefix(self):
        pair = period_pair(24000, 4301)
        first = np.tile(pair["periods"][0], 60)
        changed = first.copy()
        changed[40 * len(pair["periods"][0]):] = np.tile(pair["periods"][1], 20)
        a, ac = observe_pcm(first, 24000, packet_size=701)
        b, bc = observe_pcm(changed, 24000, packet_size=4093)
        self.assertEqual([r for r in a if r["available_sec"] <= .4],
                         [r for r in b if r["available_sec"] <= .4])
        self.assertEqual([r for r in ac if r["available_sec"] <= .4],
                         [r for r in bc if r["available_sec"] <= .4])
        self.assertEqual(len(a[-1]["envelope_scan"]), 113)
        before = np.array(a[-1]["envelope_scan"]) ** 2
        after = np.array(b[-1]["envelope_scan"]) ** 2
        distance = np.linalg.norm(np.sqrt(before / before.sum()) - np.sqrt(after / after.sum())) / np.sqrt(2)
        self.assertGreater(distance, .1)


if __name__ == "__main__":
    unittest.main()
