import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_grouping_cues import CASES, FS, PRELUDES_SEC, cue_reference, observe, stimulus
from evaluate_phrase_expectation import read_wav, write_wav


class GroupingCueTests(unittest.TestCase):
    def test_common_masker_is_identical_despite_different_precursors(self):
        baseline = stimulus("regular_all", 177001)[0]
        for case in CASES:
            if case.startswith("narrow_"):
                continue
            audio = stimulus(case, 177001)[0]
            np.testing.assert_array_equal(audio[FS // 2:], baseline[FS // 2:])
            if case != "regular_all":
                self.assertFalse(np.array_equal(audio[:FS // 2], baseline[:FS // 2]))

    def test_regular_individuals_do_not_require_shared_timing(self):
        _, envelope, truth = stimulus("regular_offset", 177001)
        cues = cue_reference(envelope, truth)
        self.assertLess(max(cues["interval_std_sec"]), 1e-15)
        signal_pairs = [pair["zero_lag_correlation"] for pair in cues["pairs"]
                        if 2 in (pair["a"], pair["b"])]
        self.assertEqual(len(signal_pairs), 4)
        self.assertLess(max(signal_pairs), 0.)

    def test_shared_irregularity_does_not_require_periodicity(self):
        _, envelope, truth = stimulus("coherent_jitter", 177001)
        cues = cue_reference(envelope, truth)
        self.assertGreater(min(cues["interval_std_sec"]), .003)
        self.assertTrue(all(abs(pair["zero_lag_correlation"] - 1.) < 1e-12 for pair in cues["pairs"]))

    def test_packetization_and_future_audio_do_not_change_available_evidence(self):
        audio, _, _ = stimulus("jitter_flankers", 177001)
        whole = observe(audio, FS, len(audio))
        self.assertEqual(whole, observe(audio, FS, 127))
        prefix = observe(audio[:round(.65 * FS)], FS)
        for name in whole:
            self.assertEqual(prefix[name], [row for row in whole[name] if row["available_sec"] <= .65])

    def test_input_availability_is_distinct_from_no_relation(self):
        audio, _, _ = stimulus("regular_all", 177001)
        observed = observe(audio, FS)
        self.assertTrue(np.any(audio != 0.))
        self.assertTrue(observed["components"])
        self.assertTrue(observed["relations"])
        self.assertTrue(all(row["status"] == "warming_up" for row in observed["relations"]))
        json.dumps(observed, allow_nan=False)

    def test_silence_and_sample_rate_are_explicit(self):
        audio, _, truth = stimulus("no_precursor", 177001, 48_000)
        self.assertEqual(len(audio), 38_400)
        self.assertEqual(truth["sample_rate"], 48_000)
        self.assertTrue(np.all(audio[:24_000] == 0.))
        self.assertTrue(np.any(audio[24_000:] != 0.))
        for case, seed, fs in [("absent", 1, FS), ("regular_all", -1, FS), ("regular_all", 1, 100)]:
            with self.assertRaises(ValueError):
                stimulus(case, seed, fs)

    def test_priming_preserves_pcm_and_sampling_phase(self):
        audio, _, _ = stimulus("regular_all", 177001)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            write_wav(path, audio)
            baseline = read_wav(path)
            for prelude in PRELUDES_SEC:
                samples = round(prelude * FS)
                self.assertEqual(samples % 240, 0)
                self.assertEqual(samples % 512, 0)
                write_wav(path, np.concatenate((np.zeros(samples), audio)))
                saved = read_wav(path)
                np.testing.assert_array_equal(saved[samples:], baseline)
                self.assertTrue(np.all(saved[:samples] == 0.))
                observed = observe(saved, FS)
                if prelude:
                    reports = [r for r in observed["relations"] if r["available_sec"] > prelude + .1]
                    self.assertTrue(reports)
                    self.assertTrue(all(r["status"] != "warming_up" for r in reports))


if __name__ == "__main__":
    unittest.main()
