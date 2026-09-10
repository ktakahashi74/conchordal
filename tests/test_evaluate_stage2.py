"""Prevent false causal and musical-closure conclusions in stage 2 assays."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_stage2 as stage2


class Stage2Tests(unittest.TestCase):
    def test_campaign_seed_removal_is_explicit_and_rejects_ambiguous_sources(self):
        source = "// Assay\nseed(21);\nwait(2.0);\n"
        self.assertEqual(stage2.campaign_seed_source(source), "// Assay\nwait(2.0);\n")
        for invalid in ("wait(2.0);", source + "seed(42);\n", "seed(user_seed);\n"):
            with self.assertRaises(ValueError):
                stage2.campaign_seed_source(invalid)

    def test_accent_pair_preserves_the_source_except_for_the_opt_in(self):
        source = (Path(__file__).resolve().parents[1] / "samples/08_murmuration.rhai").read_text()
        for amount in (0, 1):
            modified = stage2.measure_source(source, amount)
            self.assertEqual(modified.replace(f".measure_accent({amount:.1f})", ""), source)
        for source in (source.replace(".entrained()", ".metric()", 1), source + ".measure_accent(1.0)"):
            with self.assertRaises(ValueError):
                stage2.measure_source(source, 1)

    def test_fixed_frequency_is_not_the_moving_maximum(self):
        rows = [{"time_sec": i, "fmin_hz": 110, "bins_per_octave": 1, "n_bins": 3,
                 "state_scan": h, "raw_score_scan": h, "eff_score_scan": h}
                for i, h in enumerate(([.9, .4, .1], [.1, .2, .9]))]
        self.assertEqual([r["h"] for r in stage2.fixed_region(rows, 220)], [.4, .2])
        with self.assertRaises(ValueError):
            stage2.fixed_region(rows, 880)
        rows[1]["bins_per_octave"] = 2
        with self.assertRaises(ValueError):
            stage2.fixed_region(rows, 220)

    def test_autonomous_return_requires_observed_departure_and_same_voice(self):
        n = 73
        scans = [{"time_sec": t, "fmin_hz": 55, "bins_per_octave": 24, "n_bins": n,
                  "state_scan": [h]*n, "raw_score_scan": [1]*n, "eff_score_scan": [1-h]*n}
                 for t, h in enumerate((.1, .5, .6, .4, .3, .2, .2))]
        onsets = [{"time_sec": t, "population_id": 3, "voice_id": 7, "freq_hz": f}
                  for t, f in ((1, 220), (2, 220*2**(.2)), (3, 220*2**(.2)), (6, 220))]
        rows = {"habituation_scan": scans, "onset": onsets}
        events = stage2.revisit_candidates(rows)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["recovered_before_return"])
        respawn = copy.deepcopy(rows)
        respawn["onset"][-1]["voice_id"] = 8
        self.assertEqual(stage2.revisit_candidates(respawn), [])
        missing = copy.deepcopy(rows)
        missing["onset"] = [onsets[0], onsets[-1]]
        self.assertEqual(stage2.revisit_candidates(missing), [])
        persistent = copy.deepcopy(rows)
        for r in persistent["habituation_scan"]:
            r["state_scan"] = [.6]*n
        self.assertFalse(stage2.revisit_candidates(persistent)[0]["recovered_before_return"])

    def test_resolution_threshold_needs_prior_tension_and_a_full_second(self):
        def rows(values):
            return [{"time_sec": i*.5, "tension_level": x} for i, x in enumerate(values)]
        self.assertIsNone(stage2.resolution_time(rows([0]*8)))
        self.assertIsNone(stage2.resolution_time(rows([.2]*8)))
        self.assertIsNone(stage2.resolution_time(rows([.2, .2, 0, .2, 0, 0])))
        self.assertEqual(stage2.resolution_time(rows([.2, .2, 0, 0, 0])), .5)


if __name__ == "__main__":
    unittest.main()
