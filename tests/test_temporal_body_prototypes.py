import copy
import itertools
import math
import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_temporal_body_prototypes import configuration, fit


def corpus(points):
    return dict(acquisition=dict(sample_rate=48000, nfft=2048, hop_size=512),
                accent_scales=dict(means=[0., 0.], deviations=[1., 1.]),
                records=[dict(record_id=f"record-{i:03}", descriptor=dict(raw_values=[value]*6, mask=63),
                              recipe=dict(test_only=True)) for i, value in enumerate(points)])


class BodyPrototypes(unittest.TestCase):
    def test_pam_matches_exhaustive_small_objective_and_is_order_invariant(self):
        data = corpus([0., 1., 2., 8., 9., 10., 30., 31.])
        for count in (1, 2, 3, 4, 8):
            model = fit(data, count)
            points = [row["descriptor"]["raw_values"][0] for row in data["records"]]
            mean = sum(points) / len(points)
            sd = math.sqrt(sum((x-mean)**2 for x in points) / len(points))
            objective = lambda indices: sum(min(abs(x-points[j]) / sd for j in indices) for x in points)
            optimum = min(objective(indices) for indices in itertools.combinations(range(len(points)), count))
            self.assertAlmostEqual(model["objective"], optimum, places=12)
            self.assertEqual(model["initial_medoids"][0], "record-000")
            self.assertTrue(all(a-b > 1e-6 for a, b in zip(model["objective_history"], model["objective_history"][1:])))
            self.assertEqual(model["stop_reason"], "improvement_at_most_1e-6")
            reversed_data = copy.deepcopy(data)
            reversed_data["records"].reverse()
            reversed_model = fit(reversed_data, count)
            self.assertEqual(model["medoids"], reversed_model["medoids"])
            self.assertEqual(model["assignments"], reversed_model["assignments"])
            self.assertEqual(model["objective_history"], reversed_model["objective_history"])

    def test_masks_scales_and_disconnected_records_remain_explicit(self):
        data = corpus([2., 8.])
        data["records"][0]["descriptor"]["mask"] = 3
        data["records"][1]["descriptor"]["mask"] = 60
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            fit(data, 1)
        model = fit(data, 2)
        self.assertEqual(model["means"], [2., 2., 8., 8., 8., 8.])
        self.assertEqual(model["deviations"], [0.]*6)
        self.assertEqual(model["coordinate_counts"], [1]*6)
        self.assertEqual([row["common_coordinates"] for row in model["assignments"]], [2, 4])
        self.assertTrue(all(row["compatible"] for row in model["assignments"]))
        data["records"][1]["descriptor"]["mask"] = 3
        with self.assertRaisesRegex(ValueError, "every scale coordinate"):
            fit(data, 2)

    def test_frozen_model_configuration_and_provenance_change(self):
        data = corpus([0., 1., 2., 3.])
        model = fit(data, 2)
        parsed = tomllib.loads(configuration(model))
        cfg = parsed["temporal_body_prototypes"]
        self.assertEqual(cfg["model_version"], model["model_version"])
        self.assertEqual(cfg["means"], parsed["temporal_body"]["means"])
        self.assertEqual(cfg["deviations"], parsed["temporal_body"]["deviations"])
        self.assertEqual(cfg["medoids"], model["medoids"])
        self.assertEqual(fit(data, 2)["model_version"], model["model_version"])
        data["records"][0]["recipe"]["source_changed"] = True
        self.assertNotEqual(fit(data, 2)["model_version"], model["model_version"])
        with self.assertRaisesRegex(ValueError, "offline sensitivity"):
            configuration(fit(corpus(range(16)), 16))

    def test_invalid_or_empty_records_are_not_silently_discarded(self):
        data = corpus([0., 1.])
        for count in (0, 3, 17):
            with self.assertRaises(ValueError):
                fit(data, count)
        duplicate = copy.deepcopy(data)
        duplicate["records"][1]["record_id"] = duplicate["records"][0]["record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            fit(duplicate, 1)
        for mask, value in [(0, 0.), (64, 0.), (63, math.nan), (63, math.inf)]:
            bad = copy.deepcopy(data)
            bad["records"][0]["descriptor"] = dict(mask=mask, raw_values=[value]*6)
            with self.assertRaises(ValueError):
                fit(bad, 1)


if __name__ == "__main__":
    unittest.main()
