"""Independent analytic and enumerated checks for M0 reference conventions."""

import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import temporal_cognition_reference as ref


class ObservationTests(unittest.TestCase):
    def test_identical_stereo_preserves_time_energy(self):
        mono = ref.mono_observation([[1], [-0.5], None, [0]], 1)
        stereo = ref.mono_observation([[1, 1], [-0.5, -0.5], None, [0, 0]], 2)
        self.assertEqual(mono, stereo)
        self.assertAlmostEqual(mono["energy"], 1.25 / 3)
        self.assertEqual(mono["coverage"], 0.75)

    def test_antiphase_and_asymmetric_channels(self):
        self.assertEqual(ref.mono_observation([[1, -1]], 2)["energy"], 0)
        self.assertEqual(ref.mono_observation([[1, 0]], 2)["energy"], 0.25)

    def test_missing_silence_and_invalid_observed_samples_are_distinct(self):
        self.assertIsNone(ref.mono_observation([None, None], 1)["energy"])
        self.assertEqual(ref.mono_observation([[0], [0]], 1)["energy"], 0)
        self.assertIsNone(ref.mono_observation([], 1)["energy"])
        for samples in [[[math.nan]], [[math.inf]], [[]]]:
            with self.assertRaises(ValueError):
                ref.mono_observation(samples, 1)


class MemoryTests(unittest.TestCase):
    def test_known_odds_and_zero_strength_ineligibility(self):
        available = ref.episode_log_availability(3, 0, 0, 20, 4)
        self.assertAlmostEqual(ref.recognition_probability([available], [0], 0), 0.75)
        self.assertEqual(ref.recognition_probability([-math.inf], [100], -100), 0)
        self.assertEqual(ref.recognition_probability([], [], 0), 0)

    def test_long_wait_uses_log_floor_without_false_zero(self):
        available = ref.episode_log_availability(1, 1e9, 0, 2, 1)
        self.assertTrue(math.isfinite(available))
        self.assertAlmostEqual(ref.recognition_probability([available], [0], math.log(1e-300)), 0.5)

    def test_independent_time_and_interference_and_unknown_interval(self):
        baseline = ref.episode_log_availability(2, 0, 0, 20, 4)
        timed = ref.episode_log_availability(2, 20, 0, 20, 4)
        interfered = ref.episode_log_availability(2, 0, 4, 20, 4)
        self.assertAlmostEqual(baseline - timed, 1)
        self.assertAlmostEqual(timed, interfered)
        lower = ref.recognition_probability([ref.episode_log_availability(2, 20, 6, 20, 4)], [0], 0)
        upper = ref.recognition_probability([timed], [0], 0)
        self.assertLess(lower, upper)
        self.assertGreater(lower, 0)


class SectionInputTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "cumulative": [0.2] * 39, "recent": [0.5] * 39,
            "section_start": 1.0, "epoch_start": 0.0, "observed_time": 3.0,
            "known_intervals": [(0, 1.5), (1.25, 2), (2.5, 3)],
            "match_scores": [1.5, 2.0, -1.0], "match_support_end": 2.75,
            "means": [0.1] * 82, "standard_deviations": [0.5] * 82,
        }

    def test_order_raw_contrast_support_union_and_missing_imputation(self):
        self.inputs["cumulative"][3] = None
        self.inputs["recent"][4] = None
        out = ref.section_head_covariates(**self.inputs)
        self.assertTrue(out["supported"])
        self.assertEqual(len(out["covariates"]), 82)
        self.assertAlmostEqual(out["raw"][39], 0.3)
        self.assertAlmostEqual(out["covariates"][39], 0.4)
        self.assertAlmostEqual(out["raw"][78], math.log(3), places=14)
        self.assertEqual(out["raw"][79:], [0.25, 2.0, 0.5])
        self.assertEqual(out["known_duration"], 1.5)
        for index in [3, 42, 43]:
            self.assertFalse(out["valid"][index])
            self.assertEqual(out["covariates"][index], 0)
        self.assertTrue(out["valid"][4])
        # Splitting/duplicating support intervals cannot fabricate more observations.
        self.inputs["known_intervals"] = [(0, 1.5), (1.5, 2), (2.5, 2.75), (2.75, 3), (1.25, 1.9)]
        self.assertEqual(out, ref.section_head_covariates(**self.inputs))

    def test_forecast_advances_only_duration_with_frozen_context(self):
        current = ref.section_head_covariates(**self.inputs)
        later = ref.section_head_covariates(**self.inputs, forecast_time=10)
        for index in range(82):
            if index == 78:
                self.assertAlmostEqual(later["raw"][index], math.log(10))
            else:
                self.assertEqual(current["covariates"][index], later["covariates"][index])
        self.assertEqual(current["valid"], later["valid"])
        self.assertEqual(current["window"], later["window"])

    def test_known_inactivity_empty_start_clipping_and_stale_query(self):
        self.inputs["section_start"] = 2.5
        self.inputs["cumulative"] = self.inputs["recent"] = [None] * 39
        out = ref.section_head_covariates(**self.inputs)
        self.assertEqual(out["missing_fraction"], 0)
        self.assertFalse(any(out["valid"][:78]))
        self.inputs["match_support_end"] = 2.5
        out = ref.section_head_covariates(**self.inputs)
        self.assertEqual(out["valid"][80:], [False, False])
        self.inputs["match_support_end"] = 2.75
        self.inputs["match_scores"] = [1.0]
        self.assertEqual(ref.section_head_covariates(**self.inputs)["valid"][80:], [True, False])
        self.inputs["section_start"] = None
        self.assertFalse(ref.section_head_covariates(**self.inputs)["supported"])
        self.inputs["section_start"] = 3
        self.assertFalse(ref.section_head_covariates(**self.inputs)["supported"])

    def test_invalid_shapes_scales_and_future_support_are_rejected(self):
        for key, value in [("cumulative", [0] * 38), ("standard_deviations", [1] * 81),
                           ("standard_deviations", [-1] * 82), ("known_intervals", [(2, 4)]),
                           ("match_support_end", 3.1), ("recent", [math.nan] * 39)]:
            with self.subTest(key=key, value=value):
                data = dict(self.inputs, **{key: value})
                with self.assertRaises(ValueError):
                    ref.section_head_covariates(**data)


class FirstEventTests(unittest.TestCase):
    @staticmethod
    def path(hazards, types, retention=None, weight=1.0, known=True):
        return {"weight": weight, "known_at_start": known,
                "hazard_increments": hazards, "exit_types": types,
                "history_retention": retention if retention is not None else [1.0] * len(hazards)}

    @staticmethod
    def enumerate_joint(contexts, hops, kinds):
        # Enumerate terminal histories independently, then judge the earliest loss/event.
        expected = [[0.0] * (kinds + 2) for _ in range(hops + 1)]
        for row in expected:
            row[-1] = 1 - sum(context["weight"] for context in contexts)
        for context in contexts:
            choices = []
            for group in context["groups"]:
                outcomes = [(1 - sum(path["weight"] for path in group["paths"]), -1, "loss", None)]
                for path in group["paths"]:
                    mass = path["weight"]
                    if not path["known_at_start"]:
                        outcomes.append((mass, -1, "loss", None))
                        continue
                    for hop in range(hops):
                        retained = path["history_retention"][hop]
                        outcomes.append((mass * (1 - retained), hop, "loss", None))
                        mass *= retained
                        stays = math.exp(-path["hazard_increments"][hop])
                        for kind, p in enumerate(path["exit_types"][hop]):
                            outcomes.append((mass * (1 - stays) * p, hop, "event", kind))
                        mass *= stays
                    outcomes.append((mass, hops, "stay", None))
                choices.append([(mass, hop, state, kind, group["handle"])
                                for mass, hop, state, kind in outcomes if mass > 0])
            for joint in itertools.product(*choices):
                mass = context["weight"] * math.prod(item[0] for item in joint)
                for endpoint in range(hops + 1):
                    prior = [item for item in joint if item[1] < endpoint and item[2] != "stay"]
                    if not prior:
                        destination = kinds
                    else:
                        first = min(prior, key=lambda item: (item[1], item[2] != "loss", item[4]))
                        destination = kinds + 1 if first[2] == "loss" else first[3]
                    expected[endpoint][destination] += mass
        return expected

    def test_factorized_map_matches_enumerated_context_path_and_loss_histories(self):
        contexts = []
        for ci, weight in enumerate([0.55, 0.35]):
            groups = []
            for gi in range(3):
                paths = [self.path([0.1 + gi * 0.2, 0.7, 0.3], [[0.8, 0.2]] * 3,
                                   [0.9, 0.65, 1.0], weight=0.6),
                         self.path([1.0, 0.0, 0.4], [[0.1, 0.9]] * 3,
                                   [1.0, 0.95, 0.8], weight=0.3, known=not (ci == 1 and gi == 1))]
                groups.append({"handle": 9 - gi * 3, "paths": paths})
            contexts.append({"weight": weight, "groups": groups})
        out = ref.mixture_first_event(contexts, 3, 2, "heard_prefix_annotation", 8)
        expected = self.enumerate_joint(contexts, 3, 2)
        for endpoint, row in enumerate(expected):
            self.assertAlmostEqual(sum(row), 1, places=12)
            self.assertAlmostEqual(out["survival"][endpoint], row[-2], places=12)
            self.assertAlmostEqual(out["unresolved"][endpoint], row[-1], places=12)
            for kind in range(2):
                self.assertAlmostEqual(sum(x[kind] for x in out["event_mass"][:endpoint]), row[kind], places=12)
            self.assertAlmostEqual(out["cdf"][endpoint] + out["survival"][endpoint]
                                   + out["mid_unresolved"][endpoint], out["initial_known"], places=12)

    def test_ties_use_stable_handle_and_same_hop_loss_precedes_all_exits(self):
        groups = [{"handle": 8, "paths": [self.path([1000], [[1, 0]])]},
                  {"handle": 2, "paths": [self.path([1000], [[0, 1]])]}]
        contexts = [{"weight": 1, "groups": groups}]
        out = ref.mixture_first_event(contexts, 1, 2, "issued_forecast", 0)
        self.assertEqual(out["type_distribution"], [0, 1, 0, 0])
        contexts[0]["groups"] = list(reversed(groups))
        self.assertEqual(out, ref.mixture_first_event(contexts, 1, 2, "issued_forecast", 0))
        groups[0]["paths"][0]["history_retention"] = [0]
        self.assertEqual(ref.mixture_first_event(contexts, 1, 2, "issued_forecast", 0)["type_distribution"],
                         [0, 0, 0, 1])

    def test_resolved_event_survives_later_loss_but_unknown_history_cannot_reenter(self):
        paths = [self.path([1000, 1000], [[1, 0]] * 2, [1, 0], weight=0.4),
                 self.path([0, 1000], [[0, 1]] * 2, [0, 1], weight=0.3),
                 self.path([1000, 1000], [[0, 1]] * 2, weight=0.2, known=False)]
        out = ref.mixture_first_event([{"weight": 1, "groups": [{"handle": 0, "paths": paths}]}],
                                     2, 2, "heard_prefix_annotation", 8)
        for got, expected in zip(out["type_distribution"], [0.4, 0, 0, 0.6]):
            self.assertAlmostEqual(got, expected)
        self.assertAlmostEqual(out["initial_known"], 0.7)
        self.assertAlmostEqual(out["mid_unresolved"][-1], 0.3)

    def test_single_two_four_eight_groups_match_exponential_survival(self):
        for groups_count in (1, 2, 4, 8):
            for multiplier in (0.25, 1, 3):
                groups = [{"handle": gi, "paths": [self.path([0.02 * multiplier] * 10, [[1]] * 10)]}
                          for gi in range(groups_count)]
                out = ref.mixture_first_event([{"weight": 1, "groups": groups}], 10, 1, "issued_forecast", 0)
                for hop in range(11):
                    self.assertAlmostEqual(out["survival"][hop], math.exp(-groups_count * hop * 0.02 * multiplier))
                    self.assertAlmostEqual(out["cdf"][hop], 1 - out["survival"][hop])
                    self.assertAlmostEqual(out["unresolved"][hop], 0)

    def test_inactive_empty_and_distinct_scoring_snapshots(self):
        active = {"handle": 0, "paths": [self.path([1], [[1]])]}
        inactive = {"handle": 0, "paths": [self.path([0], [[1]])]}
        contexts = [{"weight": 0.8, "groups": [active]}, {"weight": 0.2, "groups": [inactive]}]
        issued = ref.mixture_first_event(contexts, 1, 1, "issued_forecast", 0)
        later = copy.deepcopy(contexts)
        later[0]["weight"], later[1]["weight"] = 0.2, 0.8
        heard = ref.mixture_first_event(later, 1, 1, "heard_prefix_annotation", 8)
        self.assertAlmostEqual(issued["cdf"][-1], 0.8 * (1 - math.exp(-1)))
        self.assertAlmostEqual(heard["cdf"][-1], 0.2 * (1 - math.exp(-1)))
        self.assertNotEqual(issued["snapshot"], heard["snapshot"])
        self.assertEqual(issued, ref.mixture_first_event(contexts, 1, 1, "issued_forecast", 0))
        self.assertEqual(ref.mixture_first_event([{"weight": 1, "groups": []}], 1, 1,
                                                "issued_forecast", 0)["type_distribution"], [0, 1, 0])
        self.assertEqual(ref.mixture_first_event([], 1, 1, "issued_forecast", 0)["type_distribution"], [0, 0, 1])

    def test_invalid_snapshot_weights_hazards_and_types_are_rejected(self):
        valid = [{"weight": 1, "groups": [{"handle": 0, "paths": [self.path([0.1], [[0.5, 0.5]])]}]}]
        for field, value in [("weight", 1.1), ("hazard_increments", [-1]),
                             ("history_retention", [1.1]), ("exit_types", [[0.4, 0.4]]),
                             ("known_at_start", 1), ("hazard_increments", [math.inf])]:
            contexts = copy.deepcopy(valid)
            contexts[0]["groups"][0]["paths"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ref.mixture_first_event(contexts, 1, 2, "issued_forecast", 0)
        with self.assertRaises(ValueError):
            ref.mixture_first_event(valid, 1, 2, "unlabelled", 0)
        duplicated = copy.deepcopy(valid)
        duplicated[0]["groups"] *= 2
        with self.assertRaises(ValueError):
            ref.mixture_first_event(duplicated, 1, 2, "issued_forecast", 0)


class DurationHazardTests(unittest.TestCase):
    def test_observed_hop_quadrature_against_independent_adaptive_integration_and_doubling(self):
        from scipy.integrate import quad
        for intercept, slope, start, hop in itertools.product(
                [-8, -2, 4], [-4, 0, 4], [0, 0.125, 8, 32, 1200], [256 / 48000, 512 / 48000, 1024 / 48000]):
            expected = quad(lambda u: math.log1p(math.exp(intercept + slope * math.log1p(start + u))),
                            0, hop, epsabs=1e-13, epsrel=1e-12)[0]
            full = ref.duration_hazard_increment(intercept, slope, start, hop)
            split = (ref.duration_hazard_increment(intercept, slope, start, hop / 2)
                     + ref.duration_hazard_increment(intercept, slope, start + hop / 2, hop / 2))
            tolerance = 1e-9 + 1e-7 * expected
            self.assertAlmostEqual(full, expected, delta=tolerance)
            self.assertAlmostEqual(full, split, delta=tolerance)

    def test_constant_hazard_calibration_and_section_standardization(self):
        for intercept in (-1000, -2, 0, 1000):
            rate = max(intercept, 0) + math.log1p(math.exp(-abs(intercept)))
            for multiplier in (0.25, 1, 4):
                self.assertAlmostEqual(ref.duration_hazard_increment(intercept, 0, 3, 0.01, multiplier),
                                       rate * 0.01 * multiplier)
        mean, sd, coefficient, bias = 1.5, 0.4, -0.7, 0.3
        intercept, slope = bias - coefficient * mean / sd, coefficient / sd
        from scipy.integrate import quad
        expected = quad(lambda u: math.log1p(math.exp(bias + coefficient * (math.log1p(8 + u) - mean) / sd)),
                        0, 512 / 48000, epsabs=1e-13)[0]
        self.assertAlmostEqual(ref.duration_hazard_increment(intercept, slope, 8, 512 / 48000), expected, places=12)

    def test_unknown_foreground_and_invalid_duration_are_not_zero_age(self):
        self.assertIsNone(ref.duration_hazard_increment(0, 1, None, 0.01))
        self.assertGreater(ref.duration_hazard_increment(0, 1, 0, 0.01), 0)
        self.assertEqual(ref.duration_hazard_increment(0, 1, 5, 0), 0)
        for start, elapsed, multiplier in [(-1, 0.01, 1), (0, -0.01, 1), (0, 0.01, 0), (math.nan, 1, 1)]:
            with self.assertRaises(ValueError):
                ref.duration_hazard_increment(0, 1, start, elapsed, multiplier)


class GapSurvivalTests(unittest.TestCase):
    def test_constant_law_zero_gap_and_logarithmic_analytic_integral(self):
        for intercept, multiplier in itertools.product((-1000, -2, 0, 1000), (0.25, 1, 4)):
            result = ref.gap_survival(intercept, 0, 8, 32, multiplier)
            expected = 32 * multiplier * (max(intercept, 0) + math.log1p(math.exp(-abs(intercept))))
            self.assertEqual(result["status"], "prior_survival")
            self.assertAlmostEqual(result["integrated_hazard"], expected)
            self.assertEqual(result["observed_exit_mass"], 0)
            self.assertAlmostEqual(result["retained_foreground_mass"] + result["unknown_current_mass"], 1)
        result = ref.gap_survival(0, 1, 8, 0.5)
        expected = 10.5 * (math.log(10.5) - 1) - 10 * (math.log(10) - 1)
        self.assertAlmostEqual(result["integrated_hazard"], expected, delta=1e-9 + 1e-7 * expected)
        empty = ref.gap_survival(0, 1, 8, 0)
        self.assertEqual(empty["evaluations"], 0)
        self.assertEqual(empty["retained_foreground_mass"], 1)

    def test_every_accepted_grid_integral_matches_independent_adaptive_quadrature(self):
        from scipy.integrate import quad
        accepted = unresolved = 0
        for intercept, slope, start, gap in itertools.product(
                [-8, -2, 4], [-8, -4, 0, 4, 8], [0, 0.125, 8, 32, 1200],
                [0.01, 0.1, 0.5, 2, 8, 32, 128, 1200]):
            result = ref.gap_survival(intercept, slope, start, gap)
            self.assertLessEqual(result["evaluations"], 64)
            self.assertEqual(result["observed_exit_mass"], 0)
            if result["status"] != "prior_survival":
                unresolved += 1
                self.assertEqual(result["unknown_current_mass"], 1)
                self.assertIsNone(result["integrated_hazard"])
                continue
            accepted += 1
            def hazard(u):
                x = intercept + slope * math.log1p(start + u)
                return max(x, 0) + math.log1p(math.exp(-abs(x)))
            expected = quad(hazard, 0, gap, epsabs=1e-13, epsrel=1e-12)[0]
            tolerance = 1e-9 + 1e-7 * expected
            self.assertAlmostEqual(result["integrated_hazard"], expected, delta=tolerance)
            self.assertLessEqual(result["error_bound"], result["tolerance"])
        self.assertGreater(accepted, 0)
        self.assertGreater(unresolved, 0)

    def test_narrow_initial_hazard_cannot_pass_by_two_nearly_zero_estimates(self):
        from scipy.integrate import quad
        def hazard(u):
            x = 20 - 40 * math.log1p(u)
            return max(x, 0) + math.log1p(math.exp(-abs(x)))
        expected = quad(hazard, 0, 1200, points=[0.5, 1, 2], epsabs=1e-12)[0]
        self.assertGreater(expected, 1)
        result = ref.gap_survival(20, -40, 0, 1200)
        self.assertEqual(result["status"], "unresolved_budget")
        self.assertEqual(result["evaluations"], 64)
        self.assertEqual(result["unknown_current_mass"], 1)
        self.assertIsNone(result["log_survival"])
        self.assertLess(result["last_integral_estimate"], expected / 100)

    def test_budget_exhaustion_and_numeric_failure_never_supply_known_survival(self):
        result = ref.gap_survival(0, 1, 0, 4, max_evaluations=4)
        self.assertEqual(result["status"], "unresolved_budget")
        self.assertEqual(result["evaluations"], 4)
        self.assertIsNone(result["integrated_hazard"])
        result = ref.gap_survival(0, 1e100, 0, 1)
        self.assertEqual(result["status"], "unresolved_numeric")
        self.assertEqual(result["unknown_current_mass"], 1)
        self.assertEqual(result["observed_exit_mass"], 0)

    def test_calibration_scales_local_integral_and_partition_preserves_prior(self):
        from scipy.integrate import quad
        mean, sd, coefficient, bias = 1.5, 0.4, -0.7, 0.3
        intercept, slope = bias - coefficient * mean / sd, coefficient / sd
        expected = quad(lambda u: math.log1p(math.exp(bias + coefficient * (math.log1p(8 + u) - mean) / sd)),
                        0, 2, epsabs=1e-13)[0]
        for multiplier in (0.25, 1, 4):
            result = ref.gap_survival(intercept, slope, 8, 2, multiplier)
            parts = [ref.gap_survival(intercept, slope, 8 + i / 2, 0.5, multiplier) for i in range(4)]
            self.assertEqual(result["status"], "prior_survival")
            self.assertAlmostEqual(result["integrated_hazard"], multiplier * expected,
                                   delta=1e-9 + 1e-7 * multiplier * expected)
            self.assertAlmostEqual(result["log_survival"], sum(p["log_survival"] for p in parts), delta=1e-8)
            self.assertAlmostEqual(result["retained_foreground_mass"],
                                   math.prod(p["retained_foreground_mass"] for p in parts), delta=1e-8)

    def test_unknown_gap_mass_is_not_a_typed_event_in_the_following_observed_window(self):
        for gap in (ref.gap_survival(0, 0, 0, 1), ref.gap_survival(20, -40, 0, 1200)):
            kept = gap["retained_foreground_mass"]
            path = FirstEventTests.path([0.2], [[0.25, 0.75]], weight=kept)
            result = ref.mixture_first_event([{"weight": 1, "groups": [{"handle": 1, "paths": [path]}]}],
                                             1, 2, "issued_forecast", 1200)
            self.assertAlmostEqual(result["unresolved"][-1], gap["unknown_current_mass"])
            self.assertAlmostEqual(result["cdf"][-1], kept * -math.expm1(-0.2))
            self.assertAlmostEqual(result["event_mass"][0][0], kept * -math.expm1(-0.2) / 4)

    def test_unknown_start_and_invalid_settings_do_not_invent_age_zero(self):
        result = ref.gap_survival(0, 1, None, 0)
        self.assertEqual(result["status"], "unknown_start")
        self.assertEqual(result["evaluations"], 0)
        self.assertEqual(result["unknown_current_mass"], 1)
        for changes in ({"start_sec": -1}, {"elapsed_sec": -1}, {"max_evaluations": 65},
                        {"absolute_tolerance": 0, "relative_tolerance": 0},
                        {"relative_tolerance": math.nan}, {"multiplier": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ref.gap_survival(**{"intercept": 0, "duration_coefficient": 1, "start_sec": 0,
                                    "elapsed_sec": 1, **changes})


class ProbabilityTests(unittest.TestCase):
    def test_first_exit_agrees_across_time_partition(self):
        full = ref.competing_event_step([2, 3], 0.2)
        half = ref.competing_event_step([2, 3], 0.1)
        self.assertAlmostEqual(full["stay"], half["stay"] ** 2)
        for a, b in zip(full["exit"], half["exit"]):
            self.assertAlmostEqual(a, b + half["stay"] * b)
        self.assertAlmostEqual(full["exit"][0] / full["exit"][1], 2 / 3)
        self.assertAlmostEqual(full["stay"] + sum(full["exit"]), 1)
        self.assertEqual(ref.competing_event_step([0, 0], 1000), {"stay": 1, "exit": [0, 0]})

    def test_ordinal_known_quintiles_extremes_and_order(self):
        cutpoints = [math.log(p / (1 - p)) for p in [0.2, 0.4, 0.6, 0.8]]
        for p in ref.ordinal_probabilities(0, cutpoints):
            self.assertAlmostEqual(p, 0.2)
        self.assertEqual(ref.ordinal_probabilities(1000, cutpoints), [0, 0, 0, 0, 1])
        self.assertEqual(ref.ordinal_probabilities(-1000, cutpoints), [1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            ref.ordinal_probabilities(0, [1, 0, 2, 3])

    def test_overlap_agrees_with_exhaustive_joint_outcomes(self):
        groups = [(0.2, 0.3, 0.5), (0.0, 0.75, 0.25), (0.4, 0.6, 0.0)]
        expected = [0.0, 0.0, 0.0]
        for states in itertools.product(range(3), repeat=len(groups)):
            mass = math.prod(g[s] for g, s in zip(groups, states))
            target = 0 if 0 in states else 1 if all(s == 1 for s in states) else 2
            expected[target] += mass
        for a, b in zip(ref.known_overlap(groups), expected):
            self.assertAlmostEqual(a, b)
        self.assertEqual(ref.known_overlap([(1, 0, 0), (0, 0, 1)]), (1, 0, 0))

    def test_periodic_modes_wrap_without_becoming_a_mean(self):
        masses = [0.0] * 32
        masses[3], masses[19] = 1, 1
        self.assertEqual(ref.timing_lookup(masses, 3.5 / 32, True), 0.5)
        self.assertEqual(ref.timing_lookup(masses, 19.5 / 32, True), 0.5)
        self.assertEqual(ref.timing_lookup(masses, 11.5 / 32, True), 0)
        self.assertEqual(ref.timing_lookup(masses, -0.1, True), ref.timing_lookup(masses, 0.9, True))

    def test_linear_overflow_is_not_renormalized_into_interior(self):
        masses = [1.0] * 32 + [32.0]
        for position in [0, 0.5, 2, 3.99, 4]:
            self.assertEqual(ref.timing_lookup(masses, position, False), 1 / 64)
        self.assertIsNone(ref.timing_lookup(masses, -0.01, False))
        self.assertIsNone(ref.timing_lookup(masses, 4.01, False))
        self.assertIsNone(ref.timing_lookup([0.0] * 32, 0, True))


class PrivateCreditAndTimingTests(unittest.TestCase):
    def test_failed_execution_and_empty_inventory_do_not_learn(self):
        key = (1, 7, "periodic")
        entry = {"key": key, "weight": 0.8, "anchor_coverage": 1}
        result = ref.executed_reference_credit([entry], 1, 1, {key}, confirmed=False)
        self.assertEqual(result, {"credits": {key: 0}, "unassigned": 1, "interference": {key: 0}})
        self.assertEqual(ref.executed_reference_credit([], 1, 1, {key})["unassigned"], 1)

    def test_fractional_credit_loss_does_not_renormalize_or_leak_epochs(self):
        a, b, old, lost = (1, 1, "phase"), (1, 2, "phase"), (0, 3, "phase"), (1, 4, "phase")
        entries = [{"key": a, "weight": 0.4, "anchor_coverage": 1},
                   {"key": b, "weight": 0.3, "anchor_coverage": 0.5},
                   {"key": old, "weight": 0.2, "anchor_coverage": 1},
                   {"key": lost, "weight": 0.1, "anchor_coverage": 1}]
        result = ref.executed_reference_credit(entries, 0.5, 1, {a, b, old})
        self.assertAlmostEqual(result["credits"][a], 0.2)
        self.assertAlmostEqual(result["credits"][b], 0.075)
        self.assertEqual(result["credits"][old], 0)
        self.assertEqual(result["credits"][lost], 0)
        self.assertAlmostEqual(result["unassigned"], 0.725)
        self.assertAlmostEqual(result["interference"][a], 0.075)
        self.assertAlmostEqual(result["interference"][b], 0.2)
        self.assertNotIn(old, result["interference"])

    def test_uniform_difference_known_triangular_and_unequal_width_cdf(self):
        for point, expected in [(-1, 0), (-0.5, 0.125), (0, 0.5), (0.5, 0.875), (1, 1)]:
            self.assertAlmostEqual(ref.uniform_difference_cdf(point, (0, 1), (0, 1)), expected)
        self.assertAlmostEqual(ref.uniform_difference_cdf(0, (0, 2), (0, 1)), 0.25)
        self.assertAlmostEqual(ref.uniform_difference_cdf(1, (0, 2), (0, 1)), 0.75)
        self.assertEqual(ref.uniform_difference_cdf(0, (0, 0), (0, 0)), 1)

    def test_full_period_uniform_and_frozen_multiple_anchor_modes(self):
        result = ref.integrate_timing_bins((0, 1), [{"weight": 1, "interval": (0, 0), "period_sec": 1}], True)
        for value in result["bins"]:
            self.assertAlmostEqual(value, 1 / 32)
        alternatives = [{"weight": 0.25, "interval": (0.125, 0.125), "period_sec": 1},
                        {"weight": 0.75, "interval": (0.625, 0.625), "period_sec": 1}]
        result = ref.integrate_timing_bins((1, 1), alternatives, True)
        self.assertEqual(result["bins"][28], 0.25)
        self.assertEqual(result["bins"][12], 0.75)
        self.assertEqual(result["bins"][20], 0)
        self.assertAlmostEqual(sum(result["bins"]), 1)

    def test_linear_overflow_and_missing_anchor_are_separate(self):
        result = ref.integrate_timing_bins((-1, 5), [{"weight": 1, "interval": (0, 0), "period_sec": 1}], False)
        self.assertAlmostEqual(result["bins"][-1], 1 / 3)
        for value in result["bins"][:-1]:
            self.assertAlmostEqual(value, 1 / 48)
        self.assertEqual(result["unsupported"], 0)
        anchors = [{"weight": 0.7, "interval": (0, 0), "period_sec": 1},
                   {"weight": 0.3, "interval": None, "period_sec": 1}]
        result = ref.integrate_timing_bins((1, 1), anchors, False)
        self.assertAlmostEqual(sum(result["bins"]), 0.7)
        self.assertAlmostEqual(result["unsupported"], 0.3)
        self.assertEqual(ref.integrate_timing_bins(None, anchors, False)["unsupported"], 1)

    def test_phase_wrap_and_linear_terminal_point(self):
        anchors = [{"weight": 1, "interval": (0, 0), "period_sec": 1}]
        self.assertEqual(ref.integrate_timing_bins((1, 1), anchors, True)["bins"][0], 1)
        self.assertEqual(ref.integrate_timing_bins((4, 4), anchors, False)["bins"][31], 1)
        self.assertEqual(ref.integrate_timing_bins((4.01, 4.01), anchors, False)["bins"][32], 1)

    def test_uniform_hop_spreads_into_wrapped_bins_without_guessing_gap(self):
        anchors = [{"weight": 1, "interval": (-0.01, 0), "period_sec": 1}]
        result = ref.integrate_timing_bins((-0.01, 0), anchors, True)
        self.assertAlmostEqual(result["bins"][0], 0.5)
        self.assertAlmostEqual(result["bins"][-1], 0.5)
        self.assertAlmostEqual(sum(result["bins"]), 1)

    def test_common_decay_and_interference_cancel_between_trace_updates(self):
        base = ref.private_bin_probabilities([1.2, 0.4, 0], [3, 1, 0], [2, 1, 0], 2, 4)
        later = ref.private_bin_probabilities([1.2, 0.4, 0], [10003, 10001, 10000], [102, 101, 100], 2, 4)
        for a, b in zip(base, later):
            self.assertAlmostEqual(a, b, places=10)
        self.assertEqual(later[-1], 0)
        self.assertIsNone(ref.private_bin_probabilities([0, 0], [0, 0], [0, 0], 2, 4))

    def test_coverage_is_applied_once_when_credit_and_timing_are_combined(self):
        key = (1, 1, "nonperiodic")
        anchors = [{"weight": 0.7, "interval": (0, 0), "period_sec": 1},
                   {"weight": 0.3, "interval": None, "period_sec": 1}]
        timing = ref.integrate_timing_bins((1, 1), anchors, False)
        credit = ref.executed_reference_credit(
            [{"key": key, "weight": 0.4, "anchor_coverage": 1 - timing["unsupported"]}], 0.5, 1, {key})
        mass = [credit["credits"][key] * b / sum(timing["bins"]) for b in timing["bins"]]
        self.assertAlmostEqual(sum(mass), 0.14)
        self.assertAlmostEqual(credit["unassigned"], 0.86)


class RetainedTimingTests(unittest.TestCase):
    def test_full_cycle_matches_exponential_integrals_not_midpoint_time(self):
        anchors = [{"weight": 1, "interval": (0, 0), "period_sec": 1}]
        result = ref.integrate_retained_timing_bins((0, 1), anchors, True, 2)
        for index, mass in enumerate(result["bins"]):
            left, right = index / 32, (index + 1) / 32
            expected = 2 * (math.exp((right - 1) / 2) - math.exp((left - 1) / 2))
            self.assertAlmostEqual(mass, expected, places=14)
        self.assertAlmostEqual(sum(result["bins"]), 2 * -math.expm1(-0.5))
        self.assertGreater(result["bins"][-1], result["bins"][0])

    def test_uniform_anchor_crossing_against_independent_midpoint_quadrature(self):
        # Independently integrate the conditional anchor CDF on a dense outcome grid.
        outcome = (-0.07, 0.09)
        anchor = (-0.03, 0.02)
        period, tau, n = 0.125, 0.2, 40000
        alternatives = [{"weight": 0.7, "interval": anchor, "period_sec": period},
                        {"weight": 0.3, "interval": None, "period_sec": period}]
        actual = ref.integrate_retained_timing_bins(outcome, alternatives, True, tau, bins=8)
        expected = [0.0] * 8
        for step in range(n):
            x = outcome[0] + (step + 0.5) * (outcome[1] - outcome[0]) / n
            for index in range(8):
                probability = 0.0
                for cycle in range(-2, 3):
                    lo = x - (cycle + (index + 1) / 8) * period
                    hi = x - (cycle + index / 8) * period
                    probability += max(0, min(hi, anchor[1]) - max(lo, anchor[0])) / (anchor[1] - anchor[0])
                expected[index] += 0.7 * probability * math.exp((x - outcome[1]) / tau) / n
        for a, b in zip(actual["bins"], expected):
            self.assertAlmostEqual(a, b, delta=2e-9)
        self.assertAlmostEqual(actual["unsupported"], 0.3)

    def test_long_time_constant_overflow_and_absolute_time_translation(self):
        anchors = [{"weight": 1, "interval": (0.1, 0.2), "period_sec": 0.5}]
        raw = ref.integrate_timing_bins((-1, 3), anchors, False)
        almost = ref.integrate_retained_timing_bins((-1, 3), anchors, False, 1e12)
        shifted = ref.integrate_retained_timing_bins((999, 1003),
            [{"weight": 1, "interval": (1000.1, 1000.2), "period_sec": 0.5}], False, 2)
        original = ref.integrate_retained_timing_bins((-1, 3), anchors, False, 2)
        for a, b in zip(raw["bins"], almost["bins"]):
            self.assertAlmostEqual(a, b, places=11)
        for a, b in zip(original["bins"], shifted["bins"]):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(sum(original["bins"]), -math.expm1(-2) / 2)
        self.assertGreater(original["bins"][-1], 0)


class StatefulPrivateTraceTests(unittest.TestCase):
    @staticmethod
    def entry(key, weight=1, anchor=0, coverage=1):
        return {"key": key, "weight": weight,
                "anchors": [{"weight": coverage, "interval": (anchor, anchor), "period_sec": 1}]}

    def test_fractional_soft_update_preserves_old_decay_and_cap_is_per_bin(self):
        key = (1, 7, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 4, 1.5)
        entry = self.entry(key)
        for event, time in enumerate([0.01, 1.01, 2.01]):
            state.observe(event, "onset", (time, time), [entry], 1, {key})
        self.assertAlmostEqual(math.exp(state.traces[key]["onset"][0]), 1.5)
        state.observe(3, "onset", (4.51, 4.51), [self.entry(key, 0.1)], 1, {key})
        expected = 1.5 * math.exp(-2.5 / 2)
        self.assertAlmostEqual(math.exp(state.traces[key]["onset"][0]), expected)
        self.assertAlmostEqual(math.exp(state.traces[key]["onset"][16]), 0.1)
        self.assertAlmostEqual(state.probabilities(key, "onset")[0], expected / (expected + 0.1))
        self.assertIsNone(state.probabilities(key, "release"))

    def test_one_physical_outcome_has_two_credits_and_interference_after_cap(self):
        a, b = (1, 1, "periodic"), (1, 2, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 1, 1.5)
        entries = [self.entry(a, 0.4), self.entry(b, 0.3, anchor=0.5)]
        first = state.observe(0, "onset", (0.01, 0.01), entries, 1, {a, b})
        self.assertAlmostEqual(first["unassigned"], 0.3)
        self.assertAlmostEqual(math.exp(state.traces[a]["onset"][0]), 0.4 * math.exp(-0.3))
        self.assertAlmostEqual(math.exp(state.traces[b]["onset"][16]), 0.3 * math.exp(-0.4))
        for event in range(1, 9):
            time = event * 0.0001 + 0.01
            state.observe(event, "onset", (time, time), entries, 1, {a, b})
        # Independent scalar recurrence retains competition on every update.
        expected = 0.4 * math.exp(-0.3)
        for _ in range(8):
            expected = min(1.5, expected * math.exp(-0.0001 / 2) + 0.4) * math.exp(-0.3)
        self.assertAlmostEqual(math.exp(state.traces[a]["onset"][0]), expected)

    def test_known_hop_age_and_coverage_are_integrated_once(self):
        key = (1, 1, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 1, 3)
        result = state.observe(0, "onset", (0.001, 0.009),
                               [self.entry(key, 0.4, coverage=0.7)], 0.5, {key})
        self.assertAlmostEqual(result["credits"][key], 0.14)
        expected = 0.14 * -math.expm1(-0.008 / 2) * 2 / 0.008
        self.assertAlmostEqual(math.exp(state.traces[key]["onset"][0]), expected)

    def test_unknown_failed_lost_epoch_and_duplicate_are_not_reinforcement(self):
        key, old = (1, 1, "periodic"), (0, 2, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 1, 3)
        entries = [self.entry(key, 0.5), self.entry(old, 0.5)]
        result = state.observe(0, "onset", (0, 0), entries, 1, {key, old}, confirmed=False)
        self.assertFalse(result["applied"])
        self.assertEqual(state.traces, {})
        state.observe(0, "onset", (0, 0), entries, 1, {key, old})
        self.assertNotIn(old, state.traces)
        self.assertAlmostEqual(math.exp(state.traces[key]["onset"][0]), 0.5)
        with self.assertRaises(ValueError):
            state.observe(0, "onset", (0, 0), entries, 1, {key, old})
        removed = state.observe(1, "release", (0.1, 0.1), [], 1, set())
        self.assertEqual(removed["removed"], [key])
        self.assertEqual(state.traces, {})

    def test_overflow_remains_in_normalization_and_only_capacity_evicts(self):
        a, b = (1, 1, "nonperiodic"), (1, 2, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 4, 3, capacity=1)
        state.observe(0, "onset", (1, 1), [self.entry(a)], 1, {a, b})
        state.observe(1, "onset", (5, 5), [self.entry(a)], 1, {a, b})
        self.assertEqual(len(state.probabilities(a, "onset")), 33)
        self.assertAlmostEqual(state.probabilities(a, "onset")[-1], 1 / (1 + math.exp(-2)))
        result = state.observe(2, "release", (10000, 10000), [self.entry(b)], 1, {a, b})
        self.assertEqual(result["evicted"], [a])
        self.assertIsNone(state.probabilities(b, "onset"))
        self.assertEqual(state.probabilities(b, "release")[0], 1)

    def test_common_wait_and_competition_preserve_conditional_shape(self):
        a, b = (1, 1, "periodic"), (1, 2, "periodic")
        state = ref.PrivateTimingTrace(1, 2, 1, 6)
        state.observe(0, "onset", (0.01, 0.01), [self.entry(a)], 1, {a, b})
        state.observe(1, "onset", (0.51, 0.51), [self.entry(a)], 1, {a, b})
        before = state.probabilities(a, "onset")
        state.observe(2, "release", (10000, 10000), [self.entry(b)], 1, {a, b})
        for actual, expected in zip(state.probabilities(a, "onset"), before):
            self.assertAlmostEqual(actual, expected, places=11)


class QuantizedObservationAuditTests(unittest.TestCase):
    def test_soft_target_loss_is_biased_under_a_known_physical_quantizer(self):
        # Four observed intervals partition one cycle. Two straddle bin borders.
        intervals = [(0.125 + j / 4, 0.375 + j / 4) for j in range(4)]
        truth = [0.8, 0.2]
        probabilities = [ref.periodic_observation_probability(truth, interval, (0, 0), 1)
                         for interval in intervals]
        for actual, expected in zip(probabilities, [0.4, 0.25, 0.1, 0.25]):
            self.assertAlmostEqual(actual, expected)
        expected_target = [0.0, 0.0]
        for interval, probability in zip(intervals, probabilities):
            soft = ref.integrate_timing_bins(interval,
                [{"weight": 1, "interval": (0, 0), "period_sec": 1}], True, bins=2)["bins"]
            for j in range(2):
                expected_target[j] += probability * soft[j]
        self.assertAlmostEqual(expected_target[0], 0.65)
        loss_truth = -sum(p * math.log(q) for p, q in zip(expected_target, truth))
        loss_biased = -sum(p * math.log(p) for p in expected_target)
        self.assertLess(loss_biased, loss_truth)
        # A proper observation likelihood instead has its population optimum at truth.
        for candidate in ([0.65, 0.35], [0.79, 0.21], [0.81, 0.19]):
            candidate_p = [ref.periodic_observation_probability(candidate, interval, (0, 0), 1)
                           for interval in intervals]
            kl = sum(p * math.log(p / q) for p, q in zip(probabilities, candidate_p))
            self.assertGreater(kl, 0)

    def test_phase_observation_likelihood_normalizes_with_uncertain_known_anchor(self):
        q = [1 / 64] * 32
        q[3] += 0.25
        q[19] += 0.25
        probabilities = [ref.periodic_observation_probability(q, (j / 64, (j + 1) / 64),
                                                            (-0.01, 0.02), 1) for j in range(64)]
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertTrue(all(p >= 0 for p in probabilities))

    def test_same_bias_exists_with_32_bins_and_512_sample_observations(self):
        hop = 512 / 48000
        truth = [0.8 / 16 if j % 2 == 0 else 0.2 / 16 for j in range(32)]
        intervals = [(j * hop, min(1, (j + 1) * hop)) for j in range(math.ceil(1 / hop))]
        observed = [ref.periodic_observation_probability(truth, interval, (0, 0), 1)
                    for interval in intervals]
        self.assertAlmostEqual(sum(observed), 1)
        expected_target = [0.0] * 32
        for interval, p in zip(intervals, observed):
            labels = ref.integrate_timing_bins(interval,
                [{"weight": 1, "interval": (0, 0), "period_sec": 1}], True)["bins"]
            for j, label in enumerate(labels):
                expected_target[j] += p * label
        self.assertGreater(max(abs(q - p) for q, p in zip(truth, expected_target)), 1e-4)
        proper_kl = sum(p * math.log(p / ref.periodic_observation_probability(
            expected_target, interval, (0, 0), 1)) for interval, p in zip(intervals, observed))
        self.assertGreater(proper_kl, 0)


class JointPhysicalObservationTests(unittest.TestCase):
    def test_one_period_reduces_to_independently_integrated_phase_likelihood(self):
        anchors = [{"weight": 1, "interval": (0, 0), "period_sec": 1}]
        records = [{"interval": (j / 64, (j + 1) / 64)} for j in range(64)]
        kernel = ref.timing_observation_kernel((0, 1), records, anchors, True)
        q = [0.8 / 16 if j % 2 == 0 else 0.2 / 16 for j in range(32)]
        key = (1, 7, "periodic")
        result = ref.private_outcome_distribution({key: kernel},
            [{"key": key, "weight": 1, "probabilities": q}], 1, {key})
        for actual, record in zip(result["probabilities"], records):
            expected = ref.periodic_observation_probability(q, record["interval"], (0, 0), 1)
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["probabilities"][-1], 0)
        self.assertEqual(result["effective_baseline_weight"], 0)

    def test_two_conflicting_references_predict_one_outcome_without_renormalizing_loss(self):
        records = [{"interval": (j / 4, (j + 1) / 4), "detection": 0.8} for j in range(4)]
        a, b, old = (1, 1, "periodic"), (1, 2, "periodic"), (0, 3, "periodic")
        kernels = {key: ref.timing_observation_kernel((0, 1), records,
            [{"weight": 1, "interval": (anchor, anchor), "period_sec": 1}], True)
            for key, anchor in [(a, 0), (b, 0.5)]}
        q = [1.0] + [0.0] * 31
        inventory = [{"key": a, "weight": 0.4, "probabilities": q},
                     {"key": b, "weight": 0.3, "probabilities": q},
                     {"key": old, "weight": 0.2, "probabilities": q}]
        result = ref.private_outcome_distribution(kernels, inventory, 1, {a, b, old})
        for actual, expected in zip(result["probabilities"], [0.38, 0.06, 0.30, 0.06, 0.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["effective_baseline_weight"], 0.3)
        lost = ref.private_outcome_distribution(kernels, inventory, 1, {a})
        self.assertAlmostEqual(lost["effective_baseline_weight"], 0.6)
        self.assertAlmostEqual(lost["probabilities"][0], 0.44)
        # A published probability vector owns its values, even if live inputs change.
        q[0], q[16] = 0, 1
        self.assertAlmostEqual(result["probabilities"][0], 0.38)

    def test_linear_overflow_and_unreachable_bins_keep_distinct_baseline_mass(self):
        records = [{"interval": (-1, 0)}, {"interval": (0, 4)}, {"interval": (4, 5)}]
        anchors = [{"weight": 1, "interval": (0, 0), "period_sec": 1}]
        kernel = ref.timing_observation_kernel((-1, 5), records, anchors, False)
        self.assertEqual(len(kernel["matrix"][0]), 33)
        self.assertEqual([row[-1] for row in kernel["matrix"]], [0.5, 0, 0.5, 0])
        self.assertEqual([row[0] for row in kernel["matrix"]], [0, 1, 0, 0])
        clipped = ref.timing_observation_kernel((0, 2), [{"interval": (0, 1)}, {"interval": (1, 2)}],
                                                anchors, False)
        self.assertEqual(clipped["unreachable"][-1], 1)
        self.assertEqual([row[-1] for row in clipped["matrix"]], [0.5, 0.5, 0])

    def test_anchor_gaps_and_acquisition_gaps_are_different_missing_mechanisms(self):
        records = [{"interval": (0, 0.25)}, {"interval": (0.5, 1)}]
        anchors = [{"weight": 0.6, "interval": (0, 0), "period_sec": 1},
                   {"weight": 0.4, "interval": None, "period_sec": 1}]
        kernel = ref.timing_observation_kernel((0, 1), records, anchors, True)
        self.assertAlmostEqual(kernel["anchor_coverage"], 0.6)
        # Bin zero is observed; unsupported anchor weight uses the complete baseline.
        for actual, expected in zip([row[0] for row in kernel["matrix"]], [0.7, 0.2, 0.1]):
            self.assertAlmostEqual(actual, expected)
        # A supported prediction inside the acquisition gap is a missing observation.
        for actual, expected in zip([row[10] for row in kernel["matrix"]], [0.1, 0.2, 0.7]):
            self.assertAlmostEqual(actual, expected)
        for column in zip(*kernel["matrix"]):
            self.assertAlmostEqual(sum(column), 1)

    def test_uncertain_anchor_integral_matches_independent_adaptive_quadrature(self):
        from scipy.integrate import quad
        records = [{"interval": (0, 0.3)}, {"interval": (0.3, 0.8), "detection": 0.7},
                   {"interval": (0.9, 1.2)}]
        kernel = ref.timing_observation_kernel((0, 1.2), records,
            [{"weight": 1, "interval": (-0.2, 0.2), "period_sec": 0.25}], False, bins=2)
        for index in range(3):
            for row, record in enumerate(records):
                def probability(anchor):
                    # Build the physical regions directly, then integrate in anchor time.
                    regions = ([(0, max(0, anchor)), (min(1.2, anchor + 1), 1.2)] if index == 2 else
                               [(max(0, anchor + index * 0.5), min(1.2, anchor + (index + 1) * 0.5))])
                    width = sum(max(0, hi - lo) for lo, hi in regions)
                    left, right = record["interval"]
                    detection = record.get("detection", 1)
                    if width == 0:
                        return (right - left) / 1.2 * detection
                    return detection * sum(max(0, min(right, hi) - max(left, lo)) for lo, hi in regions) / width
                expected = quad(probability, -0.2, 0.2, epsabs=1e-11, points=[-0.1, 0, 0.1], limit=300)[0] / 0.4
                self.assertAlmostEqual(kernel["matrix"][row][index], expected, delta=2e-9)

    def test_uniform_empty_head_and_invalid_observation_partition(self):
        key = (1, 1, "nonperiodic")
        records = [{"interval": (0, 2)}, {"interval": (2, 4)}]
        kernel = ref.timing_observation_kernel((0, 4), records,
            [{"weight": 1, "interval": (0, 0), "period_sec": 1}], False)
        result = ref.private_outcome_distribution({key: kernel},
            [{"key": key, "weight": 1, "probabilities": None}], 1, {key})
        self.assertEqual(result["probabilities"], [0.5, 0.5, 0])
        self.assertAlmostEqual(result["effective_baseline_weight"], 1 / 33)
        with self.assertRaises(ValueError):
            ref.timing_observation_kernel((0, 1), [{"interval": (0, 0.6)}, {"interval": (0.5, 1)}], [], True)


if __name__ == "__main__":
    unittest.main()
