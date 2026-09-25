"""Independent I11-2 CDF checks against analytic and adversarial fixtures."""

import copy
import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_i11_stage2.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("verify_i11_stage2", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def case(model="hazard", peak=None):
    intercept = math.log(math.expm1(2.0))
    coefficients = [0.0] * 18
    coefficients[0] = intercept
    return {"name": "constant", "config": {"model": model, "coefficients": coefficients,
            "means": [0.0] * 8, "deviations": [1.0] * 8, "horizon_sec": 4.0},
            "group": {"bus": 0, "epoch": 1, "generation": 1}, "sample_rate": 48000,
            "issued_at": 96000, "horizon_end": 288000, "elapsed_seconds": [0.1, 0.1],
            "reset_unknown": False, "context": {"peak": peak, "word": None,
            "source_start": 90000, "source_end": 95000, "available": 96000},
            "intervals": [None] * 4, "cdf": None, "unknown": None,
            "cumulative_hazard": None, "interval_evaluations": None, "evaluations": 0,
            "legacy_full_interval_integral": None}


class CdfTest(unittest.TestCase):
    def test_frozen_primary_coefficients_are_checked(self):
        import json
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "docs/roadmap/temporal-dcc/i11-2-inputs/plan.json").read_text())
        phase = case(peak={"period_seconds": 0.5, "support": 0.8, "bin": 5})
        phase["name"] = "phase_hazard"
        phase["config"] = {key: copy.deepcopy(plan["primary_hazard"][key])
                           for key in ("model", "coefficients", "means", "deviations", "horizon_sec")}
        constant = case()
        constant["name"] = "constant_hazard"
        periodic = case("periodic")
        periodic["name"] = "periodic"
        nonperiodic = case()
        nonperiodic["name"] = "nonperiodic_hazard"
        failed = case()
        failed["name"] = "integral_failed"
        failed["forced_out_of_estimator_period_range"] = True
        fixture = {"schema": "i11-2-cdf-fixture-v2",
                   "cases": [constant, phase, periodic, nonperiodic, failed]}
        verifier.verify_registered_cdf_fixture(fixture, plan)
        damaged = copy.deepcopy(fixture)
        next(case for case in damaged["cases"] if case["name"] == "phase_hazard")["config"]["coefficients"][10] = 3.0
        with self.assertRaisesRegex(ValueError, "frozen primary"):
            verifier.verify_registered_cdf_fixture(damaged, plan)

    def test_constant_hazard_without_peak_has_analytic_cdf(self):
        fixture = case()
        fixture["cdf"] = [-math.expm1(-2.0 * k / 50.0) for k in range(201)]
        fixture["cumulative_hazard"] = [2.0 * k / 50.0 for k in range(201)]
        fixture["interval_evaluations"] = [6] * 200
        fixture["evaluations"] = 1200
        fixture["legacy_full_interval_integral"] = [2.0 * k / 50.0 for k in range(201)]
        checks = verifier.verify_cdf_fixture({"schema": "i11-2-cdf-fixture-v2", "cases": [fixture]})
        self.assertTrue(all(c["passed"] for c in checks), [c for c in checks if not c["passed"]][:3])
        damaged = copy.deepcopy(fixture)
        damaged["cdf"][100] += 1e-4
        self.assertTrue(any(not c["passed"] and c.get("point") == 100
                            for c in verifier.verify_cdf_fixture(
                                {"schema": "i11-2-cdf-fixture-v2", "cases": [damaged]})))

    def test_phase_hazard_integrates_more_than_one_period(self):
        fixture = case(peak={"period_seconds": 0.5, "support": 0.8, "bin": 5})
        fixture["config"]["coefficients"] = [-2.0] + [0.0] * 9 + [4.0] + [0.0] * 7
        cdf, cumulative = verifier.reference_cdf(fixture)
        self.assertEqual(len(cdf), 201)
        self.assertGreater(cumulative[-1], 0)
        self.assertTrue(all(a <= b for a, b in zip(cdf, cdf[1:])))

    def test_periodic_step_and_unknown(self):
        fixture = case("periodic", {"period_seconds": 0.5, "support": 0.9, "bin": 3})
        fixture["cdf"], _ = verifier.reference_cdf(fixture)
        checks = verifier.verify_cdf_fixture({"schema": "i11-2-cdf-fixture-v2", "cases": [fixture]})
        self.assertTrue(all(c["passed"] for c in checks))
        fixture["elapsed_seconds"] = [0.1, 0.2]
        fixture["cdf"] = None
        fixture["unknown"] = "uncertain_elapsed"
        self.assertTrue(all(c["passed"] for c in verifier.verify_cdf_fixture(
            {"schema": "i11-2-cdf-fixture-v2", "cases": [fixture]})))

    def test_bounded_integral_failure_is_distinct_from_reference_failure(self):
        fixture = case(peak={"period_seconds": 0.001, "support": 0.9, "bin": 3})
        fixture["config"]["coefficients"][10] = 4.0
        fixture["unknown"] = "integral_failed"
        fixture["failed_interval"] = 1
        fixture["interval_evaluations"] = [62]
        fixture["evaluations"] = 62
        fixture["legacy_full_interval_integral"] = [0.0] + [None] * 200
        checks = verifier.verify_cdf_fixture({"schema": "i11-2-cdf-fixture-v2", "cases": [fixture]})
        self.assertTrue(all(c["passed"] for c in checks), checks)


def decision_pair():
    cdf = [float(k >= 66) for k in range(201)]
    group_id = {"bus": 0, "epoch": 7, "generation": 2}
    issue = {"model": "hazard", "version": 1, "bus": 0, "epoch": 7,
             "sample_rate": 1000, "issued_at": 0, "horizon_end": 4000,
             "groups": [{"group": group_id, "periodic": True,
                         "forecast": {"group": group_id, "source_start": 0,
                                      "source_end": 0, "available": 0, "issued_at": 0,
                                      "horizon_end": 4000, "model": "hazard", "version": 1,
                                      "reset_unknown": False, "elapsed_seconds": [0.1, 0.1]},
                         "cdf": cdf, "unknown": None, "evaluations": 0}] + [None] * 6}
    candidates = []
    for offset in range(-2, 21):
        at = 1200.0 + (offset * 50.0 if offset < 0 else offset * 40.0)
        base = verifier.f32(((at - 1200.0) / 800.0) ** 2)
        candidates.append({"offset": offset, "at": at, "displacement_sq": base,
                                  "context_distance": None, "pred_external_band_energy": None,
                                  "overlap": None, "external_energy": [None] * 16,
                                  "cost": base})
    off = {"type": "participation_decision", "voice_id": 5, "now": 1000,
           "sample_rate": 1000, "due_frame": 1200.0, "period_frames": 800.0,
           "width": 100.0, "earliest": 1000.0, "coupling": 1.0,
           "overlap_sensitivity": 0.0, "onset_allowed": True,
           "memory": None, "own_band_energy": [0.0] * 3, "sound_hold_sec": 0.1,
           "sound_adsr": None, "footprint_source": "body", "footprint_identity": None,
           "current_identity": None, "footprint_requested_at": None,
           "footprint_received_at": None, "footprint_d_samples": 0.0,
           "footprint_truncated": False, "footprint_delay": [0.0] * 16,
           "footprint_power": [1.0] * 16, "forecast_observed_frame": None,
           "forecast_available_through_frame": None, "skipped_cycles": 0,
           "candidates": candidates, "selected_offset": 0, "selected_at": 1200.0,
           "selected_cost": 0.0, "reference_cost": 0.0, "skipped": False}
    on = copy.deepcopy(off)
    on["arrival_weight"] = 4.0
    on["arrival"] = {"input": {"publication": {"issue": issue, "profile_key": None},
                                  "received_at": 1000, "current_epoch": 7,
                                  "owner": [5, 1, None], "current_profile_key": None,
                                  "bindings": [None] * 8, "bindings_overflow": False},
                     "input_status": "ready", "self_status": "no_binding",
                     "binding_checks": [None] * 8,
                     "groups": [{"slot": 0, "status": "used"}] + [None] * 6,
                     "eligible_count": 1, "known_decision": True}
    for candidate in on["candidates"]:
        candidate["base_cost"] = candidate["cost"]
        lower, upper = candidate["at"] - 100.0, candidate["at"] + 100.0
        probability = verifier.interpolate(cdf, upper, 0, 20.0) - verifier.interpolate(cdf, lower, 0, 20.0)
        candidate["arrival_p"] = probability
        candidate["arrival_term"] = verifier.f32(4.0 * (1.0 - probability))
        candidate["cost"] = verifier.f32(candidate["base_cost"] + candidate["arrival_term"])
    selected = on["candidates"][3]
    on["selected_offset"] = 1
    on["selected_at"] = selected["at"]
    on["selected_cost"] = selected["cost"]
    selected["base_cost"] = 0.0025
    on["selected_base_cost"] = selected["base_cost"]
    on["reference_cost"] = on["candidates"][2]["cost"]
    return off, on


class DecisionTest(unittest.TestCase):
    def test_exact_bits_include_signed_zero_and_one_ulp(self):
        self.assertFalse(verifier.same_value(0.0, -0.0))
        self.assertFalse(verifier.same_f64(0.5, math.nextafter(0.5, math.inf)))

    def test_arrival_cost_selection_and_first_divergence(self):
        off, on = decision_pair()
        self.assertNotEqual(on["candidates"][3]["base_cost"],
                            verifier.f32(on["candidates"][3]["base_cost"]))
        self.assertEqual(verifier.verify_decision(on), [])
        with tempfile.TemporaryDirectory() as directory:
            off_path = Path(directory) / "off.jsonl"
            on_path = Path(directory) / "on.jsonl"
            import json
            off_path.write_text(json.dumps(off) + "\n")
            on_path.write_text(json.dumps(on) + "\n")
            result = verifier.locate_arrival_divergence(off_path, on_path)
            self.assertEqual(result["status"], "diverged")
            self.assertEqual(result["nonarrival_differences"], [])
            self.assertEqual(result["on_arrival_failures"], [])
            self.assertTrue(result["arrival_used"])
            bad = copy.deepcopy(on)
            bad["candidates"][2]["arrival_term"] = 0.0
            self.assertIn("candidate 2 term", verifier.verify_decision(bad))
            bad = copy.deepcopy(on)
            bad["arrival"]["groups"][0]["status"] = "self"
            self.assertIn("arrival.groups", verifier.verify_decision(bad))

    def test_reported_overlap_cannot_supply_its_own_reference(self):
        _, on = decision_pair()
        on["overlap_sensitivity"] = 1.0
        on["forecast_available_through_frame"] = 0
        for candidate in on["candidates"]:
            candidate["overlap"] = 0.0
        self.assertEqual(verifier.verify_decision(on), [])
        on["candidates"][2]["overlap"] = 10.0
        self.assertIn("candidate 2 overlap", verifier.verify_decision(on))

    def test_short_energy_array_cannot_hide_by_zip_truncation(self):
        _, on = decision_pair()
        on["candidates"][2]["external_energy"].pop()
        self.assertIn("candidate 2 energy structure", verifier.verify_decision(on))

    def test_issue_and_forecast_clock_rejections_match_consumer(self):
        _, template = decision_pair()
        for field, value in (("version", 2), ("sample_rate", 999)):
            with self.subTest(field=field):
                on = copy.deepcopy(template)
                on["arrival"]["input"]["publication"]["issue"][field] = value
                self.assertIn("arrival.input_status", verifier.verify_decision(on))
        for field, value in (("reset_unknown", True), ("elapsed_seconds", [0.1, 0.2])):
            with self.subTest(field=field):
                on = copy.deepcopy(template)
                on["arrival"]["input"]["publication"]["issue"]["groups"][0]["forecast"][field] = value
                self.assertIn("arrival.groups", verifier.verify_decision(on))

    def test_cdf_offset_cannot_preserve_probabilities_while_escaping_domain(self):
        _, on = decision_pair()
        group = on["arrival"]["input"]["publication"]["issue"]["groups"][0]
        group["cdf"] = [value + 2.0 for value in group["cdf"]]
        self.assertIn("used group has invalid CDF", verifier.verify_decision(on))

    def test_false_zero_weight_zero_coupling_known_zero_and_all_unknown(self):
        off, template = decision_pair()
        self.assertNotIn("arrival", off)
        self.assertTrue(all("arrival_term" not in candidate and "base_cost" not in candidate
                            and candidate["cost"] == candidate["displacement_sq"]
                            for candidate in off["candidates"]))
        for control in ("weight_zero", "coupling_zero", "known_p_zero", "all_unknown"):
            with self.subTest(control=control):
                on = copy.deepcopy(template)
                if control == "weight_zero":
                    on["arrival_weight"] = 0.0
                elif control == "coupling_zero":
                    on["coupling"] = 0.0
                elif control == "known_p_zero":
                    on["arrival"]["input"]["publication"]["issue"]["groups"][0]["cdf"] = [0.0] * 201
                else:
                    group = on["arrival"]["input"]["publication"]["issue"]["groups"][0]
                    group["cdf"] = None
                    group["unknown"] = "integral_failed"
                    on["arrival"]["groups"][0]["status"] = "integral_failed"
                    on["arrival"]["eligible_count"] = 0
                    on["arrival"]["known_decision"] = False
                for candidate in on["candidates"]:
                    candidate["arrival_p"] = (None if control == "all_unknown" else
                                              0.0 if control == "known_p_zero" else candidate["arrival_p"])
                    candidate["arrival_term"] = (None if control == "all_unknown" else
                                                 0.0 if control in ("weight_zero", "coupling_zero") else 4.0)
                    candidate["cost"] = verifier.f32(candidate["base_cost"] + (candidate["arrival_term"] or 0.0))
                chosen = on["candidates"][2]
                on["selected_offset"] = 0
                on["selected_at"] = chosen["at"]
                on["selected_cost"] = chosen["cost"]
                on["selected_base_cost"] = chosen["base_cost"]
                on["reference_cost"] = chosen["cost"]
                self.assertEqual(verifier.verify_decision(on), [])

    def test_nongrid_interpolation_cast_and_addition_are_exact(self):
        import struct
        _, on = decision_pair()
        issue = on["arrival"]["input"]["publication"]["issue"]
        cdf = [(k / 200.0) ** 2 for k in range(201)]
        issue["groups"][0]["cdf"] = cdf
        on["due_frame"] = 1203.0
        on["width"] = 101.0
        on["coupling"] = 0.3
        on["arrival_weight"] = 1.7

        def linear(sample):
            q = sample / 20.0
            k = math.floor(q)
            return cdf[k] + (cdf[k + 1] - cdf[k]) * (q - k)

        for candidate in on["candidates"]:
            offset = candidate["offset"]
            at = 1203.0 + (offset * 101.0 / 2.0 if offset < 0 else offset * 800.0 / 20.0)
            candidate["at"] = at
            base = verifier.f32(((at - 1203.0) / 800.0) ** 2)
            candidate["displacement_sq"] = base
            candidate["base_cost"] = base
            probability = linear(at + 101.0) - linear(at - 101.0)
            candidate["arrival_p"] = probability
            term = verifier.f32(float(verifier.f32(0.3)) * 1.7 * (1.0 - probability))
            candidate["arrival_term"] = term
            candidate["cost"] = verifier.f32(base + term)
        chosen = min(on["candidates"], key=lambda candidate: candidate["cost"])
        on["selected_offset"] = chosen["offset"]
        on["selected_at"] = chosen["at"]
        on["selected_cost"] = chosen["cost"]
        on["selected_base_cost"] = chosen["base_cost"]
        on["reference_cost"] = on["candidates"][2]["cost"]
        self.assertNotEqual(1304.0 / 20.0, int(1304.0 / 20.0))
        self.assertEqual(verifier.verify_decision(on), [])
        bad = copy.deepcopy(on)
        bits = struct.unpack("<I", struct.pack("<f", bad["candidates"][2]["arrival_term"]))[0]
        bad["candidates"][2]["arrival_term"] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        self.assertIn("candidate 2 term", verifier.verify_decision(bad))

    def test_earlier_nonarrival_input_difference_fails(self):
        off, on = decision_pair()
        on["footprint_power"][0] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            off_path = Path(directory) / "off.jsonl"
            on_path = Path(directory) / "on.jsonl"
            import json
            off_path.write_text(json.dumps(off) + "\n")
            on_path.write_text(json.dumps(on) + "\n")
            result = verifier.locate_arrival_divergence(off_path, on_path)
            self.assertEqual(result["first_prior_input_difference"]["inputs"], ["footprint_power"])

    def test_first_receipt_is_stable_for_a_publication(self):
        _, on = decision_pair()
        later = copy.deepcopy(on)
        later["now"] = 1002
        later["arrival"]["input"]["received_at"] = 1001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "on.jsonl"
            import json
            path.write_text(json.dumps(on) + "\n" + json.dumps(later) + "\n")
            result = verifier.verify_report(path)
            self.assertEqual(result["failed_decisions"], 1)
            self.assertIn("first receipt changed for publication version",
                          result["examples"][0]["failures"])

    def test_empty_pair_is_not_effect_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            off = Path(directory) / "off.jsonl"
            on = Path(directory) / "on.jsonl"
            result = Path(directory) / "result.json"
            off.write_text("")
            on.write_text("")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verifier.main(["--pair", str(off), str(on), "--output", str(result)]), 1)


if __name__ == "__main__":
    unittest.main()
