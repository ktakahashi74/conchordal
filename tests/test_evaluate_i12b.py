"""Independent I12b assessor counterexamples."""

import copy
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_i12b as assay  # noqa: E402


def synthetic_decision():
    handle = {"bus": 0, "epoch": 0, "generation": 1}
    issue = {"model": "hazard", "version": 1, "bus": 0, "epoch": 0,
             "sample_rate": 48000, "issued_at": 0, "horizon_end": 192000,
             "groups": [None] * 7}
    forecast = {"group": handle, "source_start": 0, "source_end": 0,
                "available": 0, "issued_at": 0, "horizon_end": 192000,
                "model": "hazard", "version": 1, "reset_unknown": False,
                "elapsed_seconds": [0.0, 0.0]}
    issue["groups"][0] = {"group": handle, "periodic": True, "forecast": forecast,
                           "cdf": [0.0] * 50 + [1.0] * 151, "unknown": None}
    candidates = [{"offset": slot - 2, "at": slot * 8000.0,
                   "base_cost": 0.7 if slot == 0 else 1.0 if slot == 6 else 3.0}
                  for slot in range(23)]
    arrival_input = {"publication": {"issue": issue, "profile_key": None},
                     "received_at": 0, "current_epoch": 0, "owner": [1, 1, 1],
                     "current_profile_key": None, "bindings": [None] * 8,
                     "bindings_overflow": False}
    decision = {"now": 100000, "voice_id": 1, "sample_rate": 48000, "width": 1000.0,
            "candidates": candidates, "coupling": 1.0, "arrival_weight": 1.0,
            "onset_allowed": False, "forecast_available_through_frame": None,
            "own_band_energy": [1.0], "overlap_sensitivity": 0.0,
            "skipped_cycles": 0,
            "arrival": {"input": arrival_input}}
    for key in assay.NONARRIVAL_KEYS:
        decision.setdefault(key, None)
    for candidate in candidates:
        for key in assay.CANDIDATE_KEYS:
            candidate.setdefault(key, None)
    return decision


def period_observations():
    full, cut = [], []
    for sample in (100, 200, 300):
        for bus in (0, 1):
            handle = {"bus": bus, "epoch": 0, "generation": 1}
            group = {"ledger": {"group": handle}, "peaks": [1],
                     "period_source": "observed",
                     "forecast": {"group": handle, "last_accent": 20,
                                  "elapsed_seconds": [1.0, 1.0],
                                  "reset_unknown": False}}
            observation = {"bus": bus, "source_epoch": 0, "sample_rate": 48000,
                           "frame_id": sample, "support_start_sample": sample - 10,
                           "support_end_sample": sample,
                           "available_sample": sample, "input_end_sample": sample,
                           "mono_mean_square": 0.25, "acoustic": {"features": []},
                           "gesture": {"stable": True},
                           "period": {"received_at": sample, "groups": [group],
                                      "residual": None}}
            full.append({"type": "temporal_observation", "observation": observation})
            other = copy.deepcopy(observation)
            other["period_accent_cut_deliveries"] = 1
            if (bus == 1 and sample >= 200) or (bus == 0 and sample >= 300):
                other["period"]["groups"][0]["peaks"] = []
            cut.append({"type": "temporal_observation", "observation": other})
    return full, cut


class RecalculationTests(unittest.TestCase):
    def test_packet_cut_changes_cost_and_selection(self):
        decision = synthetic_decision()
        full = assay.recompute_selection(decision)
        self.assertTrue(full["valid"])
        self.assertEqual(full["selected_offset"], 4)
        self.assertEqual(full["eligible_count"], 1)
        packet = copy.deepcopy(decision["arrival"]["input"])
        packet["publication"]["issue"]["groups"][0]["periodic"] = False
        cut = assay.recompute_selection(decision, packet)
        self.assertTrue(cut["valid"])
        self.assertEqual(cut["eligible_count"], 0)
        self.assertEqual(cut["selected_offset"], -2)
        self.assertIsNone(cut["terms"][0])

    def test_weight_zero_keeps_known_p_and_zero_term(self):
        decision = synthetic_decision()
        decision["arrival_weight"] = 0.0
        outcome = assay.recompute_selection(decision)
        self.assertEqual(outcome["eligible_count"], 1)
        self.assertIsNotNone(outcome["probabilities"][6])
        self.assertTrue(assay.same_f32(outcome["terms"][6], 0.0))
        self.assertEqual(outcome["selected_offset"], -2)

    def test_invalid_receipt_and_cdf_refused(self):
        decision = synthetic_decision()
        decision["arrival"]["input"]["received_at"] = decision["now"] + 1
        self.assertFalse(assay.recompute_selection(decision)["valid"])
        decision["arrival"]["input"]["received_at"] = 0
        decision["arrival"]["input"]["publication"]["issue"]["groups"][0]["cdf"][0] = 2.0
        with self.assertRaises(ValueError):
            assay.recompute_selection(decision)


class CausalityTests(unittest.TestCase):
    def test_all_bus_period_window_does_not_move_on_other_bus_equality(self):
        full, cut = period_observations()
        result = assay.scan_period_observations(full, cut)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["first_all_bus"]["last_joint_equal_cut"], 100)
        self.assertEqual(result["first_all_bus"]["first_observed_peak_cut"], 200)
        self.assertEqual(result["first_all_bus"]["buses"], [1])
        self.assertEqual(result["first_habitat"]["first_observed_peak_cut"], 300)
        self.assertEqual(result["habitat_affected_handles"],
                         [{"bus": 0, "epoch": 0, "generation": 1}])

    def test_unchanged_peak_does_not_hide_earlier_upstream_mutation(self):
        full, cut = period_observations()
        cut[0]["observation"]["mono_mean_square"] = 0.5
        result = assay.scan_period_observations(full, cut)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"],
                         "upstream observation differs before first period peak")

    def test_real_packet_period_witness_narrows_both_observation_windows(self):
        full, cut = period_observations()
        handle = {"bus": 0, "epoch": 0, "generation": 1}
        witness = {"bus": 0, "issued_at": 150, "groups": [handle]}
        result = assay.scan_period_observations(full, cut, [witness])
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["first_all_bus"]["last_joint_equal_cut"], 100)
        self.assertEqual(result["first_all_bus"]["first_period_evidence_cut"], 150)
        self.assertEqual(result["first_all_bus"]["first_observed_peak_cut"], 200)
        self.assertEqual(result["first_habitat"]["last_equal_cut"], 100)
        self.assertEqual(result["first_habitat"]["first_period_evidence_cut"], 150)
        self.assertEqual(result["first_habitat"]["first_observed_peak_cut"], 300)
        cut[0]["observation"]["mono_mean_square"] = 0.5
        result = assay.scan_period_observations(full, cut, [witness])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"],
                         "upstream observation differs before first period peak")

    def test_packet_before_first_equal_cut_cannot_borrow_a_later_prefix(self):
        full, cut = period_observations()
        witness = {"bus": 0, "issued_at": 50,
                   "groups": [{"bus": 0, "epoch": 0, "generation": 1}]}
        result = assay.scan_period_observations(full, cut, [witness])
        self.assertFalse(result["passed"])
        self.assertIsNone(result["first"]["last_joint_equal_cut"])

    def test_upstream_at_first_observed_peak_is_unresolved_not_prior(self):
        full, cut = period_observations()
        cut[3]["observation"]["mono_mean_square"] = 0.5
        result = assay.scan_period_observations(full, cut)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["upstream_differences_at_first_observed_peak"],
                         [{"sample": 200, "bus": 1}])

    def test_missing_other_bus_cannot_become_joint_equal_cut(self):
        full, cut = period_observations()
        cut.pop(1)
        result = assay.scan_period_observations(full, cut)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "missing or unpaired observation cut")

    def test_preperiod_decision_input_change_fails_without_selection_change(self):
        full, cut = period_observations()
        left = synthetic_decision()
        left["now"] = 50
        right = copy.deepcopy(left)
        right["coupling"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "full.jsonl", Path(directory) / "cut.jsonl"
            a.write_text("\n".join(json.dumps(row) for row in
                                   full + [{"type": "participation_decision", **left}]) + "\n")
            b.write_text("\n".join(json.dumps(row) for row in
                                   cut + [{"type": "participation_decision", **right}]) + "\n")
            result = assay.backward_pair(a, b)
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"],
                             "decision prefix changed before period window")

    def test_pcm_difference_frame_and_period_prefix_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.wav", Path(directory) / "right.wav"
            for path, changed in ((left, False), (right, True)):
                values = bytearray(400 * 2)
                if changed:
                    values[80 * 2] = 1
                    values[210 * 2] = 1
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(48000)
                    output.writeframes(bytes(values))
            self.assertEqual(assay.first_wav_difference(left, right)["first_frame"], 80)
            self.assertEqual(assay.first_wav_difference(left, right, 100)["first_frame"], 210)
            self.assertLess(assay.first_wav_difference(left, right)["first_frame"], 100)

    def test_cut_report_counter_monotone_and_not_forced_to_profile_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            observations = [
                {"type": "temporal_observation", "observation": {
                    "bus": 0, "period_accent_cut_deliveries": value}}
                for value in (0, 1, 2)]
            path.write_text("\n".join(json.dumps(row) for row in observations) + "\n")
            result = assay.cut_counter_trace(path, "backward-cut", 3)
            self.assertEqual(result["last_by_bus"], {"0": 2})
            self.assertFalse(result["terminal_equality_claimed"])
            observations[2]["observation"]["period_accent_cut_deliveries"] = 0
            path.write_text("\n".join(json.dumps(row) for row in observations) + "\n")
            with self.assertRaisesRegex(ValueError, "decreased"):
                assay.cut_counter_trace(path, "backward-cut", 3)
            observations[2]["observation"]["period_accent_cut_deliveries"] = 2
            path.write_text("\n".join(json.dumps(row) for row in observations) + "\n")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                assay.cut_counter_trace(path, "backward-cut", 1)
            observations[0]["observation"]["period_accent_cut_deliveries"] = None
            path.write_text("\n".join(json.dumps(row) for row in observations) + "\n")
            with self.assertRaisesRegex(ValueError, "outside backward cut"):
                assay.cut_counter_trace(path, "full")

    def test_observed_publication_mismatch_and_receipt_rewrite_refused(self):
        decision = synthetic_decision()
        observation = {"observation": {"bus": 0, "source_epoch": 0,
                       "available_sample": 0, "support_end_sample": 0,
                       "arrival": copy.deepcopy(decision["arrival"]["input"]["publication"])}}
        self.assertTrue(assay.publication_coverage([observation], [decision])["passed"])
        changed = copy.deepcopy(decision)
        changed["arrival"]["input"]["publication"]["issue"]["groups"][0]["periodic"] = False
        result = assay.publication_coverage([observation], [changed])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"],
                         "decision issue changed observed packet or profile key")
        changed = copy.deepcopy(decision)
        changed["arrival"]["input"]["received_at"] = 1
        result = assay.publication_coverage([observation], [decision, changed])
        self.assertEqual(result["reason"], "same issue changed packet, profile key or receipt")
        changed = copy.deepcopy(decision)
        changed["arrival"]["input"]["publication"]["profile_key"] = {"id": 7}
        result = assay.publication_coverage([observation], [changed])
        self.assertEqual(result["reason"],
                         "decision issue changed observed packet or profile key")

    def test_direct_packet_swap_rejects_changed_clock_or_owner(self):
        full = synthetic_decision()
        cut = copy.deepcopy(full)
        cut["arrival"]["input"]["publication"]["issue"]["issued_at"] = 1
        self.assertEqual(assay.direct_packet_swap(full, cut)["reason"],
                         "packet lineage or clock differs")
        cut = copy.deepcopy(full)
        cut["arrival"]["input"]["owner"][0] = 2
        self.assertEqual(assay.direct_packet_swap(full, cut)["reason"], "owner differs")

    def test_missing_raw_prefix_rejected_before_period_effect(self):
        full, cut = period_observations()
        cut[0]["observation"]["acoustic"]["features"].append({"detector": {"accent": 7}})
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.jsonl"
            b = Path(directory) / "b.jsonl"
            a.write_text("\n".join(json.dumps(row) for row in full) + "\n")
            b.write_text("\n".join(json.dumps(row) for row in cut) + "\n")
            result = assay.backward_pair(a, b)
            self.assertEqual(result["reason"], "period observation chain invalid")
            self.assertEqual(result["observations"]["reason"],
                             "upstream observation differs before first period peak")

    def test_candidate_shape_is_not_zip_truncated(self):
        first = {"candidates": [{"offset": i - 2} for i in range(23)]}
        second = copy.deepcopy(first)
        second["candidates"].pop()
        self.assertFalse(assay.candidate_structure(first, second))
        first = synthetic_decision()
        second = copy.deepcopy(first)
        second["candidates"][2] = None
        self.assertIn("candidate_structure", assay.nonarrival_differences(first, second))

    def test_forward_period_prefix_mutation_refused(self):
        observation = {"type": "temporal_observation", "observation": {
            "bus": 0, "support_end_sample": 512, "period": {"groups": [None] * 7}}}
        mutated = copy.deepcopy(observation)
        mutated["observation"]["period"]["groups"][0] = {"peak": 4}
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.jsonl"
            b = Path(directory) / "b.jsonl"
            a.write_text(json.dumps(observation) + "\n")
            b.write_text(json.dumps(mutated) + "\n")
            result = assay.forward_observation_prefix(a, b, 512)
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "period/grouping/forecast prefix differs")

    def test_power_log_event_and_false_assertion_refused(self):
        good = {"command_exits": {"pmset_batt": 0, "ioreg": 0,
                                  "pmset_assertions": 0, "ps": 0},
                "power_source": "AC Power", "clamshell_state": "No",
                "pmset_batt": "Now drawing from 'AC Power'\n",
                "ioreg_clamshell": '"AppleClamshellState" = No\n',
                "build_processes": [], "caffeinate_for_python": True, "passed": True,
                "pmset_assertions": "pid 88(caffeinate): PreventUserIdleSystemSleep\n"
                                    " Details: caffeinate asserting on behalf of 'python' (pid 42)"}
        self.assertTrue(assay.valid_power(good, 42))
        self.assertFalse(assay.valid_power(good, 43))
        good["build_processes"] = [{"pid": "7", "command": "cargo"}]
        self.assertFalse(assay.valid_power(good, 42))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pmset.log"
            path.write_text("2026-09-24 10:00:05 -0700 Sleep Entering Sleep state\n")
            self.assertEqual(len(assay.control_events(path, "2026-09-24T17:00:05+00:00",
                                                       "2026-09-24T17:00:06+00:00")), 1)

    def test_run_pmset_json_exit_events_and_window(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "block-pmset.log"
            raw.write_text("2026-09-24 10:00:05 -0700 Sleep Entering Sleep state\n")
            saved = {"started_at": "2026-09-24T17:00:00+00:00",
                     "finished_at": "2026-09-24T17:00:10+00:00", "pmset_events": []}
            audit = {"command": ["pmset", "-g", "log"], "exit": 0,
                     "events": [], "raw_sha256": "a" * 64, "raw_bytes": 100,
                     "window_start": "2026-09-24T17:00:01+00:00",
                     "window_end": "2026-09-24T17:00:04+00:00"}
            self.assertTrue(assay.valid_run_pmset_audit(audit, saved, raw))
            failed = copy.deepcopy(audit)
            failed["exit"] = 1
            self.assertFalse(assay.valid_run_pmset_audit(failed, saved, raw))
            failed = copy.deepcopy(audit)
            failed["events"] = ["invented event"]
            self.assertFalse(assay.valid_run_pmset_audit(failed, saved, raw))
            failed = copy.deepcopy(audit)
            failed["window_end"] = "2026-09-24T17:00:06+00:00"
            self.assertFalse(assay.valid_run_pmset_audit(failed, saved, raw))

    def test_real_stage2_decision_shape_supports_unmodified_packet_replay(self):
        path = (Path(__file__).resolve().parents[1] /
                "target/i11-stage2-20260924/acquisition/094-modal-pulse-flow-4-body-on-render-0.jsonl")
        if not path.is_file():
            self.skipTest("sealed I11-2 report is unavailable")
        rows = assay.read_report(path, ("participation_decision",))["participation_decision"]
        matched = None
        for full in rows:
            if not full["arrival"]["eligible_count"]:
                continue
            unchanged = assay.direct_packet_swap(full, full)
            self.assertFalse(unchanged["passed"])
            self.assertEqual(unchanged["reason"], "no arrival term difference")
            cut = copy.deepcopy(full)
            for group in cut["arrival"]["input"]["publication"]["issue"]["groups"]:
                if group is not None:
                    group["periodic"] = False
            calculation = assay.recompute_selection(cut)
            if calculation["selected_offset"] == full["selected_offset"]:
                continue
            cut["arrival"].update(calculation["status"])
            for candidate, probability, term, cost in zip(
                    cut["candidates"], calculation["probabilities"],
                    calculation["terms"], calculation["costs"]):
                if candidate is not None:
                    candidate.update(arrival_p=probability, arrival_term=term, cost=cost)
            cut["selected_offset"] = calculation["selected_offset"]
            cut["skipped"] = calculation["skipped"]
            chosen = next(c for c in cut["candidates"] if c is not None
                          and c["offset"] == calculation["selected_offset"])
            cut["selected_at"] = chosen["at"]
            cut["selected_cost"] = chosen["cost"]
            cut["selected_base_cost"] = chosen["base_cost"]
            cut["reference_cost"] = cut["candidates"][2]["cost"]
            self.assertEqual(assay.verify_decision(cut), [])
            matched = assay.direct_packet_swap(full, cut)
            break
        self.assertIsNotNone(matched)
        self.assertTrue(matched["passed"])
        self.assertEqual(matched["swapped_eligible"], 0)


class PerformanceTests(unittest.TestCase):
    def test_one_sided_upper_and_signed_cut(self):
        fields = assay.PERFORMANCE_FIELDS
        profiles = {}
        for arm, value in (("old-full", 100.0), ("full", 90.0),
                           ("forward-cut", 80.0), ("backward-cut", 60.0)):
            profiles[arm] = {rep: {**{field: value for field in fields},
                                   "arrival_known_decisions": 1,
                                   "period_accent_cut_deliveries": 1}
                             for rep in (1, 2, 3)}
        result = assay.performance_cell("sine-pulse-flow-4", "report", profiles,
                                        {field: 0.0 for field in fields})
        self.assertEqual(len(result["nonregression"]), 10)
        self.assertEqual(len(result["cut_diagnostics"]), 20)
        self.assertTrue(all(check["passed"] for check in result["nonregression"]))
        self.assertTrue(any(not check["old_absolute_with_floor"] for check in result["nonregression"]))
        self.assertEqual(result["cut_diagnostics"][0]["cut_minus_full"], -10.0)

    def test_each_repetition_must_be_eligible(self):
        fields = assay.PERFORMANCE_FIELDS
        profiles = {arm: {rep: {**{field: 1.0 for field in fields},
                                "arrival_known_decisions": 1,
                                "period_accent_cut_deliveries": 1}
                          for rep in (1, 2, 3)}
                    for arm in ("old-full", "full", "forward-cut", "backward-cut")}
        profiles["backward-cut"][2]["period_accent_cut_deliveries"] = 0
        result = assay.performance_cell("modal-pulse-flow-16", "no-report", profiles,
                                        {field: 0.0 for field in fields})
        self.assertEqual(sum(not item["passed"] for item in result["eligibility"]), 1)


if __name__ == "__main__":
    unittest.main()
