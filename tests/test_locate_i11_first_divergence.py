"""Checks of the I11-1 §5.5(a) first-divergence locator; no renderer required."""

import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "locate_i11_first_divergence", SCRIPTS / "locate_i11_first_divergence.py")
loc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loc)


def decision(now, offset, power, source):
    base = {key: 0 for key in loc.DECISION_INPUTS}
    candidate = dict({key: 0 for key in loc.CANDIDATE_INPUTS}, offset=0)
    candidates = [None] * 23
    candidates[2] = candidate
    return dict(base, type="participation_decision", voice_id=1, now=now, selected_offset=offset,
                skipped=False, footprint_power=power, footprint_source=source,
                candidates=candidates)


def locate(body, proxy):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / "body.jsonl", Path(tmp) / "proxy.jsonl"]
        for path, records in zip(paths, (body, proxy)):
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return loc.locate(*paths)


def exit_status(body, proxy):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / "body.jsonl", Path(tmp) / "proxy.jsonl"]
        for path, records in zip(paths, (body, proxy)):
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with contextlib.redirect_stdout(io.StringIO()):
            return loc.main([str(path) for path in paths])


class FirstDivergenceTest(unittest.TestCase):
    def setUp(self):
        self.body = [decision(0, 0, [1.0], "body"), decision(512, 3, [1.0], "body")]
        self.proxy = [decision(0, 0, [0.5], "proxy(setting)"), decision(512, 4, [0.5], "proxy(setting)")]

    def test_a_divergence_entered_through_powers_alone(self):
        result = locate(self.body, self.proxy)
        self.assertEqual((result["status"], result["index"], result["now"]), ("diverged", 1, 512))
        self.assertTrue(result["powers_differ"])
        self.assertTrue(result["sources_valid"])
        self.assertEqual(result["other_differing_inputs"], [])
        self.assertIsNone(result["first_prior_input_difference"])
        self.assertEqual(result["diverged_outputs"], ["selected_offset"])
        self.assertEqual(exit_status(self.body, self.proxy), 0)

    def test_another_differing_input_is_named(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[1]["footprint_d_samples"] = 7
        proxy[1]["candidates"][2]["external_energy"] = 1
        result = locate(self.body, proxy)
        self.assertEqual(result["other_differing_inputs"],
                         ["footprint_d_samples", "candidates.external_energy"])
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_equal_powers_cannot_establish_power_causality(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[1]["footprint_power"] = self.body[1]["footprint_power"]
        self.assertFalse(locate(self.body, proxy)["powers_differ"])
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_a_proxy_fallback_is_not_body_evidence(self):
        body = copy.deepcopy(self.body)
        body[1]["footprint_source"] = "proxy(absent)"
        self.assertFalse(locate(body, self.proxy)["sources_valid"])
        self.assertEqual(exit_status(body, self.proxy), 1)

        proxy = copy.deepcopy(self.proxy)
        proxy[1]["footprint_source"] = "proxy(stale)"
        self.assertFalse(locate(self.body, proxy)["sources_valid"])
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_same_d_but_different_record_fails(self):
        body = copy.deepcopy(self.body)
        proxy = copy.deepcopy(self.proxy)
        body[1]["footprint_identity"] = {"recipe": "a"}
        proxy[1]["footprint_identity"] = {"recipe": "b"}
        body[1]["current_identity"] = {"recipe": "a"}
        proxy[1]["current_identity"] = {"recipe": "a"}
        self.assertEqual(body[1]["footprint_d_samples"], proxy[1]["footprint_d_samples"])
        self.assertEqual(locate(body, proxy)["other_differing_inputs"], ["footprint_identity"])
        self.assertEqual(exit_status(body, proxy), 1)

    def test_current_identity_and_truncation_differences_fail(self):
        for key, value in (("current_identity", {"recipe": "b"}),
                           ("footprint_truncated", True)):
            with self.subTest(key=key):
                proxy = copy.deepcopy(self.proxy)
                proxy[1][key] = value
                self.assertEqual(locate(self.body, proxy)["other_differing_inputs"], [key])
                self.assertEqual(exit_status(self.body, proxy), 1)

    def test_proxy_gain_inputs_must_match(self):
        for key, value in (("sound_hold_sec", 1.0),
                           ("sound_adsr", [0.1, 0.2, 0.3, 0.4])):
            with self.subTest(key=key):
                proxy = copy.deepcopy(self.proxy)
                proxy[1][key] = value
                self.assertEqual(locate(self.body, proxy)["other_differing_inputs"], [key])
                self.assertEqual(exit_status(self.body, proxy), 1)

    def test_sample_rate_and_delivery_frames_must_match(self):
        for key, value in (("sample_rate", 48000),
                           ("footprint_requested_at", 100),
                           ("footprint_received_at", 101)):
            with self.subTest(key=key):
                proxy = copy.deepcopy(self.proxy)
                proxy[1][key] = value
                self.assertEqual(locate(self.body, proxy)["other_differing_inputs"], [key])
                self.assertEqual(exit_status(self.body, proxy), 1)

    def test_prior_input_difference_is_retained_through_first_choice_difference(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[0]["footprint_delay"] = 1
        result = locate(self.body, proxy)
        self.assertEqual(result["status"], "diverged")
        self.assertEqual(result["index"], 1)
        self.assertEqual(result["first_prior_input_difference"],
                         {"index": 0, "inputs": ["footprint_delay"]})
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_skip_change_is_the_first_divergence(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[0]["skipped"] = True
        result = locate(self.body, proxy)
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["diverged_outputs"], ["skipped"])
        self.assertEqual(exit_status(self.body, proxy), 0)

    def test_tiny_input_difference_is_still_a_difference(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[1]["footprint_d_samples"] = 1e-12
        self.assertEqual(locate(self.body, proxy)["other_differing_inputs"],
                         ["footprint_d_samples"])
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_candidate_structure_mismatch_is_not_accepted(self):
        for alteration in (lambda c: c.pop(),
                           lambda c: c.__setitem__(2, None),
                           lambda c: c[2].__setitem__("offset", 1)):
            with self.subTest(alteration=alteration):
                proxy = copy.deepcopy(self.proxy)
                alteration(proxy[0]["candidates"])
                self.assertEqual(locate(self.body, proxy)["status"],
                                 "candidate_structure_differs")
                self.assertEqual(exit_status(self.body, proxy), 1)

    def test_schedule_mismatch_is_not_accepted(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[0]["now"] = 1
        self.assertEqual(locate(self.body, proxy)["status"],
                         "schedules_part_before_a_choice_differs")
        self.assertEqual(exit_status(self.body, proxy), 1)

    def test_extra_decision_is_not_silently_truncated(self):
        result = locate(self.body[:1], self.proxy)
        self.assertEqual(result["status"], "schedules_part_before_a_choice_differs")
        self.assertEqual((result["body_decisions"], result["proxy_decisions"]), (1, 2))
        self.assertEqual(exit_status(self.body[:1], self.proxy), 1)

    def test_identical_choices_do_not_diverge(self):
        result = locate(self.body[:1], self.proxy[:1])
        self.assertEqual(result, {"status": "no_divergence", "compared": 1,
                                  "first_prior_input_difference": None})
        self.assertEqual(exit_status(self.body[:1], self.proxy[:1]), 0)

    def test_no_choice_divergence_does_not_hide_input_difference(self):
        proxy = copy.deepcopy(self.proxy[:1])
        proxy[0]["memory"] = 1
        result = locate(self.body[:1], proxy)
        self.assertEqual(result["status"], "no_divergence")
        self.assertEqual(result["first_prior_input_difference"],
                         {"index": 0, "inputs": ["memory"]})
        self.assertEqual(exit_status(self.body[:1], proxy), 1)


if __name__ == "__main__":
    unittest.main()
