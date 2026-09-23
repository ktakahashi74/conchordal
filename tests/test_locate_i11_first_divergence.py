"""Checks of the I11-1 §5.5(a) first-divergence locator; no renderer required."""

import copy
import importlib.util
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
    candidate = {key: 0 for key in loc.CANDIDATE_INPUTS}
    return dict(base, type="participation_decision", voice_id=1, now=now, selected_offset=offset,
                footprint_power=power, footprint_source=source, candidates=[candidate])


def locate(body, proxy):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / "body.jsonl", Path(tmp) / "proxy.jsonl"]
        for path, records in zip(paths, (body, proxy)):
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return loc.locate(*paths)


class FirstDivergenceTest(unittest.TestCase):
    def setUp(self):
        self.body = [decision(0, 0, [1.0], "body"), decision(512, 3, [1.0], "body")]
        self.proxy = [decision(0, 0, [0.5], "proxy(setting)"), decision(512, 4, [0.5], "proxy(setting)")]

    def test_a_divergence_entered_through_powers_alone(self):
        result = locate(self.body, self.proxy)
        self.assertEqual((result["status"], result["index"], result["now"]), ("diverged", 1, 512))
        self.assertTrue(result["powers_differ"])
        self.assertEqual(result["other_differing_inputs"], [])

    def test_another_differing_input_is_named(self):
        proxy = copy.deepcopy(self.proxy)
        proxy[1]["footprint_d_samples"] = 7
        proxy[1]["candidates"][0]["external_energy"] = 1
        result = locate(self.body, proxy)
        self.assertEqual(result["other_differing_inputs"],
                         ["footprint_d_samples", "candidates.external_energy"])

    def test_identical_choices_do_not_diverge(self):
        result = locate(self.body[:1], self.proxy[:1])
        self.assertEqual(result, {"status": "no_divergence", "compared": 1})


if __name__ == "__main__":
    unittest.main()
