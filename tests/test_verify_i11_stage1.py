"""Checks of the I11-1 stage-1 independent reference; no renderer or Rust build required."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("verify_i11_stage1", SCRIPTS / "verify_i11_stage1.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)

IDENTITY = {"source_id": 3, "body_generation": 1, "recipe_hash": [1] * 32}
MEMORY = [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]
PREDICTED = [[0.02, 0.01, 0.005], [0.01, 0.01, 0.01]]
OWN = [1.0, 0.5, 0.25]


def report_fixture():
    """A body decision whose every term follows the registered formulas."""
    power = [1.0 - k / 32 for k in range(v.BINS)]
    d = 300.0
    delays = [(k + 0.5) * d / 16.0 for k in range(v.BINS)]
    powers = [v.f32(p) for p in power]
    decision = dict(
        type="participation_decision", voice_id=3, sample_rate=1000, now=100,
        due_frame=400.0, period_frames=500.0, width=60.0, earliest=100.0,
        coupling=1.0, overlap_sensitivity=0.8, onset_allowed=True, memory=MEMORY,
        own_band_energy=OWN, sound_hold_sec=0.2, sound_adsr=[0.01, 0.05, 0.6, 0.1],
        footprint_source="body", footprint_identity=IDENTITY, current_identity=IDENTITY,
        footprint_requested_at=40, footprint_received_at=80, footprint_d_samples=d,
        footprint_truncated=False, footprint_delay=delays, footprint_power=powers,
        forecast_observed_frame=50, forecast_available_through_frame=100,
        skipped_cycles=0, skipped=False,
    )
    candidates = []
    for offset in v.OFFSETS:
        shift = offset * 60.0 / 2.0 if offset < 0 else offset * 500.0 / 20.0
        at = 400.0 if offset == 0 else 400.0 + shift
        energy = [[0.02, 0.01, 0.005] if at + delay >= 100 else None for delay in delays]
        candidate = dict(
            offset=offset, at=at, displacement_sq=v.f32(((at - 400.0) / 500.0) ** 2),
            context_distance=v.hellinger(MEMORY, PREDICTED), pred_external_band_energy=PREDICTED,
            overlap=v.overlap(OWN, powers, energy), external_energy=energy,
        )
        candidate["cost"] = v.cost_of(decision, candidate, candidate["context_distance"],
                                      candidate["overlap"], powers)
        candidates.append(candidate)
    best_offset, best_cost = v.first_minimum([(c["offset"], c["cost"]) for c in candidates])
    decision.update(
        candidates=candidates, reference_cost=candidates[2]["cost"], selected_offset=best_offset,
        selected_at=candidates[best_offset + 2]["at"], selected_cost=best_cost,
    )
    footprint = dict(
        type="body_footprint", superseded=False, identity=IDENTITY, requested_at=40,
        computed_at=60, received_at=80, d_samples=300, truncated=False, state="body",
        energies=[0.0] * v.BINS, power=power,
    )
    context = dict(
        type="participation_context", voice_id=3, onset_frame=v.rust_round(decision["selected_at"]),
        selected_cost=best_cost, reference_cost=decision["reference_cost"],
        selected_offset=best_offset, footprint_source="body", footprint_received_at=80,
        forecast_observed_frame=50, target_start_frames=[decision["selected_at"] // 10 * 10 + 10, 0],
    )
    return [footprint, decision, context]


def stale_proxy_fixture(source):
    records = report_fixture()
    decision, context = records[1:]
    decision["current_identity"] = {**IDENTITY, "body_generation": 2}
    decision["footprint_source"] = source
    context["footprint_source"] = source
    powers = [v.f32(v.proxy_power(delay / decision["sample_rate"],
                                  decision["sound_hold_sec"], decision["sound_adsr"],
                                  decision["sample_rate"]))
              for delay in decision["footprint_delay"]]
    decision["footprint_power"] = powers
    for candidate in decision["candidates"]:
        candidate["overlap"] = v.overlap(OWN, powers, candidate["external_energy"])
        candidate["cost"] = v.cost_of(decision, candidate, candidate["context_distance"],
                                      candidate["overlap"], powers)
    best_offset, best_cost = v.first_minimum(
        [(candidate["offset"], candidate["cost"]) for candidate in decision["candidates"]])
    decision.update(reference_cost=decision["candidates"][2]["cost"],
                    selected_offset=best_offset,
                    selected_at=decision["candidates"][best_offset + 2]["at"],
                    selected_cost=best_cost)
    context.update(onset_frame=v.rust_round(decision["selected_at"]),
                   selected_cost=best_cost, reference_cost=decision["reference_cost"],
                   selected_offset=best_offset)
    return records


def run(records):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        findings, intervention = v.verify(*v.load(path))
    return findings.summary(), intervention


def failed(summary):
    return {name for name, entry in summary.items() if entry["failed"]}


class VerifyStageOneTest(unittest.TestCase):
    def test_a_consistent_report_passes_every_check(self):
        summary, intervention = run(report_fixture())
        self.assertEqual(failed(summary), set())
        self.assertEqual(summary["5.1 cost"]["checked"], 23)
        self.assertEqual(intervention["body_decisions"], 1)

    def test_stale_record_with_proxy_powers_accepts_both_proxy_sources(self):
        for source in ("proxy(stale)", "proxy(setting)"):
            with self.subTest(source=source):
                summary, _ = run(stale_proxy_fixture(source))
                self.assertEqual(failed(summary), set())
                self.assertEqual(summary["5.3 stale identity falls back"]["checked"], 1)

    def test_stale_record_with_body_source_is_rejected(self):
        records = stale_proxy_fixture("body")
        summary, _ = run(records)
        self.assertIn("5.3 body identity is current", failed(summary))
        self.assertIn("5.3 stale identity falls back", failed(summary))

    def test_setting_proxy_cannot_reuse_old_body_powers(self):
        records = stale_proxy_fixture("proxy(setting)")
        records[1]["footprint_power"] = records[0]["power"]
        summary, _ = run(records)
        self.assertIn("5.1 proxy powers", failed(summary))

    def test_each_violation_is_caught_by_its_check(self):
        def cost(records):
            records[1]["candidates"][5]["cost"] += 0.01

        def future(records):
            records[1]["now"] = 70

        def stale(records):
            records[1]["current_identity"] = {**IDENTITY, "body_generation": 2}

        def superseded(records):
            records[0]["superseded"] = True

        def early_energy(records):
            records[1]["forecast_available_through_frame"] = 10_000

        def selection(records):
            records[1]["selected_offset"] = 20

        def power(records):
            records[1]["footprint_power"][3] = 0.5

        cases = [
            (cost, "5.1 cost"),
            (future, "5.3 footprint received before the decision"),
            (stale, "5.3 body identity is current"),
            (superseded, "5.3 superseded record unused"),
            (early_energy, "5.3 external energy after availability"),
            (selection, "5.1 selection"),
            (power, "5.1 body powers"),
        ]
        for mutate, check in cases:
            records = copy.deepcopy(report_fixture())
            mutate(records)
            with self.subTest(check=check):
                self.assertIn(check, failed(run(records)[0]))


if __name__ == "__main__":
    unittest.main()
