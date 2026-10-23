"""Summarize immutable addendum-5 acquisitions without pooling consumer gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory
result = {}
for acquisition in ("point47", "gaussian47"):
    cases = json.loads((root / acquisition / "summary.json").read_text())
    totals = defaultdict(lambda: dict(
        fixtures=0, frames=0, pitch=[0, 0], respawn=[0, 0],
        flora47=[0, 0], flora8=[0, 0], flora1=[0, 0],
        birth=[[0.0] * 3 for _ in range(2)],
        pitch_targets=0, respawn_targets=0, flora_targets=0,
        interval_checks=0, interval_failures=0,
        near_margin_1e3=0, production_ties_1e6=0,
        worst_c_excess_over_current=0.0, worst_r_deficit_over_current=0.0))
    for case in cases:
        key = (case["variant"], case["category"], case["resolution"])
        total = totals[key]
        total["fixtures"] += 1
        for field in ("frames", "pitch_targets", "respawn_targets", "near_margin_1e3", "production_ties_1e6"):
            total[field] += case["totals"][field]
        total["flora_targets"] += case["flora_targets"]
        for field, source in (("pitch", case["totals"]["pitch"]),
                              ("respawn", case["totals"]["respawn"]),
                              ("flora1", case["totals"]["flora"]),
                              ("flora8", case["flora8_new_current"]),
                              ("flora47", case["flora47_new_current"])):
            for index in range(2):
                total[field][index] += source[index]
        for index in range(2):
            for band in range(3):
                total["birth"][index][band] += case["totals"]["birth"][index][band]
        for field in ("worst_c_excess_over_current", "worst_r_deficit_over_current"):
            total[field] = max(total[field], case[field])
    witnesses = []
    for line in (root / acquisition / "consumer-rows.jsonl").open():
        row = json.loads(line)
        total = totals[row["variant"], row["category"], row["resolution"]]
        for interval in row["intervals"]:
            total["interval_checks"] += 1
            total["interval_failures"] += not interval["pass"]
            if not interval["pass"]:
                witnesses.append(dict(fixture=row["fixture"], variant=row["variant"],
                                      frame=row["frame"], interval=interval))
    table = []
    for (variant, category, resolution), total in sorted(totals.items()):
        total.update(variant=variant, category=category, resolution=resolution)
        total["birth_mean_l1"] = [[v / total["frames"] for v in band] for band in total["birth"]]
        total["consumer_pass"] = dict(
            pitch=total["pitch"][0] <= total["pitch"][1] if total["pitch_targets"] else None,
            respawn=total["respawn"][0] <= total["respawn"][1] if total["respawn_targets"] else None,
            flora47=total["flora47"][0] <= total["flora47"][1] if total["flora_targets"] else None,
            birth=[a <= b for a, b in zip(*total["birth"])])
        total["consumer_gate_applies"] = resolution != "unresolved"
        total["interval_gate_applies"] = total["interval_checks"] > 0
        table.append(total)
    result[acquisition] = dict(table=table, interval_failure_witnesses=witnesses)

stage_rows = [json.loads(line) for line in (root / "diagnosis/stage-rows.jsonl").open()]
stage_index = {(r["fixture"]["name"], r["beta"], r["analytic"], r["frame"]): r for r in stage_rows}
comparisons = []
stage_summary = []
for row in stage_rows:
    name = row["fixture"]["name"]
    if name.endswith("_direct"):
        residual = stage_index[name.removesuffix("_direct"), row["beta"], row["analytic"], row["frame"]]
        comparisons.append(dict(fixture=name, beta=row["beta"], s1=row["analytic"], frame=row["frame"],
            field_max={key: max(abs(a - b) for a, b in zip(row[key], residual[key])) for key in ("h01", "r01", "c")}))
    if row["frame"] != 63:
        continue
    components = []
    for index, component in enumerate(row["components"]):
        channels = [c for c in row["channels"] if c["component"] == index and c["edge"] > 0]
        stats = dict(component)
        for field, name_suffix in (("power_after_gate", "power"), ("mass", "mass")):
            total = sum(c[field] * c["edge"] for c in channels)
            stats["outside_half_bin_" + name_suffix + "_fraction"] = sum(
                c[field] * c["edge"] for c in channels if abs(c["cent"]) > 6.25) / total
        stats["largest_deposits"] = sorted(channels, key=lambda c: c["mass"] * c["edge"], reverse=True)[:6]
        components.append(stats)
    stage_summary.append(dict(fixture=name, beta=row["beta"], s1=row["analytic"],
        components=components, relative_density_l1=row["relative_density_l1"], fields=row["fields"]))
result["stage_diagnosis"] = stage_summary
result["direct_vs_residual"] = comparisons
result["notes"] = [
    "Every interval uses the literal pairwise ERB classification, with no numerical tolerance.",
    "Pure unresolved-pair full consumers are diagnostics only; their interval comparisons are gates.",
    "Mixed harmonic complexes retain the original full-consumer evaluation; each unresolved pair is also recorded.",
    "No-target consumer cells are null rather than pass; no near-tie judgments are excluded.",
    "Exit zero for acquisition establishes complete data, not scientific acceptance.",
]
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
for acquisition in ("point47", "gaussian47"):
    print(acquisition)
    for row in result[acquisition]["table"]:
        if row["variant"] not in (4, 6):
            continue
        print(row["variant"], row["category"], row["resolution"], "pitch", row["pitch"],
              "respawn", row["respawn"], "flora47", row["flora47"],
              "birth_mean", row["birth_mean_l1"],
              "interval_failed/total", row["interval_failures"], row["interval_checks"])
print("direct_vs_residual_max", {key: max(r["field_max"][key] for r in comparisons) for key in ("h01", "r01", "c")})
