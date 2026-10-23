"""Keep addendum-6 consumer gates, numerical controls, and onset diagnostics separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
root = p.parse_args().directory
output = {}
for acquisition in ("boxcar", "context-reference"):
    groups = defaultdict(lambda: dict(
        fixtures=0, frames=0, pitch=[0, 0], respawn=[0, 0], flora47=[0, 0],
        birth=[[0.0] * 3 for _ in range(2)], near_far=[[0, 0], [0, 0]],
        pitch_targets=0, respawn_targets=0, flora_targets=0, near_margin_1e3=0,
        production_ties_1e6=0, interval_checks=0, interval_exceedances=0))
    cases = json.loads((root / acquisition / "summary.json").read_text())
    for case in cases:
        key = (case["variant"], case["category"], case["resolution"])
        g = groups[key]
        g["fixtures"] += 1
        for field in ("frames", "pitch_targets", "respawn_targets", "near_margin_1e3", "production_ties_1e6"):
            g[field] += case["totals"][field]
        g["flora_targets"] += case["flora_targets"]
        for i in range(2):
            g["pitch"][i] += case["totals"]["pitch"][i]
            g["respawn"][i] += case["totals"]["respawn"][i]
            g["flora47"][i] += case["flora47_new_current"][i]
            for j in range(3):
                g["birth"][i][j] += case["totals"]["birth"][i][j]
            for j in range(2):
                g["near_far"][i][j] += case["flora47_near_far_new_current"][i][j]
    witnesses = []
    for line in (root / acquisition / "consumer-rows.jsonl").open():
        row = json.loads(line)
        g = groups[row["variant"], row["category"], row["resolution"]]
        for interval in row["intervals"]:
            g["interval_checks"] += 1
            g["interval_exceedances"] += not interval["pass"]
        if row["fixture"].startswith("loo_110.0_40.0") and (
                row["new"]["pitch_errors"] or row["new"]["respawn_errors"]):
            witnesses.append(row)
    table = []
    for (variant, category, resolution), g in sorted(groups.items()):
        g.update(variant=variant, category=category, resolution=resolution,
                 consumer_gate_applies=resolution != "unresolved", interval_is_gate=False)
        g["birth_mean_l1"] = [[v / g["frames"] for v in band] for band in g["birth"]]
        g["consumer_pass"] = dict(
            pitch=g["pitch"][0] <= g["pitch"][1] if g["pitch_targets"] else None,
            respawn=g["respawn"][0] <= g["respawn"][1] if g["respawn_targets"] else None,
            flora47=g["flora47"][0] <= g["flora47"][1] if g["flora_targets"] else None,
            birth=[a <= b for a, b in zip(*g["birth"])])
        table.append(g)
    output[acquisition] = dict(table=table, resolved_pair_witnesses=witnesses)

comparisons = []
for row in output["context-reference"]["table"]:
    original = next(x for x in output["boxcar"]["table"] if
                    all(x[k] == row[k] for k in ("variant", "category", "resolution")))
    comparisons.append(dict(variant=row["variant"], category=row["category"], resolution=row["resolution"],
        same_consumer_pass=original["consumer_pass"] == row["consumer_pass"],
        fast=original["consumer_pass"], reference=row["consumer_pass"]))
output["numerical_control"] = comparisons
output["context_max_error_over_total_power"] = max(
    (json.loads(line)["context_direct_check"] or {}).get("error_over_total_power", 0)
    for line in (root / "boxcar/frame-diagnostics.jsonl").open())

trajectories = defaultdict(list)
for line in (root / "boxcar/onset-rows.jsonl").open():
    row = json.loads(line)
    if row["frame"] - 8 in (0, 1, 3, 7, 15, 31, 46):
        trajectories[row["fixture"], row["variant"]].append({k: row[k] for k in (
            "frame", "after_onset_ms", "density_mass_near_far", "truth_mass_near_far",
            "density_l1_near_far", "new_fields", "current_fields")})
output["onset_trajectories"] = [dict(fixture=k[0], variant=k[1], observations=v) for k, v in trajectories.items()]
output["selected_variant"] = None
for variant in (5, 4, 7, 6):
    rows = [r for r in output["boxcar"]["table"] if r["variant"] == variant and r["consumer_gate_applies"]]
    if rows and all(all(r["consumer_pass"][k] is True for k in ("pitch", "respawn", "flora47"))
                    and all(r["consumer_pass"]["birth"]) for r in rows):
        output["selected_variant"] = variant
        break
output["notes"] = ["Intervals are diagnostic only, including all unresolved pairs in complexes.",
    "Every full consumer gate is per category and resolution subset; no near ties are excluded.",
    "No-target cases are null. Removing far mass is an onset attribution control, never the gate input."]
(root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
print("selected_variant", output["selected_variant"])
print("fast_context_max_error_over_total_power", output["context_max_error_over_total_power"])
print("fast/reference_pass_disagreements", [r for r in comparisons if not r["same_consumer_pass"]])
for r in output["boxcar"]["table"]:
    if r["consumer_gate_applies"]:
        print(r["variant"], r["category"], r["resolution"], r["pitch"], r["respawn"],
              r["flora47"], r["birth_mean_l1"], r["near_far"], r["consumer_pass"])
