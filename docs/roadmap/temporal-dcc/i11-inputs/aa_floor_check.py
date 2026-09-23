#!/usr/bin/env python3
"""Judge the SS5.7 body-vs-none run against the A/A floor rule registered on 2026-09-23.

The rule (docs/roadmap/temporal-dcc/i11-onset-comparison.md, the section
"SS5.7の許容にA/A雑音幅の下限を加える"):

  median / p99      |body - none| <= max(0.05 * none, floor)
  max               |body - none| <= max(1 ms, floor)
  over_budget_hops  |sum(body) - sum(none)| <= floor

floor comes from the A/A acquisition and is read, not recomputed here. The original
registered tolerance stays in hop-path.json and is reported alongside, not replaced.

Output: aa-floor-verdict.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["population_us", "synthesis_us", "elapsed_us"]
REL_TOLERANCE = 0.05
MAX_ABS_TOLERANCE_US = 1000.0


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def judge(tol, floors, ob_floor):
    entry, ok = {}, True
    for field in FIELDS:
        entry[field] = {}
        for stat in ("median", "p99", "max"):
            e = tol[field][stat]
            body, none = e.get("body"), e.get("none")
            floor = floors[field][stat]
            if body is None or none is None:
                entry[field][stat] = dict(body=body, none=none, passed=None, reason="missing")
                ok = False
                continue
            diff = abs(body - none)
            allowed = (max(MAX_ABS_TOLERANCE_US, floor) if stat == "max"
                       else max(REL_TOLERANCE * none, floor))
            passed = diff <= allowed
            entry[field][stat] = dict(body=body, none=none, difference_us=diff,
                                      floor_us=floor, allowed_us=allowed,
                                      floor_binding=floor >= (allowed - 1e-12),
                                      passed=passed)
            ok = ok and passed
    ob = tol["over_budget_hops"]
    b = sum(x for x in ob["body"] if x is not None)
    n = sum(x for x in ob["none"] if x is not None)
    passed = abs(b - n) <= ob_floor
    entry["over_budget_hops"] = dict(body=ob["body"], none=ob["none"], body_total=b,
                                     none_total=n, difference=abs(b - n), floor=ob_floor,
                                     passed=passed)
    entry["passed"] = bool(ok and passed)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--hop-path", type=Path, help="hop-path.json judged (default: <root>/hop-path.json)")
    ap.add_argument("--floor", type=Path, required=True,
                    help="A/A floor JSON the rule reads (aa-floor*.json)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    hp = json.loads((args.hop_path or (args.root / "hop-path.json")).read_text())
    aa = json.loads(args.floor.read_text())
    floors, ob_floor = aa["floor_us"], aa["over_budget_hops_floor"]

    results, both, floor_only, neither = [], 0, 0, 0
    for case in hp["cases"]:
        if not case["registered_by_5_7"]:
            continue
        for mode in ("report", "no-report"):
            tol = case["modes"][mode]["tolerance_body_vs_none"]
            if tol.get("skipped"):
                continue
            verdict = judge(tol, floors, ob_floor)
            original = tol.get("passed")
            row = dict(case=case["case"], mode=mode, original_tolerance_passed=original,
                       floor_rule_passed=verdict["passed"], detail=verdict)
            results.append(row)
            if verdict["passed"] and original:
                both += 1
            elif verdict["passed"]:
                floor_only += 1
            else:
                neither += 1
            fails = []
            for field in FIELDS:
                for stat in ("median", "p99", "max"):
                    e = verdict[field][stat]
                    if e.get("passed") is False:
                        fails.append(f"{field}.{stat} {e['difference_us']:.1f}>{e['allowed_us']:.1f}us")
            if verdict["over_budget_hops"]["passed"] is False:
                fails.append(f"over_budget_hops {verdict['over_budget_hops']['difference']}"
                             f">{ob_floor}")
            print(f"{case['case']:>18} {mode:>9}: original={original} floor_rule={verdict['passed']}"
                  + ("  " + ", ".join(fails) if fails else ""), flush=True)

    out = {
        "schema": "conchordal/i11-stage1-aa-floor-verdict/1",
        "generated_at": now_iso(),
        "criterion": "SS5.7 with the A/A floor registered on 2026-09-23",
        "rule": {
            "median_p99": "|body - none| <= max(0.05 * none, floor)",
            "max": "|body - none| <= max(1000 us, floor)",
            "over_budget_hops": "|sum(body) - sum(none)| <= floor",
        },
        "floor_source": str(args.floor),
        "floor_us": floors,
        "over_budget_hops_floor": ob_floor,
        "combinations": len(results),
        "passed_under_floor_rule": both + floor_only,
        "passed_under_original_tolerance": sum(1 for r in results if r["original_tolerance_passed"]),
        "passed_under_both": both,
        "passed_under_floor_rule_only": floor_only,
        "failed_under_both": neither,
        "cases": results,
        "note": ("The floor only widens a field where 0.05 * none is below it. The original "
                 "tolerance is not replaced; hop-path.json keeps its verdict."),
    }
    dest = args.out or (args.root / "aa-floor-verdict.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nfloor rule {out['passed_under_floor_rule']}/{out['combinations']}, "
          f"original {out['passed_under_original_tolerance']}/{out['combinations']} -> {dest}")


if __name__ == "__main__":
    main()
