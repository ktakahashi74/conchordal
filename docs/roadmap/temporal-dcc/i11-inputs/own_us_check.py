#!/usr/bin/env python3
"""SS5.7 judged on the hop's own work (docs/roadmap/temporal-dcc/i11-onset-comparison.md).

Registered on 2026-09-23 in "SS5.7の`elapsed_us`をhop自身の仕事で判定する":

    own_us = elapsed_us - analysis_wait_us - listener_wait_us   (per hop)

The three elapsed items and the over-budget count are judged on own_us with the
existing tolerances -- median and p99 within max(0.05 * none, floor), max within
max(1 ms, floor), over-budget counts within floor -- and an over-budget hop is one
whose own_us exceeds the hop budget. The floor comes from the existing A/A passes
(none against none) by the same rule as the elapsed floor: the maximum of the
pooled per-combination differences.

Per-repetition statistics and the reduction across repetitions follow hop_path.py
exactly: each statistic is taken over the sorted per-hop values with the
nearest-index convention, then reduced across repetitions by its median.

    floor  --aa-pass DIR [--aa-pass DIR ...] --out own-floor.json
    judge  --root DIR --floor-own own-floor.json [--plan PLAN] --out own-us.json

`judge` also reports the raw elapsed_us statistics and the analysis_wait_us and
listener_wait_us distributions for body and none separately, which the
registration requires alongside the verdict.
"""
import argparse
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ["none", "body", "proxy"]
STATS = ["median", "p99", "max"]
WAITS = ["analysis_wait_us", "listener_wait_us"]
REL_TOLERANCE = 0.05
MAX_ABS_TOLERANCE_US = 1000.0


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quantile(values, frac):
    """hop_path.py's convention: nearest index into the sorted values."""
    if not values:
        return None
    return values[int(frac * (len(values) - 1))]


def spread(values):
    values = sorted(v for v in values if v is not None)
    return {"median": quantile(values, 0.5), "p99": quantile(values, 0.99),
            "max": values[-1] if values else None, "samples": len(values)}


def profile_stats(path):
    profile = json.loads(Path(path).read_text())
    hops = profile.get("hops") or []
    budget = profile.get("hop_budget_us")
    own = []
    for hop in hops:
        elapsed = hop.get("elapsed_us")
        if elapsed is None:
            continue
        own.append(elapsed - (hop.get("analysis_wait_us") or 0.0)
                   - (hop.get("listener_wait_us") or 0.0))
    row = {"hop_count": len(hops), "hop_budget_us": budget,
           "own_us": spread(own), "elapsed_us": spread([h.get("elapsed_us") for h in hops])}
    for field in WAITS:
        row[field] = spread([h.get(field) for h in hops])
    row["over_budget_hops_own"] = (None if budget is None
                                   else sum(1 for v in own if v > budget))
    summary = profile.get("summary") or {}
    row["over_budget_hops_elapsed"] = summary.get("over_budget_hops")
    return row


def reduce_reps(reps):
    out = {}
    for field in ["own_us", "elapsed_us", *WAITS]:
        out[field] = {}
        for stat in STATS:
            values = [r[field][stat] for r in reps if r[field][stat] is not None]
            out[field][stat] = st.median(values) if values else None
    out["over_budget_hops_own"] = [r["over_budget_hops_own"] for r in reps]
    out["over_budget_hops_elapsed"] = [r["over_budget_hops_elapsed"] for r in reps]
    out["hop_budget_us"] = next((r["hop_budget_us"] for r in reps
                                 if r["hop_budget_us"] is not None), None)
    return out


def combos(root, cases, reps, registered):
    """Reduced statistics per combination and variant."""
    out = []
    for case in cases:
        try:
            voices = int(case.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            voices = None
        for mode in ("report", "no-report"):
            per_variant = {}
            for variant in VARIANTS:
                got = []
                for rep in range(reps):
                    path = root / f"{case}-{variant}-{mode}-{rep}-profile.json"
                    if path.exists():
                        got.append(profile_stats(path))
                if got:
                    per_variant[variant] = reduce_reps(got)
            if per_variant:
                out.append({"case": case, "mode": mode, "voices": voices,
                            "registered_by_5_7": voices in registered,
                            "variants": per_variant})
    return out


def differences(entry, left="body", right="none"):
    """|left - right| per field and statistic, and for the over-budget totals."""
    a, b = entry["variants"].get(left), entry["variants"].get(right)
    if a is None or b is None:
        return None
    out = {}
    for field in ("own_us", "elapsed_us"):
        out[field] = {}
        for stat in STATS:
            x, y = a[field][stat], b[field][stat]
            out[field][stat] = None if x is None or y is None else abs(x - y)
    for key in ("over_budget_hops_own", "over_budget_hops_elapsed"):
        xs = [v for v in a[key] if v is not None]
        ys = [v for v in b[key] if v is not None]
        out[key] = abs(sum(xs) - sum(ys)) if xs and ys else None
    return out


def load_cases(root, plan_path, reps):
    plan = json.loads((plan_path or (root / "plan.json")).read_text())
    registered = set((plan.get("tolerances") or {}).get("registered_voice_counts") or [4, 16])
    return combos(root, plan["cases"], reps, registered), registered


def cmd_floor(args):
    pooled, per_pass = [], {}
    for directory in args.aa_pass:
        entries, _ = load_cases(directory, args.plan, args.reps)
        rows = []
        for entry in entries:
            if not entry["registered_by_5_7"]:
                continue
            diff = differences(entry)
            if diff is None:
                continue
            rows.append({"combo": f"{entry['case']} {entry['mode']}", **diff})
        per_pass[str(directory)] = rows
        pooled += rows
        print(f"{directory}: {len(rows)} combos", flush=True)

    floor = {"own_us": {}, "elapsed_us": {}}
    dist = {"own_us": {}, "elapsed_us": {}}
    for field in ("own_us", "elapsed_us"):
        for stat in STATS:
            values = sorted(r[field][stat] for r in pooled if r[field][stat] is not None)
            floor[field][stat] = max(values) if values else None
            dist[field][stat] = {
                "max": floor[field][stat], "p50": st.median(values) if values else None,
                "p95": quantile(values, 0.95), "samples": len(values),
                "per_pass_max": [max((r[field][stat] for r in rows
                                      if r[field][stat] is not None), default=None)
                                 for rows in per_pass.values()],
            }
    ob = [r["over_budget_hops_own"] for r in pooled if r["over_budget_hops_own"] is not None]
    ob_floor = max(ob) if ob else 0

    # Convergence, reported and not acted on: the floor from all but the last pass
    # applied to the last pass.
    names = list(per_pass)
    conv = {}
    if len(names) > 1:
        head = [r for n in names[:-1] for r in per_pass[n]]
        tail = per_pass[names[-1]]
        total = 0
        for field in ("own_us", "elapsed_us"):
            for stat in STATS:
                cut = max((r[field][stat] for r in head if r[field][stat] is not None), default=None)
                over = [r["combo"] for r in tail
                        if r[field][stat] is not None and cut is not None and r[field][stat] > cut]
                conv[f"{field}.{stat}"] = {"floor_without_last_pass": cut,
                                           "exceeded": len(over), "combos": over[:3]}
                total += len(over)
        conv["total_exceedances"] = total

    out = {"schema": "conchordal/i11-stage1-own-us-floor/1", "generated_at": now_iso(),
           "definition": ("floor[field][stat] = max of the pooled per-combination |A - A| over the "
                          "given A/A passes x 16 combinations. own_us = elapsed_us - "
                          "analysis_wait_us - listener_wait_us per hop."),
           "passes": names, "samples": len(pooled), "floor_us": floor,
           "over_budget_hops_own_floor": ob_floor, "distribution": dist,
           "convergence_check": conv, "per_pass": per_pass}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nown_us floor -> {args.out}")
    for stat in STATS:
        print(f"  own_us.{stat}: {floor['own_us'][stat]:.3f} us "
              f"(raw elapsed_us.{stat}: {floor['elapsed_us'][stat]:.3f} us)")
    print(f"  over_budget_hops (own_us): {ob_floor}")
    if conv:
        print(f"  convergence: {conv['total_exceedances']} exceedances in the last pass")


def judge_field(diff, none_value, floor, stat):
    if diff is None or none_value is None or floor is None:
        return {"passed": None, "reason": "missing"}
    allowed = (max(MAX_ABS_TOLERANCE_US, floor) if stat == "max"
               else max(REL_TOLERANCE * none_value, floor))
    return {"difference_us": diff, "none": none_value, "floor_us": floor,
            "allowed_us": allowed, "floor_binding": floor >= allowed - 1e-12,
            "passed": diff <= allowed}


def cmd_judge(args):
    entries, _ = load_cases(args.root, args.plan, args.reps)
    floors = json.loads(args.floor_own.read_text())
    own_floor = floors["floor_us"]["own_us"]
    raw_floor = floors["floor_us"]["elapsed_us"]
    ob_floor = floors["over_budget_hops_own_floor"]

    results, passed_own, passed_raw = [], 0, 0
    for entry in entries:
        if not entry["registered_by_5_7"]:
            continue
        diff = differences(entry)
        if diff is None:
            continue
        none = entry["variants"]["none"]
        body = entry["variants"]["body"]
        row = {"case": entry["case"], "mode": entry["mode"], "own_us": {}, "elapsed_us": {}}
        for field, floor in (("own_us", own_floor), ("elapsed_us", raw_floor)):
            for stat in STATS:
                row[field][stat] = judge_field(diff[field][stat], none[field][stat],
                                               floor[stat], stat)
        ob_diff = diff["over_budget_hops_own"]
        row["over_budget_hops_own"] = {
            "body": body["over_budget_hops_own"], "none": none["over_budget_hops_own"],
            "difference": ob_diff, "floor": ob_floor,
            "passed": None if ob_diff is None else ob_diff <= ob_floor}
        row["over_budget_hops_elapsed"] = {
            "body": body["over_budget_hops_elapsed"], "none": none["over_budget_hops_elapsed"]}
        row["own_passed"] = all(row["own_us"][s]["passed"] for s in STATS) and \
            bool(row["over_budget_hops_own"]["passed"])
        row["elapsed_raw_passed"] = all(row["elapsed_us"][s]["passed"] for s in STATS)
        row["waits"] = {v: {f: entry["variants"][v][f] for f in WAITS}
                        for v in ("body", "none") if v in entry["variants"]}
        row["own_us_values"] = {v: entry["variants"][v]["own_us"]
                                for v in ("body", "none") if v in entry["variants"]}
        results.append(row)
        passed_own += bool(row["own_passed"])
        passed_raw += bool(row["elapsed_raw_passed"])
        fails = [f"own_us.{s} {row['own_us'][s]['difference_us']:.1f}>"
                 f"{row['own_us'][s]['allowed_us']:.1f}us"
                 for s in STATS if row["own_us"][s]["passed"] is False]
        if row["over_budget_hops_own"]["passed"] is False:
            fails.append(f"over_budget {ob_diff}>{ob_floor}")
        print(f"{entry['case']:>18} {entry['mode']:>9}: own={row['own_passed']} "
              f"raw_elapsed={row['elapsed_raw_passed']}"
              + ("  " + ", ".join(fails) if fails else ""), flush=True)

    out = {"schema": "conchordal/i11-stage1-own-us-verdict/1", "generated_at": now_iso(),
           "criterion": "SS5.7 elapsed items on own_us, registered 2026-09-23",
           "floor_source": str(args.floor_own), "own_us_floor": own_floor,
           "raw_elapsed_floor": raw_floor, "over_budget_hops_own_floor": ob_floor,
           "combinations": len(results), "passed_on_own_us": passed_own,
           "passed_on_raw_elapsed": passed_raw, "cases": results,
           "note": ("own_us and the over-budget count are the registered judgement; the raw "
                    "elapsed_us verdict and the wait distributions are reported alongside.")}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nown_us {passed_own}/{len(results)}, raw elapsed {passed_raw}/{len(results)} "
          f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, help="registration read (default: <root or pass>/plan.json)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions read (default: %(default)s)")
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("floor", help="compute the own_us floor from A/A passes")
    f.add_argument("--aa-pass", type=Path, action="append", required=True,
                   help="directory of one A/A pass (repeatable)")
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(func=cmd_floor)

    j = sub.add_parser("judge", help="judge body against none on own_us")
    j.add_argument("--root", type=Path, required=True, help="measurement directory")
    j.add_argument("--floor-own", type=Path, required=True, help="own-floor.json from `floor`")
    j.add_argument("--out", type=Path, required=True)
    j.set_defaults(func=cmd_judge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
