#!/usr/bin/env python3
"""SS5.7: hop path and saturation, Some (body) against None (none).

From the instrument profile JSONs, per case, config variant and run mode:

  * the median / p99 / max of hops[].population_us, hops[].synthesis_us and
    hops[].elapsed_us, computed per repetition and then reduced across the 3
    repetitions by the median of each statistic;
  * summary.over_budget_hops per repetition;
  * the footprint counters of background.candidate_energy.

The tolerance check, body against none, is exactly the registered one: median
and p99 within +-5 %, max within 1 ms, over-budget counts equal. Each metric
gets its own pass/fail; nothing is tuned to make it pass.

From the report jsonl, per case and variant:

  * the distribution (median / p99 / max) of
    footprint_received_at - footprint_requested_at, in samples and in ms;
  * the fraction of participation_context records carrying each
    footprint_source value.

SS5.7 registers the alternating 3 repetitions at 4 and 16 Voices; the 64-Voice
cases are computed as well and flagged registered_by_5_7 = false.

Output: hop-path.json
"""
import argparse
import json
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ["none", "body", "proxy"]
FIELDS = ["population_us", "synthesis_us", "elapsed_us"]
REL_TOLERANCE = 0.05
MAX_ABS_TOLERANCE_US = 1000.0


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quantile(values, frac):
    """Same nearest-index convention as target/i10-reduced-resources-20260921/hop-path/acquire.py."""
    if not values:
        return None
    return values[int(frac * (len(values) - 1))]


def profile_stats(path):
    profile = json.loads(Path(path).read_text())
    hops = profile.get("hops") or []
    row = {"hop_count": len(hops)}
    for field in FIELDS:
        values = sorted(h[field] for h in hops if h.get(field) is not None)
        row[field] = dict(median=quantile(values, 0.5), p99=quantile(values, 0.99),
                          max=values[-1] if values else None, samples=len(values))
    summary = profile.get("summary") or {}
    row["over_budget_hops"] = summary.get("over_budget_hops")
    candidate = (profile.get("background") or {}).get("candidate_energy") or {}
    row["candidate_energy"] = candidate
    row["footprint_counters"] = {k: v for k, v in candidate.items() if k.startswith("footprint_")}
    return row


def reduce_reps(reps):
    """Median across repetitions of each per-repetition statistic."""
    out = {}
    for field in FIELDS:
        out[field] = {}
        for stat in ("median", "p99", "max"):
            values = [r[field][stat] for r in reps if r[field][stat] is not None]
            out[field][stat] = st.median(values) if values else None
    out["over_budget_hops"] = [r["over_budget_hops"] for r in reps]
    out["hop_counts"] = [r["hop_count"] for r in reps]
    return out


def check(body, none):
    verdicts = {}
    ok = True
    for field in FIELDS:
        entry = {}
        for stat, kind in (("median", "rel"), ("p99", "rel"), ("max", "abs")):
            b, n = body[field][stat], none[field][stat]
            if b is None or n is None:
                entry[stat] = dict(body=b, none=n, passed=None, reason="missing")
                ok = False
                continue
            if kind == "rel":
                rel = (b - n) / n if n else None
                passed = (rel is not None and abs(rel) <= REL_TOLERANCE)
                entry[stat] = dict(body=b, none=n, relative_difference=rel,
                                   tolerance=REL_TOLERANCE, passed=passed)
            else:
                diff = b - n
                passed = abs(diff) <= MAX_ABS_TOLERANCE_US
                entry[stat] = dict(body=b, none=n, difference_us=diff,
                                   tolerance_us=MAX_ABS_TOLERANCE_US, passed=passed)
            ok = ok and bool(entry[stat]["passed"])
        verdicts[field] = entry
    b_over, n_over = sorted(x for x in body["over_budget_hops"] if x is not None), \
                     sorted(x for x in none["over_budget_hops"] if x is not None)
    passed = bool(b_over) and b_over == n_over
    verdicts["over_budget_hops"] = dict(body=body["over_budget_hops"], none=none["over_budget_hops"],
                                        passed=passed, rule="equal counts")
    verdicts["passed"] = bool(ok and passed)
    return verdicts


def footprint_delay(paths):
    samples = []
    sample_rate = None
    sources = Counter()
    total = 0
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") != "participation_context":
                    continue
                total += 1
                sources[rec.get("footprint_source")] += 1
                req, got = rec.get("footprint_requested_at"), rec.get("footprint_received_at")
                if req is not None and got is not None:
                    samples.append(got - req)
                    sample_rate = sample_rate or rec.get("sample_rate")
    samples.sort()
    delay = dict(count=len(samples), sample_rate=sample_rate,
                 samples=dict(median=quantile(samples, 0.5), p99=quantile(samples, 0.99),
                              max=samples[-1] if samples else None))
    if sample_rate:
        delay["ms"] = {k: (None if v is None else 1000.0 * v / sample_rate)
                       for k, v in delay["samples"].items()}
    else:
        delay["ms"] = dict(median=None, p99=None, max=None)
    delay["footprint_source_counts"] = {str(k): v for k, v in sources.items()}
    delay["footprint_source_fraction"] = ({str(k): v / total for k, v in sources.items()}
                                          if total else {})
    delay["participation_context_records"] = total
    return delay


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--plan", type=Path, help="registration read (default: <root>/plan.json)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions to read (default: %(default)s)")
    ap.add_argument("--case", action="append", help="restrict to this case (repeatable)")
    ap.add_argument("--out", type=Path, help="output path (default: <root>/hop-path.json)")
    args = ap.parse_args()

    root = args.root.resolve()
    plan = json.loads((args.plan or (root / "plan.json")).read_text())
    cases = args.case or plan["cases"]
    registered_voices = set((plan.get("tolerances") or {}).get("registered_voice_counts") or [4, 16])

    results = []
    for case in cases:
        try:
            voices = int(case.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            voices = None
        row = dict(case=case, voices=voices, registered_by_5_7=voices in registered_voices, modes={})
        for mode in ("report", "no-report"):
            per_variant = {}
            for variant in VARIANTS:
                reps = []
                for rep in range(args.reps):
                    path = root / f"{case}-{variant}-{mode}-{rep}-profile.json"
                    if path.exists():
                        reps.append(dict(rep=rep, **profile_stats(path)))
                if reps:
                    per_variant[variant] = dict(repetitions=reps, reduced=reduce_reps(reps))
            entry = dict(variants=per_variant)
            if "body" in per_variant and "none" in per_variant:
                entry["tolerance_body_vs_none"] = check(per_variant["body"]["reduced"],
                                                        per_variant["none"]["reduced"])
            else:
                entry["tolerance_body_vs_none"] = dict(skipped="missing body or none profiles")
            row["modes"][mode] = entry

        row["footprint"] = {}
        for variant in VARIANTS:
            paths = [root / f"{case}-{variant}-report-{rep}.jsonl" for rep in range(args.reps)]
            paths = [p for p in paths if p.exists()]
            if paths:
                row["footprint"][variant] = dict(sources=[p.name for p in paths],
                                                 **footprint_delay(paths))
        results.append(row)
        verdict = row["modes"]["no-report"]["tolerance_body_vs_none"].get("passed")
        print(f"{case}: no-report body-vs-none passed={verdict} "
              f"report passed={row['modes']['report']['tolerance_body_vs_none'].get('passed')}", flush=True)

    registered = [r for r in results if r["registered_by_5_7"]]
    verdicts = [r["modes"][m]["tolerance_body_vs_none"].get("passed")
                for r in registered for m in ("report", "no-report")]
    out = {
        "schema": "conchordal/i11-stage1-hop-path/1",
        "generated_at": now_iso(),
        "criterion": "SS5.7",
        "fields": FIELDS,
        "tolerances": dict(median_relative=REL_TOLERANCE, p99_relative=REL_TOLERANCE,
                           max_absolute_us=MAX_ABS_TOLERANCE_US, over_budget_hops="equal",
                           comparison="body (Some) against none (None)"),
        "registered_voice_counts": sorted(registered_voices),
        "reps": args.reps,
        "cases": results,
        "pass": bool(verdicts) and all(v is True for v in verdicts),
        "note": ("The reduction across repetitions is the median of each per-repetition statistic. "
                 "The 64-Voice cases are reported but are outside the registered SS5.7 subset."),
    }
    dest = args.out or (root / "hop-path.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"pass={out['pass']} over {len(registered)} registered cases -> {dest}")


if __name__ == "__main__":
    main()
