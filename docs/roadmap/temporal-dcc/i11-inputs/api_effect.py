#!/usr/bin/env python3
"""SS5.5: the effect of the API on the selection and on the audio.

Per case, two comparisons of the offline render runs:

  (a) body vs proxy -- the test of the effect specific to T1. Reports how many
      participation_context decisions differ in selected_offset, the
      distribution of selected_cost - reference_cost on each side and of its
      paired difference, whether the WAV SHA-256 differs, the footprint_source
      counts, and the per-case flag t1_effect_observed, which is true when at
      least one decision differs AND the WAV differs.

  (b) body vs none -- the same statistics, labelled reference_only, because
      this difference also contains the change of subdivision count and of the
      truncation (SS5.5b).

Decisions are paired by (source_id, onset order): the identity of the sounding
source, and the index of the record within that source's sequence in file
order. Where a participation_context record carries no source_id the voice_id
is used instead; the key actually used is recorded in the output.

Output: api-effect.json
"""
import argparse
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEW_FIELDS = ["footprint_source", "footprint_bins", "footprint_truncated",
              "reference_cost", "selected_cost", "selected_offset",
              "footprint_requested_at", "footprint_received_at"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quantile(values, frac):
    """Same nearest-index convention as the I10 hop-path summary."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(frac * (len(ordered) - 1))]


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(count=0, median=None, p95=None, max=None, min=None, mean=None)
    return dict(count=len(values), median=st.median(values), p95=quantile(values, 0.95),
                max=max(values), min=min(values), mean=st.fmean(values))


def load_contexts(path):
    """participation_context records in file order, keyed by (source, onset order)."""
    seen = defaultdict(int)
    rows = {}
    order = []
    present = Counter()
    key_field = None
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "participation_context":
                continue
            if "source_id" in rec:
                source, field = rec["source_id"], "source_id"
            else:
                source, field = rec.get("voice_id"), "voice_id"
            key_field = key_field or field
            key = (source, seen[source])
            seen[source] += 1
            rows[key] = rec
            order.append(key)
            for f in NEW_FIELDS:
                if f in rec:
                    present[f] += 1
    return rows, order, present, key_field


def cost_gap(rec):
    sel, ref = rec.get("selected_cost"), rec.get("reference_cost")
    if sel is None or ref is None:
        return None
    return sel - ref


def compare(root, case, label_l, label_r, mode, rep, reference_only):
    l_path = root / f"{case}-{label_l}-{mode}-{rep}.jsonl"
    r_path = root / f"{case}-{label_r}-{mode}-{rep}.jsonl"
    if not l_path.exists() or not r_path.exists():
        return dict(comparison=f"{label_l} vs {label_r}", skipped="missing report jsonl",
                    missing=[str(p) for p in (l_path, r_path) if not p.exists()])
    l_rows, l_order, l_present, key_field = load_contexts(l_path)
    r_rows, _r_order, r_present, _ = load_contexts(r_path)
    shared = [k for k in l_order if k in r_rows]

    offsets_comparable = 0
    offsets_differ = 0
    offsets_unavailable = 0
    paired_delta = []
    for k in shared:
        lo, ro = l_rows[k].get("selected_offset"), r_rows[k].get("selected_offset")
        if lo is None or ro is None:
            offsets_unavailable += 1
        else:
            offsets_comparable += 1
            if lo != ro:
                offsets_differ += 1
        lg, rg = cost_gap(l_rows[k]), cost_gap(r_rows[k])
        if lg is not None and rg is not None:
            paired_delta.append(lg - rg)

    wav_l = root / f"{case}-{label_l}-{mode}-{rep}.wav"
    wav_r = root / f"{case}-{label_r}-{mode}-{rep}.wav"
    wav = None
    if wav_l.exists() and wav_r.exists():
        a, b = sha256(wav_l), sha256(wav_r)
        wav = dict(equal=a == b, differs=a != b, sha256={label_l: a, label_r: b})

    out = dict(
        comparison=f"{label_l} vs {label_r}",
        key_field=key_field,
        records={label_l: len(l_rows), label_r: len(r_rows)},
        paired_decisions=len(shared),
        unpaired={label_l: len(l_rows) - len(shared), label_r: len(r_rows) - len(shared)},
        selected_offset=dict(comparable=offsets_comparable, differing=offsets_differ,
                             unavailable=offsets_unavailable,
                             differing_fraction=(offsets_differ / offsets_comparable) if offsets_comparable else None),
        selected_minus_reference_cost={
            label_l: distribution([cost_gap(l_rows[k]) for k in shared]),
            label_r: distribution([cost_gap(r_rows[k]) for k in shared]),
            "paired_difference": distribution(paired_delta),
        },
        footprint_source_counts={
            label_l: dict(Counter(l_rows[k].get("footprint_source") for k in l_rows)),
            label_r: dict(Counter(r_rows[k].get("footprint_source") for k in r_rows)),
        },
        new_fields_present={label_l: dict(l_present), label_r: dict(r_present)},
        wav=wav,
    )
    if reference_only:
        out["reference_only"] = True
        out["note"] = "SS5.5b: this difference also contains the subdivision-count and truncation change."
    else:
        out["t1_effect_observed"] = bool(offsets_differ > 0 and wav is not None and wav["differs"])
        out["wav_consistency"] = (
            None if wav is None else
            ("ok" if (offsets_differ > 0) == wav["differs"] else
             "inconsistent: decisions differ but the WAV does not, or the reverse")
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--plan", type=Path, help="registration read (default: <root>/plan.json)")
    ap.add_argument("--mode", default="render", help="run mode read (default: %(default)s, the deterministic offline render)")
    ap.add_argument("--rep", type=int, default=0, help="repetition index (default: %(default)s)")
    ap.add_argument("--case", action="append", help="restrict to this case (repeatable)")
    ap.add_argument("--out", type=Path, help="output path (default: <root>/api-effect.json)")
    args = ap.parse_args()

    root = args.root.resolve()
    plan = json.loads((args.plan or (root / "plan.json")).read_text())
    cases = args.case or plan["cases"]

    results = []
    for case in cases:
        row = dict(case=case)
        row["a_body_vs_proxy"] = compare(root, case, "body", "proxy", args.mode, args.rep, False)
        row["b_body_vs_none"] = compare(root, case, "body", "none", args.mode, args.rep, True)
        results.append(row)
        a = row["a_body_vs_proxy"]
        if a.get("skipped"):
            print(f"{case}: skipped ({a['skipped']})", flush=True)
        else:
            off = a["selected_offset"]
            print(f"{case}: differing={off['differing']}/{off['comparable']} "
                  f"wav_differs={None if a['wav'] is None else a['wav']['differs']} "
                  f"t1_effect_observed={a.get('t1_effect_observed')}", flush=True)

    observed = [r["a_body_vs_proxy"].get("t1_effect_observed") for r in results
                if "t1_effect_observed" in r["a_body_vs_proxy"]]
    out = {
        "schema": "conchordal/i11-stage1-api-effect/1",
        "generated_at": now_iso(),
        "criterion": "SS5.5",
        "mode": args.mode,
        "rep": args.rep,
        "new_participation_context_fields": NEW_FIELDS,
        "cases": results,
        "t1_effect_observed_cases": sum(1 for v in observed if v),
        "cases_with_comparison": len(observed),
        "note": ("SS5.5a is the required stage-1 test: material that shows no body/proxy difference is not "
                 "enough to pass. SS5.5b is recorded for reference only."),
    }
    dest = args.out or (root / "api-effect.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"t1_effect_observed in {out['t1_effect_observed_cases']}/{out['cases_with_comparison']} cases -> {dest}")


if __name__ == "__main__":
    main()
