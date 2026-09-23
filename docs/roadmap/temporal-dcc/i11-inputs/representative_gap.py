#!/usr/bin/env python3
"""SS5.9: gap between the representative condition and the actual condition.

SKELETON. This script is deliberately incomplete.

SS5.9 asks, for every onset opportunity issued in the runs of the 12
conditions, for the ToneEnergy actually issued (actual kick, actual seed,
actual rhythms, reserved release; the same freeze as `issued` in
src/synth/energy_projection.rs) to be re-projected offline with
retained = [], bus 0, over [onset, onset + D_rep) in 16 bins, where D_rep is
the D of the representative footprint used at the same opportunity. Both sides
are normalised to power_k and compared by

    max_k |power_k^actual - power_k^rep|

plus the absolute difference of Ov(at) computed with the same external
prediction. Opportunities with a None bin in coherent_energies on the actual
side are excluded and counted; a known silence on both sides is a difference of
0; the result is a distribution (median / p95 / max), not a pass/fail.

What exists here:

  * the argument parsing;
  * the loaders for the participation_context and body_candidate_energy
    records;
  * compare_power(), which takes the two 16-value arrays, normalises each to
    power_k and returns max_k |power_actual - power_rep| together with the
    exclusion rules.

What does not exist:

  * load_actual_energies(). The report does not carry the per-bin
    coherent_energies of the actually issued tone, and there is no offline
    re-projection entry point on the Rust side. That harness has to be
    registered and implemented first (SS5.9); until then the function raises
    NotImplementedError and this script produces nothing.

TODO(SS5.9): add the Rust-side offline re-projection (a binary or a test entry
point that replays the issued ToneEnergy through project_window with
retained = [], added = the issued tone, bus 0, interval [onset, onset + D_rep),
use_coherent = true, 16 bins) and emit its 16 coherent_energies per onset
opportunity, keyed by (source_id, onset frame). Then implement
load_actual_energies() against that output, drop this notice, and let the
script write representative-gap.json.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BINS = 16


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quantile(values, frac):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(frac * (len(ordered) - 1))]


def load_participation_contexts(path):
    """participation_context records in file order, keyed by (source, onset order).

    The source key is source_id when present, voice_id otherwise; the onset
    order is the index of the record within that source's sequence.
    """
    seen = defaultdict(int)
    rows = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "participation_context":
                continue
            source = rec["source_id"] if "source_id" in rec else rec.get("voice_id")
            key = (source, seen[source])
            seen[source] += 1
            rows[key] = rec
    return rows


def load_body_candidate_energies(path):
    """body_candidate_energy records grouped by source_id, in file order."""
    rows = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "body_candidate_energy":
                continue
            rows[rec.get("source_id")].append(rec)
    return dict(rows)


def to_power(energies):
    """Normalise 16 coherent energies to power_k = E_k / max_k E_k (SS4.2).

    Returns (power, status). status is "ok", "unsupported" when any bin is
    None, or "known_silence" when every bin is a known 0 -- SS4.2 keeps the
    known silence distinct from unsupported and gives it power_k = 0.
    """
    if energies is None or len(energies) != BINS:
        return None, "unsupported"
    if any(e is None for e in energies):
        return None, "unsupported"
    arr = np.asarray(energies, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None, "unsupported"
    peak = float(arr.max())
    if peak <= 0.0:
        return np.zeros(BINS), "known_silence"
    return arr / peak, "ok"


def compare_power(actual_energies, representative_energies):
    """max_k |power_actual - power_rep| over the 16 bins.

    Excluded when either side is unsupported. Two known silences are a
    difference of 0 (SS5.9).
    """
    a_power, a_status = to_power(actual_energies)
    r_power, r_status = to_power(representative_energies)
    if a_status == "unsupported" or r_status == "unsupported":
        return dict(status="excluded", actual_status=a_status, representative_status=r_status,
                    max_abs_power_difference=None)
    if a_status == "known_silence" and r_status == "known_silence":
        return dict(status="known_silence_both", actual_status=a_status,
                    representative_status=r_status, max_abs_power_difference=0.0)
    return dict(status="compared", actual_status=a_status, representative_status=r_status,
                max_abs_power_difference=float(np.max(np.abs(a_power - r_power))))


def load_actual_energies(root, case, variant, mode, rep):
    """The 16 coherent energies of the actually issued tone, per onset opportunity.

    STUB. Needs the offline re-projection registered in SS5.9: the issued
    ToneEnergy replayed through project_window with retained = [], bus 0 and
    the interval [onset, onset + D_rep). Neither the report stream nor any
    existing binary produces this.
    """
    raise NotImplementedError("needs the offline re-projection registered in SS5.9")


def summarise(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(count=0, median=None, p95=None, max=None)
    return dict(count=len(values), median=st.median(values),
                p95=quantile(values, 0.95), max=max(values))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--plan", type=Path, help="registration read (default: <root>/plan.json)")
    ap.add_argument("--variant", default="body", help="config variant read (default: %(default)s; SS5.9 needs the body footprint)")
    ap.add_argument("--mode", default="render", help="run mode read (default: %(default)s, the deterministic offline render)")
    ap.add_argument("--rep", type=int, default=0, help="repetition index (default: %(default)s)")
    ap.add_argument("--case", action="append", help="restrict to this case (repeatable; default: every case in plan.json)")
    ap.add_argument("--out", type=Path, help="output path (default: <root>/representative-gap.json)")
    args = ap.parse_args()

    root = args.root.resolve()
    plan = json.loads((args.plan or (root / "plan.json")).read_text())
    cases = args.case or plan["cases"]

    results = []
    for case in cases:
        label = f"{case}-{args.variant}-{args.mode}-{args.rep}"
        jsonl = root / f"{label}.jsonl"
        # TODO(SS5.9): pair each onset opportunity in `contexts` with its
        # representative footprint energies and with the offline re-projection,
        # then feed both 16-value arrays to compare_power() and collect the
        # Ov(at) difference computed from the same external prediction. The
        # stub is called first on purpose, so that no run of this skeleton can
        # quietly produce an empty representative-gap.json.
        actual = load_actual_energies(root, case, args.variant, args.mode, args.rep)
        if not jsonl.exists():
            results.append(dict(case=case, label=label, skipped="missing report jsonl"))
            continue
        contexts = load_participation_contexts(jsonl)
        candidates = load_body_candidate_energies(jsonl)
        gaps = []
        ov_gaps = []
        excluded = 0
        for key, context in contexts.items():
            outcome = compare_power(actual.get(key), context.get("footprint_energies"))
            if outcome["status"] == "excluded":
                excluded += 1
                continue
            gaps.append(outcome["max_abs_power_difference"])
        results.append(dict(case=case, label=label,
                            opportunities=len(contexts),
                            candidate_sources=len(candidates),
                            excluded_unsupported=excluded,
                            max_abs_power_difference=summarise(gaps),
                            ov_absolute_difference=summarise(ov_gaps)))

    out = {
        "schema": "conchordal/i11-stage1-representative-gap/1",
        "generated_at": now_iso(),
        "criterion": "SS5.9",
        "variant": args.variant,
        "mode": args.mode,
        "bins": BINS,
        "cases": results,
        "note": "Distribution report only; SS5.9 is not a pass/fail and is material for A1.",
    }
    dest = args.out or (root / "representative-gap.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
