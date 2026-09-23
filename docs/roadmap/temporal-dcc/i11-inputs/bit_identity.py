#!/usr/bin/env python3
"""SS5.4a: the None variant must stay bit-identical to the baseline commit.

Compares, per case, the offline render of the `none` config variant in this
measurement directory against the same run produced by a baseline-commit build
in --baseline-dir (acquire.py run with the baseline binaries; this script
builds nothing).

Two comparisons:

  * the WAV SHA-256;
  * the learning-record stream, per record type, after the real-time fields
    have been removed recursively. Removed are the exact keys
    processing_us, max_processing_us, worker_resources, delivery_delay_us,
    footprint_requested_at, footprint_received_at, and every key ending in
    _us or _ns. The key paths actually removed are listed in the output
    (SS5.4: "the excluded fields are enumerated in the inspection record").

Candidate-record counts are tallied separately and are not part of the
content-equality verdict, because saturation makes them fluctuate (SS5.4).

Output: bit-identity.json
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LEARNING_TYPES = [
    "self_sound_outcome",
    "self_sound_descriptor_prediction",
    "private_participation_trace",
    "participation_outcome",
    "participation_context",
    "onset",
    "population_step",
    "body_descriptor",
    "local_prediction_match",
    "local_prediction_error",
]
EXACT_VOLATILE = {
    "processing_us",
    "max_processing_us",
    "worker_resources",
    "delivery_delay_us",
    "footprint_source", "footprint_bins", "footprint_truncated", "reference_cost", "selected_cost", "selected_offset", "footprint_requested_at",
    "footprint_received_at",
}
VOLATILE_SUFFIXES = ("_us", "_ns")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def is_volatile(key):
    return key in EXACT_VOLATILE or key.endswith(VOLATILE_SUFFIXES)


def strip_volatile(obj, removed, path=""):
    """Drop the real-time fields recursively, collecting the key paths removed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            child = f"{path}.{k}" if path else k
            if is_volatile(k):
                removed.add(child)
                continue
            out[k] = strip_volatile(v, removed, child)
        return out
    if isinstance(obj, list):
        return [strip_volatile(v, removed, f"{path}[]") for v in obj]
    return obj


def load(path, removed):
    rows = {t: [] for t in LEARNING_TYPES}
    candidates = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "body_candidate_energy":
                candidates += 1
            if t in rows:
                rows[t].append(json.dumps(strip_volatile(rec, removed), sort_keys=True))
    return rows, candidates


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="measurement directory (default: this script's directory)")
    ap.add_argument("--plan", type=Path, help="registration read (default: <root>/plan.json)")
    ap.add_argument("--baseline-dir", type=Path, required=True, help="directory holding the baseline-commit acquisition")
    ap.add_argument("--variant", default="none", help="config variant compared (default: %(default)s; SS5.4a registers None)")
    ap.add_argument("--mode", default="render", help="run mode compared (default: %(default)s, the deterministic offline render)")
    ap.add_argument("--rep", type=int, default=0, help="repetition index (default: %(default)s)")
    ap.add_argument("--case", action="append", help="restrict to this case (repeatable)")
    ap.add_argument("--out", type=Path, help="output path (default: <root>/bit-identity.json)")
    args = ap.parse_args()

    root = args.root.resolve()
    base = args.baseline_dir.resolve()
    plan = json.loads((args.plan or (root / "plan.json")).read_text())
    cases = args.case or plan["cases"]

    removed = set()
    results = []
    for case in cases:
        label = f"{case}-{args.variant}-{args.mode}-{args.rep}"
        new_wav, base_wav = root / f"{label}.wav", base / f"{label}.wav"
        new_jsonl, base_jsonl = root / f"{label}.jsonl", base / f"{label}.jsonl"
        missing = [str(p) for p in (new_jsonl, base_jsonl) if not p.exists()]
        if missing:
            results.append(dict(case=case, label=label, skipped="missing artifact", missing=missing))
            print(f"{case}: skipped (missing {missing})", flush=True)
            continue
        new_rows, new_cand = load(new_jsonl, removed)
        base_rows, base_cand = load(base_jsonl, removed)
        wav = None
        if new_wav.exists() and base_wav.exists():
            wav = dict(equal=sha256(new_wav) == sha256(base_wav),
                       sha256=sha256(new_wav), baseline_sha256=sha256(base_wav))
        types = {
            t: dict(records=len(new_rows[t]), baseline_records=len(base_rows[t]),
                    equal=new_rows[t] == base_rows[t])
            for t in LEARNING_TYPES
        }
        row = dict(
            case=case, label=label, wav=wav, types=types,
            records_equal=all(v["equal"] for v in types.values()),
            separate_tally=dict(body_candidate_energy=new_cand,
                                baseline_body_candidate_energy=base_cand,
                                note="counts only; SS5.4 keeps them out of the content-equality verdict"),
        )
        row["pass"] = bool(row["records_equal"] and (wav is None or wav["equal"]))
        results.append(row)
        print(f"{case}: wav={None if wav is None else wav['equal']} records={row['records_equal']} "
              f"candidates={new_cand}/{base_cand}", flush=True)

    checked = [r for r in results if "pass" in r]
    out = {
        "schema": "conchordal/i11-stage1-bit-identity/1",
        "generated_at": now_iso(),
        "criterion": "SS5.4a",
        "baseline_dir": str(base),
        "baseline_commit": plan.get("baseline_commit"),
        "variant": args.variant,
        "mode": args.mode,
        "learning_record_types": LEARNING_TYPES,
        "removed_fields": {
            "exact": sorted(EXACT_VOLATILE),
            "suffixes": list(VOLATILE_SUFFIXES),
            "key_paths_removed": sorted(removed),
        },
        "cases": results,
        "pass": bool(checked) and all(r["pass"] for r in checked),
        "cases_checked": len(checked),
        "cases_skipped": len(results) - len(checked),
    }
    dest = args.out or (root / "bit-identity.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"pass={out['pass']} checked={out['cases_checked']} skipped={out['cases_skipped']} -> {dest}")


if __name__ == "__main__":
    main()
