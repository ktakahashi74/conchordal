#!/usr/bin/env python3
"""Replay the registered Python moment compressor for the Rust numerical port."""

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

import temporal_descriptor_reference as ref


def raw(index, *, origin=0, rate=48000, hop=512, epoch=2, generation=3, pattern="mixed"):
    lo, hi = origin + index * hop, origin + (index + 1) * hop
    values = [math.sin(index * .19 + j * .3) + j * .1 for j in range(10)]
    intervals = [(lo, hi)]
    if pattern == "constant":
        values = [2.] * 10
    elif pattern == "large_mean":
        values = [1e8 + index * .01] * 10
    elif index % 19 == 7:
        intervals = [(lo, lo + 100), (lo + 200, lo + 300)]
        values = [None] * 10
    elif index % 23 in (10, 11):
        intervals = []
        values = [None] * 10
    else:
        values[index % 10] = None
    known = sum(b-a for a, b in intervals)
    return {"epoch": epoch, "generation": generation,
            "sample_start": lo, "sample_end": hi, "sample_rate": rate,
            "start": lo / rate, "end": hi / rate, "time": hi / rate,
            "values": values, "observed": known > 0, "gap": known < hop,
            "known_sample_intervals": intervals,
            "raw_support_start": max(0, lo - 3*hop) / rate,
            "raw_support_end": hi / rate, "available_end": (hi + hop) / rate}


def snapshot(stream):
    def packed(knot):
        return bytes(knot.data).hex()
    return {"knots_hex": bytes(stream.bank.storage[:stream.bank.count*320]).hex(),
            "pending_hex": packed(stream.pending) if stream.has_pending else None,
            "pending_gap_hex": packed(stream.gap_pending) if stream.has_gap else None,
            "reconstruction_error_bits": struct.unpack("<Q", struct.pack("<d", stream.bank.reconstruction_error))[0],
            "merges": stream.bank.merges, "frozen": stream.bank.frozen,
            "priority_coordinate_evaluations": stream.bank.priority_coordinate_evaluations,
            "max_insertions_per_hop": stream.max_insertions,
            "last_insertions_per_hop": stream.last_insertions}


def generate():
    cases = []

    def case(name, capacity, cadence, count, pattern="mixed", span_start=0, span_end=None):
        stream = ref.SpanDescriptor(2, 3, 48000, 0, 512, span_start/48000,
                                    [1., .5, 2., 0., 3., 1., .25, 1., 2., 4.],
                                    capacity=capacity, cadence=cadence,
                                    span_end=None if span_end is None else span_end/48000)
        row = {"name": name, "capacity": capacity, "cadence": cadence,
               "span_start_sample": span_start, "span_end_sample": span_end,
               "scales": list(stream.bank.scales), "steps": []}
        begin = span_start//512
        for i in range(begin, count):
            r = raw(i, pattern=pattern)
            cut = r["sample_end"] + 512
            stream.push(r, cut/48000)
            step = {"kind": "push", "raw": r, "cut_sample": cut}
            if i in (begin, begin+1, begin+3, 63, 127, 255, count-1):
                step["expected"] = snapshot(stream)
            row["steps"].append(step)
        cut = (count+1)*512
        packed = stream.finish(cut/48000)
        row["steps"].append({"kind":"finish", "cut_sample":cut, "expected":snapshot(stream)})
        cases.append(row)

    for capacity in (3, 8, 64, 128, 256):
        for cadence in (1, 2, 4):
            case(f"mixed_cap{capacity}_cadence{cadence}", capacity, cadence, max(640, capacity*cadence+128))
    case("constant_earliest_merge", 3, 1, 40, "constant")
    case("large_mean_error", 8, 1, 300, "large_mean")
    case("clipped_start_and_end", 8, 2, 4, "constant", 700, 1900)
    for explicit in (False, True):
        stream = ref.SpanDescriptor(2, 3, 48000, 0, 512, 0., [1.]*10, capacity=8)
        r = raw(0, pattern="constant")
        stream.push(r, 1024/48000)
        steps = [{"kind":"push", "raw":r, "cut_sample":1024, "expected":snapshot(stream)}]
        if explicit:
            for end in (1024,1536):
                stream.gap(end/48000, 2048/48000, 2048/48000)
                steps.append({"kind":"gap", "end_sample":end, "available_sample":2048, "cut_sample":2048, "expected":snapshot(stream)})
        r = raw(3, pattern="constant")
        stream.push(r, 2560/48000)
        steps.append({"kind":"push", "raw":r, "cut_sample":2560, "expected":snapshot(stream)})
        packed = stream.finish(2560/48000)
        steps.append({"kind":"finish", "cut_sample":2560, "expected":snapshot(stream)})
        cases.append({"name":"explicit_gap" if explicit else "implicit_gap_three_insertions", "capacity":8, "cadence":2, "span_start_sample":0, "span_end_sample":None, "scales":[1.]*10, "steps":steps})
    for case in cases:
        case["scales_bits"] = [struct.unpack("<Q", struct.pack("<d", x))[0] for x in case["scales"]]
        for step in case["steps"]:
            if "raw" in step:
                r = step["raw"]
                r["values_bits"] = [None if x is None else struct.unpack("<Q", struct.pack("<d", x))[0] for x in r.pop("values")]
                for old, new in [("raw_support_start", "source_start_sample"), ("raw_support_end", "source_end_sample"), ("available_end", "available_sample")]:
                    r[new] = round(r.pop(old)*r["sample_rate"])
                for field in ("start", "end", "time", "observed", "gap"):
                    r.pop(field)
    source = Path(ref.__file__).read_bytes()
    tree = ast.parse(source)
    return {"schema":"temporal-descriptor-fixtures-v1", "reference_source_sha256":hashlib.sha256(source).hexdigest(),
            "reference_class_ast_sha256":{name:hashlib.sha256(ast.dump(next(n for n in tree.body if getattr(n,"name",None)==name), include_attributes=False).encode()).hexdigest() for name in ("DescriptorKnot","BoundedDescriptor","SpanDescriptor")},
            "cases":cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/temporal_cognition/descriptors.json"))
    args = parser.parse_args()
    data = generate()
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"cases":len(data["cases"]), "operations":sum(len(c["steps"]) for c in data["cases"]), "checkpoints":sum("expected" in s for c in data["cases"] for s in c["steps"]), "bytes":args.output.stat().st_size}))


if __name__ == "__main__":
    main()
