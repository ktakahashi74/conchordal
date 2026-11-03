#!/usr/bin/env python3
"""Prequential GUI-1 categorical and arrival calibration; stream native JSONL."""
import argparse
import gzip
import json
import math
from pathlib import Path


def evaluate(path):
    opener = gzip.open if path.suffix == ".gz" else open
    seen = set()
    censored = set()
    known_unsupported = 0
    rows = []
    unsupported = 0
    listener_rows = active_rows = supported_rows = 0
    max_examples = 0
    for_line = opener(path, "rt")
    with for_line as source:
        for line in source:
            record = json.loads(line)
            if record.get("type") != "listener_state":
                continue
            listener_rows += 1
            current = record["prediction"]["forecast"]
            if current is not None:
                active_rows += 1
                supported_rows += current["q"] is not None
                max_examples = max(max_examples, current["examples"])
            cancelled = record["prediction"].get("censored")
            if cancelled is not None:
                f = cancelled["forecast"]
                assert f["reference_sample"] < cancelled["observed_sample"] <= f["start_sample"]
                censored.add(f["version"])
            done = record["prediction"]["completed"]
            if done is None:
                continue
            forecast = done["forecast"]
            version = forecast["version"]
            if version in seen:
                continue
            seen.add(version)
            p = forecast["q"]
            known = forecast["p_known"]
            b = forecast["marginal"]
            known_unsupported += known is None
            assert (p is None) == (b is None)
            if p is None:
                assert done["surprise_nats"] is None and not done["zero_likelihood"]
                unsupported += 1
                continue
            for distribution in (p, b, known):
                if distribution is None:
                    continue
                assert abs(sum(distribution) - 1) < 1e-5 and all(0 <= x <= 1 for x in distribution)
                impossible = (1,) if forecast["reference_bin"] is not None else (2, 3, 4)
                assert all(distribution[i] == 0 for i in impossible)
            c = forecast["coverage_score"]
            assert 0 <= c <= 1
            assert (known is None) == (c == 0)
            assert all(abs(p[i] - (c * (known or [0]*5)[i] + (1-c) * b[i])) < 1e-6 for i in range(5))
            assert abs(forecast["arrival_score"] - sum(p[1:4])) < 1e-6
            y = done["outcome"]
            assert forecast["start_sample"] < done["observed_sample"] <= forecast["end_sample"]
            assert forecast["reference_sample"] <= forecast["start_sample"]
            assert done["zero_likelihood"] == (p[y] == 0)
            if p[y]:
                assert abs(done["surprise_nats"] + math.log(p[y])) < 1e-4
            rows.append((done, p, forecast["marginal"]))
    assert not (seen & censored)
    result = {"population": "completed windows with same-reference-audibility marginal available at issuance, including all-UNKNOWN fallback",
              "censored": len(censored), "known_unsupported": known_unsupported, "completed": len(seen), "supported": len(rows), "unsupported": unsupported,
              "listener_rows": listener_rows, "active_forecast_rows": active_rows,
              "supported_forecast_rows": supported_rows, "max_retained_examples": max_examples,
              "outcomes": ["unchanged", "entry", "lower", "higher", "silence"]}
    for label, column in (("E0", 1), ("marginal", 2)):
        finite_loss = 0.0
        zeros = 0
        bins = [[0, 0.0, 0] for _ in range(10)]
        arrival_bins = [[0, 0.0, 0] for _ in range(10)]
        brier = 0.0
        arrival_brier = arrival_loss = 0.0
        arrival_zeros = 0
        failures = []
        for row in rows:
            done, p = row[0], row[column]
            y = done["outcome"]
            if p[y] == 0:
                zeros += 1
            else:
                finite_loss -= math.log(p[y])
            brier += sum((x - (i == y)) ** 2 for i, x in enumerate(p))
            top = max(range(5), key=lambda i: p[i])
            confidence = p[top]
            bucket = min(9, int(10 * confidence))
            bins[bucket][0] += 1
            bins[bucket][1] += confidence
            bins[bucket][2] += top == y
            arrival = done["forecast"]["arrival_score"] if label == "E0" else sum(p[1:4])
            arrived = y in (1, 2, 3)
            arrival_brier += (arrival - arrived) ** 2
            arrival_likelihood = sum(p[1:4]) if arrived else p[0] + p[4]
            if arrival_likelihood == 0:
                arrival_zeros += 1
            else:
                arrival_loss -= math.log(arrival_likelihood)
            bucket = min(9, int(10 * arrival))
            arrival_bins[bucket][0] += 1
            arrival_bins[bucket][1] += arrival
            arrival_bins[bucket][2] += y in (1, 2, 3)
            if top != y:
                failures.append({"version": done["forecast"]["version"], "start_sample": done["forecast"]["start_sample"], "outcome": y, "predicted": top, "confidence": confidence, "coverage": done["forecast"]["coverage_score"]})
        def curve(values):
            return [{"bin": i, "n": n, "mean_score": total/n, "observed_rate": hit/n} for i, (n, total, hit) in enumerate(values) if n]
        result[label] = {
            "mean_log_loss_nats": "Infinity" if zeros else (finite_loss/len(rows) if rows else None),
            "arrival_brier": arrival_brier/len(rows) if rows else None,
            "arrival_mean_log_loss_nats": "Infinity" if arrival_zeros else (arrival_loss/len(rows) if rows else None),
            "arrival_infinite_log_losses": arrival_zeros,
            "infinite_log_losses": zeros,
            "mean_finite_log_loss_nats": finite_loss/(len(rows)-zeros) if len(rows) > zeros else None,
            "brier": brier/len(rows) if rows else None,
            "top_category_curve": curve(bins),
            "arrival_curve": curve(arrival_bins),
            "high_confidence_errors": sorted(failures, key=lambda f: -f["confidence"])[:12],
        }
    e0, baseline = result["E0"], result["marginal"]
    rejected = e0["infinite_log_losses"] > 0 or (
        bool(rows) and not baseline["infinite_log_losses"] and
        e0["mean_log_loss_nats"] > baseline["mean_log_loss_nats"])
    result["registered_probability_rejection"] = rejected
    result["display"] = "uncalibrated score"
    result["display_reason"] = ("registered zero-likelihood or worse-log-score rejection" if rejected else
                                "necessary rejection checks alone do not establish calibration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {str(p): evaluate(p) for p in args.reports}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for path, row in result.items():
        print(f"{Path(path).name}: supported={row['supported']}/{row['completed']} E0 zeros={row['E0']['infinite_log_losses']} marginal zeros={row['marginal']['infinite_log_losses']}")


if __name__ == "__main__":
    main()
