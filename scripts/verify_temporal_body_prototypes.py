"""Replay the registered corpus through the normal prototype-enabled renderer.

Check model/corpus identity, exact audio preservation, and every published private
assignment against an independent descriptor-distance calculation. Fractions are
over unique published records, not generation decisions or held-out accuracy.
"""

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from fit_temporal_body_prototypes import configuration


def verify(binary, corpus_path, model_path, output):
    corpus = json.loads(corpus_path.read_text())
    model = json.loads(model_path.read_text())
    versioned = dict(model)
    version = versioned.pop("model_version")
    digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest(versioned) != version or digest(corpus) != model["corpus_sha256"]:
        raise ValueError("model or corpus content differs from its frozen version")
    output.mkdir(parents=True, exist_ok=False)
    extraction = corpus_path.parent / "extraction.toml"
    config = output / "model-runtime.toml"
    config.write_text(extraction.read_text().split("[temporal_body]")[0] + configuration(model))
    sources = {row["scenario"]: row for row in corpus["records"]}
    counts = Counter()
    strata = {}
    for index, (name, source) in enumerate(sorted(sources.items())):
        script = corpus_path.parent / name
        for key, path in [("scenario", script), ("report", corpus_path.parent / source["report"]),
                          ("wav", corpus_path.parent / source["wav"]), ("config", extraction)]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["hashes"][key]:
                raise ValueError(f"changed source artifact: {path}")
        wav, report = output / source["wav"], output / source["report"]
        result = subprocess.run([str(binary), str(script), "--config", str(config), "-o", str(wav), "--report", str(report)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{script.stem}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"replay failed: {name}, exit {result.returncode}")
        if wav.read_bytes() != (corpus_path.parent / source["wav"]).read_bytes():
            raise ValueError(f"prototype observation changed audio: {name}")
        seen = set()
        snapshot = None
        for line in report.read_text().splitlines():
            row = json.loads(line)
            if row["type"] == "body_observation":
                snapshot = row
                if bytes(row["prototype_model_version"]).hex() != version:
                    raise ValueError("wrong model version in runtime snapshot")
            if row["type"] != "body_descriptor":
                continue
            identity = tuple(row[key] for key in ("source_id", "source_generation", "body_generation", "bus", "end"))
            if identity in seen:
                continue
            seen.add(identity)
            candidates = []
            for medoid_index, medoid in enumerate(model["medoids"]):
                common = [i for i in range(6) if row["mask"] & medoid["mask"] & (1 << i)]
                if not common:
                    continue
                # Standardize both operands separately, matching the frozen coordinate contract.
                differences = [((row["raw_values"][i]-model["means"][i])/max(model["deviations"][i], 1e-6)
                                -(medoid["raw_values"][i]-model["means"][i])/max(model["deviations"][i], 1e-6)) for i in common]
                distance = math.sqrt(math.fsum(value*value for value in differences)/len(common))
                candidates.append((distance, -len(common), medoid["record_id"], medoid_index))
            expected = min(candidates) if candidates else None
            assignment = row["prototype_assignment"]
            eligible = expected is not None and expected[0] <= .25
            if eligible:
                if assignment is None or assignment["key"] != [expected[3], 0] or assignment["common_coordinates"] != -expected[1] or not math.isclose(assignment["distance"], expected[0], rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError(f"prototype assignment mismatch: {name} {identity}: {assignment} != {expected}")
            elif assignment is not None:
                raise ValueError(f"unsupported descriptor acquired a prototype: {name} {identity}")
            counts["records"] += 1
            counts["compatible" if eligible else "unknown"] += 1
            stratum = f"{source['recipe']['body']}/{source['recipe']['routing']}/bus{row['bus']}"
            strata.setdefault(stratum, Counter())["compatible" if eligible else "unknown"] += 1
        if snapshot is None or not snapshot["finished"] or any(snapshot[key] for key in ("capture_drops", "invalid_hops", "outside_voice_hops")):
            raise ValueError(f"incomplete prototype replay: {name}")
        counts["audio_equal_cases"] += 1
        print(f"{index+1}/{len(sources)} {name}", flush=True)
    summary = dict(model_version=version, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   counts=dict(counts), strata=strata,
                   claim="numeric assignment and audio noninterference only; same-corpus diagnostic, not action transfer or held-out acceptance")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/conchordal-render"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.binary.resolve(), args.corpus.resolve(), args.model.resolve(), args.output.resolve())
