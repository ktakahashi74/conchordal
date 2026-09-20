"""Render the preregistered I10 diagnostic corpus and freeze actual body descriptors.

Only conchordal-render writes WAVs. All artifacts and full reports are preserved in
the output directory. A failing case stops collection instead of being excluded.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path


def collect(binary, output):
    output.mkdir(parents=True, exist_ok=False)
    config = output / "extraction.toml"
    config.write_text("""[audio]
sample_rate = 48000
[analysis]
nfft = 2048
hop_size = 512
[dcc]
coupling_strength = 0.0
[temporal_body]
means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
deviations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
accent_means = [0.0, 0.0]
accent_deviations = [1.0, 1.0]
""")
    records = []
    for index, (body, hz, amplitude, route) in enumerate(itertools.product(
            ("sine", "harmonic", "modal"), (110, 440), (.04, .12), ("both", "habitat", "presentation"))):
        name = f"{index:02d}-{body}-{hz}-{amplitude}-{route}"
        script, report, wav = [output / f"{name}.{suffix}" for suffix in ("rhai", "jsonl", "wav")]
        send = "" if route == "both" else f".send({route}_bus)"
        script.write_text(f'''temporal_mode("observe");
seed(20260917);
let body = {body}().sustain().anchor().amp({amplitude}).adsr(0.01, 0.01, 1.0, 0.3){send};
let population = place(body, at({hz}.0));
wait(2.0);
release(population);
wait(0.6);
''')
        result = subprocess.run([str(binary), str(script), "--config", str(config), "-o", str(wav), "--report", str(report)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{name}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"render failed: {name}, exit {result.returncode}")
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        summary = [row for row in rows if row["type"] == "body_observation"][-1]
        if not summary["finished"] or any(summary[key] for key in ("capture_drops", "invalid_hops", "outside_voice_hops")):
            raise ValueError(f"incomplete acquisition: {name}")
        descriptors = [row for row in rows if row["type"] == "body_descriptor"]
        identities = {(row["source_id"], row["source_generation"], row["body_generation"]) for row in descriptors}
        if len(identities) != 1:
            raise ValueError(f"expected one stable actual body: {name}")
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
                  (("scenario", script), ("report", report), ("wav", wav), ("config", config))}
        for bus, cut in itertools.product((0, 1), (9600, 38400, 96000, 105600)):
            eligible = [row for row in descriptors if row["bus"] == bus and row["end"] <= cut and row["available"] <= cut]
            if not eligible:
                raise ValueError(f"missing registered cut: {name}, bus {bus}, cut {cut}")
            selected = max(eligible, key=lambda row: row["end"])
            if selected["end"] + 4800 < cut or not selected["mask"]:
                raise ValueError(f"unsupported registered cut: {name}, bus {bus}, cut {cut}")
            records.append(dict(record_id=f"{name}-bus{bus}-cut{cut:06d}", requested_cut=cut,
                                descriptor={key: value for key, value in selected.items() if key not in ("type", "standardized", "prototype_assignment")},
                                recipe=dict(body=body, frequency_hz=hz, amplitude=amplitude, routing=route,
                                            adsr=[.01, .01, 1., .3], sustain_until_sec=2.),
                                scenario=script.name, report=report.name, wav=wav.name, hashes=hashes))
        print(f"{index+1}/36 {name}", flush=True)
    corpus = dict(schema="temporal-body-development-records-v1", status="diagnostic_training_only",
                  acquisition=dict(sample_rate=48000, nfft=2048, hop_size=512),
                  accent_scales=dict(means=[0., 0.], deviations=[1., 1.]),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), records=records)
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2, allow_nan=False) + "\n")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/conchordal-render"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.binary.resolve(), args.output.resolve())
