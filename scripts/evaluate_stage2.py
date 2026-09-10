#!/usr/bin/env python3
"""Archive matched-seed measure, habituation and practical DCC comparisons."""

import argparse
import array
import collections
import copy
import datetime as dt
import html
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import tomllib
import wave

import evaluate_beta as beta
from evaluate_rt import config_text


def measure_source(source, amount):
    if source.count(".entrained()") != 2 or ".measure_accent(" in source:
        raise ValueError("sample 08 timing changed; review the accent comparison")
    return source.replace(".entrained()", f".entrained().measure_accent({amount:.1f})")


def campaign_seed_source(source):
    # Research sources keep their pinned seed; only the archived temporary copy
    # delegates seed selection to the campaign's --seed argument.
    pattern = r"(?m)^seed\([0-9]+\);\n"
    if len(re.findall(pattern, source)) != 1:
        raise ValueError("research sample must have one literal seed declaration")
    return re.sub(pattern, "", source)


def report_rows(path):
    rows = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line, object_pairs_hook=beta.strict_object, parse_float=beta.finite_float)
        rows[row["type"]].append(row)
    return rows


def fixed_region(rows, freq_hz):
    if len(rows) < 2:
        raise ValueError("missing periodic habituation scans")
    ref = rows[0]
    idx = round(math.log2(freq_hz / ref["fmin_hz"]) * ref["bins_per_octave"])
    if not 0 <= idx < ref["n_bins"]:
        raise ValueError("fixed frequency outside Log2Space")
    for row in rows:
        if any(row[key] != ref[key] for key in ("fmin_hz", "bins_per_octave", "n_bins")):
            raise ValueError("Log2Space changed within the assay")
    return [{"time_sec": r["time_sec"], "h": r["state_scan"][idx],
             "raw_score": r["raw_score_scan"][idx], "eff_score": r["eff_score_scan"][idx]}
            for r in rows]


def revisit_candidates(rows):
    # Fixed 50-cent cells; require observed displacement of at least 100 cents.
    # Missing onsets never count as departure. Identity must survive the excursion.
    voices = collections.defaultdict(list)
    for row in rows["onset"]:
        if row["population_id"] == 3:
            voices[row["voice_id"]].append(row)
    events = []
    for voice, onsets in voices.items():
        for cell in sorted({round(24 * math.log2(r["freq_hz"] / 55)) for r in onsets}):
            center = 55 * 2 ** (cell / 24)
            last_visit = departure = None
            for onset in onsets:
                cents = abs(1200 * math.log2(onset["freq_hz"] / center))
                if cents <= 25:
                    if departure is not None and onset["time_sec"] - departure >= 2:
                        series = fixed_region(rows["habituation_scan"], center)
                        before = [r for r in series if r["time_sec"] <= departure]
                        gap = [r for r in series if departure < r["time_sec"] < onset["time_sec"]]
                        if before and len(gap) >= 2:
                            initial = before[-1]["h"]
                            minimum = min(r["h"] for r in gap)
                            events.append({"voice_id": voice, "frequency_hz": center,
                                           "last_visit_sec": last_visit, "departure_sec": departure,
                                           "return_sec": onset["time_sec"], "h_at_departure": initial,
                                           "h_min_during_absence": minimum,
                                           "recovered_before_return": initial >= .1 and minimum < .8 * initial})
                    last_visit, departure = onset["time_sec"], None
                elif cents >= 100 and last_visit is not None and departure is None:
                    departure = onset["time_sec"]
    return events


def pitch_metrics(rows, population_id):
    series = [r for r in rows["population_step"] if r["population_id"] == population_id and r["alive_count"] == 1]
    if len(series) < 100 or any(r["mean_freq_hz"] <= 0 for r in series):
        raise ValueError("missing single-Voice pitch trajectory")
    steps = [(b["time_sec"], abs(1200 * math.log2(b["mean_freq_hz"] / a["mean_freq_hz"])))
             for a, b in zip(series, series[1:])]
    late = series[-1]["time_sec"] - 5
    return {"path_cents": math.fsum(x for _, x in steps),
            "late_path_cents": math.fsum(x for t, x in steps if t >= late),
            "final_freq_hz": series[-1]["mean_freq_hz"],
            "series": [[r["time_sec"], r["mean_freq_hz"]] for r in series]}


def resolution_time(listener, start_sec=.5):
    # A diagnostic threshold, not a claim about perceived musical resolution.
    elevated = False
    low_start = None
    for row in listener:
        if row["time_sec"] < start_sec:
            continue
        if row["tension_level"] > .05:
            elevated, low_start = True, None
        elif elevated:
            if low_start is None:
                low_start = row["time_sec"]
            if row["time_sec"] - low_start >= 1:
                return low_start - start_sec
    return None


def measure_pair(output, cases, seed):
    off, on = [next(c for c in cases if c["seed"] == seed and c["sample"] == "measure_" + label)
               for label in ("off", "on")]
    rows = [report_rows(output / c["case"] / "report.jsonl") for c in (off, on)]
    intervention = next((r["time_sec"] for r in rows[1]["onset"] if r["strength"] != 1), None)
    if intervention is None:
        raise ValueError("no generated accent in the enabled condition")
    for kind in ("spawn", "onset", "population_step", "listener_state", "rhythm_observation", "habituation_scan"):
        traces = [[r for r in group[kind] if r["time_sec"] < intervention] for group in rows]
        if traces[0] != traces[1]:
            raise ValueError(f"pre-accent {kind} differs")
    formats, pcm = [], []
    for case in (off, on):
        with wave.open(str(output / case["case"] / "audio.wav"), "rb") as stream:
            formats.append((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()))
            data = array.array("h", stream.readframes(stream.getnframes()))
            if sys.byteorder != "little":
                data.byteswap()
            pcm.append(data)
    if formats[0] != formats[1] or formats[0][1:3] != (1, 2):
        raise ValueError("paired mono PCM16 formats or durations differ")
    prefix = math.floor(intervention * formats[0][0])
    if pcm[0][:prefix] != pcm[1][:prefix]:
        raise ValueError("audio differs before the first accent")
    differences = [float(b - a) for a, b in zip(*pcm)]
    if not any(differences):
        raise ValueError("accent never reached rendered PCM")
    return {"seed": seed, "first_accent_hop_sec": intervention, "equal_prefix_frames": prefix,
            "pcm_difference_rms": math.sqrt(math.fsum(x*x for x in differences) / len(differences)) / 32768,
            "on_vs_off_rms_db": 20 * math.log10(on["audio"]["rms"] / off["audio"]["rms"])}


def analyze(case_dir, assay, sample, strength):
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "ok":
        raise ValueError(f"failed render: {case_dir}")
    for name in ("audio.wav", "report.jsonl"):
        if beta.sha256(case_dir / name) != metrics["artifact_sha256"][name]:
            raise ValueError(f"artifact changed after validation: {case_dir / name}")
    rows = report_rows(case_dir / "report.jsonl")
    result = {"audio": metrics["audio"], "artifact_sha256": metrics["artifact_sha256"]}
    if metrics["audio"]["all_silent"] or metrics["audio"]["clipping_fraction"]:
        raise ValueError(f"silent or clipped comparison: {case_dir}")
    if sample in ("habituation_field_assay", "dcc_fixed_input", "dcc_closed_loop_probe"):
        anchors = {1, 2, 3, 4} if sample == "dcc_fixed_input" else {1, 2}
        end = {"habituation_field_assay": 60, "dcc_fixed_input": 8.5, "dcc_closed_loop_probe": 20.5}[sample]
        if any(r["population_id"] in anchors and r["time_sec"] < end - .02 for r in rows["death"]):
            raise ValueError("a supposedly persistent reference died during the assay")
        late = {r["population_id"] for r in rows["population_step"]
                if end - .1 < r["time_sec"] < end and r["alive_count"] == 1}
        if not anchors <= late:
            raise ValueError("missing reference population at the end of the assay")
        if any(sum(r["population_id"] == population for r in rows["onset"]) != 1 for population in anchors):
            raise ValueError("a sustained reference became a repeated onset source")
        result["persistent_references_verified"] = True
    if assay == "measure":
        rhythm = rows["rhythm_observation"]
        for r in rhythm:
            if (type(r["measure_ratio"]) is not int or r["measure_ratio"] not in (0, 2, 3, 4)
                    or not 0 <= r["measure_confidence"] <= 1
                    or not math.isfinite(r["measure_phase"])
                    or r["measure_hz"] is not None and r["measure_hz"] <= 0):
                raise ValueError("invalid production measure observation")
        strengths = [r["strength"] for r in rows["onset"]]
        if not strengths or min(strengths) < .65 - 1e-6 or max(strengths) > 1.35 + 1e-6:
            raise ValueError("accent strength outside the bounded fixture range")
        if sample.endswith("off") and any(x != 1 for x in strengths):
            raise ValueError("disabled accent is not neutral")
        result.update(onset_strength=beta.stats(strengths),
                      production_measure_ratios=dict(collections.Counter(r["measure_ratio"] for r in rhythm)),
                      production_measure_confidence=beta.stats([r["measure_confidence"] for r in rhythm]),
                      listener_measure_confidence=beta.stats([r["measure_confidence"] for r in rows["listener_state"]]))
    elif assay == "habituation":
        if sample == "habituation_recovery_probe":
            series = fixed_region(rows["habituation_scan"], 220)
            result["fixed_220hz_series"] = series
            result["probe_h"] = [next(r["h"] for r in series if sec <= r["time_sec"] < sec + .03)
                                 for sec in (9, 25, 35)]
        else:
            events = revisit_candidates(rows)
            result.update(revisits=events, revisit_count=len(events),
                          recovered_revisit_count=sum(e["recovered_before_return"] for e in events))
        if strength == 0:
            if any(any(r["state_scan"]) or r["raw_score_scan"] != r["eff_score_scan"]
                   for r in rows["habituation_scan"]):
                raise ValueError("disabled habituation is not identity")
    else:
        states, pressures = rows["listener_state"], rows["dcc_pressure"]
        if not states or len(states) != len(pressures):
            raise ValueError("missing DCC observations")
        for state, pressure in zip(states, pressures):
            expected = state["tension_level"] * strength
            if (state["time_sec"] != pressure["time_sec"]
                    or abs(pressure["tension_pressure"] - expected) > 1e-6
                    or abs(pressure["temperature_bonus"] - .1 * expected) > 1e-6
                    or not 0 <= pressure["temperature_bonus"] <= .1 * strength + 1e-6):
                raise ValueError("DCC pressure formula, time or bound mismatch")
        result.update(temperature_bonus=beta.stats([r["temperature_bonus"] for r in pressures]),
                      tension=beta.stats([r["tension_level"] for r in states]),
                      time_to_low_tension_sec=(None if sample == "dcc_fixed_input" else resolution_time(states)),
                      pitch=pitch_metrics(rows, 5 if sample == "dcc_fixed_input" else 3))
    return result


def audition(output, cases, assay):
    if assay not in ("measure", "dcc"):
        return
    rng = random.SystemRandom()
    mapping, buttons = [], []
    for seed in sorted({c["seed"] for c in cases}):
        if assay == "measure":
            pair = [c for c in cases if c["seed"] == seed]
        else:
            pair = [c for c in cases if c["seed"] == seed and c["sample"] == "dcc_closed_loop_probe"
                    and c["condition"] in ("zero", "quarter", "full")]
        rng.shuffle(pair)
        buttons.append(f"<h2>seed {seed}</h2>")
        for label, case in zip("ABC", pair):
            name = f"seed-{seed}-{label}.wav"
            shutil.copy2(output / case["case"] / "audio.wav", output / name)
            mapping.append({"seed": seed, "label": label, "case": case["case"],
                            "sha256": beta.sha256(output / name)})
            buttons.append(f'<button data-src="{name}">seed {seed} · {label}</button>')
    title = "小節アクセント A/B" if assay == "measure" else "DCCの探索量 A/B/C"
    question = ("一拍ずつの揃い方に加え、数拍ごとに繰り返す強弱が聞こえるかを比べてください。"
                "AとBで違いが聞こえるか、どちらを音楽として選ぶかを分けて判断してください。"
                if assay == "measure" else
                "音高の動きが落ち着くまでと、後半に動き続ける量を比べてください。"
                "違いの有無と、どの動きを音楽として選ぶかを分けて判断してください。")
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>:root{{color-scheme:dark;font:18px/1.7 system-ui;background:#152020;color:#e6efeb}}
body{{max-width:760px;margin:3rem auto;padding:0 1rem}}button,textarea{{font:inherit;padding:.6rem;margin:.3rem}}
button{{cursor:pointer}}audio,textarea{{width:100%;box-sizing:border-box}}h2{{font-size:1.1rem}}</style>
<main><h1>{title}</h1><p>{question}</p>
<p>同じseedでは初期条件が共通です。音源ごとの音量調整はありません。プレイヤーの音量を揃えて比較してください。
専用の拍打ちは追加していません。</p>{''.join(buttons)}
<p id="status" aria-live="polite">ボタンから再生できます。</p><audio id="player" controls></audio>
<p>途中から聞く場合、プレイヤーの再生位置を変更できます。条件を切り替えると先頭へ戻ります。</p>
<label for="notes">試聴メモ（seed、違い、好み）</label><textarea id="notes" rows="4"></textarea>
<p>メモはこのブラウザー内に保持されます。チャットに回答しても構いません。</p>
<p><small>{html.escape(output.name)} · 試聴判定は未記入</small></p></main>
<script>
const player=document.querySelector('#player'), notes=document.querySelector('#notes');
const key='conchordal-stage2:'+location.pathname; let request=0;
try{{notes.value=localStorage.getItem(key)||'';}}catch(_){{}}
notes.addEventListener('input',()=>{{try{{localStorage.setItem(key,notes.value);}}catch(_){{}}}});
document.querySelectorAll('[data-src]').forEach(button=>button.onclick=async()=>{{
const id=++request; player.pause(); player.src=button.dataset.src; player.load();
try{{await player.play();if(id===request)document.querySelector('#status').textContent=button.textContent+' 再生中';}}
catch(error){{if(id===request)document.querySelector('#status').textContent='再生できません: '+error.message;}}
}});
</script></html>'''
    (output / "audition.html").write_text(page, encoding="utf-8")
    beta.dump_json(output / "audition_mapping.json", {"assignment": mapping, "normalization": "none",
                                                    "listening_status": "unreviewed"})


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assay", choices=("measure", "habituation", "dcc"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 21, 42])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**64 for s in args.seeds):
        parser.error("require a new output path and distinct u64 seeds")
    output.mkdir(parents=True)
    manifest = {"assay": args.assay, "seeds": args.seeds, "status": "running",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "listening_status": "unreviewed"}
    beta.dump_json(output / "stage2_manifest.json", manifest)
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="conchordal-stage2-", dir="/tmp") as temporary:
            temp = Path(temporary)
            base = tomllib.loads((root / "config.toml").read_text(encoding="utf-8"))
            sources = {}
            if args.assay == "measure":
                original = (root / "samples/08_murmuration.rhai").read_text(encoding="utf-8")
                sources = {f"measure_{label}": measure_source(original, amount)
                           for label, amount in (("off", 0), ("on", 1))}
                conditions = [("paired", 0, None)]
            elif args.assay == "habituation":
                sources = {name: (root / f"samples/research/{name}.rhai").read_text(encoding="utf-8")
                           for name in ("habituation_recovery_probe", "habituation_field_assay")}
                conditions = [("off", 0, 8), ("normal", 1, 8), ("slow", 1, 80)]
            else:
                sources["dcc_closed_loop_probe"] = (root / "samples/research/dcc_closed_loop_probe.rhai").read_text(encoding="utf-8")
                conditions = [("default", 0, None), ("zero", 0, None), ("tenth", .1, None),
                              ("quarter", .25, None), ("half", .5, None), ("full", 1, None)]
            paths = []
            for name, source in sources.items():
                path = temp / (name + ".rhai")
                if args.assay != "measure":
                    source = campaign_seed_source(source)
                path.write_text(source, encoding="utf-8")
                paths.append(str(path))
            binary_hash = None
            for condition, strength, recovery in conditions:
                config = copy.deepcopy(base)
                config.setdefault("psychoacoustics", {})["habituation"] = {
                    "enabled": args.assay == "habituation" and strength > 0,
                    "satiation_sec": 5, "recovery_sec": recovery or 8, "ref_drive": .25}
                config["dcc"] = {"max_temperature_bonus": .1}
                text = config_text(config, strength if args.assay == "dcc" else 0)
                if condition == "default":
                    text = text.replace('"coupling_strength" = 0\n', '')
                config_path = temp / f"{condition}.toml"
                config_path.write_text(text, encoding="utf-8")
                campaign = output / condition
                if beta.main(["--samples", *paths, "--seeds", *map(str, args.seeds), "--config", str(config_path),
                              "--output", str(campaign), "--timeout", "300"]):
                    raise ValueError(f"render campaign failed: {condition}")
                child = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
                if binary_hash is not None and child["binary"]["sha256"] != binary_hash:
                    raise ValueError("binary changed between conditions")
                binary_hash = child["binary"]["sha256"]
                summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
                for case in summary["cases"]:
                    directory = campaign / case["case"]
                    result = analyze(directory, args.assay, case["sample"], strength)
                    cases.append({"condition": condition, "strength": strength, "seed": case["seed"],
                                  "sample": case["sample"], "case": str(directory.relative_to(output)), **result})
                beta.dump_json(output / "stage2_results.json", {"cases": cases})
            manifest["binary_sha256"] = binary_hash
            manifest["seed_policy"] = "temporary copies delegate seed to CLI; repository research samples retain pinned seeds"
        if args.assay == "dcc":
            outcome = beta.execute(
                ["env", "CONCHORDAL_RESEARCH_SEEDS=" + ",".join(map(str, args.seeds)),
                 "cargo", "test", "--lib", "dcc_fixed_input_campaign", "--", "--ignored", "--nocapture"],
                root, output / "fixed_input.log", 300)
            manifest["fixed_input_command"] = outcome
            if outcome["status"] != "ok":
                raise ValueError("fixed-input DCC campaign failed")
            log = (output / "fixed_input.log").read_text(encoding="utf-8")
            reports = re.findall(r"(?m)^fixed input report: (.+\.jsonl)$", log)
            if len(reports) != 5 * len(args.seeds):
                raise ValueError("incomplete fixed-input campaign")
            executable = re.search(r"Running unittests src/lib.rs \(([^)]+)\)", log)
            if executable is None:
                raise ValueError("missing fixed-input test executable identity")
            (output / "bin").mkdir()
            shutil.copy2(root / executable[1], output / "bin" / "fixed-input-tests")
            manifest["fixed_input_binary_sha256"] = beta.sha256(output / "bin" / "fixed-input-tests")
            manifest["fixed_input_scope"] = "default AppConfig; SeqGate references with 20-second lifetime; real deterministic worker; offline test binary"
            for report_name in reports:
                report = root / report_name
                match = re.fullmatch(r"strength-([0-9.]+)-seed-([0-9]+)", report.stem)
                if match is None:
                    raise ValueError("unexpected fixed-input report name")
                strength, seed = float(match[1]), int(match[2])
                directory = output / "fixed_input" / report.stem
                directory.mkdir(parents=True)
                shutil.copy2(report, directory / "report.jsonl")
                shutil.copy2(report.with_suffix(".wav"), directory / "audio.wav")
                metrics = {"status": "ok", "audio": beta.wav_metrics(directory / "audio.wav"),
                           "report": beta.parse_report(directory / "report.jsonl", seed, 2),
                           "artifact_sha256": {name: beta.sha256(directory / name) for name in ("audio.wav", "report.jsonl")}}
                beta.dump_json(directory / "metrics.json", metrics)
                result = analyze(directory, "dcc", "dcc_fixed_input", strength)
                cases.append({"condition": {0: "zero", .1: "tenth", .25: "quarter", .5: "half", 1: "full"}[strength],
                              "strength": strength, "seed": seed, "sample": "dcc_fixed_input",
                              "case": str(directory.relative_to(output)), **result})
            beta.dump_json(output / "stage2_results.json", {"cases": cases})
            for seed in args.seeds:
                fixed = [c for c in cases if c["seed"] == seed and c["sample"] == "dcc_fixed_input"]
                traces = [report_rows(output / c["case"] / "report.jsonl")["listener_state"] for c in fixed]
                if (len(fixed) != 5 or any(trace != traces[0] for trace in traces[1:])
                        or len({c["artifact_sha256"]["audio.wav"] for c in fixed}) != 1):
                    raise ValueError("fixed listener input changed across DCC conditions")
                for sample in sources:
                    a, b = [c for c in cases if c["seed"] == seed and c["sample"] == sample
                            and c["condition"] in ("default", "zero")]
                    if a["artifact_sha256"]["audio.wav"] != b["artifact_sha256"]["audio.wav"] or a["pitch"] != b["pitch"]:
                        raise ValueError("explicit zero differs from default coupling")
            manifest["fixed_listener_equality"] = True
            manifest["default_equals_explicit_zero"] = True
        elif args.assay == "measure":
            manifest["paired_controls"] = [measure_pair(output, cases, seed) for seed in args.seeds]
        else:
            for seed in args.seeds:
                normal, slow = [next(c["probe_h"] for c in cases if c["seed"] == seed
                                    and c["sample"] == "habituation_recovery_probe" and c["condition"] == condition)
                                for condition in ("normal", "slow")]
                if not (normal[0] > .15 and normal[1] < .15 * normal[0] and normal[2] > .7 * normal[0]
                        and slow[1] > .5 * slow[0] and slow[1] > 3 * normal[1]):
                    raise ValueError("controlled fixed-region recovery comparison failed")
            manifest["fixed_region_recovery"] = True
        audition(output, cases, args.assay)
        manifest["status"] = "complete"
        beta.dump_json(output / "stage2_manifest.json", manifest)
        print(f"stage 2 results: {output}", flush=True)
        return 0
    except (ValueError, OSError, KeyError, StopIteration, wave.Error) as error:
        manifest.update(status="failed", error=str(error))
        beta.dump_json(output / "stage2_manifest.json", manifest)
        print(f"stage 2 evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
