#!/usr/bin/env python3
"""Render a blinded, matched-seed flow-amplitude comparison for sample 12."""

import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import wave

import evaluate_beta as beta
import evaluate_resolution as resolution
import sample12


INTERVENTION_SEC, RELEASE_SEC = sample12.WINDOWS["early_resolution"]
EXCERPT_START_SEC = sample12.FLOW_START_SEC
AMPLITUDES = {"unchanged": 0.020, "reduced": 0.014}


def amplitude_source(source, condition):
    sample12.validate_source(source)
    if (source.count("let flow_particles = modal()\n    .amp(0.020)\n") != 1
            or len(re.findall(r"\bflow\.amp\s*\(", source)) != 1):
        raise ValueError("sample 12 flow amplitude changed; review the comparison")
    return source.replace("    flow.amp(0.014);\n",
                          f"    flow.amp({AMPLITUDES[condition]:.3f});\n", 1)


def publish_audition(output, variants, seed):
    cases = json.loads((output / "summary.json").read_text(encoding="utf-8"))["cases"]
    if len(cases) != 2 or any(c["status"] != "ok" or c["seed"] != seed for c in cases):
        raise ValueError("both amplitude conditions must render successfully with the same seed")
    directories, formats, windows = {}, set(), []
    for condition, spec in variants.items():
        case = next(c for c in cases if c["sample"] == spec["sample"])
        directory = output / case["case"]
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for name in ("audio.wav", "report.jsonl"):
            if beta.sha256(directory / name) != metrics["artifact_sha256"][name]:
                raise ValueError(f"artifact changed after render validation: {name}")
        if metrics["audio"]["all_silent"] or metrics["audio"]["clipping_fraction"]:
            raise ValueError(f"{condition}: silent or clipped comparison audio")
        with wave.open(str(directory / "audio.wav"), "rb") as stream:
            formats.add((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                         stream.getnframes(), stream.getcomptype()))
        with (directory / "report.jsonl").open(encoding="utf-8") as stream:
            population = [r for line in stream if (r := json.loads(line))["type"] == "population_step"]
        for label in ("tension", "early_resolution"):
            windows.append({"condition": condition, "seed": seed,
                            **resolution.window_metrics(metrics["report"]["listener"]["series"],
                                                        population, directory / "audio.wav", label)})
        directories[condition] = directory
    if len(formats) != 1:
        raise ValueError("paired audio formats or durations differ")
    failures = resolution.flow_pre_intervention_checks(
        output, variants, [seed], end_sec=INTERVENTION_SEC,
        pairs=[("unchanged", "reduced")], filename="amplitude_pre_intervention.json")
    if failures:
        raise ValueError("amplitude comparison differs before intervention")
    beta.dump_json(output / "amplitude_windows.json", {"windows": windows, "listening_status": "unreviewed"})

    order = list(variants)
    random.SystemRandom().shuffle(order)
    audio_dir = output / "audition_audio"
    audio_dir.mkdir()
    mapping = {}
    for label, condition in zip(("A", "B"), order):
        source = directories[condition] / "audio.wav"
        full = audio_dir / f"{label}_full.wav"
        excerpt = audio_dir / f"{label}_excerpt.wav"
        shutil.copy2(source, full)
        with wave.open(str(source), "rb") as stream:
            start = math.ceil(EXCERPT_START_SEC * stream.getframerate())
            end = math.ceil(RELEASE_SEC * stream.getframerate())
            stream.setpos(start)
            pcm = stream.readframes(end - start)
            if len(pcm) != (end - start) * stream.getnchannels() * stream.getsampwidth():
                raise ValueError("incomplete audition excerpt")
            with wave.open(str(excerpt), "wb") as clip:
                clip.setparams(stream.getparams())
                clip.writeframes(pcm)
        mapping[label] = {"condition": condition, "amp_after_intervention": AMPLITUDES[condition],
                          "case": str(directories[condition].relative_to(output)),
                          "full_sha256": beta.sha256(full), "excerpt_sha256": beta.sha256(excerpt)}
    campaign = {"id": output.name, "seed": seed, "intervention_sec": INTERVENTION_SEC,
                "excerpt_interval_sec": [EXCERPT_START_SEC, RELEASE_SEC]}
    beta.dump_json(output / "audition_mapping.json", {
        **campaign, "presentation_order": ["A", "B"], "assignment": mapping,
        "normalization": "none; PCM copied without gain changes", "listening_status": "unreviewed"})
    page = AUDITION_HTML.replace("__CAMPAIGN_JSON__", json.dumps(campaign, ensure_ascii=True).replace("<", "\\u003c"))
    page = page.replace("__CAMPAIGN_TEXT__", html.escape(output.name))
    (output / "audition.html").write_text(page, encoding="utf-8")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=Path, default=root / "config.toml")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**64 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require a u64 seed and a positive finite timeout")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or root / "target" / "resolution-evaluation" / f"flow-amplitude-{stamp}").resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    source_path = root / "samples" / "12_emergence_and_resolution.rhai"
    manifest = None
    try:
        source = source_path.read_text(encoding="utf-8")
        variants = {}
        with tempfile.TemporaryDirectory(prefix="conchordal-flow-amplitude-", dir="/tmp") as temporary:
            paths = []
            for condition, amp in AMPLITUDES.items():
                path = Path(temporary) / f"flow_amplitude_{condition}.rhai"
                path.write_text(amplitude_source(source, condition), encoding="utf-8")
                variants[condition] = {"sample": path.stem, "amp": amp, "sha256": beta.sha256(path)}
                paths.append(str(path))
            result = beta.main(["--samples", *paths, "--seeds", str(args.seed),
                                "--config", str(args.config), "--output", str(output),
                                "--timeout", str(args.timeout), "--reserve-runtime-ids-through",
                                str(resolution.RESERVE_RUNTIME_IDS_THROUGH)])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["amplitude_experiment"] = {
            "source_sample": str(source_path), "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "variants": variants, "intervention_sec": INTERVENTION_SEC, "release_sec": RELEASE_SEC,
            "excerpt_start_sec": EXCERPT_START_SEC, "dedicated_beat_carrier": False,
            "scope": "flow amplitude patch only; identical pre-intervention PCM/state required; no audio normalization",
            "listening_status": "unreviewed"}
        if result:
            raise ValueError("amplitude rendering failed; see per-case logs")
        publish_audition(output, variants, args.seed)
        manifest["amplitude_experiment"]["status"] = "complete"
        manifest["amplitude_experiment"]["audition_sha256"] = {
            name: beta.sha256(output / name) for name in ("audition.html", "audition_mapping.json")}
        beta.dump_json(output / "manifest.json", manifest)
        print(f"amplitude audition: {output / 'audition.html'}", flush=True)
        return 0
    except (ValueError, KeyError, StopIteration, OSError, EOFError, wave.Error) as error:
        print(f"amplitude evaluation failed: {error}", file=sys.stderr)
        if manifest is not None:
            manifest["status"] = "failed"
            manifest.setdefault("amplitude_experiment", {}).update(status="failed", error=str(error))
            beta.dump_json(output / "manifest.json", manifest)
        return 1


AUDITION_HTML = """<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sample 12 — 音量 A/B</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; background: #101819; color: #e5eeeb; }
body { max-width: 760px; margin: 3rem auto; padding: 0 1.2rem 4rem; line-height: 1.7; }
h1 { font-size: 1.8rem; } h2 { font-size: 1.2rem; }
section { background: #192526; border: 1px solid #344746; border-radius: 12px; padding: 1.3rem; margin: 1.4rem 0; }
button, select, textarea { font: inherit; padding: .6rem .9rem; border-radius: 6px; border: 1px solid #657e78; }
button { background: #304a44; color: #fff; cursor: pointer; margin: .3rem .5rem .3rem 0; }
button:focus-visible, select:focus-visible, textarea:focus-visible { outline: 3px solid #b3db9c; }
audio, textarea { width: 100%; box-sizing: border-box; } audio { margin-top: 1rem; }
label { display: block; margin: .7rem 0; } select { max-width: 100%; }
.muted { color: #b2c4bf; font-size: .9rem; } #playing { min-height: 1.7em; }
</style>
<main>
<p class="muted">CONCHORDAL · SAMPLE 12</p><h1>音量の違いを聴く — A/B</h1>
<p>専用の拍打ちを除いた構成です。AとBでは、途中からflowの音量だけが異なります。
glideや基準音の帰還などは共通です。条件名は伏せてあります。</p>
<section aria-labelledby="listen"><h2 id="listen">1. 同じ音量で聴き比べる</h2>
<p>まずA、次にBを聴いてください。抜粋は15.0〜23.6秒。最初の5.3秒は同じ音で、その後の3.3秒が比較区間です。
音源ごとの音量調整はありません。再生機器とプレイヤーの音量を揃えたまま比較してください。</p>
<label for="mode">再生範囲</label><select id="mode"><option value="excerpt">前後の抜粋（8.6秒）</option>
<option value="full">全体（約36秒）</option></select>
<div><button type="button" data-label="A">Aを最初から聴く</button>
<button type="button" data-label="B">Bを最初から聴く</button></div>
<audio id="player" controls preload="metadata"></audio>
<p id="playing" aria-live="polite">Aから再生できます。</p></section>
<section aria-labelledby="judge"><h2 id="judge">2. 二つの判断を分けて記録する</h2>
<form id="answers">
<label for="difference">AとBの音の違いは聞き取れますか？</label>
<select id="difference" required><option value="">選択してください</option><option>聞き取れる</option>
<option>聞き取れない</option><option>判断できない</option></select>
<label for="settling">どちらが協和へ落ち着くと感じますか？</label>
<select id="settling" required><option value="">選択してください</option><option>A</option><option>B</option>
<option>同程度</option><option>判断できない</option></select>
<label for="notes">気づいた音や時刻（任意）</label><textarea id="notes" rows="3"></textarea>
<p><button type="submit">記録を保存（JSON）</button><span id="saved" aria-live="polite"></span></p>
</form><p class="muted">回答はこのブラウザー内に保持されます。保存ボタンでファイルをダウンロードできます。
回答の自動送信はありません。チャットに二つの判断を直接返しても構いません。</p></section>
<p class="muted">試聴セット: __CAMPAIGN_TEXT__</p></main>
<script>
const campaign = __CAMPAIGN_JSON__;
const player = document.getElementById('player'), mode = document.getElementById('mode');
const playing = document.getElementById('playing'), form = document.getElementById('answers');
const fields = ['difference', 'settling', 'notes'];
const storageKey = 'conchordal-flow-amplitude:' + campaign.id;
let events = [], activeLabel = null, request = 0;
try {
  const prior = JSON.parse(localStorage.getItem(storageKey));
  if (prior) { fields.forEach(key => document.getElementById(key).value = prior.answers[key] || ''); events = prior.playback || []; }
} catch (_) {}
function record() {
  return {campaign, recorded_at: new Date().toISOString(),
    answers: Object.fromEntries(fields.map(key => [key, document.getElementById(key).value])), playback: events};
}
function persist() { try { localStorage.setItem(storageKey, JSON.stringify(record())); } catch (_) {} }
document.querySelectorAll('[data-label]').forEach(button => button.addEventListener('click', async () => {
  const currentRequest = ++request;
  player.pause(); activeLabel = button.dataset.label;
  player.src = 'audition_audio/' + activeLabel + '_' + mode.value + '.wav';
  player.load();
  try { await player.play(); if (currentRequest === request) playing.textContent = activeLabel + 'を再生中'; }
  catch (_) { if (currentRequest === request) playing.textContent = '再生できません。プレイヤーの再生ボタンで再試行してください。'; }
}));
player.addEventListener('play', () => {
  events.push({label: activeLabel, mode: mode.value, offset_sec: player.currentTime,
    volume: player.volume, muted: player.muted, at: new Date().toISOString()}); persist();
});
player.addEventListener('volumechange', () => {
  events.push({event: 'volumechange', volume: player.volume, muted: player.muted,
    at: new Date().toISOString()}); persist();
});
player.addEventListener('ended', () => { playing.textContent = activeLabel + 'の再生終了'; });
mode.addEventListener('change', () => { ++request; player.pause(); player.removeAttribute('src'); player.load();
  activeLabel = null; playing.textContent = '範囲を変更しました。AまたはBを選んでください。'; });
form.addEventListener('input', persist);
form.addEventListener('submit', event => {
  event.preventDefault(); persist();
  const url = URL.createObjectURL(new Blob([JSON.stringify(record(), null, 2) + '\\n'], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = campaign.id + '-listening.json';
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  document.getElementById('saved').textContent = 'ダウンロードを開始しました。';
});
</script></html>
"""


if __name__ == "__main__":
    sys.exit(main())
