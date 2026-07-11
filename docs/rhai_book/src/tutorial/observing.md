# Observing a Performance

A scenario script is open-loop: it says where voices enter and how they
behave, not what the terrain will do with them. The composing loop closes it
by hand — run, listen, read what actually happened, adjust the numbers, run
again.

`conchordal` (the instrument) never writes audio to disk, in any build
profile — performances are ephemeral by design. It can, however, write
analysis. That asymmetry is deliberate: the sound is a one-time event, but the
evidence about how the ecology behaved is worth keeping.

## Reports

Run with `--report`, pointing at a file to write:

```bash
cargo run --release -- samples/10_generations.rhai --report run.jsonl
```

`--report` is a flag on the `conchordal` instrument (it also works headless,
with `--nogui`). It is not available on `conchordal-render`, which renders
audio rather than observing a live run.

The file is JSON Lines — one record per line, tagged by `type`. The record
families are:

- `meta` — the effective scenario seed, written first.
- `scene_marker` — your `section("name", || { ... })` boundaries, so records
  can be read section by section.
- `spawn` / `respawn` / `death` — population turnover per voice: `spawn` and
  `respawn` carry the entry frequency (and, for `respawn`, the parent voice);
  `death` carries configured nominal endurance, energy-depletion time,
  observable `lifetime_sec` (including the envelope tail), an early-life
  consonance snapshot (`first_k_mean`), and phase-locking at death
  (`plv_at_death`).
- `onset` — per-voice onset time, strength, frequency, phase-locking, and
  scaffold context.
- `population_step` — active Population size (including `alive_count: 0` while
  awaiting replacement), mean frequency, Consonance Field score and level, and
  frequency entropy over time.
- `listener_state` — the `ListenerTwin`'s four perceptual levels
  (`stability_level`, `resolvability_level`, `tension_level`, and
  `attention_level`), beat/subdivision/measure tracking, and analysis lag.
- `rhythm_observation` / `rhythm_summary` — Kuramoto order, onset density,
  inter-onset-interval regularity, and burstiness, globally and per Population.
- `listener_confidence_summary` — peak and late-window beat confidence.
- `dcc_pressure` — listener-derived tension pressure and the pitch-temperature
  bonus applied by DCC.
- `phonation_gate_open` — when a voice's phonation gate opens, and the
  consonance value it opened at.

Reading the raw JSONL answers narrow questions well: "when did Population 3's
membership actually turn over?" or "what did `tension_level` do right after
that anchor entered?"

## Reading a report

The stream is plain JSONL, so section-level questions are one filter away
with any JSON tool. For example, every death with its lifetime and early-life
consonance:

```bash
jq -c 'select(.type == "death")
       | {time_sec, population_id, configured_endurance_sec,
          energy_depletion_sec, lifetime_sec, first_k_mean}' run.jsonl
```

Bucket those by your `scene_marker` times and "did the colony starve in
section IV, and at what consonance?" becomes a direct read. There is no
bespoke digest tool: which summaries matter depends on the piece, and a
one-off filter (or a script) shaped to the question beats a fixed report
format.

## Replaying a performance

Every run logs its seed:

```text
scenario seed: 3821650944810716341 (replay with --seed 3821650944810716341)
```

Top-level études start from a fresh, unseeded run every time by design — that
is what makes them études rather than fixed pieces. When a run is worth
keeping, replay it exactly with the logged value:

```bash
cargo run --release -- samples/10_generations.rhai --seed 3821650944810716341
```

`--seed` works the same way on `conchordal-render` for turning a kept
performance into a WAV. A script-level `seed(...)` call always wins over the
flag, since it runs during script evaluation and overwrites whatever seed the
run started with — reach for it only when a script itself must be
reproducible regardless of how it is invoked.

## The GUI

The GUI shows the same landscape and listener-twin state live, as it runs.
Reports exist for reading *after* listening — for the moments you noticed
something but couldn't hold every number in your head at once.
