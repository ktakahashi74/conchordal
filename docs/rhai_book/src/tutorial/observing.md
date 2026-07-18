# Performance

[Quick Start](quick_start.md) explains how to write and run a minimal
scenario. This chapter describes the actual performance workflow: starting
the real-time instrument, ending a run, performing without the GUI, and then
recording or replaying what happened.

## Performing with the GUI

Start a scenario with the real-time instrument:

```bash
cargo run --release -- samples/10_generations.rhai
```

The scenario is compiled, the GUI opens, and playback begins automatically.
The script supplies the macro-timeline: `place()` introduces Populations,
`wait()` advances script time, and `section()` or `play()` releases the
Populations created inside its scope. During that timeline, Voice behavior,
the Landscape, and their feedback continue to evolve in real time.

The progress bar shows scenario time. While playback is running, the control
at its right exits the performance; closing the window or pressing Ctrl-C also
stops it. It is not a pause button, and the GUI is not a live script editor.
To change the scenario, stop the run, edit the Rhai file, and run it again.

To prepare the GUI first and begin on a cue, add `--wait-user-start`. Press
Space or click the start control when ready:

```bash
cargo run --release -- samples/10_generations.rhai --wait-user-start
```

By default the GUI remains open when the scenario finishes, allowing the
final state to be inspected. Close it with the playback control or window
close button. Add `--wait-user-exit=false` when it should close automatically
at the end.

## Performing without the GUI

`--nogui` starts immediately and exits when the scenario finishes. It still
plays through the default audio device:

```bash
cargo run --release -- samples/10_generations.rhai --nogui
```

Use `--play=false` only for a silent simulation or report-only run:

```bash
cargo run --release -- samples/10_generations.rhai --nogui --play=false --report run.jsonl
```

`--nogui` disables both wait-for-start and wait-for-exit. Offline WAV output
is a different operation provided by `conchordal-render`; the real-time
instrument itself never records audio.

## Listen, inspect, revise

A scenario specifies which Populations enter, when they enter, and the
policies that guide their behavior. It does not predetermine the exact pitches,
rhythmic synchronization, survival, or respawn outcome of a performance;
those emerge from interaction with the changing Landscape. Run the scenario,
listen to the result, inspect what happened, adjust the script, and run it
again.

`conchordal` (the instrument) never writes audio to disk, in any build
profile. A performance is designed as an ephemeral event that leaves no audio
inside the instrument after it ends.

## Replaying a performance

Every run logs its seed:

```text
scenario seed: 3821650944810716341 (replay with --seed 3821650944810716341)
```

Top-level samples start from a fresh, unseeded run every time so they expose
the system's variation rather than prescribe one fixed result. When a run is
worth keeping, replay it exactly with the logged value:

```bash
cargo run --release -- samples/10_generations.rhai --seed 3821650944810716341
```

`--seed` works the same way on `conchordal-render` for turning a kept
performance into a WAV. A script-level `seed(...)` call always wins over the
flag, since it runs during script evaluation and overwrites whatever seed the
run started with — reach for it only when a script itself must be
reproducible regardless of how it is invoked.

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
- `scene_marker` — the start of each `section("name", || { ... })`, so later
  records can be grouped by the most recent marker.
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
- `rhythm_observation` — instantaneous global Kuramoto and environment rhythm
  state. `rhythm_summary` reports onset density, inter-onset-interval
  regularity, and burstiness globally and per Population; its Kuramoto summary
  is global only.
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

## The GUI

The GUI shows the same landscape and listener-twin state live, as it runs.
Reports exist for reading *after* listening — for the moments you noticed
something but couldn't hold every number in your head at once.
