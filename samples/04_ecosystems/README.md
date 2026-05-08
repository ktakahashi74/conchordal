# Ecosystem Samples

This directory contains v0.4 rhythm/harmony ecology candidates plus research
comparison scripts. The redesigned rhythm path is now present as compile-only
and headless-report candidates; audition and tuning still decide whether each
script is release curated.

## Redesigned v0.4 Rhythm Path

```bash
cargo run --release -- samples/04_ecosystems/metric_beat_foundation.rhai
cargo run --release -- samples/04_ecosystems/entrained_beat.rhai
cargo run --release -- samples/04_ecosystems/flow_timing_field.rhai
cargo run --release -- samples/04_ecosystems/conchordal_ecology.rhai
```

| Script | Role | Approx. duration | What it should show |
|---|---|---:|---|
| `metric_beat_foundation.rhai` | Metric beat candidate | 39s | Stable beat, gate-snap articulation, and consonance-biased pitch placement. |
| `entrained_beat.rhai` | Entrained beat candidate | 43s | Agent-phase timing, social onset feedback, vitality coupling, and attack reward. |
| `flow_timing_field.rhai` | Flow timing candidate | 40s | Seeded non-metric onset flow over consonance-biased regions. |
| `conchordal_ecology.rhai` | Integrated flagship candidate | 55s | Metric beat, entrained beat, flow timing, consonance movement, viability, and respawn in one ecology. |

## Paused v0.4 Alpha Path

These scripts are useful candidates, but they are not a complete v0.4 alpha
path:

```bash
cargo run --release -- samples/04_ecosystems/consonance_ecology.rhai
cargo run --release -- samples/04_ecosystems/pulse_foundation.rhai
```

| Script | Role | Approx. duration | What it should show |
|---|---|---:|---|
| `consonance_ecology.rhai` | Consonance ecology candidate | 75s | A population reorganizing under field changes, viability pressure, respawn, and mirror shifts. |
| `pulse_foundation.rhai` | Rhythm source material | 52s | Pulse articulation, gated durations, vitality coupling, and attack-phase reward. |

## Required v0.4 Rhythm Families

The final curated path should be rebuilt around:

- metric beat: a Western-music-like pulse surface
- entrained beat: synchronization emerging from agent coupling and survival
- flow timing: rain/river-like non-metric onset texture
- integrated rhythm/harmony ecology: the final flagship

See `docs/roadmap/v0.4.0-rhythm-redesign.md`.

## Headless Reports

The current non-audition report gate is recorded in
`docs/roadmap/v0.4.0-rhythm-report-runs.md`. Regenerate the reports with:

```bash
target/debug/conchordal --nogui --play false --report target/rhythm_reports/metric_beat_foundation.jsonl samples/04_ecosystems/metric_beat_foundation.rhai
target/debug/conchordal --nogui --play false --report target/rhythm_reports/entrained_beat.jsonl samples/04_ecosystems/entrained_beat.rhai
target/debug/conchordal --nogui --play false --report target/rhythm_reports/flow_timing_field.jsonl samples/04_ecosystems/flow_timing_field.rhai
target/debug/conchordal --nogui --play false --report target/rhythm_reports/conchordal_ecology.jsonl samples/04_ecosystems/conchordal_ecology.rhai
```

## Research Comparisons

These scripts remain useful, but they are not the first v0.4 user path.

| Scripts | Role |
|---|---|
| `consonance_field_control.rhai` | Demoted harmony-control sketch; useful for mechanism comparison, but not clear enough as a first-listen demo. |
| `mirror_dualism*.rhai` | Compact mirror/field comparison sketches. |
| `temporal_scaffolding_*.rhai` | External scaffold comparisons for rhythm-coupling behavior. |
| `hereditary_adaptation_*.rhai` | Reference assays for selection, heredity, and approximate leave-one-out variants. |
| `drift_flow.rhai`, `emergent_harmony.rhai`, `symbiotic_field.rhai` | Earlier ecosystem sketches retained for mechanism exploration. |

## Audition Gate

Before cutting v0.4.0 alpha, rebuild and audition the redesigned rhythm/harmony
demo set and record:

- whether each script should remain curated, be revised, or be demoted
- whether the duration and density are acceptable
- whether the metric, entrained, and flow rhythm families are each audible
- whether the integrated flagship sounds like one Conchordal ecology rather
  than separate rhythm and harmony layers
