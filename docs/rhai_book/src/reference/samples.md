# Curated Samples

The curated v0.4.0 listening path is the rhythm/harmony ecology set under
`samples/04_ecosystems/`. Run each with:

```bash
cargo run --release -- samples/04_ecosystems/metric_beat_foundation.rhai
cargo run --release -- samples/04_ecosystems/entrained_beat.rhai
cargo run --release -- samples/04_ecosystems/flow_timing_field.rhai
cargo run --release -- samples/04_ecosystems/rhythm_harmony_ecology.rhai
cargo run --release -- samples/04_ecosystems/conchordal_ecology.rhai
cargo run --release -- samples/04_ecosystems/conchordal_flagship.rhai
```

## What to listen for

**`metric_beat_foundation.rhai`** — the metric region of the coupling
continuum. A Western-music-like pulse should surface and be immediately
legible.

**`entrained_beat.rhai`** — the entrained region. Synchronization emerges from
agent phase, social onset feedback, vitality coupling, and attack reward; the
beat drifts with the population.

**`flow_timing_field.rhai`** — the flow region. Non-metric but structured
timing over consonance-biased placement.

**`rhythm_harmony_ecology.rhai`** — functional integration demo. The metric
groove stays stable while entrained agents, non-metric flow, consonance
movement, and harmonic field changes coexist on one field. Judge it as
integration, not as the musical showcase.

**`conchordal_ecology.rhai`** — feature-integration coverage. Metric beat,
entrained beat, flow timing, consonance movement, viability, and respawn
coexist on one field. Judge it as coverage, not as the musical showcase.

**`conchordal_flagship.rhai`** — the dedicated musical etude, *Emergence and
Resolution*. A single directed arc: a `harmonic_mirror` arch (consonant →
dissonant → consonant) and register transposition shape the tension/release
drama, while harmony emerges from the agents. The pulse is a deep emergent
attractor: a metric heartbeat (an `accent` role with high `meter_stability`)
drives the shared meter, and a living colony locks to that *same* emergent
beat — its life is harmonic (which pitches snap in and survive each onset),
not rhythmic drift. Flow appears only as non-metric shimmer at the tension
peak. It uses only the features that serve the form.

## Other sample directories

- `samples/01_fundamentals/` — minimal mechanics: spawning, tone generation,
  basic placement.
- `samples/02_mechanisms/` — single-mechanism studies (hill-climb convergence,
  rhythmic sync, modal timbres, spectral gaps).
- `samples/03_structures/` — directed harmonic structures.
- The rest of `samples/04_ecosystems/` contains research comparisons and
  mechanism sketches (e.g. `consonance_ecology.rhai`, `pulse_foundation.rhai`,
  the `hereditary_adaptation_*` and `temporal_scaffolding_*` assay sets).

All samples are compile-checked by the test suite, so they always match the
current API.

## Offline rendering

The `conchordal` instrument never writes audio to disk — performances are
ephemeral by design. For offline WAV rendering use the separate
`conchordal-render` binary, which shares the core engine but is not the
instrument.
