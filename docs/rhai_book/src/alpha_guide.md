# v0.4 Alpha Guide

Conchordal v0.4.0 is aimed at research composers who want to work with
Conchordal concepts directly. It is not trying to hide the model behind common
music-production vocabulary.

The release center is being reset around rhythm/harmony ecology: Consonance
Field, Consonance Density, Consonance Movement, Consonance Viability, and
rhythm as a living time structure. The previous rhythm-foundation framing is no
longer sufficient for v0.4.0.

## Current Candidate Path

The redesigned v0.4.0 rhythm/harmony path is now the active candidate set.
These scripts still require audition and tuning before release curation:

```bash
cargo run --release -- samples/04_ecosystems/metric_beat_foundation.rhai
cargo run --release -- samples/04_ecosystems/entrained_beat.rhai
cargo run --release -- samples/04_ecosystems/flow_timing_field.rhai
cargo run --release -- samples/04_ecosystems/rhythm_harmony_ecology.rhai
cargo run --release -- samples/04_ecosystems/conchordal_ecology.rhai
```

`metric_beat_foundation.rhai` should make a Western-music-like pulse surface
immediately legible.

`entrained_beat.rhai` should make synchronization emerge from agent phase,
social onset feedback, vitality coupling, and attack reward.

`flow_timing_field.rhai` should make non-metric but structured timing audible
over consonance-biased placement.

`rhythm_harmony_ecology.rhai` is a functional integration demo. It should keep
the metric groove stable while entrained agents, non-metric flow, consonance
movement, and harmonic field changes coexist on one field. Do not judge it as
the musical showcase.

`conchordal_ecology.rhai` is candidate material for the musical showcase. Listen
for whether metric beat, entrained beat, flow timing, consonance movement,
viability, and respawn form one ecology, or whether a dedicated etude is needed
instead.

A separate musical etude or showcase is still needed. It should use only the
features that serve the musical form, rather than trying to demonstrate every
v0.4.0 API concept at once.

The rest of `samples/04_ecosystems/` contains research comparisons and older
mechanism sketches. `consonance_ecology.rhai` and `pulse_foundation.rhai`
remain useful source material, but they are no longer sufficient as the v0.4.0
alpha path. `consonance_field_control.rhai` remains there as a demoted
harmony-control sketch.

## Core API Concepts

`peaks(root_hz)` places voices near high Consonance Field positions around
a root. Use it when the musical thought is "start from this harmonic center".

`density(min_hz, max_hz)` samples from Consonance Density. Use it
when the musical thought is "let the field seed a population inside this range".

`seek_consonance()` makes a voice move by free hill-climb behavior with
glide defaults. Use `glide(tau_sec)` when the musical thought is
"same movement idea, slower or faster pitch motion".

`consonance_viability(low, high)` defines the consonance window that feeds
life-cycle viability. It defaults to environment-relative scoring, so a voice
is evaluated against the field with its own footprint approximately removed.
Use `viability_scope("total")` only when total-field viability is the intended
comparison.

`respawn_consonance()` lets a population replace itself through
consonance-biased heredity. Combine it with `respawn_capacity(count)` and
`respawn_settle(strategy)` for ecology-scale scripts.

`harmonic_mirror(value)` changes the overtone/undertone balance of the
harmonicity field. It is the most direct way to bend the harmonic terrain during
a scene.

## Rhythm Redesign

The v0.4.0 rhythm surface is being redesigned around three families:

- metric beat: a Western-music-like pulse surface
- entrained beat: synchronization emerging from agent coupling and survival
- flow timing: rain/river-like non-metric onset texture

The new entry points are `metric_beat(rate_hz)`, `entrained_beat(rate_hz)`,
and `flow_timing(mean_rate_hz[, depth])`.

The current low-level tools remain useful while this is designed. Use
`repeat()`, `pulse(freq_hz)`, and `cycles(n)` for explicit attacks. Use
`rhythm_coupling_vitality(lambda_v, v_floor)` and
`rhythm_reward(rho_t, "attack_phase_match")` when timing should affect survival
and reorganization.

The scaffold functions are external comparison controls:

- `set_scaffold_off()`
- `set_scaffold_shared(freq_hz)`
- `set_scaffold_scrambled(freq_hz, seed)`

They are useful for research assays and demos, but they are not the final
rhythm-composition abstraction.

## Research Controls

Some functions remain available because they are useful for assays and
mechanism-level work:

- `selection_approx_loo(enabled)`
- `pitch_mode(name)`, `pitch_core(name)`, `pitch_apply_mode(name)`
- `set_control_update_mode(name)`
- `set_pitch_objective(name)`

Prefer the higher-level v0.4.0 names in curated demos unless the script is
explicitly testing a mechanism.

## Deferred To Later

The v0.4.0 target does not require a finished rhythm/harmony fusion API. It
requires a foundation that can grow into one.

The v0.4.0 target also does not require backward compatibility with older alpha
names. Clean research-composer-facing API names take priority.

General musician onboarding, polished production workflows, and broader product
documentation are deferred until the v0.4.0 harmony core is stable.
