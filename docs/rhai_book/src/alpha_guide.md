# v0.4 Alpha Guide

Conchordal v0.4.0 is aimed at research composers who want to work with
Conchordal concepts directly. It is not trying to hide the model behind common
music-production vocabulary.

The release center is being reset around rhythm/harmony ecology: Consonance
Field, Consonance Density, Consonance Movement, Consonance Viability, and
rhythm as a living time structure. The previous rhythm-foundation framing is no
longer sufficient for v0.4.0.

## Current Candidate Path

The redesigned v0.4.0 rhythm/harmony path is now the active candidate set. The
three rhythm families pass the initial separation audition (metric beat is a
stable pulse, entrained beat drifts with the population, flow timing stays
non-metric). Meter correspondence, musicality, and a dedicated flagship showcase
are still being tuned, so these scripts are candidates, not final curation:

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

The v0.4.0 rhythm surface is organized around three families:

- metric beat: a Western-music-like pulse surface
- entrained beat: synchronization emerging from agent coupling and survival
- flow timing: rain/river-like non-metric onset texture

The entry points are `metric_beat(rate_hz)`, `entrained_beat(rate_hz)`, and
`flow_timing(mean_rate_hz[, depth])`. They are Tier-1 phonation presets and
work on both a species (`Material`) and a draft group (`Participant`). Rates
accept integer or float literals: `metric_beat(2)` and `metric_beat(2.0)` are
the same.

Below them sit the explicit lower-level controls. Use `repeat()`,
`pulse(freq_hz)`, and `cycles(n)` for explicit attacks. Use
`rhythm_coupling_vitality(lambda_v, v_floor)` and
`rhythm_reward(rho_t, "attack_phase_match")` when timing should affect survival
and reorganization.

The scaffold functions are external comparison controls:

- `set_scaffold_off()`
- `set_scaffold_shared(freq_hz)`
- `set_scaffold_scrambled(freq_hz, seed)`

They are useful for research assays and demos, but they are not the final
rhythm-composition abstraction.

## Listener Twin And DCC

Every voice feeds two independent buses (see the API reference *Routing*
section):

- the **presentation bus** is the work as heard — cpal output, recording, and
  UI metering.
- the **habitat bus** is what the ALife ecology senses through NSGT analysis and
  the landscape.

Keeping these separate is a composition tool. A hidden anchor on
`send(habitat_bus)` can shape how the population organizes without ever becoming
audible, and presented decor on `send(presentation_bus)` can be heard without
perturbing the ecology.

conchordal also keeps a `ListenerTwin`: a listener-side model of the *presented*
sound only. It never reads the habitat bus, so hidden scaffolds cannot create
fake listener tension. It reports five state values:

- `stability_level`: how stable / consonant the current audible sound is.
- `resolvability_level`: whether a nearby audible state offers a plausible, more
  stable continuation.
- `tension_level`: `(1 - stability) * resolvability` — unstable now, but with a
  reachable path toward improvement.
- `attention_level`: presentation-derived onset / spectral-flux salience.
- `neural_rhythms`: presentation-derived listener-side rhythm (delta/theta).

There is no scripting verb for the twin. It is observed, not commanded: when you
run with reporting enabled it emits `listener_state` records, and the GUI shows
the same state. Use it to check whether the tension you hear matches what the
twin reports before coupling it back into generation.

That optional coupling is **DCC**, configured in `conchordal.toml`, not in
script:

```toml
[dcc]
# Listener pressure is report/UI-only by default.
# coupling_strength = 0.0
# max_exploration_bonus = 0.10
```

- `coupling_strength` (`0.0`-`1.0`, default `0.0`): at `0.0` the twin is
  report/UI-only and generation is unchanged. Above `0.0` it applies
  `tension_pressure = tension_level * resolvability_level * coupling_strength`
  as a transient pitch-exploration bonus only. It never sets target pitches or
  changes rhythm synchronization.
- `max_exploration_bonus` (default `0.10`): ceiling on that transient bonus.

Raise `coupling_strength` gradually, and only after the reported
`listener_state` looks musically legible.

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
