# v0.4 Alpha Guide

Conchordal v0.4.0 is aimed at research composers who want to work with
Conchordal concepts directly. It is not trying to hide the model behind common
music-production vocabulary.

The release center is harmony-first composition: Consonance Field, Consonance
Density, Consonance Movement, Consonance Viability, and the first rhythm
foundation that can later fuse with harmony.

## Suggested Path

Run these scripts in order:

```bash
cargo run --release -- samples/04_ecosystems/consonance_ecology.rhai
cargo run --release -- samples/04_ecosystems/pulse_foundation.rhai
```

`consonance_ecology.rhai` is the flagship demo. Listen for a population that is
not merely placed on a chord, but reorganizes as the consonance terrain, mirror,
viability pressure, and respawn rules interact.

`pulse_foundation.rhai` is a rhythm-foundation sketch. It is not the final
rhythm/harmony fusion. Listen for pulse articulation, gated durations,
vitality-modulated coupling, and attack-phase reward.

The rest of `samples/04_ecosystems/` contains research comparisons and older
mechanism sketches. `consonance_field_control.rhai` remains there as a demoted
harmony-control sketch, but it is not part of the first v0.4 alpha path.

## Core API Concepts

`consonance(root_hz)` places voices near high Consonance Field positions around
a root. Use it when the musical thought is "start from this harmonic center".

`consonance_density(min_hz, max_hz)` samples from Consonance Density. Use it
when the musical thought is "let the field seed a population inside this range".

`consonance_movement()` makes a voice move by free hill-climb behavior with
glide defaults. Use `movement_glide(tau_sec)` when the musical thought is
"same movement idea, slower or faster pitch motion".

`consonance_viability(low, high)` defines the consonance window that feeds
life-cycle viability. It defaults to environment-relative scoring, so a voice
is evaluated against the field with its own footprint approximately removed.
Use `viability_scope("total")` only when total-field viability is the intended
comparison.

`respawn_consonance()` lets a population replace itself through
consonance-biased heredity. Combine it with `respawn_capacity(count)` and
`respawn_settle(strategy)` for ecology-scale scripts.

`set_harmonic_mirror(value)` changes the overtone/undertone balance of the
harmonicity field. It is the most direct way to bend the harmonic terrain during
a scene.

## Rhythm Foundation

Use `repeat()`, `pulse(freq_hz)`, and `gates(n)` for explicit attacks. Use
`rhythm_coupling_vitality(lambda_v, v_floor)` and
`rhythm_reward(rho_t, "attack_phase_match")` when timing should affect
survival and reorganization.

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
- `pitch_mode(name)`, `pitch_core(name)`, `pitch_apply_mode(name)`,
  `pitch_glide(tau_sec)`
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
