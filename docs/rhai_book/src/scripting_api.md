# Quick Start Guide

Conchordal v0.4.0-dev is a research-composer scripting surface. The central
idea is not note scheduling. The central idea is shaping a perceptual consonance
field, then letting populations move, survive, and reorganize inside it.

For a guided listening path, see the [Alpha Guide](alpha_guide.md). For the full
function list, see the [API Reference](reference/life.md).

## Minimal Sound

```ts
create(sine, 1).freq(440.0);
wait(2.0);
```

`create(species, count)` spawns a group. `wait(seconds)` advances the scenario
time and flushes pending groups.

## Basic Objects

- **Species** are voice templates made with `derive()` from presets such as
  `sine`, `harmonic`, `modal`, `saw`, `square`, and `noise`.
- **Groups** are live or draft collections returned by `create()`.
- **Strategies** decide spawn frequencies: `consonance()`,
  `consonance_density()`, `random_log()`, and `linear()`.
- **Scenes** scope groups and release them automatically.

```ts
let voice = derive(harmonic)
    .amp(0.08)
    .sustain()
    .brightness(0.35);

scene("plain entry", || {
    create(voice, 3).place(linear(220.0, 440.0));
    wait(4.0);
});
```

## Consonance Field

`consonance(root_hz)` places voices at high Consonance Field positions around a
root. The field is shaped by what the system perceives.

```ts
let anchor = derive(harmonic)
    .brain("drone")
    .amp(0.06)
    .sustain()
    .pitch_mode("lock");

let voice = derive(harmonic)
    .amp(0.04)
    .sustain();

scene("field placement", || {
    create(anchor, 1).freq(110.0);
    wait(1.0);

    create(voice, 6)
        .place(consonance(110.0).range(1.0, 4.0).min_dist(0.9));
    wait(6.0);
});
```

`set_harmonic_mirror(value)` bends the harmonicity field from overtone emphasis
toward undertone emphasis.

```ts
set_harmonic_mirror(0.0);
wait(4.0);
set_harmonic_mirror(1.0);
wait(4.0);
```

## Consonance Density

`consonance_density(min_hz, max_hz)` samples from the density view of the field.
Use it when the musical thought is a population seeded by the current terrain.

```ts
let cloud = derive(harmonic).amp(0.035).sustain();

create(cloud, 10)
    .place(consonance_density(90.0, 1200.0).min_dist(0.8));
wait(8.0);
```

Density is not just "random but harmonic". It is a normalized distribution
derived from the consonance model, and it remains well-defined inside the
requested range.

## Consonance Movement

Use `consonance_movement()` when voices should actively seek better field
positions. It sets free hill-climb movement with glide defaults.

```ts
let mover = derive(harmonic)
    .amp(0.045)
    .sustain()
    .consonance_movement()
    .movement_glide(0.35)
    .crowding(0.6)
    .global_peaks(8, 70.0)
    .ratio_candidates(5);

create(mover, 8)
    .place(consonance_density(80.0, 900.0));
wait(12.0);
```

Advanced controls such as `pitch_mode()`, `pitch_core()`,
`pitch_apply_mode()`, and `pitch_glide()` remain available for mechanism-level
work. Prefer `consonance_movement()` in curated v0.4.0 scripts.

## Consonance Viability And Respawn

Viability makes field fit matter over time. `consonance_viability(low, high)`
defines the consonance window, and `viability_rate(rate)` controls continuous
recharge. By default, viability uses environment-relative scoring.

```ts
let settle = consonance_density(70.0, 1100.0).min_dist(0.8);

let ecology = derive(harmonic)
    .amp(0.04)
    .repeat()
    .pulse(1.5)
    .gates(3)
    .consonance_movement()
    .movement_glide(0.45)
    .initial_energy(0.7)
    .energy_cap(1.0)
    .metabolism(0.09)
    .action_cost(0.012)
    .viability_rate(0.18)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle);

create(ecology, 14)
    .place(consonance_density(70.0, 1100.0));
wait(30.0);
```

Use `viability_scope("total")` when the compositional question is explicitly
total-field viability. The default `viability_scope("environment")` keeps
v0.4.0 ecology centered on whether a voice is supported by its surroundings.
Use `selection_approx_loo(false)` only for older reference assays that need the
previous implementation-level control.

## Rhythm Redesign

The v0.4.0 rhythm surface is being redesigned so rhythm is a core part of the
same ecology as consonance, viability, movement, and respawn. The redesign must
cover metric beat, entrained beat, and flow timing.

The new entry points name the musical timing intent directly:

```ts
let beat = derive(harmonic)
    .metric_beat(2.0)
    .accent(0.7)
    .gates(2);

let entrained = derive(harmonic)
    .entrained_beat(2.0)
    .gates(2);

let flow = derive(harmonic)
    .flow_timing(3.0, 0.7)
    .gates(1);
```

The current low-level tools remain useful for mechanism-level scripts:

```ts
let pulse_voice = derive(harmonic)
    .repeat()
    .pulse(2.0)
    .gates(2)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");
```

Scaffold functions are external comparison controls:

```ts
set_scaffold_off();
set_scaffold_shared(2.0);
set_scaffold_scrambled(2.0, 17);
```

They are useful for demos and assays. They are not the final rhythm-composition
abstraction.

## Modal Bodies

The `modal` preset can use inharmonic mode patterns. Landscape-aware modes can
sample the live field.

```ts
let shimmer_modes = landscape_density_modes()
    .count(10)
    .range(1.0, 5.5)
    .gamma(1.6)
    .min_dist(0.7);

let shimmer = derive(modal)
    .amp(0.025)
    .sustain()
    .consonance_movement()
    .modes(shimmer_modes)
    .brightness(0.7);
```

Mode constructors include `harmonic_modes()`, `odd_modes()`,
`power_modes(beta)`, `stiff_string_modes(stiffness)`,
`custom_modes([ratios])`, `modal_table(name)`, `landscape_density_modes()`, and
`landscape_peaks_modes()`.

## Live Patching

Some group methods patch running voices. Others are draft-only and must be set
before the first `flush()` or `wait()`.

```ts
let g = create(harmonic, 3)
    .amp(0.04)
    .place(consonance(220.0));
flush();

g.amp(0.02);
g.movement_glide(0.8);
wait(3.0);
release(g);
```

See the [API Reference](reference/life.md) for the live-patchable and
draft-only method lists.

## Current Candidate Path

The active v0.4.0 candidate path is the redesigned rhythm/harmony set:

```bash
cargo run --release -- samples/04_ecosystems/metric_beat_foundation.rhai
cargo run --release -- samples/04_ecosystems/entrained_beat.rhai
cargo run --release -- samples/04_ecosystems/flow_timing_field.rhai
cargo run --release -- samples/04_ecosystems/conchordal_ecology.rhai
```

These scripts cover metric beat, entrained beat, flow timing, and the integrated
rhythm/harmony ecology. They still need audition before final release curation.
`consonance_ecology.rhai`, `pulse_foundation.rhai`, and
`consonance_field_control.rhai` remain useful research comparisons, but they are
not the first alpha-user path.
