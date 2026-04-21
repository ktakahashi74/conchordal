# Rhai Scripting API Reference (v0.4.0-dev)

This reference documents the complete Rhai scripting API for Conchordal's Life
Engine. Scenarios are scripts that define species, spawn voice groups, control
the timeline, and tune global parameters.

---

## Registered Types

| Type | Description |
|------|-------------|
| `SpeciesHandle` | Species template, built with the builder pattern |
| `GroupHandle` | Handle to a voice group (draft or live) |
| `SpawnStrategy` | Frequency allocation strategy for placement |
| `ModePattern` | Frequency pattern for modal synthesis bodies |

---

## Preset Species

Global constants available in every script:

| Preset | Body | Notes |
|--------|------|-------|
| `sine` | Sine | Pure sine wave |
| `harmonic` | Harmonic | Harmonic series synthesis |
| `saw` | Harmonic | brightness = 0.85 |
| `square` | Harmonic | brightness = 0.65 |
| `noise` | Harmonic | brightness = 1.0, motion = 1.0 |
| `modal` | Modal | Resonator-based modal synthesis |

```ts
let voice = derive(harmonic).amp(0.4).brightness(0.7);
```

---

## Species Definition

### derive(parent) -> SpeciesHandle

Clone a preset or existing species for modification.

```ts
let pluck = derive(harmonic).amp(0.3).adsr(0.01, 0.1, 0.3, 0.2);
```

All species methods are chainable and return `SpeciesHandle`.

### Body and Amplitude

| Method | Description |
|--------|-------------|
| `amp(value)` | Amplitude 0.0--1.0 |
| `freq(value)` | Frequency lock in Hz |
| `brightness(value)` | Spectral brightness 0.0--1.0 (harmonic/modal bodies) |
| `spread(value)` | Detuning spread |
| `unison(count)` | Number of unison detuning copies |
| `modes(pattern)` | Set mode pattern (modal body); see [Mode Patterns](#mode-patterns) |

```ts
let pad = derive(harmonic).amp(0.5).brightness(0.6).spread(0.02).unison(3);
let bell = derive(modal).modes(stiff_string_modes(0.02).count(8));
```

### Pitch Control

| Method | Description |
|--------|-------------|
| `pitch_mode(name)` | `"free"` or `"lock"` |
| `pitch_core(name)` | `"hill_climb"` or `"peak_sampler"` |
| `pitch_apply_mode(name)` | `"gate_snap"` or `"glide"` |
| `pitch_glide(tau_sec)` | Glide time constant in seconds |
| `pitch_smooth(tau_sec)` | Pitch smoothing time constant |

```ts
let glider = derive(sine).pitch_mode("free").pitch_core("hill_climb")
    .pitch_apply_mode("glide").pitch_glide(0.1);
```

### Hill-Climb Tuning

| Method | Description |
|--------|-------------|
| `landscape_weight(value)` | Weight of landscape objective |
| `neighbor_step_cents(value)` | Step size for neighbor exploration |
| `tessitura_gravity(value)` | Gravity toward tessitura center |
| `exploration(value)` | Exploration tendency |
| `persistence(value)` | Persistence bias |
| `anneal_temp(value)` | Annealing temperature |
| `temperature(value)` | Alias for `anneal_temp` |
| `improvement_threshold(value)` | Minimum improvement to accept move |
| `move_cost(coeff)` | Cost multiplier for pitch changes |
| `move_cost_exp(exp)` | Exponent for move cost |
| `move_cost_time_scale(name)` | `"legacy"`/`"integration_window"` or `"proposal"`/`"proposal_interval"` |
| `proposal_interval(seconds)` | Proposal generation interval |

### Peak Sampler Tuning

| Method | Description |
|--------|-------------|
| `window_cents(width)` | Search window width in cents |
| `top_k(count)` | Top-k candidates |
| `sigma_cents(spread)` | Gaussian spread in cents |
| `random_candidates(count)` | Number of random candidates |
| `global_peaks(count)` | Global peak candidates (min_sep = 0) |
| `global_peaks(count, min_sep_cents)` | Global peak candidates with separation |
| `ratio_candidates(count)` | Enable ratio-based candidates (0 to disable) |

### Crowding

| Method | Description |
|--------|-------------|
| `crowding(strength)` | Auto-sigma from roughness kernel |
| `crowding(strength, sigma_cents)` | Explicit sigma |
| `crowding_target(same_visible, other_visible)` | Target visibility (booleans) |
| `leave_self_out(enabled)` | Subtract own spectral contribution |
| `leave_self_out_mode(name)` | `"approx"`/`"approx_harmonics"` or `"exact"`/`"exact_scan"` |
| `leave_self_out_harmonics(count)` | Number of harmonics for self-subtraction |

```ts
let social = derive(harmonic)
    .crowding(0.8)
    .crowding_target(true, true)
    .leave_self_out(true).leave_self_out_mode("approx");
```

### Brain (Articulation)

| Method | Description |
|--------|-------------|
| `brain(name)` | `"entrain"`, `"seq"`, or `"drone"` |

| Brain | Behavior |
|-------|----------|
| `entrain` | Synchronizes with detected rhythms in the field (default) |
| `seq` | Fixed-duration note sequencing |
| `drone` | Sustained tone with slow frequency sway |

### Phonation

Phonation is configured in three tiers of increasing detail.

**Tier 1 -- Presets:**

| Method | Description |
|--------|-------------|
| `sustain()` | Sustain mode (default) |
| `repeat()` | Repeated/pulsed mode |

**Tier 2 -- Explicit when/duration:**

| Method | Description |
|--------|-------------|
| `once()` | Single trigger |
| `pulse(rate_hz)` | Pulse at given rate |
| `while_alive()` | Duration: while voice alive |
| `gates(n)` | Duration: n theta gates |
| `field()` | Duration: field-based |

**Tier 3 -- Expert tuning:**

| Method | Description |
|--------|-------------|
| `sync(depth)` | Sync depth 0--1 (requires `pulse`) |
| `social(coupling)` | Social coupling 0--1 (requires `pulse`) |
| `field_window(min, max)` | Field theta window 0--1 (requires `field`) |
| `field_curve(k, x0)` | Field curve parameters (requires `field`) |
| `field_drop(gain)` | Field drop gain (requires `field`) |

```ts
let pulsed = derive(harmonic).repeat().pulse(3.0).sync(0.6).social(0.3);
let fielded = derive(sine).field().field_window(0.2, 0.8);
```

### Routing

Each voice contributes to two independent mono buses:

- **listener bus** → cpal output / wav / UI metering (what the audience hears)
- **perceptual bus** → NSGT analysis → landscape (what other voices perceive)

By default both buses receive the voice. Opt out per axis:

| Method | Effect |
|--------|--------|
| `mute()` | Voice bypasses the listener bus; still contributes to the perceptual landscape. |
| `unperceived()` | Voice bypasses the perceptual bus; still audible to the listener. |

```ts
// Reference anchor: sensed by the population, silent to the audience.
let anchor = derive(harmonic).brain("drone").mute();

// (Hypothetical) audience-only decor that does not influence the ecosystem.
let decor = derive(sine).unperceived();
```

### Lifecycle

| Method | Description |
|--------|-------------|
| `metabolism(rate)` | Energy consumption rate |
| `adsr(attack_sec, decay_sec, sustain_level, release_sec)` | ADSR envelope |

```ts
let pluck = derive(harmonic).metabolism(0.5).adsr(0.01, 0.1, 0.3, 0.2);
```

### Rhythm

| Method | Description |
|--------|-------------|
| `rhythm_coupling(mode)` | `"temporal"` / `"temporal_only"` |
| `rhythm_coupling_vitality(lambda_v, v_floor)` | Vitality-modulated coupling |
| `rhythm_reward(rho_t, metric)` | `"attack_phase_match"` or `"none"` |

### Respawn

| Method | Description |
|--------|-------------|
| `respawn_random()` | Random respawn locations |
| `respawn_hereditary(sigma_oct)` | Hereditary with frequency variance in octaves |

### Telemetry

| Method | Description |
|--------|-------------|
| `enable_telemetry(first_k)` | Enable for first k voices |
| `enable_plv(window)` | Enable Phase Locking Value computation |

### Sustain Drive

| Method | Description |
|--------|-------------|
| `sustain_drive(value)` | Continuous drive level |

---

## Mode Patterns

Mode patterns define frequency relationships for modal synthesis bodies.

### Constructors

All constructors return `ModePattern`.

| Function | Description |
|----------|-------------|
| `harmonic_modes()` | Harmonic series: f, 2f, 3f, ... |
| `odd_modes()` | Odd harmonics: f, 3f, 5f, ... |
| `power_modes(beta)` | Power law: f * n^beta |
| `stiff_string_modes(stiffness)` | Stiffness-adjusted harmonics |
| `custom_modes(ratios)` | Custom frequency ratios from array |
| `modal_table(name)` | Named mode table lookup |
| `landscape_density_modes()` | Modes from landscape density |
| `landscape_peaks_modes()` | Modes from landscape peaks |

### Modifiers

All modifiers are chainable and return `ModePattern`.

| Method | Description |
|--------|-------------|
| `.count(n)` | Number of modes |
| `.range(min_mul, max_mul)` | Frequency range (landscape modes only) |
| `.min_dist(d)` | Min ERB distance (landscape modes only) |
| `.gamma(g)` | Gamma parameter (landscape_density only) |
| `.jitter(cents)` | Randomization in cents |
| `.seed(s)` | Random seed |

```ts
let bell_modes = stiff_string_modes(0.03).count(12).jitter(5.0);
let adaptive = landscape_density_modes().count(6).range(1.0, 8.0).gamma(2.0);
```

---

## Spawn Strategies

Strategies define how frequencies are allocated when spawning a group.

### Constructors

All constructors return `SpawnStrategy`.

| Function | Description |
|----------|-------------|
| `consonance(root_freq)` | Highest consonance positions (default range 1x--4x) |
| `consonance_density_pmf(min_freq, max_freq)` | Weighted-random from density PMF |
| `random_log(min_freq, max_freq)` | Log-uniform random |
| `linear(start_freq, end_freq)` | Linear interpolation |

### Modifiers

| Method | Applies to | Description |
|--------|------------|-------------|
| `.range(min_mul, max_mul)` | `consonance` | Multiplier range relative to root |
| `.min_dist(d)` | `consonance`, `consonance_density_pmf` | Min ERB distance between voices |
| `.reject_targets(anchor_hz, targets_st, exclusion_st, max_tries)` | Any | Reject positions near specified targets |

`reject_targets` wraps any strategy: `targets_st` is an array of semitone offsets
from `anchor_hz`, `exclusion_st` is the exclusion zone width in semitones, and
`max_tries` is the retry limit.

```ts
let strat = consonance(220.0).range(1.0, 3.0).min_dist(0.9);
create(harmonic, 6).place(strat);

let density = consonance_density_pmf(100.0, 800.0).min_dist(1.0);
create(sine, 8).place(density);

// Avoid octaves and fifths of 220 Hz
let filtered = random_log(100.0, 1000.0)
    .reject_targets(220.0, [0.0, 12.0, 7.0, 19.0], 0.5, 50);
```

---

## Group Creation and Operations

### create(species, count) -> GroupHandle

Create a draft group of `count` voices. The group spawns on the next `flush()` or
`wait()` call.

```ts
let g = create(sine, 4).amp(0.3);
flush();  // spawns the group
```

### place(strategy) -> GroupHandle

Set spawn strategy (draft only).

```ts
create(harmonic, 6).place(consonance(220.0).range(1.0, 4.0));
```

### release(group)

Release a live group. Agents enter their release phase and fade out.

```ts
let g = create(sine, 1).freq(440.0);
flush();
wait(2.0);
release(g);
```

### Draft vs. Live Methods

Group methods work in two contexts:

- **Draft** (before `flush()`/`wait()`): configures the group before spawning.
- **Live** (after spawning): patches parameters on running voices.

**Live-patchable** (work in both draft and live):

`amp`, `freq`, `landscape_weight`, `neighbor_step_cents`,
`tessitura_gravity`, `sustain_drive`, `pitch_smooth`, `exploration`,
`persistence`, `anneal_temp`, `temperature`, `move_cost`, `move_cost_exp`,
`improvement_threshold`, `proposal_interval`, `window_cents`, `top_k`,
`sigma_cents`, `random_candidates`, `brightness`, `spread`, `unison`,
`pitch_glide`, `leave_self_out`, `leave_self_out_mode`,
`leave_self_out_harmonics`, `crowding`, `crowding_target`, `global_peaks`,
`ratio_candidates`, `move_cost_time_scale`, `pitch_apply_mode`

**Draft-only** (ignored with a warning if called on a live group):

`brain`, `pitch_mode`, `pitch_core`, `sustain`, `repeat`, `once`,
`pulse`, `while_alive`, `gates`, `field`, `sync`, `social`, `field_window`,
`field_curve`, `field_drop`, `metabolism`, `adsr`, `rhythm_coupling`,
`rhythm_coupling_vitality`, `rhythm_reward`, `respawn_random`,
`respawn_hereditary`, `modes`, `place`

```ts
let g = create(harmonic, 4)
    .brightness(0.7)          // draft: sets initial value
    .brain("entrain")         // draft-only
    .place(consonance(220.0));
flush();

g.brightness(0.3);            // live: patches running voices
wait(2.0);
release(g);
```

---

## Timeline and Control Flow

### wait(seconds)

Flush pending groups, then advance the timeline cursor.

```ts
create(sine, 1).freq(440.0);
wait(2.0);  // spawn + wait 2 seconds
```

### flush()

Commit pending draft groups without advancing time.

```ts
create(sine, 1).freq(440.0);
flush();  // spawn immediately at current cursor
```

### seed(value)

Set the random seed for reproducible scenarios.

```ts
seed(42);
```

### scene(name, callback)

Named scope with automatic group release when the callback returns.

```ts
scene("Intro", || {
    create(sine, 2).freq(220.0);
    flush();
    wait(4.0);
    // all groups auto-released here
});
```

### play(callback [, args...])

Scoped callback execution with automatic cleanup.
Accepts 0--3 positional arguments, or an array.

```ts
let chord = |root| {
    create(sine, 1).freq(root);
    create(sine, 1).freq(root * 1.25);
    create(sine, 1).freq(root * 1.5);
    flush();
};

play(chord, 220.0);
wait(2.0);
// groups auto-released when play scope ends
```

### parallel(callbacks)

Execute closures on parallel timelines. Each branch starts at the current cursor.
The cursor advances to the latest branch end.

```ts
parallel([
    || {
        create(sine, 1).freq(220.0);
        wait(1.0);
    },
    || {
        create(sine, 1).freq(330.0);
        wait(3.0);
    },
]);
// cursor is now at +3.0 seconds
```

---

## Global Parameters

| Function | Description |
|----------|-------------|
| `set_harmonicity_mirror_weight(value)` | Overtone/undertone balance 0.0--1.0 |
| `set_roughness_k(value)` | Roughness tolerance |
| `set_global_coupling(value)` | Agent interaction strength |
| `set_pitch_objective(name)` | `"consonance"`/`"positive"` or `"dissonance"`/`"negative"` |

```ts
set_harmonicity_mirror_weight(0.0);  // overtone-dominant (major)
wait(4.0);
set_harmonicity_mirror_weight(1.0);  // undertone-dominant (minor)
```

---

## Complete Example

```ts
seed(1);

// Define species
let anchor = derive(sine).amp(0.6).sustain();
let voice = derive(harmonic)
    .amp(0.3)
    .brightness(0.5)
    .pitch_mode("free")
    .pitch_core("hill_climb")
    .pitch_apply_mode("glide").pitch_glide(0.08)
    .crowding(0.5)
    .metabolism(0.4)
    .adsr(0.05, 0.1, 0.7, 0.3)
    .repeat().pulse(2.25).sync(0.5);

scene("Opening", || {
    // Anchor drone
    let root = create(anchor, 1).freq(220.0);
    flush();
    wait(1.0);

    // Consonant placement
    let strat = consonance(220.0).range(1.0, 3.0).min_dist(0.9);
    create(voice, 6).place(strat);
    wait(4.0);

    // Shift to undertone mode
    set_harmonicity_mirror_weight(1.0);
    wait(4.0);
});

scene("Development", || {
    let dense = derive(voice).exploration(0.8).anneal_temp(0.5);
    let strat = consonance_density_pmf(100.0, 800.0).min_dist(0.8);
    create(dense, 12).place(strat);
    wait(8.0);
});
```
