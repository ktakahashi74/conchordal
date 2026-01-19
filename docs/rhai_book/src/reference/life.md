# Conchordal Scripting API Reference

This document describes the Rhai scripting API for Conchordal's Life Engine.

## Overview

The scripting API allows you to compose scenarios by:
- Defining **species** (agent templates with body method, timbre, brain type)
- Creating **groups** of agents from species
- Controlling **timeline** (waiting, flushing, parallel execution)
- Managing **placement** strategies (consonance-based, random, linear)
- Adjusting **global parameters** (harmonicity mirror weight, coupling, roughness)

## Types

### SpeciesHandle

Represents a species template for creating agents. Created by deriving from presets or other species.

### GroupHandle

A handle to a group of agents. Returned by `create()`. Used to control live agents or configure draft groups before spawning.

### SpawnStrategy

Defines how frequencies are assigned when creating multiple agents. Created by strategy constructors like `consonance()`, `consonance_density()`, etc.

## Preset Species

Global variables for common body methods:

| Preset | Body Method | Description |
|--------|-------------|-------------|
| `sine` | Sine | Pure sine wave |
| `harmonic` | Harmonic | Harmonic series synthesis |
| `saw` | Harmonic | Sawtooth-like (brightness=0.85, width=0.2) |
| `square` | Harmonic | Square-like (brightness=0.65, width=0.1) |
| `noise` | Harmonic | Noise-like (brightness=1.0, motion=1.0, width=0.35) |

## Species Definition

### derive(parent: SpeciesHandle) → SpeciesHandle

Clone a species to create a new template.

```ts
let my_species = derive(sine).amp(0.5).phonation("hold");
```

### Species Methods (Chainable)

These methods modify a `SpeciesHandle` and return it for chaining.

#### amp(value: float) → SpeciesHandle
#### amp(value: int) → SpeciesHandle

Set amplitude (0.0–1.0).

```ts
let loud = derive(sine).amp(0.8);
```

#### freq(value: float) → SpeciesHandle
#### freq(value: int) → SpeciesHandle

Set default frequency in Hz.

```ts
let a440 = derive(sine).freq(440.0);
```

#### brain(name: string) → SpeciesHandle

Set brain type. Valid values:
- `"entrain"` - Entrainment brain (default): responds to rhythm field
- `"seq"` - Sequencer brain: fixed duration notes
- `"drone"` - Drone brain: sustained with sway modulation

```ts
let drone = derive(sine).brain("drone");
```

#### phonation(name: string) → SpeciesHandle

Set phonation type. Valid values:
- `"hold"` - One-shot sustain tied to lifecycle
- `"decay"` - Interval-based retriggering
- `"grain"` - Field-based retriggering

```ts
let voice = derive(harmonic).phonation("hold");
```

#### timbre(brightness: float, width: float) → SpeciesHandle

Set timbre parameters (0.0–1.0 each).
- `brightness`: Spectral brightness (higher = brighter)
- `width`: Spectral width (higher = wider)

```ts
let bright = derive(harmonic).timbre(0.9, 0.3);
```

#### metabolism(rate: float) → SpeciesHandle

Set metabolism rate (energy consumption/recovery rate).

```ts
let slow = derive(sine).metabolism(0.3);
```

#### adsr(attack: float, decay: float, sustain: float, release: float) → SpeciesHandle

Set ADSR envelope parameters (time in seconds for A/D/R, level 0.0–1.0 for S).

```ts
let pluck = derive(harmonic).adsr(0.01, 0.1, 0.3, 0.2);
```

## Group Creation

### create(species: SpeciesHandle, count: int) → GroupHandle

Create a draft group of `count` agents from the given species. Returns a handle for further configuration. The group is spawned when `flush()` or `wait()` is called.

```ts
let g = create(sine, 4).freq(220.0).amp(0.5);
```

## Timeline Control

### wait(seconds: float)
### wait(seconds: int)

Flush pending groups, advance the timeline cursor by the given duration, and commit all scheduled events.

```ts
wait(2.0);  // Wait 2 seconds
```

### flush()

Flush pending draft groups immediately without advancing time. Commits all scheduled events at the current cursor position.

```ts
create(sine, 1).freq(440.0);
flush();  // Spawn immediately
```

### seed(value: int)

Set the random seed for reproducible placement strategies.

```ts
seed(12345);
```

## Group Control

### GroupHandle Methods (Chainable)

These methods can be used in two contexts:
1. **Draft mode** (before `flush()`/`wait()`): Configure the group before spawning
2. **Live mode** (after spawning): Update parameters of live agents

#### amp(value: float) → GroupHandle
#### amp(value: int) → GroupHandle

Set or update amplitude.

```ts
let g = create(sine, 1).amp(0.5);  // Draft
flush();
g.amp(0.8);  // Live update
```

#### freq(value: float) → GroupHandle
#### freq(value: int) → GroupHandle

Set or update frequency. In draft mode, overrides any placement strategy.

```ts
let g = create(sine, 1).freq(220.0);  // Draft
flush();
g.freq(440.0);  // Live update (pitch slides)
```

#### brain(name: string) → GroupHandle

Set brain type (draft only). See species `brain()` for valid values.

```ts
create(sine, 1).brain("drone");
```

#### phonation(name: string) → GroupHandle

Set phonation type (draft only). See species `phonation()` for valid values.

```ts
create(harmonic, 1).phonation("decay");
```

#### timbre(brightness: float, width: float) → GroupHandle

Set or update timbre.

```ts
let g = create(harmonic, 1).timbre(0.8, 0.2);
flush();
g.timbre(0.5, 0.3);  // Live update
```

#### metabolism(rate: float) → GroupHandle

Set metabolism rate (draft only).

```ts
create(sine, 1).metabolism(0.5);
```

#### adsr(attack: float, decay: float, sustain: float, release: float) → GroupHandle

Set ADSR envelope (draft only).

```ts
create(harmonic, 1).adsr(0.05, 0.1, 0.7, 0.2);
```

#### place(strategy: SpawnStrategy) → GroupHandle

Set placement strategy (draft only). Assigns frequencies to agents in the group based on the strategy.

```ts
let strat = consonance(220.0).range(1.0, 4.0);
create(sine, 4).place(strat);
```

### release(handle: GroupHandle)

Release a group of agents. Agents enter their release phase and fade out.

```ts
let g = create(sine, 1).freq(440.0);
flush();
wait(2.0);
release(g);  // Begin release
```

## Spawn Strategies

Strategies define how frequencies are assigned when spawning multiple agents.

### consonance(root_freq: float) → SpawnStrategy

Pick highest consonance positions within the multiplier range of the root frequency.

**Default range:** `[1.0, 4.0]` (root to 2 octaves above)
**Default min_dist:** `1.0` ERB

```ts
let strat = consonance(220.0);
create(sine, 4).place(strat);
```

### consonance_density(min_freq: float, max_freq: float) → SpawnStrategy

Weighted-random placement based on consonance density in the frequency range.

**Default min_dist:** `1.0` ERB

```ts
let strat = consonance_density(100.0, 800.0);
create(sine, 8).place(strat);
```

### random_log(min_freq: float, max_freq: float) → SpawnStrategy

Uniform distribution in logarithmic frequency space (equal probability per octave).

```ts
let strat = random_log(100.0, 1000.0);
create(sine, 6).place(strat);
```

### linear(start_freq: float, end_freq: float) → SpawnStrategy

Linear interpolation between start and end frequencies.

```ts
let strat = linear(200.0, 800.0);
create(sine, 5).place(strat);  // Evenly spaced
```

## Strategy Modifiers

### range(strategy: SpawnStrategy, min_mul: float, max_mul: float) → SpawnStrategy

Set multiplier range for `consonance()` strategy. Ignored for other strategies.

```ts
let strat = consonance(220.0).range(1.0, 3.0);  // 220 to 660 Hz
```

### min_dist(strategy: SpawnStrategy, distance: float) → SpawnStrategy

Set minimum distance between agents in ERB units (critical band spacing). Applies to `consonance()` and `consonance_density()`.

```ts
let strat = consonance(220.0).min_dist(1.5);  // Wider spacing
```

## Scope Management

### scene(name: string, callback: closure)

Create a named scene marker. All groups created within the callback are automatically released when the scene ends (scoped cleanup).

```ts
scene("Intro", || {
    create(sine, 2).freq(220.0);
    wait(2.0);
    // Groups auto-released here
});
```

### play(callback: closure)
### play(callback: closure, arg1)
### play(callback: closure, arg1, arg2)
### play(callback: closure, arg1, arg2, arg3)
### play(callback: closure, args: array)

Execute a closure with optional arguments. Creates a new scope for groups (scoped cleanup).

```ts
let make_chord = |root| {
    create(sine, 1).freq(root);
    create(sine, 1).freq(root * 1.25);
    create(sine, 1).freq(root * 1.5);
    flush();
};

play(make_chord, 220.0);
wait(2.0);
// Groups auto-released
```

### parallel(callbacks: array)

Execute multiple closures in parallel timelines. Each closure starts at the current cursor position. The cursor advances to the maximum end time of all branches.

```ts
parallel([
    || {
        create(sine, 1).freq(220.0);
        wait(1.0);
    },
    || {
        create(sine, 1).freq(330.0);
        wait(2.0);  // This branch takes longer
    }
]);
// Cursor now at 2.0 seconds
```

## Global Parameters

### set_harmonicity_mirror_weight(value: float)

Set the overtone/undertone balance (0.0–1.0).
- `0.0` = Pure overtone series (major/bright)
- `1.0` = Pure undertone series (minor/dark)
- `0.5` = Balanced

```ts
set_harmonicity_mirror_weight(0.0);  // Major mode
wait(2.0);
set_harmonicity_mirror_weight(1.0);  // Minor mode
```

### set_global_coupling(value: float)

Set global agent-agent interaction strength (0.0–1.0). Higher values increase mutual influence between agents.

```ts
set_global_coupling(0.5);
```

### set_roughness_k(value: float)

Set roughness tolerance parameter. Controls how strongly agents avoid dissonant (rough) regions.

```ts
set_roughness_k(0.8);
```

## Example: Complete Scenario

```ts
// Define species
let drone = derive(sine).amp(0.6).phonation("hold");
let voice = derive(harmonic).amp(0.3).phonation("hold");

scene("Opening", || {
    // Create anchor drone
    let anchor = create(drone, 1).freq(220.0);
    flush();
    wait(1.0);

    // Spawn consonant voices
    let strat = consonance(220.0).range(1.0, 3.0).min_dist(0.9);
    create(voice, 4).place(strat);
    wait(2.0);

    // Modulate harmonicity
    set_harmonicity_mirror_weight(1.0);
    wait(2.0);
});
```

## Brain Types Reference

| Brain | Behavior |
|-------|----------|
| `entrain` | Synchronizes with detected rhythms in the field (default) |
| `seq` | Fixed-duration note sequencing |
| `drone` | Sustained tone with slow frequency sway |

## Phonation Types Reference

| Phonation | Behavior |
|-----------|----------|
| `hold` | One-shot sustain tied to agent lifecycle |
| `decay` | Interval-based retriggering |
| `grain` | Field-based granular retriggering |
