# Quick Start Guide

This page provides a quick introduction to Conchordal's scripting API. For the complete API reference, see [Reference](reference/life.md).

## Basic Concepts

- **Species**: Templates for creating agents (defined with `derive()`)
- **Groups**: Collections of agents created from a species (created with `create()`)
- **Timeline**: Sequential and parallel execution controlled by `wait()`, `flush()`, and `parallel()`
- **Strategies**: Methods for placing multiple agents in frequency space

## Minimal Example

```ts
// Create a single sine wave at 440 Hz
create(sine, 1).freq(440.0);
wait(2.0);  // Play for 2 seconds
```

## Common Patterns

### Define a Custom Species

```ts
let voice = derive(harmonic)
    .amp(0.5)
    .phonation("hold")
    .timbre(0.7, 0.2);
```

### Create Multiple Agents with Consonance Strategy

```ts
let strat = consonance(220.0).range(1.0, 4.0);
create(sine, 4).place(strat);
flush();  // Spawn immediately
```

### Scene with Automatic Cleanup

```ts
scene("Introduction", || {
    let drone = derive(sine).amp(0.6).phonation("hold");
    create(drone, 1).freq(110.0);
    wait(3.0);
    // All groups auto-released when scene ends
});
```

### Parallel Timelines

```ts
parallel([
    || {
        // Bass line
        create(sine, 1).freq(110.0);
        wait(2.0);
    },
    || {
        // Melody
        create(harmonic, 1).freq(440.0);
        wait(1.0);
        create(harmonic, 1).freq(550.0);
        wait(1.0);
    }
]);
```

### Live Parameter Updates

```ts
let g = create(sine, 1).freq(220.0);
flush();
wait(1.0);

g.freq(440.0);  // Slide to new frequency
flush();
wait(1.0);
```

### Modulate Global Parameters

```ts
// Start with overtone series (major)
set_harmonicity_mirror_weight(0.0);
let strat = consonance(261.63).range(1.0, 3.0);
create(sine, 4).place(strat);
wait(2.0);

// Switch to undertone series (minor)
set_harmonicity_mirror_weight(1.0);
wait(2.0);
```

## Preset Species

| Preset | Description |
|--------|-------------|
| `sine` | Pure sine wave |
| `harmonic` | Harmonic series |
| `saw` | Sawtooth-like timbre |
| `square` | Square-like timbre |
| `noise` | Noise-like timbre |

## Brain Types

| Type | Behavior |
|------|----------|
| `"entrain"` | Responds to rhythm field (default) |
| `"seq"` | Fixed-duration notes |
| `"drone"` | Sustained with sway |

## Phonation Types

| Type | Behavior |
|------|----------|
| `"hold"` | Sustain tied to lifecycle |
| `"decay"` | Interval retriggering |
| `"grain"` | Field-based granular |

## Next Steps

See the [complete API reference](reference/life.md) for detailed documentation of all functions, types, and parameters.
