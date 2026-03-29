# Quick Start Guide

Conchordal v0.4.0-dev scripting API overview. For full details, see the [API Reference](reference/life.md).

## Basic Concepts

- **Species**: Voice templates created with `derive()` from presets (`sine`, `harmonic`, `saw`, `square`, `noise`, `modal`).
- **Groups**: Collections of voices spawned from a species via `create(species, count)`. Returns a `GroupHandle`.
- **Timeline**: `wait(seconds)` advances the clock. `flush()` forces immediate dispatch. `scene()` scopes groups with auto-release. `parallel()` runs branches concurrently.
- **Strategies**: Placement methods for multi-voice frequency assignment: `consonance()`, `consonance_density_pmf()`, `random_log()`, `linear()`.

## Minimal Example

```ts
// Single sine voice at 440 Hz for 2 seconds
create(sine, 1).freq(440.0);
wait(2.0);
```

## Common Patterns

### Custom Species Definition

```ts
let voice = derive(harmonic)
    .amp(0.5)
    .sustain()
    .brightness(0.7)
    .spread(0.2)
    .unison(4);
```

Body parameters are individual methods: `brightness(v)`, `spread(v)`, `unison(n)`.

### Multiple Voices with Consonance Strategy

```ts
let base = derive(harmonic).amp(0.1).sustain();

// Place 4 voices using harmonic consonance relative to 220 Hz
create(base, 4)
    .place(consonance(220.0).range(1.0, 4.0).min_dist(1.0));
wait(2.0);

// Or use landscape-derived density placement
create(base, 4).place(consonance_density_pmf(110.0, 880.0));
wait(2.0);
```

### Scene with Auto Cleanup

```ts
let drone = derive(sine).amp(0.3).sustain();
let seeker = derive(sine).amp(0.2).sustain();

scene("Introduction", || {
    let a = create(drone, 1).freq(110.0);
    let s = create(seeker, 1).freq(220.0);
    wait(3.0);
    // All groups auto-released when scene ends
});
```

### Parallel Timelines

```ts
parallel([
    || {
        // Bass layer
        create(sine, 1).freq(110.0).sustain();
        wait(2.0);
    },
    || {
        // Melody layer
        create(harmonic, 1).freq(440.0).sustain();
        wait(1.0);
        create(harmonic, 1).freq(550.0).sustain();
        wait(1.0);
    }
]);
```

### Modal Body with Custom Modes

The `modal` preset uses inharmonic mode patterns for bell/bar/glass timbres.
Configure modes with `modes(pattern)` and control brightness.

```ts
let bell = derive(modal)
    .sustain()
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.7)
    .modes(modal_table("vibraphone_1").count(5));

create(bell, 1);
wait(2.0);
```

Mode pattern constructors: `harmonic_modes()`, `odd_modes()`, `power_modes(beta)`,
`stiff_string_modes(stiffness)`, `custom_modes([ratios])`, `modal_table(name)`.
Chain `.count(n)`, `.jitter(cents)`, `.seed(n)` to refine patterns.

Landscape-aware modes sample from the live perceptual field:

```ts
let adaptive = derive(modal)
    .sustain()
    .pitch_mode("free")
    .amp(0.05)
    .brightness(0.7)
    .modes(landscape_density_modes().count(16).range(1.0, 3.5).min_dist(0.9));
```

### Phonation Control

Phonation follows a tiered system.

**Tier 1 -- Presets** set sensible defaults for common use:

```ts
derive(sine).sustain();   // Sustained tone, tied to lifecycle
derive(sine).repeat();    // Interval-based retriggering
```

**Tier 2 -- Explicit when/duration** for finer control:

```ts
derive(sine).once();              // Single trigger
derive(sine).pulse(2.0);         // Retrigger at 2 Hz
derive(sine).while_alive();      // Duration spans full lifecycle
derive(sine).gates(4);           // Hold for 4 gate cycles
derive(sine).field();            // Field-driven duration
```

These compose: `derive(sine).pulse(3.0).gates(2)` pulses at 3 Hz, each lasting 2 gates.

### Pitch Control

```ts
// "lock" keeps initial frequency; "free" allows hill-climbing
derive(sine).pitch_mode("lock");
derive(sine).pitch_mode("free");

// "gate_snap" applies pitch on gate boundaries; "glide" interpolates
derive(sine).pitch_apply_mode("gate_snap");
derive(sine).pitch_apply_mode("glide").pitch_glide(0.05);
```

### Live Parameter Patching

Groups support live updates on spawned voices:

```ts
let g = create(sine, 1).freq(220.0).sustain();
flush();
wait(1.0);

g.freq(440.0);   // Slide to new frequency
g.amp(0.3);      // Adjust amplitude
flush();
wait(1.0);
```

### Global Parameter Modulation

```ts
// Overtone series emphasis (major quality)
set_harmonicity_mirror_weight(0.0);
create(sine, 4).place(consonance(261.63).range(1.0, 3.0));
wait(2.0);

// Shift toward undertone series (minor quality)
set_harmonicity_mirror_weight(1.0);
wait(2.0);
```

## Preset Species

| Preset | Body | Description |
|--------|------|-------------|
| `sine` | Sine | Pure sine wave |
| `harmonic` | Harmonic | Harmonic series synthesis |
| `saw` | Harmonic | Sawtooth-like (bright) |
| `square` | Harmonic | Square-like (odd harmonics) |
| `noise` | Harmonic | Noise-like (full spectrum) |
| `modal` | Modal | Inharmonic mode synthesis |

## Next Steps

See the [API Reference](reference/life.md) for the complete function list, type details, and advanced parameters.
