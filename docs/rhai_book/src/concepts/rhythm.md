# Rhythm: One Coupling Continuum

Rhythm in conchordal is **one coupling continuum on a shared emergent meter**,
not a set of independent clocks. The population drives a single production
meter (a coupled-oscillator beat); each voice is a phase oscillator that
entrains its onset phase to that emergent beat with a coupling strength. There
is no externally imposed grid — coherence (or its absence) emerges from how
tightly each voice locks to the meter the population itself drives.

The three rhythm "families" are just three regions of the continuum:

- **metric**: high coupling — a deep attractor, reads as a stable pulse.
- **entrained**: medium coupling — synchronization emerges over time, still
  drifts.
- **flow**: near-zero coupling — a free renewal process, non-metric texture.

Rhythm is part of the same ecology as consonance, viability, movement, and
respawn: timing can affect survival, and survival reorganizes timing.

## Director-level terrain

The director shapes the rhythmic terrain, symmetric to the consonance-field
operations. These are soft priors, never a schedule — emergence still does the
work:

- `meter_stability(value)` — attractor depth in `[0,1]`: how readily a pulse
  forms. It only deepens the basin for a *real* periodicity; it never
  fabricates a beat from non-metric input.
- `temporal_basin(min_hz, max_hz)` — the tempo region the emergent beat
  gravitates toward (the time-axis analogue of `density(min, max)`). It shapes
  the terrain; it does not place a beat, and it never forces a measure.

## Per-voice presets and modifiers

The Tier-1 presets take **no rate argument**: `metric()`, `entrained()`, and
`flow()`. The tempo region is a property of the terrain, set once at the
director level, not per voice. They work on both a `Material` and a draft
`Participant`.

Per-voice modifiers refine where on the continuum a voice sits:

- `entrainment(strength)` — coupling in `[0,1]`, free (`0`) .. locked (`1`).
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — the voice's
  metrical job. `accent` emits a stronger onset that drives the shared meter
  harder, so a recurring downbeat can seed an emergent measure.
- `microtiming(amount)` — a signed beat-phase offset in `[-0.5, 0.5]`. `0.5`
  places a voice a half-beat off, reading as syncopation.

```rhai
meter_stability(0.85);     // attractor depth: how readily a pulse forms
temporal_basin(1.8, 2.2);   // tempo region the emergent beat gravitates toward

let beat = harmonic()
    .metric()
    .rhythm_role("accent")  // a strong onset that drives the shared beat
    .cycles(2);

let entrained = harmonic()
    .entrained()
    .cycles(2);

let drift = harmonic()
    .flow()
    .cycles(1);

let offbeat = harmonic()
    .metric()
    .microtiming(0.5)       // a half-beat offset reads as syncopation
    .cycles(2);

place(beat, at(110.0));
place(entrained, peaks(110.0).count(3));
place(drift, density(300.0, 1200.0).count(4));
place(offbeat, at(220.0));
wait(12.0);
```

Calls on the same axis are last-write-wins: the last timing mode and the last
duration mode determine the final behavior. Modifiers are remembered and
applied when their matching preset is selected, so `entrainment(0.8).metric()`
and `metric().entrainment(0.8)` are equivalent. The same applies to
`duration_range(...).adaptive_duration()` and the reverse order.

## Explicit when/duration (Tier 2)

Below the presets sit explicit controls: `once()`, `pulse(rate_hz)`,
`while_alive()`, `cycles(n)`, and `adaptive_duration()` (with
`duration_range`, `duration_curve`, and `shorten_on_drop` for tuning).

## Timing as survival (Tier 3)

Use `rhythm_coupling_vitality(lambda_v, v_floor)` and
`rhythm_reward(rho_t, "attack_phase_match")` when timing should affect
survival and reorganization, and `rhythm_freq(freq_hz)` to set the internal
oscillator directly:

```rhai
let pulse_voice = harmonic()
    .repeat()
    .pulse(2.0)
    .cycles(2)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");

place(pulse_voice, at(165.0));
wait(8.0);
```

## Scaffolds (research controls)

The scaffold functions impose an *external* pulse for comparison assays:

```rhai
set_scaffold_off();
set_scaffold_shared(2.0);
set_scaffold_scrambled(2.0, 17);
```

They are useful for demos and assays. They are not the rhythm-composition
abstraction: composed rhythm should come from the continuum and the terrain.
