# Rhythm: Evidence and Participation

A beat is a recurrent temporal reference inferred from sound. Sharing that
reference does not require every Voice to attack at the same phase.

The three presets express different timing intentions:

- **metric**: explicit attraction to the shared production beat. Strong coupling
  favors synchronized onsets.
- **entrained**: a bodily pace adjusted by remembered acoustic context and a
  prediction of energetic overlap with surrounding sound.
- **flow**: weak acoustic participation with clustered renewal intervals around
  an intrinsic scale that does not follow recurrence.

For entrained and flow timing, a causal observer reads past habitat audio. Each
Voice also retains the sound it rendered, so the energy attributed to itself can
be excluded from a predicted competitor. The Voice compares sounding near its due
time, waiting up to a participation cycle, and omitting the current event. Overlap is
integrated across an estimate of the planned hold and ADSR release. Successive
omissions become more costly, so overlap alone cannot prescribe permanent silence.
Temporal memory changes only after an onset is emitted. For entrained timing, the cadence
can follow a supported recurrence near the Voice's intrinsic scale, while preserving
its own cycle position. The observer does not prescribe a common onset phase.
Hold lengths remain tied to the intrinsic period, so cadence adaptation does not
stretch each sound. The overlap preview uses the same hold duration.
Flow keeps its intrinsic renewal scale while still waiting or omitting events in
response to surrounding sound; its executed intervals therefore remain variable.
This three-band energy proxy is an engineering approximation, with musical
acceptance still under evaluation.

## Director-level terrain

The director shapes the rhythmic terrain, symmetric to the consonance-field
operations. These operations shape the production meter used by metric timing. They do not
set the bodily rate or recurrence candidates of entrained/flow participants:

- `meter_stability(value)` — attractor depth in `[0,1]`: how readily a pulse
  forms. It only deepens the basin for a *real* periodicity; it never
  fabricates a beat from non-metric input.
- `temporal_basin(min_hz, max_hz)` — the tempo region the emergent beat
  gravitates toward (the time-axis analogue of `consonance(min, max)`). It shapes
  the terrain; it does not place a beat, and it never forces a measure.

## Per-voice presets and modifiers

The Tier-1 presets take **no rate argument**: `metric()`, `entrained()`, and
`flow()`. Configure them on a `PopulationSpec` before calling `place()`. The temporal
basin shapes metric timing. Entrained/flow participants begin with their own
2 Hz intrinsic prior. Entrained cadence can follow acoustic recurrence without
changing that prior or assigning a shared onset phase. Flow keeps that intrinsic
scale for its irregular renewal intervals.

Do not confuse `.entrained()` with `brain("entrain")`. The preset controls
onset timing; the brain selects the voice's articulation life and metabolism.
They are independent and may be used together. See
[Population — A Persistent Unit of Voices](voice_life.md).

Per-voice modifiers refine these intentions:

- `entrainment(strength)` — influence in `[0,1]`: acoustic context for
  entrained/flow, beat-phase attraction for metric. Zero keeps intrinsic pacing.
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — the voice's
  metrical job. `accent` emits a stronger onset that drives the shared meter
  harder, so a recurring downbeat can seed an emergent measure.
- `microtiming(amount)` — a signed metric beat-phase offset in `[-0.5, 0.5]`.
  A value of `0.5` targets the half-beat. It does not assign fixed positions to
  entrained/flow participants.
- `measure_accent(amount)` — weak onset emphasis following the detected measure,
  in `[0,1]` (default `0`). The onset strength multiplier is
  `1 + 0.35 * amount * confidence * cos(phase)`. The phase origin follows observed
  accents, and no detected measure means no emphasis. It changes strength, not
  onset scheduling, and works independently of `rhythm_role`.

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
place(entrained, consonance(110.0).peak().count(3));
place(drift, consonance(300.0, 1200.0).count(4));
place(offbeat, at(220.0));
wait(12.0);
```

Calls on the same axis are last-write-wins: the last timing mode and the last
duration mode determine the final behavior. `cycles(n)` counts the intrinsic
bodily period for entrained/flow and the effective beat period for metric timing;
flow retains its existing threefold hold multiplier. Modifiers are remembered and
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
articulation oscillator directly. These are legacy articulation controls:
Gated phonation disables autonomous attacks and applies ordinary consonance
recharge to emitted onsets, without the phase-match reward:

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
abstraction: use them only when explicit synchronization is essential to the demo or assay.
A dedicated beat carrier is not required for entrained/flow participation.
