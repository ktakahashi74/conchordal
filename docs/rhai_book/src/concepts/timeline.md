# Timeline and Structure

A scenario does not schedule individual audio samples. It advances a script
cursor, places Populations at that cursor, and describes when they are patched
or released.

## `place`, `wait`, and `flush`

`place()` schedules founder Voices immediately at the current cursor and
returns their stable `Population` handle. `wait(seconds)` emits pending live
patches and moves the cursor forward. `flush()` emits pending live patches
without moving time.

```rhai
let spec = harmonic()
    .amp(0.04)
    .brightness(0.35)
    .sustain();

let population = place(spec, consonance(220.0).count(3));
population.amp(0.025); // Live patch at the same script time.
flush();               // Emit it; the cursor does not move.
wait(2.0);             // Advance two seconds.
release(population);
```

The semantic boundary is `place()`, not `wait()` or `flush()`. Initial body,
behavior, lifecycle, and respawn settings belong to `PopulationSpec`; only
live patches and release belong to `Population`.

## `release`, `section`, and `play`

`release(population)` is explicit. `section(name, callback)` and
`play(callback, ...)` create scopes and automatically release Populations
created inside them when the callback returns. Scope exit emits any pending
live patches before the automatic release.

Use `section` for named form. Its name becomes a `scene_marker` in reports.
Use `play` for a reusable gesture that takes arguments.

```rhai
let gesture = |root_hz, duration_sec| {
    place(
        harmonic().amp(0.035).sustain(),
        consonance(root_hz).peak().count(3)
    );
    wait(duration_sec);
};

section("two gestures", || {
    play(gesture, 110.0, 2.0);
    play(gesture, 165.0, 2.0);
});
```

The Populations created by each `play` are released at the end of that call.
The outer `section` provides the report marker and owns anything created
directly inside it.

## Parallel timelines

`parallel([callbacks])` forks several cursors from the current time. Every
branch starts together. After all branches are described, the main cursor
continues at the end of the longest branch. Each branch is also a scope: its
Populations are released when that branch returns.

```rhai
section("overlap", || {
    parallel([
        || {
            place(sine().amp(0.05).sustain(), at(220.0));
            wait(3.0);
        },
        || {
            wait(1.0);
            place(harmonic().amp(0.03).sustain(), at(330.0));
            wait(1.0);
        }
    ]);
});
```

The first branch ends three seconds after the fork. The second ends two
seconds after it, so the main cursor advances by three seconds. Their Voices
overlap only for the interval in which both branch scopes are active.

## Script time and runtime behavior

The script cursor decides when control events occur. It does not predict the
exact state of the terrain at those times. Movement, coupled rhythm, death,
and respawn continue inside the runtime while `wait()` advances the scenario.

This distinction is central:

- The script authors macro-structure.
- The Community produces moment-to-moment behavior.
- Reports show what actually happened during the authored interval.

Use the workflow in [Performance](../tutorial/observing.md) to inspect the
result and revise the scenario. Use `seed(...)` or `--seed` only when a
particular realization must be replayed.

## Scene settings and timed changes

Global parameter names describe when they apply. Bare names set scene-wide initial
conditions: `meter_stability`, `temporal_basin`, `control_update_mode`, `scaffold_off`,
`scaffold_shared`, `scaffold_scrambled`, `seed`, and `output_gain`. Their last declaration wins,
even after `wait()`. `temporal_mode` is also a scene setting, but must be declared
once at top level before events.

The `set_` verbs schedule parameter changes at the current cursor:
`set_roughness_aversion`, `set_global_coupling`, and `set_pitch_objective`.
Use `wait()` to choose when those changes take effect. Placement, release and
scope verbs continue to describe actions rather than parameter settings.

## Scene playback level

`output_gain(db)` sets one playback gain for the whole scene. It accepts integer
and floating-point dB values; omission means `0` dB. As with `seed`, the last
declaration wins, including declarations after `wait()`. It does not schedule a fade.

```rhai
output_gain(3.0);
play(|| {
    place(sine().amp(0.04).sustain(), at(220.0));
    wait(2.0);
});
```

The scene value adds to `[audio].output_gain_db` (default `+4.5` dB). The output
amplitude multiplier is `10^((config dB + scene dB)/20)`. Both the instrument's
speakers and the offline renderer's WAV apply it immediately before their output
limiter, including when the limiter is disabled. A scene value of `0` preserves
the configured playback level. The limiter can reduce peaks at higher gains.

This controls playback only: analysis, ListenerTwin, ecology, reports and UI bus
meters receive the original audio levels. To change a Population's emitted sound
and its effect on the ecology, use its `amp()` instead.

The value must be finite and convert to a finite, positive `f32` amplitude
multiplier. The sum with the config gain must meet the same condition. Invalid
values report the declaration's source position. These limits follow `f32`
representation; there is no additional artistic dB range or loudness target.

## Expression toward a known end

`expressive_section(name, arch_weight, ritard_weight, callback)` gives the final
explicit interval a loudness arch and a common slowing clock. It has the same
ownership and automatic releases as `section`. Both weights are finite,
nonnegative and initial-only. Zero is neutral; values at or above one saturate
at full application. Omitting the term changes nothing, and two zero weights
use the existing `section` path.

```rhai
temporal_basin(1.6, 2.4);
meter_stability(0.55);
let pluck = modal().metric()
    .modes(stiff_string_modes(0.004).count(12))
    .brightness(0.42).adsr(0.004, 0.18, 0.0, 0.35).amp(0.04);
let pulse = place(pluck, line(196.0, 392.0).count(3));
section("opening", || { wait(4.0); });
expressive_section("ending", 1.0, 1.0, || {
    wait(8.0);
    release(pulse);
});
```

This uses candidate curves from Todd (1992, single-layer intensity) and
Friberg–Sundberg (1999, q=3 timing). Assigning them to an authored interval,
adding the amplitude floor and freezing the generation clock are author rules.
It does not infer a phrase or assert a perceived ending. No listener preset
activates it automatically.

The initial supported scene contains Once Voices without free-pitch
rearticulation, or non-sustained synchronized `metric()` Voices. Flora,
participating clocks, sustained metric Voices, `follow_own_sequence`, and `rest`
are outside this scope. Unsupported scenes warn and ignore the entire expression
effect, preserving their other controls. One nonzero interval is allowed, and
all populations must be released by its end. Outer populations need explicit
release, as in the example; scope exit releases only its own populations.
There can be no later spawn or patch. Invalid weights or intervals also warn
and ignore the effect. At device setup, colliding Once onset/release hops and
transformed durations above 600 seconds are diagnosed before applying the map.

The arch changes emitted source PCM in both buses, including already sounding
tails; it does not multiply articulation energy directly. At full application
its midpoint amplitude is multiplied by square root of two. Ritard stretches
nominal time by about 1.403 at full application and reaches 0.3 of its entry
rate. It freezes each synchronized clock's entry phase velocity and renewal,
while sensory observation, ecology and body envelopes retain physical time.
Gate durations follow the same time map; releases start later but their fades
retain physical seconds. Pending Off commands issued before entry stay at their
original physical ticks. A positive ritard, however small, freezes the clock;
zero ritard leaves the original clock active. Section names and the known end
are never evidence for ListenerTwin's perception of closure.

Once voices must start immediately; `phonate_when_viable()` on Once is outside this initial support.
