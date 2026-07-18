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
