# Scenario Scripting API (v2)

## Core
- `spawn(tag, count[, opts])`
- `wait(dt)`
- `scene(name)`
- `set(target, patch)`
- `release(target, duration)`
- `remove(target)`
- `end()`: end at the current cursor time. Terminal: further scheduling calls (spawn/set/wait/scene/after/at/parallel) are errors.
- `end_at(t_abs)`: end at an absolute time (seconds). Terminal: further scheduling calls are errors.
  - To change the end time, move the end declaration; keep a single end declaration at the end.
- Prelude helpers: `after`, `at`, `parallel`, `repeat`, `every`, `spawn_every`

## Advanced (Global Controls)
- `set_global_coupling(x)`
- `set_roughness_tolerance(x)`
- `set_rhythm_vitality(x)`
- `set_harmonicity(map)`

## Strict Keys
- `spawn` opts: `amp`, `method`, `life`
- `set` patch: `amp`, `freq`, `drift`, `commitment`

## Phonation types
Use in patches via `phonation: #{ type: "hold", ... }`.
- `none`: disable phonation events
- `interval`: retriggering (interval/accumulator)
- `clock`: retriggering (clock-gated)
- `field`: retriggering (timing field)
- `hold`: one-shot sustain tied to lifecycle; ignores density/sync/legato

## Minimal examples
- `spawn("drones", 5); end_at(20);`
- `wait(2); end();`
