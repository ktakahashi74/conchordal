# Quick Start

Run a scenario script with the real-time instrument (release mode is
recommended for real-time DSP):

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

## Minimal Sound

```rhai
place(sine().amp(0.08).sustain(), at(440.0));
wait(2.0);
```

`place(population_spec, placement)` creates a `Population` immediately at the
current script time. The returned handle can patch or release that living
population.

## Basic Objects

- **PopulationSpecs** describe founder voices and population policy. Start one
  with `sine()`, `harmonic()`,
  `modal()`, `saw()`, `square()`, and `noise()`.
- **Variants** clone a specification with `variant(population_spec)`.
- **Placements** decide where populations enter: field targets
  `consonance()`, `dissonance()`, `edge()`, `gap()` (cloud by default,
  `.peak()` for the extremum), plus `random()`, `at()`, and `line()`.
- **Populations** are the stable handles returned by `place()`. Their founder
  voices exist from that boundary; live methods update the current voices.
- **Sections** scope populations and release them automatically.

A Population is one stable handle for a placed population. A placement with `.count(6)`
creates six founder voices under that handle. If voices later die and respawn,
the Population remains the same while its members and generations change. See
[Population — A Persistent Unit of Voices](../concepts/voice_life.md) for the complete object and
lifecycle model.

```rhai
let population_spec = harmonic()
    .amp(0.08)
    .sustain()
    .brightness(0.35);

section("plain entry", || {
    place(population_spec, line(220.0, 440.0).count(3));
    wait(4.0);
});
```

## Placing Into the Field

`consonance(root_hz).peak()` places voices at high Consonance Field positions
around a root. The field is shaped by what the system perceives — an anchor
changes where the peaks are.

```rhai
let anchor = harmonic()
    .brain("drone")
    .amp(0.06)
    .sustain()
    .anchor();

let voice = harmonic()
    .amp(0.04)
    .sustain();

section("field placement", || {
    place(anchor, at(110.0));
    wait(1.0);

    place(voice, consonance(110.0).peak().range(1.0, 4.0).count(6).spacing(0.9));
    wait(6.0);
});
```

How the field works — and how voices move, survive, and respawn inside it — is
the subject of [Consonance Field — A Terrain for Evaluating Pitch](../concepts/consonance.md).

## Live Patching

Configure body, behavior, lifecycle, and respawn on a `PopulationSpec` before
calling `place()`. The returned `Population` exposes only operations that make
sense after placement: live patches and release. The
[API Reference](../reference/api.md) marks the two method sets.

```rhai
let spec = harmonic().amp(0.04).sustain();
let population = place(
    spec,
    consonance(220.0).peak().count(3)
);
population.amp(0.02); // live patch on running voices
population.glide(0.8);
wait(3.0);
release(population);
```

## A Complete Miniature

```rhai
seed(7);

let anchor = harmonic()
    .brain("drone")
    .amp(0.05)
    .sustain();

let colony = harmonic()
    .amp(0.035)
    .sustain()
    .seek_consonance()
    .glide(0.4)
    .avoid_neighbors(0.6);

section("emergence", || {
    place(anchor, at(110.0));
    wait(2.0);

    place(colony, consonance(90.0, 900.0).count(8).spacing(0.8));
    wait(8.0);
});
```

## Where to go next

- [Editor Setup](editor_setup.md) — completion, hover docs, and diagnostics
  for the whole scripting surface.
- [Performance](observing.md) — reports, filtering, and
  replaying a run with `--seed`.
- [Population — A Persistent Unit of Voices](../concepts/voice_life.md) — population specs, voices, brains,
  phonation, survival, and release.
- [Voice and Landscape — Sound–Environment Feedback](../concepts/ecological_loop.md) — how sound changes
  the terrain that changes the voices.
- [Consonance Field — A Terrain for Evaluating Pitch](../concepts/consonance.md) — field, density,
  movement, viability, respawn.
- [Rhythm](../concepts/rhythm.md) — the coupling continuum and the director's
  rhythmic terrain.
- [Routing and the Listener Twin](../concepts/routing.md) — what the ecology
  senses, what the audience hears, and what is observed.
- [Timeline and Structure](../concepts/timeline.md) — placement boundaries,
  scopes, reusable gestures, and parallel branches.
- [Curated Samples](../reference/samples.md) — the guided listening path.
