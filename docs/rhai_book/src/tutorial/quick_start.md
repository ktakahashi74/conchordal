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

`place(material, placement)` stages a `Participant` at the current script time.
It is committed by `wait(seconds)` or `flush()`, then can be patched while it
is alive.

## Basic Objects

- **Materials** are voice templates made with `sine()`, `harmonic()`,
  `modal()`, `saw()`, `square()`, and `noise()`.
- **Variants** clone a material with `variant(material)`.
- **Placements** decide where participants enter: `at()`, `peaks()`,
  `density()`, `random()`, and `line()`.
- **Participants** are the handles returned by `place()`. Before the next
  `wait()` or `flush()`, participant builder methods still shape the initial
  spawn; after that, patchable methods update running voices.
- **Sections** scope participants and release them automatically.

```rhai
let voice = harmonic()
    .amp(0.08)
    .sustain()
    .brightness(0.35);

section("plain entry", || {
    place(voice, line(220.0, 440.0).count(3));
    wait(4.0);
});
```

## Placing Into the Field

`peaks(root_hz)` places voices at high Consonance Field positions around a
root. The field is shaped by what the system perceives — an anchor changes
where the peaks are.

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

    place(voice, peaks(110.0).range(1.0, 4.0).count(6).spacing(0.9));
    wait(6.0);
});
```

How the field works — and how voices move, survive, and respawn inside it — is
the subject of [The Consonance Field](../concepts/consonance.md).

## Live Patching

Some participant methods patch running voices; others only shape the draft and
must be set before the group is committed. The
[API Reference](../reference/api.md) tags every method as live-patchable or
draft-only.

```rhai
let g = place(
    harmonic().amp(0.04).sustain(),
    peaks(220.0).count(3)
);
wait(2.0);     // commit: the group is now live

g.amp(0.02);   // live patch on running voices
g.glide(0.8);
wait(3.0);
release(g);
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

    place(colony, density(90.0, 900.0).count(8).spacing(0.8));
    wait(8.0);
});
```

## Where to go next

- [Editor Setup](editor_setup.md) — completion, hover docs, and diagnostics
  for the whole scripting surface.
- [The Consonance Field](../concepts/consonance.md) — field, density,
  movement, viability, respawn.
- [Rhythm](../concepts/rhythm.md) — the coupling continuum and the director's
  rhythmic terrain.
- [Curated Samples](../reference/samples.md) — the guided listening path.
