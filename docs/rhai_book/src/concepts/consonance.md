# The Consonance Field

Conchordal's perception core (the **Landscape**) listens to the habitat bus,
transforms it into log-frequency space, and computes two potentials:
**roughness** (sensory dissonance from amplitude fluctuations within critical
bands) and **harmonicity** (periodicity and template matching). Their
combination is the **Consonance Field**: an evaluation terrain over frequency
that placement, movement, prediction, and survival all read from.

Because the field is computed from what the system actually hears, every voice
deforms the terrain for every other voice. That feedback loop — not a chord
chart — is where harmony comes from.

## Field placement: `peaks`

`peaks(root_hz)` places voices at high Consonance Field positions around a
root. Use it when the musical thought is "start from this harmonic center".

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

## Consonance Density: `density`

`density(min_hz, max_hz)` samples from the density view of the field. Use it
when the musical thought is "let the field seed a population inside this
range".

```rhai
let cloud = harmonic().amp(0.035).sustain();

place(cloud, density(90.0, 1200.0).count(10).spacing(0.8));
wait(8.0);
```

Density is not just "random but harmonic". It is a normalized distribution
derived from the consonance model, and it remains well-defined inside the
requested range.

## Consonance Movement: `seek_consonance`

Use `seek_consonance()` when voices should actively seek better field
positions. It sets free hill-climb movement with glide defaults. Use
`glide(tau_sec)` when the musical thought is "same movement idea, slower or
faster pitch motion".

```rhai
let mover = harmonic()
    .amp(0.045)
    .sustain()
    .seek_consonance()
    .glide(0.35)
    .avoid_neighbors(0.6)
    .global_peaks(8, 70.0)
    .ratio_candidates(5);

place(mover, density(80.0, 900.0).count(8));
wait(12.0);
```

`avoid_neighbors(strength)` adds crowding repulsion so movers spread out
instead of collapsing onto the same peak.

The opposite of movement is `anchor()`: an anchored voice holds its pitch and
only deforms the terrain for others. Voices placed with `at()` or given
`freq()` are anchored implicitly; use `anchor()` to freeze strategy-placed
voices at their settled position.

How movement lands is resolved from phonation: sustained movers glide,
re-attacking movers (`pulse()`, `metric()`, `entrained()`, `flow()`) snap to
their new pitch at each onset. Override with `pitch_apply_mode()` when a
script needs the other behavior. Mechanism-level controls (`pitch_core()` and
the hill-climb / peak-sampler tuning in the
[API Reference](../reference/api.md)) remain available for research scripts;
prefer `seek_consonance()` and `glide()` in curated work.

## Consonance Viability and Respawn

Viability makes field fit matter over time. `consonance_viability(low, high)`
defines the consonance window, and `viability_rate(rate)` controls continuous
recharge: a voice in a well-fitting place is sustained, a voice in a poor
place starves.

By default viability uses **environment-relative** scoring: a voice is
evaluated against the field with its own footprint approximately removed. Use
`viability_scope("total")` only when the compositional question is explicitly
total-field viability.

Respawn closes the loop into an ecology: when voices die, replacements appear
according to a respawn policy. `respawn_consonance()` draws them from
consonance-biased parental peaks; `respawn_capacity(count)` keeps the
population bounded; `respawn_settle(placement)` decides where replacements
settle.

```rhai
let settle = density(70.0, 1100.0).spacing(0.8);

let ecology = harmonic()
    .amp(0.04)
    .repeat()
    .pulse(1.5)
    .cycles(3)
    .seek_consonance()
    .glide(0.45)
    .initial_energy(0.7)
    .energy_cap(1.0)
    .metabolism(0.09)
    .action_cost(0.012)
    .viability_rate(0.18)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle);

place(ecology, density(70.0, 1100.0).count(14));
wait(30.0);
```

The full lifecycle surface (energy, metabolism, costs) and the respawn
policies are documented in the [API Reference](../reference/api.md).

## Landscape-aware timbre

The field can shape timbre as well as pitch. The `modal()` body takes a mode
pattern, and landscape-aware patterns sample the live field, so a bell's
partials can sit where the terrain already supports them.

```rhai
let shimmer_modes = landscape_density_modes()
    .count(10)
    .range(1.0, 5.5)
    .gamma(1.6)
    .spacing(0.7);

let shimmer = modal()
    .amp(0.025)
    .sustain()
    .seek_consonance()
    .modes(shimmer_modes)
    .brightness(0.7);

place(shimmer, density(200.0, 1600.0).count(4));
wait(8.0);
```

Mode constructors include `harmonic_modes()`, `odd_modes()`,
`power_modes(beta)`, `stiff_string_modes(stiffness)`,
`custom_modes([ratios])`, `modal_table(name)`, `landscape_density_modes()`,
and `landscape_peaks_modes()` — see the
[API Reference](../reference/api.md#mode-patterns).

## A director's tension dial: `harmonic_tension`

`harmonic_tension(value)` is a director-level tension–release dial, not a
foundational field operation. Raising it degrades the harmonicity terrain's
coherence for overtone-radiating bodies, heard as harmonic tension; lowering it
restores consonant gravity. Reach for it to shape drama across a scene.

A caveat on what it is: mechanically it blends the terrain from overtone (`0.0`)
toward an *undertone* projection (`1.0`) — a leftover of an abandoned
harmonic-dualism hypothesis. There is no undertone series in nature and no
perceptual mechanism behind it, so this is a useful artifact (a tension knob that
happens to work), not a tonality (major/minor) switch and not a grounded terrain
operation. Do not read musical meaning into the projection itself.

```rhai
let voice = harmonic().amp(0.04).sustain();
place(voice, peaks(110.0).count(5));

harmonic_tension(0.1);   // consonant gravity
wait(4.0);
harmonic_tension(0.8);   // tension: the terrain loses coherence
wait(4.0);
harmonic_tension(0.1);   // release
wait(4.0);
```
