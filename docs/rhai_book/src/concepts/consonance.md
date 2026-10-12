# Consonance Field — A Terrain for Evaluating Pitch

Conchordal's perception core (the **Landscape**) listens to the habitat bus,
transforms it into log-frequency space, and computes two potentials:
**roughness** (sensory dissonance from amplitude fluctuations within critical
bands) and **harmonicity** (periodicity and template matching). Their
combination is the **Consonance Field**: an evaluation terrain over frequency
that placement, movement, prediction, and survival all read from.

Because the field is computed from the habitat bus, every voice routed to that
bus deforms the terrain for every other voice. A presentation-only voice does
not. That feedback loop — not a chord chart — is where harmony comes from. See
[Voice and Landscape — Sound–Environment Feedback](ecological_loop.md) for the relationship between
potential, score, level, mass, and density.

## Placing into the field: `consonance`, `dissonance`, `edge`, `gap`

A field-relative placement names a **target** — a region of the field — and is
realized either as a cloud (the default) or as a deterministic extremum with
`.peak()`. The targets are:

- `consonance` — high-consonance positions (harmonic centers, fusion).
- `dissonance` — low-consonance positions (tension, clusters, color).
- `edge` — the consonance/dissonance boundary (the metastable middle).
- `gap` — empty registers (fill the room, avoid masking).

`consonance(root)` takes a harmonic window around a root (multiples via
`range()`); every target also takes an absolute `(min_hz, max_hz)` range.

Range bounds must be finite and within the synthesizer's 1–20000 Hz range;
reversed bounds are accepted. Initial placement stays inside those bounds,
including jitter and all-zero or fully occupied fallbacks. Field targets use
the intersection with the analysis band (currently 55–8000 Hz). If there is
no intersection, placement falls back to log-uniform sampling in the requested
range because no field evidence is available there. `random()` samples the
requested range directly. Placement does not constrain subsequent
`seek_consonance()` movement to that range.

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

A target with no modifier is a **density cloud**: not "random but harmonic" but
a normalized distribution derived from the consonance model, well-defined inside
the range. Placement is independent of behavior — enter at `dissonance` and
`anchor()` for a held cluster, or enter at `dissonance` and `seek_consonance()`
for a resolution gesture.

```rhai
let cloud = harmonic().amp(0.035).sustain();

place(cloud, consonance(90.0, 1200.0).count(10).spacing(0.8));
wait(8.0);
```

## Placement tension: `tension(τ)`

Without `tension`, Consonance Placement uses its ordinary realization:
`.peak()` chooses the strongest peak, while the default density produces the
ordinary Consonance cloud. `tension(τ)` biases that Placement toward a
field-score step **below** the maximum. `τ ∈ [0, 1]` is the tension degree;
`0` leaves the ordinary realization unchanged, while larger values target
progressively weaker, metastable steps
(`target = L_max − τ·(L_max − L_min)` over the range, in field score). It is the
placement twin of movement's search `temperature` — the dial for *how resolved* a
spawn should sit — and it reads the field's score directly, so the degree rides
the terrain's own scale rather than the count of peaks that happen to be
sounding. With `.peak()` it snaps to the nearest step; as a cloud it concentrates
the distribution around the target.

```rhai
let tense = harmonic().amp(0.035).sustain();

// A metastable step below the strongest peak — placed, not resolved.
place(tense, consonance(110.0, 1200.0).peak().tension(0.4).count(6).spacing(0.8));
wait(6.0);
```

The field-agnostic placements are `random(min_hz, max_hz)` (log-uniform) and the
geometric `at(hz)` and `line(start_hz, end_hz)`.

## Naming frequencies: ratios of a root

`at(hz)` and `line(start_hz, end_hz)` take absolute frequencies, and the
canonical idiom for filling them in is to name one sounding root in Hz and
derive every other pitch as a ratio of it: `root_hz * 1.5` for a fifth,
`root_hz * 4.0/3.0` for a fourth, `root_hz * 2.0` for a register lift. A ratio
is the physical quantity the field itself reads, so writing intervals this way
keeps the thinking in frequency relationships.

Writing an equal-tempered decimal instead — `146.83` for a D3 — still runs,
but it re-imports the twelve-tone symbol grid through the back door: the
number stands in for a note name rather than describing a relationship to a
sounding partial. Prefer ratios of a root that is actually sounding in the
scene.

```rhai
let root_hz = 110.0;
let voice = harmonic().amp(0.04).sustain();

place(voice, at(root_hz * 1.5));
wait(2.0);
```

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

place(mover, consonance(80.0, 900.0).count(8));
wait(12.0);
```

`avoid_neighbors(strength)` adds crowding repulsion so movers spread out
instead of collapsing onto the same peak.

The opposite of movement is `anchor()`: an anchored voice holds its pitch and
only deforms the terrain for others. `freq()` sets an anchored pitch; use
`anchor()` to freeze strategy-placed voices at their settled position.

`at(hz)` holds a fixed pitch unless the final PopulationSpec explicitly requests
`seek_consonance()` and remains free. For that moving Voice, it sets only the
initial position, just as `line(hz, hz)` does, and keeps the placement count.
For example, `place(mover, at(220.0))` starts the mover defined above at 220 Hz
without cancelling its movement or glide settings. It specifies initial body
pitch, not a guaranteed first audible pitch or a promise of movement.

Builder order matters when changing the pitch policy:

| Final builder order | Meaning of `at(hz)` |
| --- | --- |
| `seek_consonance()` then `freq()` | Fixed pitch |
| `freq()` then `seek_consonance()` | Initial position |
| `seek_consonance()` then `anchor()` | Fixed pitch |
| `anchor()` then `seek_consonance()` | Initial position |

Without explicit movement, `at()` still fixes pitch. A live Population's
`freq()` still locks pitch during performance.

How movement lands is resolved from phonation: sustained movers glide,
re-attacking movers (`pulse()`, `metric()`, `entrained()`, `flow()`) snap to
their new pitch at each onset. `pitch_apply_mode("gate_snap")` and
`pitch_apply_mode("glide")` override that choice.

For sustained movers without a participation clock, `pitch_apply_mode("rearticulate")` gives each newly
adopted pitch target a fresh attack. The old Tone releases while the new Tone
starts directly at the adopted pitch, at the sample when movement begins.
Rejected proposals, unchanged targets and intermediate glide updates do not
create notes. The proposal interval controls opportunities to choose a target,
not a fixed note clock; rearticulation does not wait for a beat.

```rhai
let moving = harmonic().sustain().seek_consonance()
    .pitch_apply_mode("rearticulate");
let voices = place(moving, line(220.0, 440.0).count(3));
wait(12.0);
voices.pitch_apply_mode("glide");
wait(12.0);
```

A live switch between glide and rearticulate preserves the sounding Tone and
does not trigger a note by itself. Unrendered rearticulation requests are
cancelled by that switch; the new mode handles subsequent adoptions. The mode
also belongs to the Population template used for replacements. `glide(tau)`
sets the sliding time constant in glide mode and below a rearticulation boundary. Anchored Voices are
unaffected by rearticulate.

Rearticulate requires free pitch and Hold phonation. Combining free-pitch
Gated phonation with it is a script error, checked on the final placement spec
regardless of builder order; an incompatible live call also fails before any
state is changed. Use `sustain()`, or choose gate_snap or glide with Gated
phonation. Hold without participation does not offer a separate same-pitch attack request.
Sustained participation can re-attack its current body pitch at a sounded opportunity;
autonomous per-note selection is not provided.

### Choosing how pitch changes become notes

These choices distinguish target adoption from a participation opportunity.
The boundary is a composer parameter of the founder Voice's articulation, not
a universal auditory threshold. Values such as 50 or 100 cents are examples;
the composer chooses them.

| Intention | Explicit form | When an attack occurs |
|---|---|---|
| Legato | `sustain().pitch_apply_mode("glide")` | Initial onset; subsequent pitch movement slides |
| Every adopted target | `sustain().pitch_apply_mode("rearticulate")` | Each adoption, with the boundary omitted or zero |
| Small moves slide | `sustain().pitch_apply_mode("rearticulate").rearticulate_above(50)` | Adoption strictly above 50 cents from the last sounded onset |
| Sustained participation with sliding | `sustain().entrained().pitch_apply_mode("glide")` | A sounded opportunity at its current body pitch; other movement slides |
| Sustained participation with jumps | `sustain().flow().pitch_apply_mode("gate_snap")` | A sounded opportunity consumes the latest pending target |
| Sustained participation with a movement boundary | `sustain().entrained().pitch_apply_mode("rearticulate").rearticulate_above(50)` | Small adoptions slide; large ones wait for a sounded opportunity |

Entrained and flow can each be used in any participation row. See
[Rhythm](rhythm.md#sustained-sound-with-participation-opportunities) for their clocks.

`intrinsic_rate(hz)` sets the body's initial scale in any participation row,
independently of the pitch-application choice. It does not command an onset interval
or place a shared beat. `proposal_interval(seconds)` still controls pitch proposals,
not that bodily scale. See [Setting the body's intrinsic rate](rhythm.md#setting-the-bodys-intrinsic-rate).

Without participation, a positive `rearticulate_above(cents)` measures net displacement from the last actual onset,
not accumulated path length. At or below the boundary, the existing Tone glides
with `glide(tau)`; above it, a new Tone attacks. That choice lasts until the next
adoption. Exact Log2Space bin centers use their bin-distance in cents; off-grid
endpoints use the continuous log2 distance, without rounding the target or adding
an epsilon. Earlier legal attacks in the same output hop provide the reference,
and cancelled requests never replace the last materialized onset.

Without participation, the boundary defaults to zero, retaining every-adoption rearticulation. Negative
and nonfinite values become zero; no upper limit is imposed. Changing the boundary
live causes no attack and applies at the next adoption. A call outside Rearticulate
warns and is ignored. Builder order is independent at placement. Replacement
Voices inherit the boundary, with no inherited onset-pitch memory.

```rhai
let moving = harmonic().sustain().seek_consonance()
    .pitch_apply_mode("rearticulate").rearticulate_above(50).glide(0.03);
let voices = place(moving, consonance(220.0, 660.0).count(2));
wait(3.0);
voices.rearticulate_above(100);
wait(3.0);
release(voices);
```

With a participation clock, Rearticulate applies the same distance measure to each
adoption but uses the clock for attacks. At or below a positive boundary, movement
starts gliding immediately without an attack and cancels any older pending target.
Above it, an active glide stops, body pitch is held, and the latest target waits
for a sounded opportunity. That opportunity consumes the target and attacks there;
with tie omitted or zero, it re-attacks the current body pitch without a target, which may have changed during
a glide. Waiting and skipping retain sound and pending state while existing glides
continue. Both kinds of attack update the last actual onset reference. Same-sample
adoption precedes the opportunity. No adoption-time candidate is added, and the
boundary does not filter clock attacks. Zero or omission uses the GateSnap behavior,
including zero-distance adoptions. See [Rhythm](rhythm.md#sustained-sound-with-participation-opportunities)
for mode switching.

The positive-boundary sustained-participation form also accepts initial-only
`articulation_persistence(p)` to retain Base/Connect across existing events.
This changes policy inertia, not the movement boundary or a prescribed legato
proportion. See [Retaining an articulation policy](rhythm.md#retaining-an-articulation-policy).

For gate_snap/rearticulate participation, `tie(p)` can extend a living Tone without
a pending target instead of re-attacking. It leaves pending-target attacks intact,
including small GateSnap moves. See [Extending opportunities](rhythm.md#extending-opportunities-without-a-new-pitch-destination).

### Cutting notes and leaving a rest

For sustained participation, `rest()` adds a finite end to every sounded note
and an intentional pause after its sound has finished. Supply the note length
with `cycles(n)` or `adaptive_duration()` and its modifiers. This length, explicit
`sustain()`, `entrained()` or `flow()`, and `rest()` are retained independently of
builder order. No finite length is supplied implicitly. Explicit `while_alive()`
clears the finite length; placement then fails unless another finite length follows.

The scheduled Off starts the normal release, leaving tails intact. The rest begins
only after the renderer confirms that every Tone of that Voice's source ID and
generation has finished. `rest()` then prohibits new notes for one intrinsic bodily
period; `rest(periods)` uses the composer's chosen multiple of the period at that
completion. A known period with `rest(0)` waits for the tails only. An unknown period
keeps waiting, as does unfinished sound, without a fallback or timeout. Negative
and nonfinite arguments become zero; finite nonnegative values have no upper limit.

`intrinsic_rate(hz)` changes this bodily period and the finite note-length grid
together. The same `cycles` or `duration_range` and `rest(periods)` therefore produce
different note lengths and rests in seconds at different bodily rates, independently
of entrained cadence adaptation.

At the end of the rest, the existing participation decision can sound, wait or skip;
the end itself does not force an attack. GateSnap retains the latest adopted target
through sounding, release and rest, and consumes it at the next sounded opportunity.
Rearticulate applies its small-adoption cancellation and large-target deferral
during release and rest too. Its last actual onset remains the reference until
the next sounded opportunity. Sliding changes body pitch after Off but never
retunes released tails. Glide leaves released tails at their released pitch. Release, death and removal
take precedence. Replacements inherit controls, with fresh performance state.

```rhai
let notes = harmonic().sustain().entrained()
    .adaptive_duration().duration_range(0.2, 0.2).rest(2);
let voices = place(notes, at(330.0));
wait(3.0);
release(voices);
```

Here 0.2 cycles and two periods are example inputs, not recommended values.
`rest` is initial-only, like the note-length controls. It is valid only with explicit
sustain plus entrained/flow and a finite note length; Glide, GateSnap and
Rearticulate are all available. Incompatible
placements fail before state changes. Without `rest`, the existing rules remain.
This form ends every note; choosing to sustain or release each individual note is
not implemented by this control.

Mechanism-level controls (`pitch_core()` and
the hill-climb / peak-sampler tuning in the
[API Reference](../reference/api.md)) remain available for research scripts;
prefer `seek_consonance()` and `glide()` in curated work.

## Consonance Viability and Respawn

Viability makes field fit matter over time. `consonance_viability(low, high)`
defines the consonance window, while `recovery(seconds)` states how long a
full-scale recharge takes: a voice in a well-fitting place is sustained, a
voice in a poor place approaches its nominal `endurance(seconds)`.

By default viability uses **environment-relative** scoring: a voice is
evaluated against the field with its own footprint approximately removed. Use
`viability_scope("total")` only when the compositional question is explicitly
total-field viability.

Respawn closes the loop into an ecology: when voices die, replacements appear
according to a respawn policy. `respawn_consonance()` selects high field-score
peaks with a bias around an energy-weighted living parent;
`respawn_capacity(count)` sets the maximum living membership (defaulting to,
and never lower than, the founder count); `respawn_settle(placement)` adds
that Placement to the replacement candidate pool rather than replacing the
policy's baseline candidate.

```rhai
let settle = consonance(70.0, 1100.0).spacing(0.8);

let ecology = harmonic()
    .amp(0.04)
    .repeat()
    .pulse(1.5)
    .cycles(3)
    .seek_consonance()
    .glide(0.45)
    .endurance(8.0)
    .recovery(4.0)
    .attack_cost_fraction(0.017)
    .attack_recharge_fraction(0.70)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle);

place(ecology, consonance(70.0, 1100.0).count(14));
wait(30.0);
```

The full lifecycle surface (time-domain endurance/recovery and normalized
per-attack fractions) and the respawn policies are documented in the
[API Reference](../reference/api.md). [Population — A Persistent Unit of Voices](voice_life.md)
separates articulation, phonation, pitch behavior, and survival; the
[Voice and Landscape — Sound–Environment Feedback](ecological_loop.md) follows the complete energy and respawn
cycle.

## Landscape-aware timbre

The field can shape timbre as well as pitch. The `modal()` body takes a mode
pattern. `landscape_density_modes()` deterministically chooses the strongest
separated density-mass positions; `landscape_peaks_modes()` chooses the
strongest separated local Field-level peaks. Thus a bell's partials can sit
where the terrain already supports them.

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

place(shimmer, consonance(200.0, 1600.0).count(4));
wait(8.0);
```

Mode constructors include `harmonic_modes()`, `odd_modes()`,
`power_modes(beta)`, `stiff_string_modes(stiffness)`,
`custom_modes([ratios])`, `modal_table(name)`, `landscape_density_modes()`,
and `landscape_peaks_modes()` — see the
[API Reference](../reference/api.md#mode-patterns).
