# Tension as a Composable Degree

Status: Design (v0.5 candidate). Supersedes the habituation-only sketch.
Scope: make the *degree* of harmonic tension a composer parameter — **one
operated axis (`τ`), with resolvability left emergent** — without warping the
terrain.

## Problem

We want a grounded composer dial for *how much tension*. Prior attempts failed:

- **`harmonic_tension`** (removed): warped the harmonicity terrain from outside.
  No perceptual mechanism, did not close the production loop, mathematically
  redundant (Section 3.3.3, technote).
- **register lift**: pushed the terrain via `ground.freq`; the ecology's own
  audio dominates the normalized field, so a single director voice cannot move
  it. External pushing is fragile by design.

## Principle: tension is the distance to a predicted resolution

Tension is not "dissonance". It is the gap to a predicted resolution:
`err = pred(resolution) − perc(now)`. The tension note (add9, 7th, sus) is a
*metastable rung below* the chord tone, not a clash.

Conceptually this has **two components** — the listener-twin's
`(1 − stability) · resolvability`, kept *un-collapsed*:

- **depth** — how far below the resolution the voice rests (instability).
- **resolvability** — how reachable a higher consonance is (expectation).

They are genuinely two-dimensional, and collapsing them to one scalar loses
information: *deep + resolvable* (dominant → tonic — a strong tension that
clearly wants to resolve) is musically different from *deep + unresolvable* (an
atonal cluster — unstable with no resolution in sight). We keep them separate.

## One operated axis, one emergent — and why

The two components are **not symmetric to implement**:

- **depth is the operated axis (`τ`).** It changes only the *target* of the
  voice's hill-climb — local and cheap.
- **resolvability is emergent, not operated.** A hill-climb *already enacts*
  resolvability: climbing the local gradient toward a higher consonance **is**
  resolvability in action. Computing it explicitly — scanning each bin's
  neighbourhood for a higher peak — duplicates what the hill-climb does every
  tick, and nesting that scan inside candidate evaluation makes movement a step
  heavier for no new behaviour. **The weight is the symptom: it is the same work
  twice.** So resolvability stays implicit in the hill-climb and is only
  *observed* (listener-twin, reports, UI) — never operated or pre-computed.

  (Putting resolvability into a `placement` target would also break
  placement ⊥ movement — it predicts how a voice will *move* — another sign the
  axis does not belong on the operation surface.)

So the design rule is: **operate `τ`; let resolvability emerge.**

## Design: `τ` — target a rung below the peak

The consonance field has local maxima of varying height. The strongest reachable
one is the resolution `L_max` (= `pred`). `τ ∈ [0,1]` sets a target consonance
level:

```
L_target = L_max − τ · (L_max − L_floor)
```

`L_floor` = the weakest metastable rung in range (or 0). The voice's hill-climb
aims for the metastable maximum nearest `L_target` instead of always climbing to
`L_max`:

- `τ = 0` → `L_max`: full resolution (today's `seek_consonance`).
- `τ` larger → a lower metastable consonance: a *held* tension.

Because `L_target < L_max`, a higher peak always remains — `resolvability > 0` by
construction (the emergent axis is satisfied for free). The voice rests (a
maximum, so stable) below its resolution (so tense). Live tension is
`err = L_max − L_current`.

**The terrain is never touched** — `τ` only chooses which height of the real,
ecology-built field the voice targets. Both technote tests pass: the perceptual
mechanism is prediction error toward a resolution; the production loop stays
closed (the voice reads the real terrain and only re-aims on it).

This is a *local* change to the hill-climb's target — the same computational
order as today's seek. **That it stays light is the sign it fits conchordal:**
the natural unit is "a voice hill-climbs its local terrain", and `τ` only re-aims
that climb. The heavyweight alternative (explicit resolvability) was the sign of
a conceptual mismatch, not just a performance cost.

## Resolution

Release = lowering `τ` (raising the target): the voice climbs from its rung to
the resolution. Compose the arc by moving `τ` over time (tension → release), or
hold it for standing tension. Étude 4 becomes a `τ` arc that touches no
frequency.

## API (the true successor to `harmonic_tension`)

`tension(τ)` on a voice / group / director, `τ ∈ [0,1]`. Same one-dial ergonomics
as the removed dial, but it **selects a target on the real terrain instead of
warping the terrain**.

## Habituation — autonomous complement (secondary, optional)

`τ` is the *static degree*. Habituation supplies *when* a held tension releases
on its own: sustained occupancy erodes the held peak's fusion
(`H_eff = H·(1 − depth·h)`, `h` a leaky per-bin integral of
`subjective_intensity`), so the rung stops rewarding and the voice must climb to
a fresher one — an autonomous release the composer did not script; vacated bins
recover. A complement, not the tension mechanism, and it keeps the closed loop
(occupancy → erosion → movement → new occupancy).

## Placement

No `resolvability()` placement target (heavy, and breaks placement ⊥ movement).
To seed tension at spawn, the existing `dissonance()` / `edge()` targets suffice
— static, local, cheap. Whether a seeded voice can resolve emerges afterward from
its hill-climb, not from a placement-time prediction.

## Implementation

- Rank the consonance-field local maxima in the voice's reachable range (extend
  the existing `global_peaks(n, min_sep)` machinery, which already finds top-n
  peaks).
- `pred = L_max`; `perc = L_current`; `err = L_max − perc` is the observed
  tension (feeds listener-twin / reports). Computed once per proposal for the
  voice's own position — **not** a per-bin resolvability scan.
- Seek targets the maximum nearest `L_target`; move-cost / crowding break ties as
  today.
- `tension(τ)` → a pitch-control parameter (reuse the `landscape_weight` /
  `temperature` knob plumbing).
- Determinism: terrain unchanged (deterministic); target selection deterministic.

## Open questions

- **`τ` meaning — discrete rungs vs continuous slope.** Ranking maxima by height
  is intuitive for *selection* (`tension(0.5)` = "the middle metastable
  consonance"), but the *strength* is the level gap `err`, and rungs are uneven —
  so a fixed `τ` gives uneven strength across scenes. Accept it as a *relative*
  degree, or normalize by the rung heights.
- **Resolution scope** — `L_max` local (the voice's reachable window) vs global
  (the scene's best consonance).
- **Scope of `τ`** — per-voice vs group/director. Chord-level tension
  (distributing rungs across voices so the *aggregate* sits below resolution) is
  the richer, harder case, and the one that matches "tension chord".
- **`τ` vs habituation when both on** — `τ` sets the floor a voice relaxes
  toward; habituation perturbs around it and forces eventual release.
- **No higher peak** (voice already at the global max): tension is
  undefined/zero — "no resolution to point at, no tension".
