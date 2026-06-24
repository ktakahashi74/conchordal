# Tension: Temperature in Movement, Relative Level in Placement

Status: Design (v0.5 candidate). Final form; supersedes the rung / level /
temperature-only sketches in this file's history.
Scope: make the *degree* of harmonic tension a composer parameter across both
phases — where a voice is placed, and how it moves — without warping the terrain
or adding mechanisms conchordal does not already have.

## Problem

`harmonic_tension` was removed (it warped the terrain from outside: no perceptual
mechanism, no closed production loop, mathematically redundant). We still want a
grounded handle for *how much tension*. A long exploration — target consonance
level, a rung below `L_max`, percentile, resolvability — kept hitting walls:
sigmoid distortion of `field_level`, "is the target a peak or a slope?", absolute
vs relative to a shifting `L_max`, heavy per-bin scans. The recurring
weight/difficulty was itself the signal that tension was being framed as an
*external target constraint* on the voice — the same shape as `harmonic_tension`.

Two heuristics cut through (both the user's): (1) computational weight is a sign
of conceptual mismatch; (2) *persistent* difficulty in finding a natural form
means the premise is wrong.

## Principle

Tension is the distance to a predicted resolution (a strong consonance) — a
metastable state below it. It shows up in two phases, and conchordal already has
the right primitive for each:

- **Movement** (how a placed voice moves): tension = **temperature**.
- **Placement** (where a voice is spawned): tension = **relative level (τ) ×
  sharpness (temperature)**.

The shared substrate is the continuous consonance field plus a single notion of
temperature.

## Movement: tension = temperature

Treat the consonance field as a potential `U = −score`; a voice's pitch
distribution is Boltzmann:

```
P(f) ∝ exp( score(f) / T )
```

- `T → 0`: settles on the most consonant point (resolution).
- `T` large: spreads into lower-score regions (fluctuation = tension).

This **is** the pitch core's existing search temperature (HillClimb Metropolis /
PeakSampler softmax); DCC already routes listener tension into a
`temperature_bonus`. **No new movement API — `temperature(v)` is it.**
`seek_consonance` is the *pull* (wins at low `T`); temperature is the
*fluctuation* (wins at high `T`), so resolution is built in and the voice's
autonomy / the closed loop are preserved.

Scope: temperature is set per species and copied per voice; per-voice differences
come only from the DCC bonus today (per-voice operation is an open question).

Why temperature dissolves the movement difficulties: it acts on the energy
`score`, not the sigmoid `field_level`; it names no point (a distribution spread,
so the peak/slope question vanishes); `exp(Δscore/T)` makes a deep well hold and
a shallow one fluctuate (relativity built in, no `L_max` detection); one
continuous scalar.

## Placement: relative level (τ) × sharpness (temperature)

Placement is a *one-shot selection* among the field's peaks at spawn, so the
difficulties that killed a movement "rung" do not arise (no continuous climb, no
slope to slip down). Two orthogonal axes:

### 1. Relative level τ — the tension degree

Rank the consonance-field peaks in the voice's range by height. Let `L_max` /
`L_min` be the highest / lowest, in **`field_score`** (the linear scale — NOT the
sigmoid `field_level`). The target consonance level is:

```
target = L_max − τ · (L_max − L_min)      τ ∈ [0,1]
```

- `τ = 0` → `L_max`: the strongest consonance (resolved placement).
- `τ` large → a weaker, metastable consonance (tense placement).

Place at the peak nearest `target`. `τ` is "how far below the strongest reachable
consonance", normalized in `field_score` so the sigmoid never distorts it and
`0..1` reads intuitively.

**Why relative level, not rank.** rank (top-XX%) depends on the *number* of
peaks, which is scene-dependent (the same rank points at different steps as the
sounding changes); relative level is by *height*, peak-count-independent. rank is
a meta-structure over the field; relative level rides the field's own continuous
scale, and it generalizes the existing density machinery (peak = the `τ=0`
special case). A continuous `τ` maps onto the discrete peaks, so the musical
"step" feel comes for free.

### 2. Sharpness — temperature

How tightly the spawn concentrates around `target`: `peak()` (sharp,
deterministic ≈ `T→0`) ↔ `density` (broad, stochastic ≈ `T>0`). This is the
**same temperature** as movement, in its placement guise — `peak`/`density` are
its two extremes, and any sharpness between them is a temperature.

## Unification

| phase | tension degree | temperature |
|---|---|---|
| **placement** | relative level `τ` (which consonant step) | sharpness (`peak`↔`density`: how concentrated) |
| **movement** | (emerges from the climb) | fluctuation (how restless once placed) |

**Temperature runs through both phases** (sharpness at placement, fluctuation in
movement). Placement adds the explicit degree `τ` (which step to aim at); in
movement the degree emerges from the climb under its temperature. Neither warps
the terrain — both read the real, ecology-built field — so both pass the
technote's two tests and avoid `harmonic_tension`'s trap.

## API

Movement:
```
temperature(v)        # existing; species / group / DCC. Low = settle, high = restless.
```

Placement:
```
consonance(110.0).peak()               # τ=0, sharp: strongest consonance (resolved)
consonance(110.0).peak().tension(0.4)  # τ=0.4 step, sharply (tense placement)
consonance(110.0).tension(0.4)         # same step, density (broader)
```

`tension(τ)` here is the *placement degree*; movement uses `temperature`. If a
single verb is preferred, `tension` could also set a voice's temperature — but
that is naming, not a new mechanism.

## Implementation

- Placement degree: reuse `global_peaks(n, min_sep)` to get range peaks;
  `L_max`/`L_min` = max/min of their `field_score`; `target = L_max −
  τ·(L_max−L_min)`; select the peak nearest `target`.
- Placement sharpness: the existing `peak`/`density` sampling (a continuous
  temperature form is future work).
- Movement: the existing `temperature` / `set_temperature` path (the one DCC's
  bonus already feeds).
- Always normalize in `field_score` (linear), never the sigmoid `field_level`.
  No `L_max` detection in movement, no per-bin resolvability scan, no rung
  ranking as a movement axis.
- Determinism: terrain unchanged; peak selection and Boltzmann draws are
  deterministic in render.

## What we are deliberately NOT building

- No `harmonic_tension`-style terrain warp (removed; the reason this note exists).
- No rung as an *operated movement* axis (movement uses temperature).
- No `resolvability()` placement target or per-bin resolvability scan (heavy,
  breaks placement ⊥ movement; resolvability stays emergent, observed by
  listener-twin).
- Habituation (occupancy erodes a held peak) is a *possible later* autonomous-
  release complement, not part of this.

## Open questions

- `τ → step` when peaks are sparse: nearest peak — how far off is acceptable.
- Placement sharpness as a continuous temperature vs the current `peak`/`density`
  pair (unify now, or keep the pair).
- Per-voice movement temperature (today: species + DCC bonus).
- Combining composer `temperature` with DCC's listener-driven `temperature_bonus`.
- **Chord-level tension**: distributing `τ` across voices (some at `τ=0` resolved,
  some tense) — the tension-chord case, realized through placement.
