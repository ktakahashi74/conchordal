# Tension as Temperature

Status: Design (v0.5 candidate). Supersedes the rung/level sketches in this file's
history.
Scope: define the *degree* of harmonic tension as the voice's pitch-search
**temperature** — reusing existing machinery, and dissolving the
level/rung/sigmoid difficulties that earlier framings hit.

## Problem

`harmonic_tension` was removed (it warped the terrain from outside — no
perceptual mechanism, no closed production loop). We still want a composer dial
for *how much tension*. Successive attempts to define it as a target consonance
*level*, or a *rung* below the resolution, kept hitting walls:

- the `sigmoid` in `field_level` distorts any level-based degree;
- "is the target a peak or a slope?" — a level names a point that may be
  unstable;
- absolute level vs relative to a shifting `L_max` (the peak depends on what is
  sounding);
- discrete rungs vs continuous.

The *persistent* difficulty was itself the sign of a mismatch: all those framings
treat tension as an **external target constraint** on the voice's climb — the
same shape as the failed `harmonic_tension`.

## Principle: tension = temperature

Treat the consonance field as a potential, `U(f) = −score(f)` (the existing
`consonance_field_energy`; lower = more consonant). A voice's stationary pitch
distribution is Boltzmann:

```
P(f)  ∝  exp( score(f) / T )
```

- `T → 0`: the voice concentrates on the most consonant point (U minimum) —
  full resolution (greedy settling).
- `T` large: the distribution spreads into lower-score (less consonant) regions
  — fluctuation and excursion, i.e. tension.

**Tension is the temperature `T`.** Nothing else.

## Why temperature dissolves every earlier difficulty

| earlier difficulty | under temperature |
|---|---|
| sigmoid distortion of `field_level` | `T` acts on the energy `score`, never on the sigmoid `field_level` |
| target a peak or a slope? | `T` is the *spread* of a distribution; it names no point, so the question never arises |
| absolute vs relative to a shifting `L_max` | `exp(Δscore/T)` is a ratio of energy-gap to temperature — a deep consonant well holds even at high `T`, a shallow one fluctuates even at low `T`. **Relativity is built in; no `L_max` detection.** |
| discrete rungs vs continuous | `T` is one continuous scalar — no quantization, no ranking |

All of it collapses into a single scalar.

## Existing machinery — conchordal already does this

- `temperature` is already the pitch core's search temperature (HillClimb
  Metropolis, PeakSampler softmax; defaults 0.0 / 0.08). `exp(score/T)` is
  already implemented in the cores.
- **DCC already routes `listener tension → temperature_bonus`**
  (`src/dcc_coupler.rs`): higher listener tension raises the search temperature.

So this is not a new mechanism. "Define tension as temperature" *promotes the
existing temperature axis to be the meaning of tension*. The implementation is
minimal.

## Resolution is built in (no external target)

- `seek_consonance` is the **pull** toward consonance — dominant at low `T`.
- temperature is the **fluctuation** — dominant at high `T`.

Low `T` resolves (the pull wins); high `T` holds tension (fluctuation wins). The
voice is never told "stop at this rung" — its autonomy is intact and temperature
only adds jitter. Both technote tests pass (perceptual mechanism = stochastic
exploration under a real potential; production loop stays closed), and
`harmonic_tension`'s external-warp trap is avoided.

## API

`tension(τ)` on a voice / group / director, mapping to the search temperature.
`τ = 0` → cool: the voice settles into consonance (release). Raising `τ` heats it
(tension). Time-vary `τ` for a tension–release arc, or hold it for standing
tension. Same one-dial ergonomics as the removed `harmonic_tension`, but it is
the voice's **own search temperature**, not a terrain deformation.

## Caveat: fluctuating tension, not a held chord

Temperature gives a *fluctuating* tension (restless, wants to move), not a
*static* one (a voice parked on a specific dissonant chord tone). But a deep
consonant well **plus** temperature gives "pulled toward consonance while
jittering", which suits conchordal's living ecology. If a held, named tension (a
tension *chord*) is ever genuinely needed, add discrete rungs then — not now.

## Habituation — autonomous complement (unchanged role)

`τ`/temperature is the *static degree* of restlessness; habituation (sustained
occupancy erodes a held peak's fusion) supplies *when* a settled voice is forced
to move again. A complement, not the tension mechanism.

## Placement

Seed tension at spawn with the existing `dissonance()` / `edge()` targets
(static, local, cheap). Whether a seeded voice resolves emerges afterward from
its climb under its temperature — not from a placement-time prediction.

## Implementation

- Map `tension(τ)` to the pitch-core search temperature (reuse the `temperature`
  / `set_temperature` plumbing — the same path DCC's `temperature_bonus` already
  feeds).
- No `L_max`, no rung ranking, no per-bin resolvability scan, no sigmoid
  handling.
- Observed tension for reports = the voice's current `T` (or, richer, how far
  below the local consonant peak it actually sits on average).
- Determinism: temperature is a parameter; with a fixed seed the
  Metropolis / softmax draws are already deterministic in render.

## Open questions

- **`τ → T` mapping**: linear, or a curve (the Metropolis/softmax response to `T`
  is nonlinear, so a little `τ` may already loosen a tight resolution).
- **Scope**: per-voice vs group/director temperature. DCC sets a global
  `temperature_bonus` today; a composer `τ` could be per-voice.
- **Composition with the core's intrinsic `temperature`** (PeakSampler's 0.08):
  does `tension(τ)` add to it or override it?
- **DCC interaction**: composer `tension(τ)` and listener-driven
  `temperature_bonus` both heat the search — decide how they combine.
- **Static held tension** (discrete rungs) deferred until a concrete need appears.
