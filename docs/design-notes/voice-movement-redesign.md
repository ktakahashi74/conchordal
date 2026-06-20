# Voice Movement Redesign: Occupancy as a Unified Drive

Status: Design proposal, verified by a standalone toy; **not yet implemented**.
No code change is made by this note. Distillation into the technote ledger is
deferred until adoption.
Scope: how a voice decides *where to move* (pitch behavior) — unifying crowding,
habituation, and the proposed "restless escape" into one principle.
Date: 2026-06-19

This is the behavior-side successor to `docs/design-notes/crowding-unison-avoidance.md`
(which decided to keep the crowding *penalty* as-is) and a companion to
`docs/design-notes/listener-twin.md` (perception layers). The crowding note's
"Open direction" — escape-as-search-modulation — was prototyped, found to be a
redundant extra layer (its own degeneracy detector plus a parallel code path),
and reverted. That failure motivated stepping back to first principles: what is
the *single* reason a voice moves?

## Problem: the movement decision has accreted six overlapping mechanisms

Today `propose_with_scorer` (`src/life/pitch_core.rs`) answers one question —
"move or stay, and to where?" — with six mechanisms layered on top of each other:

| mechanism | role | kind |
|---|---|---|
| `improvement_threshold` | move only if the best candidate beats current by this | threshold |
| `persistence` × `satisfaction` | tendency to stay (satisfaction rises with consonance) | probability |
| `exploration` | bias toward moving / randomizing | probability |
| `anneal_temp` | accept downhill moves (simulated annealing) | probability |
| `crowding_penalty` | deform scores to repel from neighbor fundamentals | score term |
| (proposed) restless escape | detect overlap, force a jump to a consonant ratio | discrete path |

They interact in ways that are hard to reason about. The deepest symptom is the
**satisfaction trap**: `satisfaction` is defined from the consonance score, which
is *maximal at unison*, so the stay-probability is highest exactly where we least
want a voice to sit. Each new "make it move" knob is a patch around that trap.

The accretion is the signature of a missing unifying principle.

## First principle: movement is descent on one potential field

A voice moves to occupy a position that is **consonant and not already explained
by the perceived scene**. Formally it descends a single field:

```
U(p) = consonance(p)  −  F(p)  −  effort(p)
```

Each term maps one-to-one onto a conchordal commitment, and the three are
orthogonal:

- **consonance(p)** — the perception model (Landscape `H`/`R`). Pragmatic value:
  where fusion/consonance is good. Full-spectrum.
- **F(p)** — *redundancy / occupancy*. Predictive-coding epistemic value: how much
  position `p` is already accounted for by the rest of the scene. On fundamentals.
- **effort(p)** — embodiment (move cost / inertia).

In active-inference terms `U = preference(consonance) + information_gain(−F) −
effort`. The voice prefers consonant states and is driven toward pitches that add
information the scene does not already carry.

## The unification: diversity and habituation are one field F

The central claim of this note: **diversity (crowding) and habituation
(predictive-coding boredom) are the same quantity** — the occupancy of perceived
fundamentals — differing only in *source* and *timescale*:

```
F(p) = Σ_other_voices  occ(p − f0_other)            // spatial slice  = "diversity"
     + Σ_self_history   decay · occ(p − f0_self_past)  // temporal slice = "habituation"
```

- **Spatial slice** (other voices' current fundamentals): occupying the same f0
  as a neighbor is redundant — two voices fuse into one perceived source. This is
  exactly the old crowding repulsion.
- **Temporal slice** (the voice's own decaying history): dwelling makes a position
  "already predicted" — boredom — which drives renewal. With decay, abandoned
  positions become fresh again, so a voice can return later.

Same physics; the only difference is whose contribution and over what time.

### Critical constraint: F is over *fundamentals*, not the full spectrum

`occ(·)` is a narrow kernel on **fundamental** distance. This is what keeps the
unification from eating consonance — the failure mode `crowding-unison-avoidance.md`
Finding 1 warned about (a single full-spectrum aggregation lands on equal spacing
and destroys just intervals). If shared *harmonics* counted as redundancy, the
consonant intervals (which exist precisely because partials coincide) would be
penalized. By restricting `F` to perceived fundamentals (virtual pitch), unison
(f0 coincidence) is redundant while a fifth (f0 distinct, harmonics shared) is
not.

The kernel half-width `σ` is the one essential parameter of `F`: the width within
which two fundamentals count as "the same perceived pitch." ≈ 50 cents (a
quartertone neighborhood) matches the perceptual fusion zone and the degeneracy
zone used in the prior note.

## How the six mechanisms collapse

- `crowding_penalty` + restless escape → **one field `F`**. The penalty *is* `F`'s
  spatial slice used as a score term; "escape" is not a separate path but an
  emergent consequence of `F` plus the search.
- `improvement_threshold` / `persistence` / `exploration` / `anneal_temp` → **one
  search temperature**, ideally driven by `F`'s temporal slice (boredom raises the
  temperature). The free probabilistic knobs become one quantity with a clear
  meaning.
- The **satisfaction trap dissolves**: the drive to stay comes from low boredom
  (low temporal `F`), not from a high consonance score. A voice at high consonance
  still accrues boredom and eventually leaves — no special escape needed.

The dozen-odd pitch knobs reduce to roughly five meaningful quantities: the three
field weights (consonance / F / effort), the occupancy width `σ`, and a search
temperature. Defaults keep simple things simple: `seek_consonance()` alone should
yield "gather on consonance, drift with boredom, individuate by diversity."

## Method (reproducible toy)

A standalone model (no engine code), reproducible from this spec alone. It mirrors
the experiments in `crowding-unison-avoidance.md`.

**Voices.** Harmonic complex tones, K=6 partials at `k·f0` (k=1..6), amplitude
`1/k`. Fundamentals in cents; `f = base · 2^(cents/1200)`, base ≈ 1000 Hz.

**Consonance (Sethares, leave-self-out).** For two partials `(f1,a1),(f2,a2)` with
`fmin=min(f1,f2)`: `s = 0.24/(0.0207·fmin + 18.96)`, `df = |f1−f2|`,
`d = a1·a2·(exp(−3.5·s·df) − exp(−5.75·s·df))`. A voice's consonance at `p` is
`−Σ d` over pairs between its partials and all *other* voices' partials. Maximal
(=0) at exact unison — the trap.

**Occupancy.** `occ(Δ_cents) = exp(−Δ²/(2σ²))`, σ=50c, on **fundamental**
distance. `F_spatial(p) = Σ_others occ(p − f0_other)`. `F_temporal` is a decaying
per-grid-bin history: each step multiply by `decay` and add `occ(p − chosen)`.

**Objective.** `U(p) = consonance(p) − strength·F(p) − effort·|p − current|`,
with `strength = factor · Cspan` (`Cspan` = consonance span over the grid vs a
single reference) and `effort` as a fraction of `Cspan` per octave. Grid
−50..1250c at 1c. Exp A uses 60 rounds of coordinate ascent; Exp B/D time-step by
global argmax of `U`.

Experiments: **A** static self-organization (4 voices from near-unison, spatial F
only); **B** boredom-only wandering (1 voice vs a fixed drone, temporal F only);
**C** kernel sanity (occupancy at named intervals); **D** unified F (3 voices,
spatial + temporal) with an effort sweep.

## Findings (toy; directional, not quantitative)

**Exp C — consonance is preserved.** The fundamental-occupancy kernel flags only
near-unison as redundant:

```
occ(unison 0c)=1.00  occ(m2 100c)=0.14  occ(m3)=occ(M3)=occ(P4)=occ(P5)=occ(8ve)≈0.00
```

A perfect fifth is ~non-redundant, so `F` does not erode just intervals.

**Exp A — spatial slice yields diversity without touching consonance.**

```
factor=0.0   → 0, 0, 0, 0            (collapsed to unison; min pairwise 0c)
factor≥0.3   → 0 / P4 / M6 / 8ve     (a real consonant chord; min pairwise m3=316c)
```

**Exp B — temporal slice dissolves the satisfaction trap.** One voice vs a drone:

```
factor=0.0   → unison 200/200 steps        (1 well: the trap)
factor=0.4   → 13 distinct consonant wells  (wanders among consonant points)
```

Boredom alone produces ongoing movement among *consonant* positions — no separate
restless mechanism.

**Exp D — unified F: no collapse and continuous movement; field vs search
separate.** 3 voices from near-unison, effort = inertia:

```
factor=0.0 effort=0.0  → unison-dwell=100%  move/step=12c    (collapse)
factor=0.4 effort=0.0  → unison-dwell=  0%  move/step=1339c
factor=0.4 effort=0.5  → unison-dwell=  0%  move/step= 669c
factor=0.4 effort=1.5  → unison-dwell=  3%  move/step= 395c   final 0/P4/m6
```

The same `F` gives 0% unison dwell *and* alive movement; `effort` (a search/inertia
control) tunes the movement magnitude without reintroducing collapse. This
confirms the field encodes *where to want to be* and the search/temperature
controls *how much to move* — they are cleanly separable.

## What this establishes

1. **The unification holds.** Diversity and habituation are one fundamental-
   occupancy field `F`; one narrow kernel does both.
2. **Consonance is preserved** as long as `F` is on fundamentals (σ ≈ fusion
   width), not the full spectrum.
3. **The movement objective collapses to three terms** `U = consonance − F −
   effort`, with the probabilistic stay/move apparatus replaced by a single
   boredom-driven temperature.
4. **Field and search are orthogonal**: incentives (`U`) vs effort/temperature.

## Recommendation

Adopt the principle for the redesign: one occupancy field `F` (fundamental,
spatial + temporal), `U = consonance − F − effort`, and a search temperature
driven by `F`'s temporal slice. Treat the existing crowding penalty as `F`'s
spatial slice and the existing adaptation (boredom/familiarity) as its temporal
slice, rather than as independent mechanisms. Do **not** add a separate escape
path.

## Open implementation questions

- **Representation.** Express `F` as one Log2Space occupancy `_scan` fed by all
  voices' fundamentals plus a decaying self/scene history. Can `src/life/adaptation.rs`
  (already fed by `subjective_intensity`, the whole-scene density) be repurposed as
  the temporal slice on a *fundamental* grid? `harmonicity_kernel` provides the
  virtual-pitch basis for "fundamental."
- **One field, many consumers.** `F` may subsume leave-self-out (marginal
  contribution) and spawn density (occupancy masks) — both are occupancy queries.
  Worth checking whether they can read the same `F`, conchordal-style.
- **Search layer.** Replace global argmax with local search + a boredom-driven
  temperature to tame movement into musical (not teleporting) motion.
- **Balance.** `strength` (F vs consonance) sets adventurous-vs-sticky character;
  σ sets the fusion width. Both need auditioning.
- **pred/perc.** A deeper layer: act on the *predicted* occupancy (zero-latency)
  and correct by `err = perc − pred`. Out of scope for the first pass.

## Implementation investigation (substrate audit, 2026-06-19)

A code audit shows the redesign is **consolidation, not new construction**: every
piece of `F` already exists, scattered across different representations and paths.

| existing component | what it is | role in `F` | repr | pred/perc |
|---|---|---|---|---|
| `landscape.harmonicity` | Env->Roots virtual-pitch salience, computed per frame | **fundamental occupancy itself** | Log2Space scan | perc (delayed) |
| `crowding_penalty` (`pitch_core.rs`) | ERB repulsion from neighbor fundamentals, per candidate | **spatial slice** | ERB, per-candidate | pred (known f0) |
| `adaptation` (per voice) | `h_fast`(boredom)/`h_slow`(familiarity), leaky, env+self blend, added to objective | **temporal slice** | Log2Space scan | mixed |
| `neighbor_pitch_log2` (`population.rs`) | other voices' current f0 (self-excluded, visibility-filtered, split-sign) | spatial-slice **source** | log2 | pred |
| `is_range_occupied` / `consonance_density_mass` | ERB occupancy mask at birth | **spawn consumer** of occupancy | ERB | pred |
| `leave_self_out` | subtract a voice's own contribution from its consonance eval | "don't repel yourself" | — | — |

### Key finding: adaptation is wired correctly but **fed the wrong signal**

`adaptation.update` takes its environment term from
`FeaturesNow::from_subjective_intensity(...)` — **full-spectrum** perceived density.
This violates the note's core constraint (`F` must be on fundamentals). Because
consonant intervals deposit energy at shared-harmonic bins, full-spectrum density
marks those bins as "occupied", so **the default-on adaptation may already mildly
repel consonant intervals** — a small instance of Finding 1's failure. The fix is
one repoint: feed adaptation from `harmonicity` (fundamental occupancy) instead of
`subjective_intensity`. This is the single highest-value change and reuses an
existing field.

### Two clean unifications discovered

1. **LSO == self-exclusion on the occupancy field.** A shared field
   `F_all = Σ_all_voices occ(f0)` (decaying) evaluated as `F_all − own_contribution`
   *is* leave-self-out. Crowding's self-exclusion and LSO become one operation.
2. **Spatial slice == temporal slice in the τ→0 limit.** Depositing every voice's
   current f0 each frame and decaying gives both the instantaneous (others) and the
   historical (boredom) behavior from one leaky occupancy field.

### Staged migration (each stage independently auditionable and reversible)

- **Stage 0 (no behavior change):** add an `F` scan to `landscape`, computed per
  frame as a decaying deposit of all live voices' fundamentals (predictive). Not
  wired to behavior; visualize only.
- **Stage 1 (minimal, high-value):** repoint adaptation's env feed from
  `subjective_intensity` to fundamental occupancy (`harmonicity` / `F`). Audition:
  does consonance-erosion vanish while boredom-driven movement persists?
- **Stage 1.5:** feed adaptation's env from a *predictive* fundamental-occupancy
  scan built from visible neighbors' f0 (zero-latency, self-excluded), rather than
  the perceptual, delayed, root-salience `harmonicity`.
- **Stage 2:** replace the crowding penalty's ERB per-candidate eval with sampling
  the shared `F` minus self (= LSO on occupancy). Unifies spatial slice + LSO.
- **Stage 3:** collapse `improvement_threshold` / `persistence` / `exploration` /
  `anneal_temp` into one boredom-driven search temperature.
- **Stage 4 (optional):** spawn density reads the same `F`; split pred/perc sources.

**Implementation status (2026-06-19):** Stage 1 landed (rename
`FeaturesNow::from_subjective_intensity` -> `from_occupancy_scan`, feed
`landscape.harmonicity`), then immediately refined into **Stage 1.5**: the
adaptation env term is now built per voice in `PitchController` from
`neighbor_pitch_log2` (Gaussian deposit, σ≈50c, windowed), so it is f0-based,
zero-latency, and self-excluded. With this, adaptation cleanly splits into env =
others' occupancy (spatial slice) and self = own history (temporal slice). Tests
and clippy pass. Crowding is still the separate Channel A penalty
(`avoid_neighbors`, default off); Stage 2 will collapse it onto the same scan.

**Audition (2026-06-20):** A/B render of `05_settling` confirms the principle in
real audio. *Before* (full-spectrum feed) keeps residual tension to the end —
voices that reach a consonant interval are pushed back off it because their own
shared-harmonic energy marks those bins "occupied" (Finding 1's erosion, audible).
*After* (Stage 1.5) resolves fully to consonance: the "Settling" étude actually
settles. So the change is a real, audible fix, not just a cleanup. The
over-resolution concern (does it freeze pieces that want sustained life into
static consonance?) was checked against `08_murmuration`: the after render keeps
its movement/life, so boredom-driven drift survives — no freezing observed.
Stage 1.5 is therefore adopted. Coefficient balance (`strength` / `rho_self`)
remains available if a future piece needs more or less drift.

**Stage 2 (2026-06-20):** the crowding penalty and the adaptation occupancy feed
now share one definition — `occupancy_contribution` (a Log2 Gaussian of width
`crowding_sigma_cents` with the pairwise split bias). The ERB roughness-complement
crowding path is gone (`crowding_sigma_erb`, `crowding_runtime_delta_erb`, and the
`hz_to_erb` use in pitch behavior are removed). This is the "single evaluation,
multiple responses" the redesign set out to reach: crowding (a per-candidate
penalty) and adaptation (boredom/drift) read the same occupancy field, and
`crowding_strength = 0` by default so the penalty is opt-in while the boredom
response is always on. The narrower Gaussian also fixes Finding 4 (the old wide
reach bled penalty into close consonant wells). Residual cleanup (**Stage 2b**):
`crowding_sigma_from_roughness` is now vestigial — still plumbed but ignored — and
should be removed from the config / Rhai surface, with the `avoid_neighbors`
one-arg docs updated (the width no longer derives from the roughness kernel).
Tests and clippy pass. **Audition (2026-06-20):** Stage 2 is behavior-neutral —
`05_settling` and `08_murmuration` render essentially the same as Stage 1.5
(s2 ≈ s15), confirming the kernel swap unified the definition without changing
behavior. Stage 2 adopted.

**Octave-convergence finding (2026-06-20).** Auditioning the endings exposed a
real gap: `05_settling` converges to near octave/unison (the deepest, most-fused
consonance), and no parameter cleanly prevents it. The occupancy field is on raw
f0 distance, so it repels unison (Δf0≈0) but is blind to octaves (Δf0=1200c →
occupancy≈0), which are also the most consonant wells. So a composer can already
tune *overall* convergence (`landscape_weight`, `avoid_neighbors`,
`persistence`/`exploration`/`anneal_temp`, `move_cost`, `range_oct`) but cannot
forbid octave-doubling to land a *distinct* chord. Octave-convergence as intent is
fine; octave-convergence as the only reachable outcome is not.

**Octave-aware occupancy (implemented 2026-06-20).** An octave-equivalence (chroma)
term was added to the shared occupancy kernel `occupancy_contribution`, weighted by
a composer knob `octave_avoidance` (Rhai `octave_avoidance(weight)`, draft-only):
`occupancy = Gaussian(Δf0) + octave_avoidance · Gaussian(Δ_chroma)` where
`Δ_chroma` folds Δf0 into one octave. `0` = octaves allowed (converge), `>0` =
octaves repelled (distinct chords). One knob feeds both consumers (crowding penalty
and adaptation occupancy), so it extends the single occupancy definition rather than
adding a parallel mechanism.

*Audition:* `05_settling` with `octave_avoidance(1.0)` no longer collapses to
octave/unison. With 7 seekers the result is a distinct, consonant, *moving*
texture (octave-doubling is forbidden, so 7 voices cannot fold into a 3-note triad,
and packing that many distinct pitches has no still equilibrium — judged fine). With
3 seekers it settles into a distinct triad in the final 1–2 s. So the composer now
chooses octave-convergence vs distinct chords via the knob, and voice count vs the
`octave_avoidance` weight trade off doubling against distinctness. Adopted as-is
(the chroma term is intentionally applied to both crowding and the adaptation feed;
the resulting drift is wanted).

*Window fix (A/B, 2026-06-20).* A review caught that the adaptation occupancy scan
used a local ±(0.5+4σ)≈0.7-octave deposit window, which does not reach the octave
(±1.0), so octave avoidance was effectively crowding-only (the crowding penalty is
windowless and was correct). An A/B render compared (a) the local window vs (b)
covering the whole scan when `octave_avoidance > 0` (the chroma term is
octave-periodic, so it cannot be captured by a single local window). (b) was chosen
by ear, so the scan now deposits across the full Log2Space when octave avoidance is
on, making the adaptation feed genuinely octave-aware as intended. (Cost: O(n_bins ×
neighbors) per proposal when the knob is on; acceptable at control rate, optimizable
later to per-octave windows if needed.)

**Remaining cleanup — Stage 2b.** The vestigial `crowding_sigma_from_roughness`
(plumbed but ignored since Stage 2) should still be removed from the core / config /
Rhai / docs surface. Deferred to keep the octave_avoidance change contained.

### Mismatches to resolve

- **ERB vs Log2:** crowding/spawn use ERB; `harmonicity`/adaptation use Log2Space.
  Unify `F` on Log2Space (`_scan`, invariants F1–F4). Since shape "barely matters"
  (`crowding-unison-avoidance.md` Finding 3), a plain Gaussian on Log2 (σ≈50c)
  replaces the ERB roughness-complement — a simplification.
- **pred vs perc:** `harmonicity` is delayed; for zero-latency movement prefer a
  predictive deposit from voice f0, with `harmonicity` as the perceptual correction.
- **per-voice vs shared:** adaptation is per-voice; decide between a shared spatial
  `F_all` + per-voice self-history, or fully shared + LSO.

## Caveats

The evidence is a Hz-domain Sethares toy with harmonic tones, greedy global-argmax
search, ≤4 voices, deterministic. It is directional, not quantitative; production
is ERB/Log2, dynamic, multi-voice with leave-self-out. The structural conclusions
(diversity = habituation = fundamental occupancy; consonance preserved iff `F` is
on fundamentals; field and search separable) were robust across the strengths and
efforts swept, but exact thresholds (σ, strength, decay, temperature) are not
transferable and must be retuned and auditioned in the engine. The toy source used
for the figures above lives at `target/toy/occ_toy.rs` (git-ignored).
