# Crowding and Unison Avoidance Design

Status: Investigation complete; decision = keep the current crowding design (no
code change recommended). In scope for v0.4.0 only as documentation.
Scope: conchordal pitch behavior — crowding, the consonance landscape, and how
voices avoid collapsing to unison.
Date: 2026-06-18

This memo records why the current crowding mechanism is kept as-is, after an
investigation that started from the suspicion that `crowding_runtime_delta_erb`
(the roughness-complement repulsion) was theoretically wrong. It is the
behavior-side companion to `docs/design-notes/listener-twin.md` (which narrows
the perception/landscape layers); here the concern is voice *placement*.

## Problem

Two facts pull in opposite directions:

- **Fundamentals overlapping is boring** — two voices at the same f0 fuse into
  one perceptual source (degeneracy), which we want to avoid.
- **Harmonics overlapping is consonant** — shared upper partials (consonant
  intervals) are exactly what makes a chord fuse pleasantly.

Is this a genuine trade-off that needs two opposing terms (the current
`consonance − crowding`), or can it be unified into one principle? And is the
current crowding *shape* (`crowding_runtime_delta_erb` = analytic complement of
the roughness kernel, `CROWDING_REACH_FACTOR = 2.1`, flat-topped at Δ=0) the
right one?

## How crowding actually works today

- Crowding is applied per-candidate, per-neighbor, on **f0 distance** — not baked
  into the consonance field: `base = weighted_score − penalty − gravity_penalty −
  crowding_penalty` (`src/life/pitch_core.rs:949`), with
  `crowding_penalty = crowding_strength · Σ_neighbors crowd(Δ_erb)`
  (`pitch_core.rs:937–945`). Agents **maximize** `base`.
- `crowd(·)` defaults to `crowding_runtime_delta_erb` (the roughness complement)
  when `crowding_sigma_from_roughness` is true (`src/core/roughness_kernel.rs:137`).
- The "roughness + crowding" sum in `plot_roughness_plus_crowding_matches_no_dip`
  is a **diagnostic proxy** for the near-field cost, not a live landscape.
- Defaults are conservative: `crowding_strength = 0`, `anneal_temp = 0`,
  `exploration = 0`, `temperature = None` (`src/life/control.rs`). Candidate
  generation is independently stochastic: 3 Gaussian random candidates at σ=30
  cents every frame; `ratio_candidates` / `global_peaks` default **off**
  (`pitch_core.rs:13–20, 296–323`).
- In shipped samples, `avoid_neighbors` runs at 0.4–1.2; only `10_generations`
  and the `research/` scenarios add `ratio_candidates` / `global_peaks`; the
  annealing **temperature is unused** (one scenario sets `exploration(0.10)`).

## Method (reproducible)

The supporting experiments are a small standalone model (no engine code) and can
be reproduced from this spec alone.

**Model.** Each voice is a harmonic complex tone: K=6 partials at k*f0 (k=1..6),
amplitude 1/k. Pitch in cents; f = base * 2^(cents/1200), base ~= 1000 Hz.

**Roughness / dissonance (Sethares).** For two partials (f1,a1),(f2,a2) with
fmin=min(f1,f2): s = 0.24/(0.0207*fmin + 18.96); df = |f1-f2|;
d = a1*a2*(exp(-3.5*s*df) - exp(-5.75*s*df)). Total roughness R = sum of d over
partial pairs. A voice's leave-self-out (LSO) consonance score
= -(sum of d between its partials and all *other* voices' partials); higher =
more consonant with the rest. Note this is high at unison -- the core problem.

**Spectral density.** A(x) on a cents grid (~3c) = sum of each partial's
amplitude deposited as a Gaussian (sigma ~= 12c). Coverage_p = sum A(x)^p
(p<1 concave/saturating, p>1 convex).

**Experiment 1 - static self-organization.** Coordinate-ascent each voice's f0
from a near-unison start (e.g. 0/7/13/19 c) to maximize an objective; report the
final intervals. Objectives compared: `-R` (naive); `ALPHA*Coverage_0.5 - R` and
`ALPHA*Coverage_2 - R` (concave vs convex, full spectrum); `-R +
GAMMA*Coverage_0.5` over the *fundamentals only*; `-R - strength*sum_pairs
exp(-|df0|/sigma_c)`.

**Experiment 2 - crowding-shape landscape.** Two voices, one fixed; sweep
separation d in [0,1400] c. Objective = `-R_dyad(d) - strength*crowd(d)`. Crowd
shapes: complement = clamp(0, 1 - R(d/kappa)/Rpeak), kappa=2.1 (flat-topped, the
current shape); cusp = exp(-d/40); reach = exp(-d/180). Find local maxima
(attractors), whether d=0 is one, and each maximum's objective value.

**Experiment 3 - dynamics (search modulation).** N=3 voices, time-stepped
(T=160, 24 trials), near-unison start. Degeneracy = min neighbor |df0| < D_deg
(=50 c). Candidates per step: current + 6 Gaussian random (sigma=30c), optionally
ratio candidates at neighbor +/- {316,386,498,702,884,1200} c. Policies:
P0 = greedy on consonance; PA / PAr = greedy on `consonance - strength*sum
exp(-|df0|/80c)` (PAr also offers ratio candidates); PB = pure consonance with
the degeneracy zone excluded from candidates, plus a forced jump to the best
consonant (ratio) candidate while currently degenerate, else greedy-settle.
strength ~= 0.8. Metrics: unison-dwell %, escape %, escape step, final intervals.

## Findings (toy Sethares model; directional, not quantitative)

Conclusions below held across the parameters swept.

1. **Not unifiable into one spectral measure.** A single concave (submodular)
   aggregation over the *full* spectrum (`∫ A^0.5 − roughness`) avoids unison but
   destroys consonant-interval structure: it lands on ~equal 88-cent spacing for
   every coverage weight tried, not on just intervals.

2. **Cleanly separable by order.** Putting the diversity term on the
   **fundamentals only** (`−roughness + γ·∫ A_f0^0.5`) yields real chords
   (e.g. 5:4 / 3:2 / 16:9) and avoids unison. Consonance is a property of the
   **full partial spectrum**; diversity is a property of the **fundamentals
   (perceived pitch)**. The two act on different variables, so they coexist at
   consonant intervals and only collide at the unison singularity. This is
   exactly the structure the current code already has (consonance field +
   crowding on f0 distance).

3. **Crowding shape barely matters.** complement (flat-top), `exp(−d/40)` (cusp),
   and `exp(−d/180)` (cusp + reach) give nearly identical landscapes. In all
   three, **unison stays a shallow local attractor** — not because of the
   crowding shape but because the *consonance curve itself* has a steep
   beating-moat at unison: detuning identical partials makes the coincident
   pairs beat immediately, so roughness rises sharply from Δ=0. That moat
   out-slopes any reasonable crowding cusp, so a cusp does **not** turn unison
   into a local repeller (an earlier hypothesis in this investigation — refuted).

4. **Wider reach is mildly harmful.** `exp(−d/180)` bleeds penalty into the
   close consonant wells (6:5 dropped 0.241 → 0.196), so widening reach to
   "bridge the gap" works against landing on near consonant intervals.

5. **The real levers are strength + candidate jumps.** Crowding strength lowers
   unison's depth so consonant intervals become the **global** optimum (they
   already are at strength ≥ 0.5·Cspan in the toy). Escape from the shallow
   unison attractor happens by **jumping across the beating moat** via random /
   ratio / global candidates — not via the crowding gradient. The engine already
   does this.

## Decision

- **Keep `crowding_runtime_delta_erb` as-is.** Do not switch to a cusp, and do
  not change `CROWDING_REACH_FACTOR`. The shape is nearly irrelevant to behavior;
  the cusp+reach refinement that this investigation initially favored gives no
  benefit and wider reach can erode close consonant intervals.
- **Treat the design as two terms on different orders, by intent:** consonance on
  the full spectrum (`R`, `H`) and diversity on the fundamentals (crowding on f0
  distance). This is correct and should not be "unified away" into a single
  spectral functional.
- **The effective anti-collapse mechanism is strength + candidate engineering**
  (`crowding_strength` plus `ratio_candidates` / `global_peaks` that let the
  search jump the beating moat to a consonant well), not the crowding gradient or
  the (implemented-but-unused) annealing temperature. Scenarios that need precise
  consonant convergence should enable ratio/global candidates (as
  `10_generations` and the research scenarios do).
- Residual property: unison remains a shallow local attractor surrounded by the
  roughness beating-moat. This is intrinsic to harmonic-timbre consonance, not a
  crowding defect; it is bypassed in practice by candidate jumps and made
  globally non-optimal by crowding strength.

## Open direction: unison as search modulation (promising, not yet adopted)

Because escape from the unison attractor is by *jumps across the beating moat*,
the natural lever is the search dynamics, not the landscape: treat unison/
degeneracy as a trigger that raises restlessness (lowers persistence, raises
jump probability) and offers *targeted* consonant (ratio) candidates — while
leaving the consonance landscape pure (no crowding penalty).

A toy dynamics comparison (Experiment 3 in Method, 3 harmonic voices started
near unison, 24 trials) supports this:

| policy | unison dwell | escaped | escape step | example final |
|---|---|---|---|---|
| pure consonance, greedy | 100% | 0% | — | unison collapse |
| consonance − crowding (random cands) | 0.8% | 100% | ~1.9 | 5:4 / 3:2 |
| consonance − crowding + ratio cands | 0.0% | 100% | 0 | 3:2 / 2:1 |
| **pure consonance + degeneracy-restless + ratio jump + degeneracy-zone exclusion** | **0.0%** | **100%** | **0** | 3:2 / 2:1 |

The proposed search-modulation policy **matches the best current setup** while
keeping the consonance landscape undistorted (no penalty bleeding into close
consonant wells). Two design points are essential: (1) restlessness must be
triggered by *degeneracy* (small Δf0 / leave-self-out marginal ≈ 0), **not** by
consonance score — score is highest at unison, so a score-driven "satisfaction"
would settle there; (2) the escape jump must be *targeted* (ratio candidates) so
it lands on a consonant interval rather than wandering. The reuse is clean: the
leave-self-out marginal, which fails as an *objective* term, works well as the
*restlessness trigger*.

Hard vs soft is a design choice: a hard degeneracy-zone exclusion gives 0%
dwell (competitive exclusion); a soft restlessness/jump-probability bump leaves
a little emergent dwell but a more organic feel. This is a behavior change
(it alters dwell time / escape rate, not just fixed points), so it needs an
engine-side prototype and audition before adoption. Maps onto existing knobs:
`persistence`/satisfaction, `exploration`, `anneal_temp`, and `ratio_candidates`
made state-dependent on degeneracy (overriding the greedy shortcut when degenerate).

## Caveats

The supporting evidence is a Hz-domain static Sethares toy with 2–4 harmonic
voices and greedy search. The production landscape is ERB/Log2, dynamic, and
multi-voice with leave-self-out. The structural conclusions (shape is nearly
irrelevant; beating-moat dominates the unison neighborhood; strength and
candidate jumps are the real levers; full-spectrum unification fails while
order-separated terms succeed) were robust across the shapes and strengths
swept, but exact thresholds are not transferable.
