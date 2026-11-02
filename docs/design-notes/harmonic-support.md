# Harmonic Support Features: Valuation Beyond One Consonance Scalar

Status: design adopted for implementation (HV-2, 2026-11-01). Implementation unit HV-3.
Scope: the vertical body of the author's decision to extend harmonic valuation from one
consonance scalar to several axes. Stage 1, `set_harmonicity_preference`, is implemented
(ledger §9.3.116). The temporal part is designed in parallel (HV-4); it is not deferred.

## Decision and aim

The author asked whether developments that harmony theory treats symbolically (substitute
chords, the flat fifth, seventh to thirteenth extensions) could become parameters of the
cognitive landscape, and adopted an extension of the harmonic valuation space. The success
condition is non-symbolic sonic color (condition A). Reproducing jazz categories or harmonic
function is not a success condition, but every phenomenon of the harmonic system is assigned
to a place below, so that nothing is dropped silently.

One scalar `field_score` per bin cannot express these developments. Reweighting H inside it
(stage 1) changes how strongly the ecology seeks harmonic positions; it does not separate a
note that adds related content while the support stays focused from an isolated note or a
duplicate. The features below add those distinctions.

## Coverage of the harmonic system

Each phenomenon is split across places. V = vertical features (this note). T = temporal DCC
(HV-4). L3 = layer-3 familiarity. X = not represented, because it needs symbolic input or
depends on pitch-class or spelling properties; the replacement is named.

1. Chord tones, triads, compound intervals: V (interference, integer-ratio support, support
   focus). X: chord names and degree numbers.
2. Extensions and tensions (7, 9, 11, 13, ♭9, #11): V (related new content under shared
   support, roughness, register). T (preparation, holding, resolution). L3 (familiarity).
   X: degree labels relative to a named root or scale.
3. Avoid notes: V (interference that depends on placement, level and timbre). T. L3.
   X: per-pitch-class prohibition tables.
4. Flat fifth and altered notes: V (support and interference as an interval moves
   continuously). T. X: the spelling of what was altered.
5. Diminished, augmented, whole-tone and other symmetric structures: V (spread and
   conflicting support, repetition in actual pitch). L3. X: symmetry of pitch-class sets.
6. Suspensions and quartal stacks: V. T (suspension and release). L3.
7. Rootless voicings and missing fundamentals: V (support at low positions that do not
   sound). T.
8. Upper structures, polychords, polytonality: V (coexisting support for different
   explanations). T. L3. Perceived root counts are not estimated.
9. Inversion, bass, pedal, doubling, omission: V (absolute register, lowest partials,
   overlap). T (holding the bass and common tones). Equal pitch-class sets are not identified.
10. Diatonic substitution: V (momentary acoustic similarity). T (another sonority that fits
    the same context). L3. X: in-key substitution tables.
11. Tritone substitution: V (shared pitches, support and interference). T (approach and voice
    motion). X: the rule "move the dominant root by a tritone".
12. Function, cadence, resolution, tension arcs: V (momentary roughness, support, conflict).
    T (deviation from expectation, arrival). L3. X: fixed tonic/dominant/subdominant labels.
13. Modulation, pivots, tonicization: T (shift of the remembered support distribution, fit
    to several contexts). L3.
14. Voice leading and common tones: V (the reached configuration). T (voice continuity,
    holding, motion size). L3. X: prohibition tables such as parallel fifths.
15. Passing, neighbor, anticipation and appoggiatura tones: T (role from context and dwell).
    V (momentary support and interference).
16. Harmonic rhythm, sequences: T. L3. `harmonic_rhythm` only arranges opportunities.
17. Tuning, microtones, beating, enharmonics: V (continuous ratios and interference).
    X: different spellings of the same pitch.
18. Keys, modes, scale membership, stylistic chord vocabulary: L3 (continuous distributions
    from listening experience). T. X: fixed pitch-name tables.

L3 is not implemented under the current policy: fixed tables are presets and are removed,
and learning layer 3 during a performance is deferred (§9.3.56). Implementing L3 needs a
separate author policy decision.

## Representation

`e` is the existing auditory density on Log2Space (after self-exclusion where the consumer
uses it). `A` is the existing downward projection of the H kernel, so `r = A e` is the
template-support distribution (previously internal). Sums include the existing bin width.
`p = r / sum(r)` when `sum(r) > 0`.

`r` is template support. It is not a perceived root, a root probability or a count of roots;
a single sine already shows support at f/2 and f/3 with weights 0.76 and 0.64. Sub-root
peaks are kept; no peak is removed, and no feature is read as a root count.

Silence: `α(e) = z(e) / max(z(e), ε)`, where `z` is the total mass used by the existing
density normalization and `ε` is its shared floor (`src/core/stream/potentials.rs`). The
floor applies to total mass only; it is not a per-bin presence threshold. All features are
zero for exact silence.

Discrete measure. The new readouts live in the H projection's own coordinate. `A`
shift-adds density samples on the constant log2 grid, so every new sum, normalization and
inner product uses the constant log2 bin width: `M = Σ_i r_i · Δlog2`,
`p_i = r_i / Σ_j r_j`, the cosines `J0`, `J`, `K`, the quadratic forms and the sums in `N`
(including `Σ e` and `Σ q`). The constant width cancels in every normalization and cosine,
so these are plain vector operations on the grid, and shifts in log frequency stay exact
grid translations. Only `α` uses the existing ERB mass `z(e) = Σ_i e_i · du_i`, because it
inherits the existing silence floor defined on that mass. `M` is reported in grid units and
is not the existing loudness mass.

## Features

All three are author rules (§9.3.104): they are new readouts without a published formal
model. Their ranges are `[0, 1]`.

1. Support focus `C`. With `g_x` the existing H smoothing Gaussian (σ = 3 cents) placed at
   x and `J0(x, y)` the cosine between `g_x` and `g_y`:
   `C(e) = α(e)^2 · p' J0 p`.
   It is larger when support gathers in one resolution cell of the existing representation.
   σ is reused so that positions the existing smoothing does not separate are not counted
   separately; this note does not claim σ is perceptually justified.
2. Support conflict `D`. With `t_x` the harmonic template at root position x (the existing
   harmonic count, weights and smoothing) and `J(x, y)` the cosine between `t_x` and `t_y`:
   `D(e) = α(e)^2 · (1 − p' J p)`.
   It is larger when the supported templates explain different spectra. `C` measures
   position, `D` template overlap; the two quadratic forms are not proportional in general.
3. Related novelty `N`. With `a_y = A δ_y` the support of a unit component at y and
   `K(x, y)` the cosine between `a_x` and `a_y`, the candidate density `q` contributes only
   its part not already covered by the existing density:
   `q_new(y) = max(0, q(y) − e(y))`,
   `N(e, q) = α(e) · α(q) · 4 · Σ_y (K e)(y) · q_new(y) / (Σ e + Σ q)^2`.
   The factor 4 follows from the bound `N ≤ 1`. Duplicated components (covered by equal or
   greater existing density) contribute nothing, even when other component pairs are
   harmonically related. New components that share support with the existing sound
   contribute positively. Components that share no support contribute nothing, and adding
   them only enlarges the denominator. Resolution is the one already encoded in the
   auditory density; no presence constant is introduced.

## Candidate scans

For a fixed reference `e` during one decision and a candidate body density `q_f` at pitch f:

```
ΔC(f) = C(e + q_f) − C(e)
ΔD(f) = D(e + q_f) − D(e)
V(f)  = w_C · ΔC(f) + w_D · ΔD(f) + w_N · N(e, q_f)
```

When the body template shifts in log frequency (`q_f = T_f q`), `A q_f = T_f A q`, so the
numerators of `C` and `D` over all f are cross-correlations of fixed patterns with `r`
(linear correlation with zero padding; circular correlation is forbidden). `N` is
nonlinear in `q_f` through `q_new`; it is evaluated directly over the body template's
support, which is sparse. Boundary templates correct mass and norms per candidate so that
a truncated body does not gain.

The additive density `e + q_f` is an acoustic surrogate: the front end is nonlinear, so it is
not the density of the mixed PCM. It is named so in reports and diagnostics.

**Orchestrator revision after HV-6.** For harmonic, modal and noise-excited bodies the additive
density surrogate reverses candidate rankings against the actual mixed PCM even with the measured
candidate density, because the front end's critical-band compression is not additive. Candidate
evaluation therefore adds the candidate in the power domain before compression: observers publish
the per-bin pre-compression quantities the steady-state S4–S6 formula needs, the body template is a
pre-compression power pattern shifted in log2, absolute-frequency factors (A weighting, ERB and
critical-band widths) are applied after the shift, and the compressed density is recomputed only on
the affected bins before `A` and the features. Phase interference and transients remain outside the
surrogate.

Acceptance of a fast path is measured on the consumer's decision, not on raw ranking reversals. For each
decision rule (PeakSampler softmax at its temperature, Metropolis acceptance, deterministic peak
argmax, density spawn with `β`), the total-variation distance between the decision distribution from the
surrogate and from exact mixed-PCM evaluation of the same candidates must not exceed the existing error
budget: the distance between decision distributions from exact evaluations at consecutive hops of
steady sound. No new tolerance constant is introduced. Outside the accepted range, consumers with small
candidate sets may use exact render evaluation within the hop budget; otherwise they fall back to the
labeled probe.

`V` is reference-dependent. It is not a fixed global potential, so sequential updates do
not guarantee global improvement, and temperature, habituation and births remove
monotonicity in any case.

Weight scale: each feature difference between two candidates is at most 1, so weight 1
lets one axis separate candidates by at most one `field_score` unit, the full span of the
default H term (`a = 1`). This is a scale statement, not a claim that the axes have equal
reach.

**Orchestrator revision after HV-6, second pass (supersedes the counterfactual surrogate above).**
The power-domain surrogate certified only sine bodies for peak spawn, exact render evaluation costs
84–116 ms per candidate, and power publication on every observer broke the declared load. These are the
signs the author names: heavy computation and a persistent lack of a natural form mean the premise is wrong.
AGENTS.md already says why: a Voice acts on local perception, and exact pre-hearing predictions or
exhaustive candidate tables are not built unless a consumer needs them. The counterfactual mixture
prediction, and the orchestrator's acceptance by agreement with mixed PCM, made the features a
prediction problem.

The features are therefore published as first-order scans from the heard sound, with the same epistemic
status as H and R. With `a_y = A δ_y`, `|a_y| = Σ a_y`, `Σr = sum(r)`:

```
gC(y) = ∂C/∂e_y = α^2 (2/Σr) [ (Aᵀ(J0 p))_y − (pᵀ J0 p) |a_y| ]
gD(y) = ∂D/∂e_y = −α^2 (2/Σr) [ (Aᵀ(J p))_y − (pᵀ J p) |a_y| ]
k(y)  = (K e)(y)
```

Each is one or two existing up/down projections per observer, close to the cost of H. A body reads them at
its own partial positions, as body fitness already integrates fields over partials: with the body's
partial positions `y_k` and partial densities `q_k` at its level from its self-report,

```
V_body(f) = Σ_k q_k [ w_C gC(y_k) + w_D gD(y_k) ]
          + w_N · 4 Σ_k k(y_k) · max(0, q_k − e(y_k)) / (Σe + Σq)^2
```

The first line is the first-order ΔU; the N term reads the related-novelty definition at the partials.
No mixed-PCM prediction, power publication or exact render evaluation is used. Bodies without a
self-report read the scans as a single partial at f, labeled as such.

Acceptance: zero weights bit-exact; definitions (each gradient equals the limit of finite differences of
C and D for vanishing additions; N at the partials as defined); causal difference; real time at the declared
load (Voices with and without self-exclusion and Flora with its default self-exclusion). Agreement of
candidate decisions with mixed PCM is not required. One author audition on the final configuration.

## Consumers

1. Voice movement reads `V` only when pitch self-exclusion is on (`leave_self_out(true)`).
   The features measure relations to the sound of others; a Voice's current sound will not
   be present after it moves, so the mixed reference would reward relating to itself.
   The move compares `score(f) − score(current) + V(f) − V(current)` with the same
   reference, body and level. Without self-exclusion, a positive weight warns once and
   movement ignores `V`. No mixed-minus-self density is used anywhere.
2. Spawn reads `V` against the mixed reference, because a spawn really adds sound. Peak
   sampling ranks `existing score + V`; `tension(τ)` ranks the same choice score. Density
   sampling adds `β · V` to the log mass (β is the existing level steepness, default 2);
   zero existing mass stays zero. No new temperature word is added.
3. Flora lane placement and replacement read `V` with the lane atom as `q` and the Flora's
   existing reference. Flora growth keeps its base level; feeding the new value into growth
   would let partial count and level feed themselves.
4. Habituation drive and ListenerTwin keep their existing inputs.
5. The UI shows the global `M`, `C`, `D` and the selected body's `ΔC`, `ΔD`, `N` separately
   from the landscape; candidate-dependent values are not drawn as one universal terrain.

## Bodies

No ecology code branches on body kind. A body supplies its auditory density template at its
level (or a few basis patterns) as a fast path, and conformance tests check it against the
body's own render after the front end. Without a template, a listener-side narrowband probe
may rank proposals, labeled as a probe; it is not the body's exact `ΔU`.

## Composer words

Cursor events, default 0, finite values only, evidence label Author rule. All three at zero
bypass the new paths, including random draws, so every output is bit-exact with the
previous build. No upper clamp exists until failure or saturation gives one.

- `set_support_focus_preference(w)`: real; positive favors focused support (`C`).
- `set_support_conflict_preference(w)`: real; positive favors conflicting support (`D`).
- `set_related_novelty_preference(w)`: nonnegative; favors related new content (`N`).

## Cost and real time

No new observers. Existing observers publish `e` and `r` once; features and correlations
reuse them. FFT plans, halos, body templates, scans and double buffers are preallocated;
hop processing reuses scratch. Real-time acceptance is measured at a declared load
(Voice count, body bases, self-exclusion count) within the 10.67 ms hop at 48 kHz / 512.
Load beyond it is reported as capacity, not hidden by weight bounds.

## Acceptance

1. All weights zero: outputs, choices, events and random sequences bit-exact.
2. Definitions: ranges, silence, floor continuity, direct versus correlated evaluation,
   boundaries, levels. Acoustic-structure controls: a single tone's sub-root series,
   focused support, separated conflicting support, broad support, a fully duplicated
   candidate (`N = 0`), a related new addition (`N > 0`), an unrelated addition (`N` at
   kernel-tail level). No chord name is used as a criterion.
3. Causal difference: with the same state and random stream, changing one weight changes
   the declared candidate ranking or placement.
4. Surrogate check, the one evaluation whose result changes a decision: compare candidate
   rankings from `e + q_f` against the actual mixed PCM for the same candidates. The result
   decides where the fast correlated path may be used; the front end's nonlinearity means
   formulas alone cannot answer it.
5. Real time and allocation at the declared load.
6. One author audition on the final configuration: whether sonic color beyond the single
   scalar is heard and whether the three controls are distinguishable. No value ranking.

## Temporal contract

This unit publishes, with timestamps, the support distribution `p`, `M`, `C` and `D` for
temporal consumers. Substitution, function, resolution, modulation and voice continuity
are designed in HV-4 on top of this contract.

### Endless harmony observation reservations

Endless scenes publish vertical support scans from sample zero, including while all
three valuation weights are zero. The temporal observer is separately reserved by
`observe_harmonic_relations(max_sources)` before time advances. The positive integer
bounds simultaneous Voice and Flora lane sources, including retiring sources; its
ceiling is the existing structural-record capacity of 38,400, not a musical limit.
Both signed nonzero temporal controls require this declaration; missing or late
reservation is a positioned script error. Finite scenes warn and ignore the declaration,
retaining event pre-scanning for observer and source capacity. Observation does not
activate either valuation weight. The declaration is an Author rule and does not
change the existing 600-second rolling history or the supported event density.
