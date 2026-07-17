# Habituation Field — Landscape-Level Adaptation

Status: Design, revised after external review (codex, 2026-07-17). Not
implemented. The review approved an assay/prototype phase conditional on the
blockers below; permanent core integration and the composer API stay gated on
the assay result. This design revisits — and will leave a verdict on — the
"Deferred" note in `docs/design-notes/lifecycle-time-domain.md` (it does not
pre-authorize the life-coupling continuum), and moves toward discharging the
largest ledger gap in `web/content/technote.md` §9.1 ("Adaptation and
expectation").

Scope: add a landscape-level habituation mechanism so that terrain the ecology
has been sounding into loses value over time and recovers after release —
"consonance is a meal, not a place" (technote §9.2). The consonance *evaluation
terrain* erodes under sustained perceived consonance; the physical spectral
analysis (H01/R01) is untouched. Assay precedes the core change, following the
Phase A "assay-before-model-change" precedent of the lifecycle work — with one
difference: no habituation mechanism exists yet, so the assay runs against a
feature-gated experimental implementation, not the current engine.

## Why this is necessary

Three independent arguments, all of which must hold for the mechanism to earn
its place rather than being merely nice to have.

1. **Model completeness (principle).** The landscape claims to *be* a model of
   perception (Manifesto: generation without symbolic intermediaries). Real
   perception habituates — stimulus-specific adaptation is textbook auditory
   cortex behavior: an unchanging percept fades from salience and recovers after
   withdrawal. A terrain that never adapts models a listener who never tires. It
   is not a missing feature; it is a known falsehood in the core claim. This is
   why the technote ledger marks it "the largest gap" — every other open row is
   "unimplemented," this one is "implements something false."

2. **Function (mechanism).** The consonance field is recomputed statelessly
   every hop from fresh H01/R01, so the terrain is static; the hill-climb settles
   on a consonant peak and parks there. Death/respawn does not fix this — the
   density field is static too, so respawns land in the same consonant places and
   re-instantiate the same texture. Temperature (movement tension) heats or cools
   the search but does not move the peaks. Only making the field time-varying
   *from within* breaks the fixed point; erosion-under-excitation is the minimal
   endogenous driver, because the sounding itself creates the pressure to move.

3. **Division of labor (architecture).** The Manifesto assigns the meso layer
   (phrase scale, ~8 s) to the ecology and the macro layer (scene, 15–30 s) to the
   human director. The meso layer currently has no mechanism, so stasis avoidance
   leaks up into authored pacing — the author is forced to do the ecology's job,
   breaking "no central conductor" at the meso scale. Habituation is the meso-layer
   mechanism that lets the author own only the macro boundary. The 2026-06-11
   flagship audition failed across pulse, harmonic arch, and *form*; meso-scale
   stasis is a leading cause of the form failure.

**The gate (technote §9.3).** Any terrain operation must pass both tests: a
perceptual mechanism must exist, and the production loop must close. Habituation's
perceptual mechanism is the best-attested of any terrain operation in the system.
Loop closure is the central design risk and is what the assay must prove. Crucially
(review blocker 1), closure is **not** automatic from a bounded state variable:
recovery of `h` only removes a *penalty*; it does not recreate a raw attractor.
The raw consonance peak `C_raw` depends on the voices' own spectra, so when a voice
leaves, the H/R evidence that built its peak may also vanish. The loop therefore
closes only when the raw attractor persists independently of the voice occupying it
— i.e. on a **shared Community terrain** sustained by other voices or external
input. A single isolated voice that is the sole excitation source cannot close the
loop and is outside the closure claim. If the loop does not close under the assay's
shared-terrain conditions, habituation is rejected for the same reason that killed
mirror dualism and `harmonic_tension`.

## Design decisions

Each decision below was chosen against alternatives; the rejected options and the
reason are recorded so the choice is not re-litigated. Decisions marked
**(assay-gated)** are provisional pending the assay.

### Point of action — erode the kernel output, not the physical analysis

The habituation state scales the **consonance evaluation terrain**
(`consonance_field_score` and `consonance_density_mass`), while H01/R01 remain the
untouched physical analysis. This is the literal reading of "consonance is a meal":
a place that has been eaten loses value, but perception itself does not vanish.
Consumers reading through the field (ListenerTwin, prediction) follow automatically.

- Rejected — erode H01: conflates "harmonic-fusion physical analysis" with
  "salience decay"; the harmonicity scan should stay a physical quantity.
- Rejected — separate independent scan read individually by each consumer: more
  wiring, and it desynchronizes the hill-climb terrain from ListenerTwin/prediction.

### Excitation input — perceived-consonance activity in root coordinate (revised)

**Revised from the original "raw `subjective_intensity`" after review blocker 2.**
`subjective_intensity[i]` is spectral energy at frequency bin `i`;
`consonance_field_score[i]` evaluates bin `i` as a candidate fundamental/root after
harmonic projection. They live on the same Log2Space grid but in different semantic
coordinates: a partial at `2f` helps build a consonance peak at `f`, so eroding the
raw energy bin `2f` would fail to habituate the actual pitch attractor at `f`
(worst for missing-fundamental and modal-body spectra). `subjective_intensity` is
also peak-picked into sparse bins (`landscape_spectral.rs`), so multiplying it
directly punches bin-width notches into an otherwise smooth terrain, causing
adjacent-bin chatter and grid-resolution dependence.

The excitation must therefore be expressed in the **field's root coordinate**. The
recommended drive is the **perceived-consonance activity**, the product ListenerTwin
already treats as perceived salience:

```
drive_raw[bin] = consonance_field_level[bin] * subjective_intensity_projected[bin]
```

where `subjective_intensity_projected` is the excitation mapped into root space via
the same harmonic template that builds the field (or, equivalently, driving `h` from
the perceived field activity rather than raw bin energy). This makes "what has been
sounding *as consonance*" the thing that erodes — matching the meal metaphor — and
keeps the drive and the eroded quantity in the same coordinate. The mild
self-reference (`field_level` appears in the drive) is bounded because `field_level ∈
[0,1]` and `h` is a slow leaky integrator. **(assay-gated: the exact projection and
whether to include `field_level` in the drive are resolved by the spectrum-matrix
assay below.)**

- Rejected — raw `subjective_intensity` at the same bin: coordinate mismatch (above).
- Rejected — occupancy scan (voice fundamentals): model-side and zero-latency, but
  ignores partials and external sound, and cannot be computed on the listener side
  where occupancy is an ecology-internal quantity.

### Decay form — relax toward the neutral baseline (revised)

**Revised from "multiplicative `score*(1-h)`" after review blocker 4.** Multiplying a
*signed* score by `(1-h)` has wrong edge behavior: for `score < 0`, raising `h` moves
the score toward zero, making a dissonant location *more* attractive to a consonance
maximizer (and voices do reach such places via temperature, dissonance objectives,
external input, partials, and fixed placement). Zero is also not a representation-
invariant baseline: at full erosion `field_level = sigmoid(-beta*theta) = 0.5` under
defaults, not 0.

The fix keeps the multiplicative spirit but relaxes the score toward its **level-
neutral baseline `theta`** (the sigmoid center, where `field_level = 0.5`):

```
field_score_eff[bin]  = theta + (field_score[bin] - theta) * (1 - h[bin])
density_mass_eff[bin] = density_mass[bin] * (1 - h[bin])      # non-negative, no sign issue
```

Full erosion drives a location to perceptual neutrality (`level → 0.5`), which is the
correct reading of "fades from salience" — a habituated percept reads as neither
notably consonant nor dissonant, not as maximally aversive. The baseline is thus
chosen deliberately (`theta`), discharging the review's "define the baseline" demand.
`density_mass` is already non-negative, so it keeps the plain multiplicative form.

- Rejected — subtractive `score - k*h`: erosion decoupled from score, and `k`
  reintroduces the roughness-scale calibration arbitrariness the consonance-density
  design explicitly avoids.
- Rejected — density only, field unchanged: existing voices stay on fixed peaks;
  satisfies only half the meso-layer requirement.

### State update — asymmetric leaky integrator, driven by a bounded transfer (revised)

The state is a per-bin `h ∈ [0,1]` advanced once per unique analysis interval.
Reuses the `exp(-dt/tau)` EMA *equation* from the per-voice `AdaptationContext`, but
does **not** hoist or generalize that type (review: it has two memories, normalized-
occupancy input, and familiarity/boredom weights; sharing one line of math is not a
shared abstraction under the anti-bloat rules).

```
drive[bin] = transfer(drive_raw[bin])            # -> [0,1], see below
tau        = (drive > h[bin]) ? tau_e : tau_r     # asymmetric: satiation vs recovery
h[bin]    += (drive[bin] - h[bin]) * (1 - exp(-dt / tau))
```

- **Bounded transfer (review blocker 3).** `drive_raw` is unbounded (derived from
  `ref_power`, the loudness exponent, and ERB-bin width). `transfer()` must be an
  explicit, gain-stable map to `[0,1]` (candidates for the assay: soft-knee
  saturation referenced to the same `ref_power`, or a running-percentile normalizer).
  Because the steady state is `h → drive`, any persistent floor in `drive` leaves an
  erosion floor and recovery is toward that residual, not toward 0. `transfer()` must
  define the zero-drive condition so recovery can reach the raw terrain again.
- `tau_e` is derived from `satiation_sec` and `tau_r` from `recovery_sec` in one
  place (the lifecycle "derive runtime rates once" boundary). The exact
  `satiation_sec → tau_e` mapping is fixed during the assay so that "stale in N
  seconds" is an observable contract with a stated reference-drive condition, a
  numeric "stale" threshold on the effective-field reduction, and a recovery endpoint
  tolerance.
- When habituation is disabled, `h ≡ 0` and the effective field equals the raw field
  exactly — current behavior is bit-for-bit unchanged.

### State scope — per-`Landscape`, global to the Community

`h` is a per-bin scan on `Landscape`, shared by all Populations of that Community.
Stimulus-specific adaptation is a listener-side phenomenon that does not distinguish
sources; a per-Population state would be "per-group memory," a different ontology.
There are effectively two `Landscape` instances (ecology on the habitat bus,
ListenerTwin on the presentation bus); each drives its own `h` from its own analysis
bus, so the presentation/habitat separation (acceptance criterion 14) is preserved
automatically.

### Time constants — `satiation(sec)` primary, `recovery(sec)` secondary

Physiology requires two constants: adaptation onset and recovery are distinct
processes (asymmetry is real and sets the limit-cycle duty ratio — fast satiation
with slow recovery gives sparse texture, the reverse gives dense). A single tau is
physiologically wrong. Usability is preserved by expressing both as seconds with
direct perceptual meaning (no dimensionless calibration, following the lifecycle
rate-archaeology fix) and by giving `recovery` a good default so the minimal
scenario sets only `satiation`.

- `satiation(sec)`: seconds until the terrain goes stale under a reference
  continuous excitation. Default ~5 s (the prediction window, per technote §9.2).
- `recovery(sec)`: seconds to recover after release. Secondary, with a physiology-
  motivated default. Unlike lifecycle's `recovery` (absence = disabled), habituation
  recovery cannot be "off" — permanent erosion drives everything dead — so it
  defaults rather than disabling.

**(assay-gated: exposing only two taus does not by itself guarantee a benign regime.
The assay must map the regime — see failure modes below — before these become the
final composer surface.)**

- Rejected — single sec + fixed internal ratio: kills the duty-ratio expressive axis.
- Rejected — fully internal (no composer surface): a good default already provides
  the simplicity; hiding it costs the expressive axis for no gain.

## Mechanism integration

### Persistent-state ownership (new, review)

`Landscape` is currently a cloneable "data snapshot"; the main loop drains to the
newest analysis frame and the listener path recomputes a fresh local frame and
discards it. `h` is *persistent state* and must not live on a disposable snapshot.
The design must name a persistent owner per bus (ecology and listener) and guarantee:

- `h` advances exactly **once per unique analysis interval**; draining multiple
  queued hops must advance by the summed elapsed `dt`, not once per drain and not per
  discarded frame.
- Parameter-only recomputation (`apply_pending_landscape_update` → `recompute_
  consonance` when kernels change) must **not** advance `h`.
- Resize (`resize_to_space`) and reset semantics for `h` are deterministic and
  covered by tests; `h` participates in the `_scan` length assertions (F1/F2).
- The listener `h` is not reset each time a presentation snapshot arrives.

### Per-hop update order (new, review)

`recompute_consonance` currently builds score, level, energy, density mass, and the
normalized PMF together. With habituation the order must be explicit so no derived
view goes stale:

1. Compute raw `field_score` and raw `density_mass` from the kernel.
2. Advance `h` once for the elapsed analysis interval (using the drive from step 1's
   level and the projected excitation).
3. Compute `field_score_eff` and `density_mass_eff`.
4. Derive `field_level` and `field_energy` **from `field_score_eff`**.
5. Normalize `density_mass_eff` into the PMF.

### Consumption and naming (revised, review)

Raw and effective quantities are **distinct named fields**, not a value that is both
"retained" and "replaced in place." CLAUDE.md defines `consonance_field_score` and
`consonance_density_mass` as the kernel formulas; those names keep meaning the raw
kernel output. The eroded views are new names (`*_eff`), and consumers are switched
to read the effective views:

- Movement: `evaluate_pitch_score_log2` / `evaluate_pitch_score` (the hill-climb
  chokepoint) reads `field_score_eff`.
- Spawn/respawn: `field_density_mass`, `SpawnStrategy::Field` branches, respawn peak
  bias read the effective mass/score.
- Prediction: `GeneratorModel::observe_consonance_field_level` observes the effective
  level.
- ListenerTwin: `observe_presentation_landscape` reads the effective level.

Consumers requiring explicit attention (review — the inventory the original spec
missed):

- **Exact leave-self-out** (`pitch_core.rs`) recomputes a *raw* kernel score scan,
  bypassing the evaluation chokepoint. It must apply the same erosion (share `h`) or
  the LOO path will read un-eroded terrain and silently defeat habituation for the
  exact-LOO branch.
- **Inverting placements** — `NegativeConsonance`, dissonance placement, edge
  placement (`community.rs`) — transform/invert the level; on eroded terrain a *stale
  consonant* location can become a desirable *dissonant* target. Behavior must be
  defined and assayed, not left implicit.
- **Field-derived modal patterns** read mass/level and inherit erosion.
- **Per-voice `AdaptationContext`** (boredom/familiarity) remains additive on top of
  global habituation. The combined system can double-count adaptation; the assay runs
  with per-voice adaptation both enabled and disabled to isolate the effect.

### Prediction bias (new, review)

The `perc_*` field is eroded, but the terrain predictor extrapolates from two
observed levels without modelling `h`, so it will show phase-conditioned
undershoot/overshoot at departure and at recovery reversals — and predicted level
gates phonation, forming another feedback path. The design does **not** add predictor
state now; the assay measures signed prediction error conditioned on `dh/dt` and cycle
phase, and only then is predictor `h`-awareness considered.

### Telemetry

JSONL report of raw vs effective (observed erosion amount) and the effective field
consumers actually read. The raw-vs-effective difference is **not** an `err_*`
quantity (`err_*` is reserved for `perc − pred`); it needs its own name.

## Composer API

- Tier: **Core** (perceptual seconds contract, peer to lifecycle
  `endurance`/`recovery`).
- Set at the scenario/director level (state is Community-global, so not
  per-Population); belongs with director terrain operations (technote §6.3.6).
- `satiation(sec)` primary (default ~5 s), `recovery(sec)` secondary (defaulted).
- **Opt-in**: active only in a scenario that sets `satiation`. Default is disabled
  (`h ≡ 0`, current behavior). Rationale: protect sacred étude outputs
  (`autumn_cycle` and the numbered samples) and avoid a default-on core behavior
  before the assay proves loop closure. Promotion to default-on is a separate
  decision made after the assay passes, recorded in the technote ledger.
- The same parameters apply to both the ecology and the listener `Landscape` (both
  are listeners; both habituate).
- **The API and permanent core integration are gated on the assay.** Only the
  feature-gated experimental path and the assay/acceptance criteria are authorized
  before the assay result.

## Assay and verification (before the core change)

Build `samples/research/habituation_field_assay.rhai` and its acceptance criteria
**first**, against a feature-gated experimental implementation. A controlled
open-loop drive trace is only a component test; production-loop closure must be
proven on the **real chain**:

`voice audio → habitat analysis → projected perceived-consonance drive → h →
effective terrain → movement/lifecycle → voice audio`

The assay must show, on a shared terrain (multiple sources / persistent input so the
raw attractor survives a voice's departure — see the gate above):

- **Causal control.** Identical seed, zero search temperature: habituation-off
  settles; habituation-on moves *because of erosion*, not crowding or random
  exploration.
- **Defined cycle events.** Source erosion → departure by a declared pitch distance
  → drop in that source's drive → measurable decline in `h` → recovery of the old
  effective basin's rank/prominence → **actual return** of a voice/cohort to the
  previously vacated basin. Return-to-vacated-basin, not mere movement, is the
  closure criterion.
- **Repeated recurrence.** Several cycles over a duration many times `max(tau_e,
  tau_r)`, with no secular drift in mean `h`, frequency centroid, occupied-band
  entropy, population size, or mean effective consonance.
- **Failure metrics (must stay bounded).** Migration synchrony (cohorts leaving the
  shared peak together), adjacent-bin reversal/chatter rate, deaths/extinction,
  density uniform-fallback count, total effective mass, PMF entropy, and time spent
  below viable consonance.
- **Robustness matrix.** Multiple seeds, population sizes, `tau_e:tau_r` ratios,
  movement cost/proposal interval, body spectra (sine / harmonic / modal / missing-
  fundamental — to prove the excited perceptual object and the eroded attractor
  coincide), signal gain, `ref_power`, `bins_per_oct`, anchored/Seq/Drone/external
  producers (outside the closure claim — confirm they degrade gracefully), and per-
  voice adaptation on/off.
- **Prediction/listener checks.** Signed prediction error conditioned on `dh/dt` and
  cycle phase; and whether ListenerTwin tension actually *rises* with staleness —
  this is not automatic, since `tension = (1 − stability) * resolvability` and uniform
  erosion can drive resolvability toward 0, which would *lower* tension.
- **Determinism.** Fixed seeds, fixed control-update mode, golden summary outputs.
  One golden trajectory proves reproducibility, not closure — closure is proven by
  the predeclared thresholds above.

The assay stays in the tree as the regression instrument for the core change.

## Vocabulary, telemetry, documentation

- Vocabulary: `perc_habituation_state_scan` (audio-derived → `perc_` per Axis A;
  `_scan` per the Log2Space convention). Effective views named `*_eff`; the raw
  kernel names keep their CLAUDE.md meaning. No new Axis B representation family.
- No-compat: alpha policy, but because habituation is opt-in and defaults to
  disabled, all existing samples are behaviorally unchanged.
- Docs (after the assay passes): update the technote §9.1 ledger row "Adaptation and
  expectation" to **"partial — landscape habituation implemented"** (not fully
  discharged: expectation and the larger DCC path remain independent), and the §9.2
  prose; add the Rhai Book API; regenerate `.d.rhai` and `api.md` from the docs
  registry; record the outcome in the lifecycle design note's deferred section,
  including an explicit verdict on whether habituation actually required the life-
  coupling continuum.

## Non-goals

- The life-coupling continuum / mortal-terrain generalization of the brain axis.
  Review confirms the separation is clean: habituation is listener/landscape memory,
  not a new articulation-life architecture, and does not require the mortal-terrain
  continuum. This design **revisits and will leave a verdict on** the deferred
  question (per the lifecycle note's request); it does not supersede or pre-authorize
  the generalization.
- Default-on habituation. Deferred to a post-assay ledger decision.
- Closing the full DCC biosignal loop (`coupling_strength`); habituation only makes
  ListenerTwin tension reflect staleness, one fragment of that larger direction.
- Closure for fixed/exogenous producers (Seq, Drone, anchored, external input) that
  do not respond to the terrain; they are outside the closure claim and only need to
  degrade gracefully.
