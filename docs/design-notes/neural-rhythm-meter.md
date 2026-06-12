# Neural Rhythm and Meter Design

Status: Phases 1-4 landed (emergent meter core, accent/measure shaping, coupling-
continuum composer API, backend collapse); musical audition of the curated demos
is the remaining acceptance step. In scope for v0.4.0
Scope: conchordal rhythm architecture, DCC time-axis coupling, composer API
Date: 2026-06-04 (Phase 1-2 outcomes appended 2026-06-05; Phases 3-4 landed 2026-06-10)

Phase 3-4 outcome (2026-06-10): the externally-imposed fixed clock is gone.
Production timing is now a per-voice phase-coupling clock (`CouplingClock`) that
entrains to the shared production meter; one `coupling` strength spans the
flow / entrained / metric continuum. The composer API is `metric()` /
`entrained()` / `flow()` presets plus `entrainment()` / `rhythm_role()` /
`microtiming()` modifiers, with director-level `meter_stability()` (attractor
depth) and `temporal_basin()` (tempo region) shaping the emergent meter. Accent
role emits a stronger onset that drives the meter, so the measure subharmonic
can emerge from a recurring downbeat (the Phase-2 gap). `metric_beat(Hz)` /
`entrained_beat(Hz)` / `flow_timing(Hz)` / `beat_strength` / the `FixedRate`
backend are removed.

This memo records the working design position for conchordal's rhythm layer
before implementation. It is the implementation-facing companion to
`docs/design-notes/dcc-neurocognitive-hierarchy.md`, narrowing Layer 2 (rhythm
entrainment) the same way `docs/design-notes/listener-twin.md` narrows Layers 1,
3, and 4.

It supersedes the tactical rhythm fixes recorded in
`docs/roadmap/v0.4.0-demo-audition.md` ("Engine fix landed 2026-06-03"): those
fixes made the families measurably distinct but did so with an externally
imposed fixed clock, which this memo argues is the wrong foundation.

## Core Position

Rhythm in conchordal should be an emergent property of neural-rhythm
entrainment, not an externally imposed clock. A legible, stable beat is a deep
metrical attractor in a network of coupled neural oscillators, not the output of
a metronome. The composer shapes the temporal landscape (how deep the attractor,
roughly where it sits, which levels are emphasized); the population settles into
it. This is the time-axis analogue of the existing consonance landscape on the
frequency axis.

## The Problem With the Current Foundation

The 2026-06-03 fix introduced `PhonationClockConfig::FixedRate { rate_hz }` and
routed the metric-beat family through an isochronous wall-clock grid anchored to
absolute tick 0. It also keeps `metric_beat(beat_hz)` as the scripting surface.

This works perceptually but contradicts the Manifesto on three points:

- The Manifesto rejects "grids of symbols" as the origin of music. `beat_hz` is a
  number placed on the time axis: a symbol grid.
- The Manifesto states rhythm should form "spontaneously rather than through
  externally imposed meter" and that there is "no central conductor." A fixed
  external clock is exactly an imposed meter and a conductor.
- DCC means the engine should be a direct model of music cognition. A wall-clock
  grid is a DSP scheduler, not a cognitive model.

The deeper diagnosis: the perceived "shared wobble" across families was **not**
caused by the absence of a clock. It was caused by the absence of a stable
metrical attractor. A single adaptive oscillator (`AdaptiveRhythm`) has no
metrical hierarchy and no stable limit cycle, so its entrainment well is shallow
and drifts with density and texture. Adding an external clock hides this; it does
not fix the model. The correct fix is to give the oscillator network the
structure that makes a beat self-sustaining.

## Neuroscience Mechanism (the level conchordal needs)

The bands (delta, theta, alpha, beta) are distinct oscillatory systems, not
harmonics of one fundamental. Their relationships are built by coupling, and two
mechanisms matter for music:

### Regime A: Integer-ratio mode-locking within the pulse range

Within the perceptible pulse range (roughly 0.5-8 Hz), coupled oscillators
mode-lock at integer ratios (2:1, 3:1; Arnold-tongue mode-locking). This is the
basis of Large's neural resonance theory of meter: a network resonates at the
beat and its metrically related layers, and meter emerges as a resonance pattern
without an external grid. The brain also imposes meter endogenously, adding a
slow **subharmonic** of the beat for the measure level (Nozaradan et al. 2011),
rather than adding faster harmonics.

Mapped to conchordal:

- delta (~1-4 Hz): beat / tactus.
- theta (~4-8 Hz): subdivision / tatum (and the syllabic chunking analogue;
  Giraud & Poeppel 2012).
- measure: a slow subharmonic of delta, internally generated.

These are genuine entraining oscillators in integer-ratio relationships, and
integer-ratio phase coupling is the right mechanism **here only**.

### Regime B: Phase-amplitude coupling for accent and groove

Across to faster bands, the dominant mechanism is phase-amplitude coupling
(PAC): the phase of a slower oscillation modulates the amplitude/precision of a
faster one (Lakatos et al. 2005 oscillatory hierarchy; Lisman-Jensen
theta-gamma). The faster bands are **not** faster pulse layers:

- beta (~13-30 Hz) is an auditory-motor timing/prediction signal whose power
  drops after an onset and rebounds in anticipation of the next beat (Fujioka et
  al. 2012). Its envelope is modulated at the beat (delta) rate. This is the home
  of **groove / microtiming**, expressed as modulation of the beat frame.
- alpha (~8-13 Hz) is attentional gating/inhibition. The musical function the
  Manifesto assigns to it ("pulse and accent within phrases") is better realized
  as the beat phase weighting event salience (**accent**), not as an 8-13 Hz
  pulse oscillator.

So the Manifesto's four-band intent is honored, but alpha and beta are realized
as **modulators of the beat frame**, not as independent faster oscillators. This
is both more correct and more economical (no extra pulse oscillators to add).

### Regime C: Entrainment is not evoked

Spectral power at the beat frequency can be produced by a repeated evoked
response, not only by intrinsic entrainment (Nozaradan; Doelling & Poeppel 2015;
caution in Damsma et al. 2025). Because conchordal claims to be a *direct*
cognition model, the listener-side salience must encode the signatures of true
entrainment: phase prediction, persistence through silence, anticipation (phase
leading the stimulus), and response to omitted events. Salience must not be raw
drive amplitude. This is the one place where scientific rigor is a hard product
requirement, not an approximation.

### What conchordal does not need

- Biophysical fidelity (ion channels, exact PAC spectra, gamma band).
- Treating every band with the same mathematical object. The hierarchy memo's
  Practical Rule applies: "do not force the same mathematical object onto every
  layer." Regime A (mode-locking) and Regime B (PAC) are deliberately different
  mechanisms, and that difference is the design, not a compromise.

## Design: The Temporal Landscape

conchordal already has a frequency-axis landscape: a consonance potential field
with peaks that agents hill-climb. The rhythm design makes the time axis
symmetric.

| | Frequency axis (existing) | Time axis (this memo) |
|---|---|---|
| Field | consonance potential | metric potential (phase attractors) |
| Generator | roughness / harmonicity kernels | coupled neural-oscillator network |
| Agent action | hill-climb consonance | entrain onset phase to the meter |
| Env <-> inhabitant | density deforms terrain | onsets drive oscillators; field guides onsets |
| Attractor depth | sharpness of consonance peak | metric stability (groove depth) |

A stable beat is a deep metrical attractor (the network settled into a
mode-locked lattice). A wobble is a shallow/unsettled field. Flow is agents that
do not entrain (renewal). The same mutual environment <-> inhabitant loop the
Manifesto describes for consonance applies to time.

### Engine

A small network of coupled nonlinear oscillators implementing resonance-theory
meter:

```text
dphi_i/dt = omega_i
          + sum_j K_ij * sin(m_ij * phi_j - n_ij * phi_i)   # Regime A: n:m mode-lock
          + input_i(t)                                       # acoustic drive (flux)
```

- Oscillators carry a stable limit cycle (amplitude is a state, not just driven),
  so the beat is self-sustaining and persists through gaps (Regime C).
- delta and theta are coupled at integer ratios; the measure is a slow
  subharmonic of delta. The ratio is selected by resonance (the ratio with the
  best phase coherence wins), not by a comb filter or an external tempo number.
- Slow plasticity of `K_ij` lets the network learn and hold the active meter.
- `input_i(t)` is the acoustic drive (spectral flux from `DorsalStream`). Drive
  is the force term only; it is not the salience.

### Accent and groove (Regime B)

- accent: the beat (delta) phase weights subdivision-event salience.
- groove / microtiming: a beat-phase-modulated timing offset and precision on
  fast events. Realized as PAC over the beat frame, not as a fast oscillator.

### Salience and confidence (Regime C)

Each band reports a confidence derived from phase prediction quality (a
PLV/coherence measure over a window, plus persistence/anticipation), not from
drive amplitude. This drives both UI brightness and behavioral weight.
Consequences:

- A clean isochronous beat reads strong even when quiet.
- A wobbling or ambiguous beat reads weak even when loud.
- A non-metric texture (rain) reads weak in all bands: the correct null result.
- Content gating is automatic: theta confidence stays low when there is no
  subdivision content to lock to. There is no separate gating rule.

Names must keep drive and confidence distinct so reports and GUI do not blur the
two. Use `attention_level` for onset / spectral-flux salience (drive), and
`beat_confidence` / `subdivision_confidence` / `measure_confidence` for
entrainment confidence per metrical level. Retire `theta_mag` / `delta_mag`
style names, which are ambiguous between flux magnitude and entrainment
confidence.

## Families Collapse Into One Continuum

The three families are not separate backends. They are regions of one axis:
coupling strength / attractor depth.

- Metric: strong coupling, deep attractor. A stable beat emerges (a groove fully
  locked in), not a metronome.
- Entrained: medium coupling. Synchronization rises over time; the lock-in
  process is itself the audible/visible content.
- Flow: weak coupling / renewal. Agents do not entrain; low confidence in all
  bands.

This replaces the `FixedRate` vs `ThetaGate` vs renewal split with one mechanism
parameterized along one perceptual axis. The named families stay as presets over
this continuum, not as separate backends:

```text
flow:      meter_stability low,    entrainment free,          confidence low
entrained: meter_stability medium, entrainment loose..joined, confidence grows over time
metric:    meter_stability high,   entrainment locked,        stable beat emerges
```

## Composer API

Goal: a composer learns the small core concept (two cognitive landscapes; agents
that seek consonance and entrain; directing the ecosystem over time) and goes
straight to composing. The vocabulary must map transparently onto those concept
primitives so that using it teaches the model. It is deliberately not dressed in
everyday or genre vocabulary (no `walking` / `dancing` tempo labels, no `triple`
/ `four` meter labels): hiding the model behind colloquial words both betrays the
stance in `alpha_guide.md` ("not trying to hide the model behind common
music-production vocabulary") and smuggles the symbol grid back in.

The scenario author never sees neural or engine vocabulary (Hz, PLV, Kuramoto
coupling, `FixedRate`, `ThetaGate`); those stay in the engine, reports, and debug
views.

Field level (Scenario / director), symmetric to consonance-field operations:

- `meter_stability`: attractor depth -- how strongly a pulse wants to form. A
  property of the temporal terrain, not a tempo.
- temporal basin: the time scale the pulse gravitates toward, expressed as the
  time-axis analogue of `density(min_hz, max_hz)` -- a period/rate region that
  shapes the terrain (a basin), never a scheduled `beat_hz`. Numeric ranges are
  fine here for the same reason `density` ranges are: they parameterize a field,
  they do not place events.
- `layer_emphasis`: relative salience of metrical levels (beat / subdivision /
  accent).

Voice / population level, expressed as concept primitives:

- `rhythm_role`: which level of the metrical hierarchy the voice expresses
  (beat / subdivision / accent / texture).
- entrainment coupling: how strongly the voice locks to the emergent pulse
  (free <-> locked) -- the agent's coupling strength, a model quantity.
- microtiming: signed timing deviation / elasticity (the Regime B beat-phase
  modulation), as an amount, not a genre label.

A voice does not write `beat_hz`. The beat location and tempo emerge from the
population's mutual entrainment and the temporal-basin terrain. Metrical grouping
(duple / triple / ...) is not a composer symbol: it emerges, or arises from
inter-voice relationships and accent emphasis. "Keep simple things simple": the
minimal voice just joins with default coupling; role / coupling / microtiming are
opt-in. The named families (`metric` / `entrained` / `flow`) are presets over the
continuum (see "Families Collapse Into One Continuum"), not separate backends.

`habitat_bus` and `presentation_bus` remain public composer routing (audible
output vs hidden scaffold); that is a legitimate compositional choice and is not
removed. Only `ListenerTwin`'s input is type-locked to presentation-derived
analysis (see `listener-twin.md`), so the twin cannot subscribe to a hidden bus.

## Perception vs Production: Two Engine Roles

The two existing `RhythmEngine` instances map cleanly onto two cognitively
distinct roles. They must not be conflated, and the perception engine must not
form a closed self-driving loop.

- ListenerTwin engine = pure perception. Reads the presentation bus. Runs the
  full meter model (Regime A + B + C). This is the GUI mandala and the
  listener-side confidence. It must not be fed generator vitality or habitat
  scaffold state (see `listener-twin.md`).
- Generator-side engine = production (auditory-motor coupling). Only the
  Entrained family uses it to time onsets. It is the auditory-motor model: a
  voice's motor oscillator couples to the perceived beat. Keep its self-driving
  feedback weak so it does not re-create the closed-loop wobble.
- Metric and Flow do not use generation clocks at all in the final design; their
  surface emerges from coupling strength on the same population substrate.

## What This Replaces

- Remove `PhonationClockConfig::FixedRate` as the metric foundation.
- Remove `metric_beat(beat_hz)` / `beat_hz` from the scripting surface (no
  backward-compat shim; alpha policy).
- Fold the existing delta IOI band-routing hack (in `RhythmEngine::update`) into
  the perception engine's mode-locked coupling.
- Replace drive-magnitude salience (`mag` from `tanh(500*flux)`) with
  entrainment confidence for UI and behavior.

## Phased Implementation

Land the well-supported core first; defer speculative layers.

1. **Perception meter core (Regime A + C) -- DONE (2026-06-05).** A coupled
   beat (tactus) limit-cycle oscillator entrained by onset drive, integer-ratio
   subdivision detection, and windowed-PLV entrainment confidence, wired into the
   ListenerTwin perception path and reported. The **measure subharmonic and
   grouping selection moved to Phase 2**: they are degenerate from onset timing
   alone (a beat-only stream supports every harmonic equally), so they need the
   accent layer's off-beat support to be real rather than inert. UI brightness is
   not yet driven from confidence (no headless way to verify); reports carry it.
2. **Accent and groove (Regime B) -- measure layer landed (2026-06-05).**
   Endogenous slow-subharmonic measure detection driven by exogenous accent:
   onset strength relative to a leaky baseline accumulates an accent-weighted
   resultant at candidate subharmonic phases, and the measure ratio is the
   argmax, gated so an unaccented beat induces no measure. Exposed as
   `measure_hz / measure_ratio / measure_confidence`. Subdivision microtiming and
   the alpha/beta intent mapping remain for later in this phase.
3. Composer API: replace `metric_beat(Hz)` with relational voice verbs and field
   `meter_stability` / tempo-region / layer-emphasis operations. Rewrite the
   sample `.rhai` files.
4. Family continuum: drive metric/entrained/flow purely from coupling strength on
   the production substrate; remove `FixedRate` and `ThetaGate` special cases.

Each phase ends with the mandatory cargo test/clippy procedure and a re-audition
of the three families plus the integration demo.

### Phase 1 Outcome (2026-06-05)

Implemented as an isolated `src/core/meter.rs` (`MeterNetwork`), driven from the
existing `salience_level` (`tanh(500*flux)`) inside `ListenerTwin`, exposing
`beat_hz / beat_phase / beat_confidence / subdivision_ratio /
subdivision_confidence` on `ListenerState`. The production path (`DorsalStream` /
generator `RhythmEngine`) is untouched; this is perception only.

Confidence separates the families, and does so independently of drive. On the
metric/entrained/flow demos, read as a **late-stable** statistic (the honest
readout; see below):

| Family | beat confidence (late mean) | peak | drive (attention) |
|---|---:|---:|---:|
| Metric | 0.50 | 1.00 | ~0.21 |
| Entrained | 0.08 | 0.47 | ~0.15 |
| Flow | 0.035 | 0.45 | ~0.03 |

Metric and Entrained carry nearly the same drive yet differ ~6x in confidence:
the Regime C requirement (salience = entrainment, not loudness) holds on real
material.

Calibration decisions:

- **Read confidence as a trajectory, not a grand mean.** Entrainment is a
  process; a run-long mean blurs warmup, settled lock, and voice die-off. The
  report emits a `listener_confidence_summary` record (peak + late-window mean).
  Late-stable mean is the headline statistic; peak is noisier (it captures
  lucky finite-sample alignments) and barely separates entrained from flow.
- **Do not scale confidence to flatter the metric demo.** Metric reading ~0.5
  (not ~1.0) is honest: that demo mixes 2 Hz and 3 Hz layers over a drone, so the
  listener hears an ambiguous meter. The confidence *scale* is correct (a clean
  synthetic isochronous beat reads ~1.0). Raising the demo reading is a
  composition fix (present a cleaner pulse), not an engine gain.
- **Presence gate requires ~4 onsets of evidence** before confidence is trusted
  (`smoothstep(1.0, 4.0, plv_count)`), matching beat induction needing a few
  cycles. This suppresses low-count spurious confidence but does not erase it.
- **Residual: finite-sample PLV bias.** A leaky window of ~6 onsets gives random
  phases a baseline resultant of ~0.2-0.3, so flow's *instantaneous* early
  confidence flickers up before phases spread. The late-window reading excludes
  this, so the *reported* number is honest. A PLV bias correction (e.g. PPC, an
  unbiased pairwise estimator) is the principled fix but is deferred until
  instantaneous confidence has a consumer (live UI); there is no current headless
  way to verify a real-time display.
- `theta_mag` / `delta_mag` are **retained alongside** the new `beat_confidence`
  for now; their retirement (per "What This Replaces") is a UI/behavior migration
  belonging to a later phase, not done here.

### Phase 2 Outcome (2026-06-05)

The measure layer accumulates an accent-weighted resultant at candidate
subharmonic phases of the beat (`TAU * beat_cycles / M`, `M in {2,3,4}`), where
accent is onset strength above a leaky baseline. The ratio is the argmax
magnitude, and `measure_confidence` multiplies the resultant's coherence by an
accent-presence gate and the beat presence. Two unit tests pin the mechanism: an
alternating strong/weak 2 Hz stream yields `measure_ratio == 2`, and a uniform
2 Hz beat yields no measure (`< 0.1`) while still locking the beat.

On the three demos the measure layer stays **correctly near-silent**:

| Family | measure conf (late mean) | peak | late ratio winner |
|---|---:|---:|---|
| Metric | 0.033 | 0.41 | none stable ({3,4,2} spread) |
| Entrained | 0.000 | 0.39 | none |
| Flow | 0.000 | 0.03 | none |

This is honest, not a miss. The demos carry **no measure-level accent** -- no
downbeat that recurs every N beats. In `metric_beat_foundation`, the `downbeat`
(`beat_strength 0.80`) and `consonant_pulse` (`0.55`) both fire on the *same*
2 Hz grid, so there is no beat-to-beat strength alternation to seed a slow
subharmonic; and the single downbeat voice is further washed out under the
eight-voice 2 Hz cluster in the global spectral-flux salience. The metric demo's
sporadic ratio hits spread across {3,4,2} -- the signature of finite-sample
noise on residual flux fluctuation, not a stable meter. Subdivision, by
contrast, *is* alive and family-specific (late `subdivision_ratio` is dominated
by 2 for Metric/Flow and by 3/4 for Entrained).

Calibration decisions:

- **Do not lower the accent gate to manufacture a measure on these demos.** Per
  the design-first principle, a silent measure on accent-free input is the
  correct behavior, not a band-aid target. Spurious low-confidence ratio flicker
  is suppressed by the accent-presence gate, not by forcing a winner.
- **This is the empirical motivation for Phase 3.** The composer API needs a verb
  to place a *recurring downbeat* (measure-level accent at `beat_hz / N`),
  distinct from per-onset `beat_strength`. Today an author can vary strength but
  cannot express "accent every other beat," so no measure can emerge from the
  current samples.
- **Endogenous subjective metricization is out of scope here.** Listeners impose
  meter on a truly isochronous, unaccented sequence (Brochard et al. 2003;
  Nozaradan's imagined-meter SSEPs). That is a separate, deeper mechanism than
  the exogenous-accent layer built in Phase 2; it is noted as a future Regime-B
  extension, not attempted now.

## Non-Goals

- Do not model gamma, biophysics, or exact PAC spectra.
- Do not add alpha/beta as independent pulse oscillators.
- Do not let the perception engine drive its own input (no closed self-loop).
- Do not reintroduce an external fixed clock or a `beat_hz` symbol as the
  foundation.
- Do not preserve backward compatibility for removed clock configs.
- Do not expose neural / engine vocabulary (Hz, PLV, Kuramoto, `FixedRate`,
  `ThetaGate`) to scenario authors. Keep `habitat_bus` / `presentation_bus` as
  composer routing; type-lock only the `ListenerTwin` input.
- Do not let composer-facing priors (`meter_stability`, the temporal basin,
  `layer_emphasis`) accumulate into a de-facto grid. Keep them low-dimensional
  and genuinely soft; emergence must still do real work.
- Do not dress the vocabulary in everyday or genre wrappers (no `walking` /
  `dancing` tempo labels, no `triple` / `four` meter labels). The primitives must
  map transparently onto the concept; learning the concept is the on-ramp.
- Do not expose metrical grouping (duple / triple / ...) as a composer symbol.
  It is an emergent subharmonic of the attractor, not a seeded meter.
- Do not extend this redesign into a frequency-side API refactor. A unified
  `field.harmony()` / `field.time()` grammar is a possible future north star,
  out of v0.4 scope.

## Open Questions

- Confidence metric: is windowed PLV sufficient, or is an explicit omission/
  anticipation test needed before coupling confidence into behavior?
- Prior budget: where is the line between shaping the attractor and imposing
  meter? Define a headless check that confirms emergence still does real work
  once `meter_stability` + the temporal basin are both active. (The basin
  resolves "how to express a tempo region without Hz" as a soft preference, not
  a target.) Metrical grouping is intentionally left to emergence, not a seeded
  meter symbol; the same check must confirm grouping arises as a subharmonic
  rather than being implicitly forced by the basin shape.
- Entrained production path: does the Entrained family time onsets via weak
  `DccCoupler` pressure, or via a separate generator-side auditory-motor engine?
  The two-engine split (perception vs production) is not yet resolved.
- Cold-start: emergent rhythm needs audible output to entrain to, but audible
  output needs rhythm to start. Reconcile with the bootstrapping path in
  `listener-twin.md`.
- Production coupling strength: what keeps the generator-side engine from
  re-creating the closed-loop wobble while still allowing emergent entrainment?
- DccCoupler applies resolvability twice: `tension_level =
  (1-stability)*resolvability` (`listener-twin.md`) then `tension_pressure =
  tension_level * resolvability * coupling_strength` (`src/dcc_coupler.rs:32`),
  giving `resolvability^2`. Collapse to a single resolvability term before
  raising `coupling_strength` above 0. Currently inert (default 0.0).

## Reference Anchors

- Resonance theory of meter: Large & Snyder 2009, Ann NY Acad Sci; Large &
  Kolen 1994.
- Beat/meter tracking: Nozaradan et al. 2011, J Neurosci, DOI
  `10.1523/JNEUROSCI.0411-11.2011`; Doelling & Poeppel 2015, PNAS, DOI
  `10.1073/pnas.1508431112`.
- Oscillatory hierarchy / PAC: Lakatos et al. 2005, J Neurophysiol, DOI
  `10.1152/jn.00263.2005`; Lisman & Jensen 2013, Neuron (theta-gamma).
- Auditory-motor beta: Fujioka et al. 2012, J Neurosci, DOI
  `10.1523/JNEUROSCI.4107-11.2012`.
- Speech delta/theta nesting: Giraud & Poeppel 2012, Nat Neurosci, DOI
  `10.1038/nn.3063`.
- Entrainment vs evoked caution: Damsma et al. 2025.
