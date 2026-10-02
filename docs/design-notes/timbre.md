# Timbre: Bodies Under a Listener Model

Status: Design (v0.5 direction), revised 2026-09-28. This version replaces
"Phenotype, Excitation × Resonance, State Leakage" (2026-07). Consequences 1–2 of
that version are implemented: energy-driven `damping` in the scheduled harmonic
path, and body-ratio leave-one-out in `ApproxHarmonics`. Whole-body fitness is in
progress under the [body-aware fitness plan](body-aware-fitness-plan.md)
(2026-09-26), which separates source-level self-exclusion from whole-body
candidate evaluation; the ratio-based LOO is not whole-body fitness. The
principles below are adopted; the decisions of 2026-09-29 are marked where they
apply; everything else is open and sequenced in the
[implementation plan](../superpowers/plans/2026-09-29-timbre-synthesis.md). The
revision reconciles two independent answers (Claude, Astra) to the author's
question below, and a second Astra review of the decisions and plan. The code
changes of 2026-09-29 (valuation-side roughness aversion, habitat-only meter
onsets) live on branch `timbre-valuation-meter` until the next integration point.

## The question

Auditory cognition predates music. Music arose where that cognition met the
sounds bodies drew from instruments, and instruments bound those sounds to their
physical structure. Most electronic synthesis still reproduces that structure:
integer-ratio partials, ADSR envelopes. Conchordal need not. Under Direct
Cognitive Coupling (DCC), sound may address auditory cognition directly. Hearing
probably adapted to natural sounds, so structure taken from them is not out of
place; but natural sounds are not instrument sounds. What, then, should
Conchordal's synthesis be?

Two corrections sharpen the premise before it is used.

- **Not every constraint is physical.** Integer-ratio partials are not an
  instrument convention: any periodic vibration has them, whatever the material
  (a Fourier series). They mark where periodic natural sources, voices above
  all, meet the periodicity-based pitch mechanism. ADSR is not physics either;
  it is a 1960s analog-synthesizer control convention fitted to a keyboard gate.
  Per-note voice allocation, waveform templates that transpose with f0, and
  perfectly periodic oscillators belong to the same layer. That layer carries
  the note, the symbolic grid the Manifesto rejects, into synthesis.
- **Freedom from instrument physics is not new.** Electronic music has long
  built perceptually designed, non-physical sound, including timbres and
  tunings designed together: Stockhausen's *Studie II* (inharmonic sine
  mixtures on a 25th-root-of-5 scale), Pierce's synthetic timbres made
  consonant in 8-TET, Risset's inharmonic tones, Chowning's *Stria*
  (golden-ratio spectra and scale). Sethares formalized the timbre–scale
  correspondence. Conchordal's contribution is not unheard timbre. It is a loop
  in which what is heard changes how sources move, survive and reproduce.

## Principle

A Voice's synthesis is **a body in a world whose physics is a listener model.**
Sims's virtual creatures evolved their morphology under Newtonian physics;
Conchordal's bodies take shape under auditory cognition as the Landscape models
it. Material physics is dropped as a law, and the listener model takes its
place. Three sources of structure stay apart (not to be confused with the
listener-model layers of technote ledger §9.3.56):

1. **Material contingency**: strings, tubes, bars and their losses. Not a law
   here. The instrument tables in `src/core/mode_pattern.rs` (`xylophone`,
   `vibraphone`, `wine glass`) may seed founders; they are not a destination.
2. **Regularities of natural sound that hearing exploits**: periodicity, common
   onset and coherent envelopes within one source, ring-down after excitation,
   envelope statistics of textures. They enter synthesis as the grammar that
   lets a body be heard as a source and as initial conditions for search, never
   as a second copy of what the Landscape already encodes (ERB spacing,
   harmonic templates).
3. **The listener model**: R, H, the meter, and the survival policy C built on
   them. It stands for the human in the room, so the population's own sound
   does not rewrite it; it changes only through the scenario or through modeled
   listener dynamics such as habituation. The Manifesto keeps it replaceable
   across cultures and individuals; templates that adapt to the population are
   a separate research line.

Two cautions keep the listener model honest:

- **H carries an integer-ratio prior; it is not an established neural
  mechanism.** Pitch draws on both resolved components and temporal cues, and
  human cortical pitch regions respond mainly to resolved harmonics
  (Norman-Haignere et al. 2013). Fusion of consonant intervals appears even
  where consonance is not preferred (McPherson et al. 2020), which supports
  modeling fusion; it does not settle how H should compute it.
- **C is a chosen survival policy, not a measurement.** R and H describe part
  of perception. Making low consonance starve is a design decision, and the
  preference for consonance varies across cultures (McDermott et al. 2016).
  Whatever timbre the ecology selects is C's verdict under this model, not a
  finding about human ears.

## What follows for the body

### Timbre is free; the listener is fixed

R is defined for any partial set; H applies its integer prior. An inharmonic
body therefore does not escape the listener: R's valleys move with the spectrum,
while H pulls toward periodicity. Instruments explored this tension only in
narrow corners: bells tuned by shaving metal, piano octaves stretched for string
inharmonicity, gamelan scales that Sethares relates to metallophone spectra.
Here the body's partial ratios, spectral slope and decays are ecological traits,
a phenotype; the listener model is not.

### Import the process, not the product

Instruments are fossils of a slow co-adaptation between timbre and interval
under human hearing. Conchordal should run that process inside the ecology
instead of importing its products (integer spectra, ADSR, fixed tables). This
makes heredity of timbre the center of the design, but it comes late in order:
selection needs a body-level fitness first, or timbre only drifts.

### One sound, three computations

The previous version had one partial set radiate, perceive and identify. These
are three separate computations:

- **Radiation**: the body's actual output. On the habitat bus it passes through
  the NSGT into R/H/C and deforms the terrain every Voice reads. This already
  works and must be kept.
- **Body fitness**: how the body would fit if it sounded here, evaluated over
  the whole body in the shared listener model. It does not give each Voice its
  own ear.
- **Source identity**: what a listener attributes to one source. It belongs to
  the listener side (the source-state contract in the technote ledger: body
  candidates, excitation, sounding state, attribution hypotheses) and is
  inferred from presented audio. It is never handed the generator's Voice IDs
  or true ratios.

### Input, persistent state, output

A body's grammar is state-space: input drives a persistent internal state that
radiates. The reason to keep it is not that strings and larynxes work this way.
Hearing tracks sources by this regularity (a body keeps sounding after its
excitation stops, and the next excitation adds to its tail), and the
listener-side contract uses the same state equation. Nor must every body be a
passive resonator; a self-sustaining oscillator is also valid if the ecology
pays for sustaining it. Parameters are free of material law: decay need not
shorten with frequency, and mode ratios need not be physically realizable.

ADSR leaves the definition of timbre. Where an onset or offset shape is needed,
it becomes excitation control; click prevention and output gain stay as output
processing. `OscillatorBank` and `ModalEngine` both remain. What gets unified is
the meaning of input and state, not waveform or drive type. Today the meanings
differ: `Sine` has no drive and fakes an attack by lifting output gain
(`sine_impulse_boost`), `Harmonic` takes deterministic drive and `Modal` noisy
drive (`DriveMode`, `src/life/sound/any_backend.rs`).

Per-note allocation is not by itself a bug. Each `ToneCmd::On` creates a new
`Tone` keyed by `(source_id, tone_id)` (`src/life/schedule_renderer.rs`). For a
linear modal body with fixed coefficients, overlapping tails superpose exactly
as re-exciting one body would, as the ledger's resonator check confirms. The
question is how far this allocation can represent one continuing source:

- the ADSR release cuts tails with a gain curve;
- after a change of pitch or coefficients, the old tail rings on as another
  `Tone` at the old pitch, so one individual can sound two pitches;
- fluctuation is seeded per `Tone` (`modal_phase_seed(source_id, onset,
  tone_id)`), so a Voice's micro-fluctuation restarts at every note instead of
  continuing.

`ToneCmd::On / Off / Update` must state what they start, stop or change:
excitation, internal state and audibility are three different things.
These are questions about the current implementation, not an adopted monophony
requirement. The revised Task 2 proposal retunes only open excitation handles;
ordinary new phonations create new Tones while closed tails keep their last
coefficients. That deliberately permits old/new pitch overlap across phonations.
Its Off coefficient policy, lowering rules and capacity envelope still require
author decisions A1–A3; no renderer change follows from the proposal alone.

### An observability map replaces the two timbre domains

The previous version split timbre into an ecological domain (partials, spectral
envelope) and a presentation domain (transients, noise, space). That fixed split
is withdrawn. Transients already reach the ecology through the spectral-flux
onset signal that drives the meter, and texture recognition depends on subband
envelope statistics, not only on the mean spectrum (McDermott & Simoncelli
2011). "Noise is decoration, partials are ecology" was a property of the current
analysis, not of timbre.

In its place, keep an observability map: for each feature of a body, what the
current analysis detects, and which consumer (movement, metabolism, respawn,
meter, `ListenerTwin`) it returns to. The test the previous version applied to
new couplings (a perceptual mechanism exists, and the production loop closes)
now governs every piece of synthesis structure:

- A feature that is detected and returned to behavior is an ecological trait;
  the ecology finds its value.
- A feature that is not detected stays fixed at a setting chosen by audition and
  does not vary as a heritable trait. A change a person hears but the ecology
  never evaluated is a change outside DCC.
- The map describes the current model, not the scope of timbre. When the
  analysis loses a difference that matters, add the mechanism that detects it,
  and only that one.

The map as of this revision:

- **Partial positions and spectral envelope**: detected by R/H; returned to
  movement, metabolism and respawn at the fundamental only, until whole-body
  fitness is integrated.
- **Onset sharpness**: detected through spectral flux; returned to the meter.
  The meter also takes the phonation onset strengths of habitat-routed Voices,
  a motor-side auxiliary input; presentation-only Voices stopped reaching it on
  2026-09-29. That input still carries the commanded accent, not the rendered
  level, so a habitat-routed onset too quiet to hear still counts.
- **Common onset and envelope coherence (grouping)**: not detected. Components
  that change together group even when far apart (Elhilali et al. 2009), so an
  inharmonic set that starts and moves together can form one source, and a
  harmonic set is not thereby one individual. These cues become ecological only
  with listener-side attribution.
- **Micro-fluctuation (`motion`)**: not detected as such; R/H see at most a
  smeared spectrum. Modulation makes a harmonic set more prominent and
  voice-like (Chowning's sung-vowel synthesis; McAdams 1989), but whether
  frequency-modulation coherence is itself a grouping cue, rather than acting
  through harmonicity, is disputed (Carlyon 1991). The implementation applies
  it in common across a body's partials and independently across bodies, with
  default 0.0. `motion` is almost pure vibrato: a 5 Hz sinusoid of relative
  depth 0.02·`motion` (24.5 cents RMS at 1.0), while its pink-noise jitter is
  0.03–0.1 cents RMS at 1.0, below any audible level (measured from the
  implementation). Two questions are kept apart (decision D5 of the plan).
  Whether irregular fluctuation should be present while a body is driven, and
  at what level, is decided by blind audition at equal RMS against periodic
  FM. Where fluctuation lives (excitation input or body state), who owns its
  random state, and what the tail keeps are contract decisions for this body
  family; in many driven sources irregularity comes from the excitation
  mechanism while a freely ringing linear resonator keeps its frequencies, but
  that does not settle the choice for self-sustaining bodies.
- **Texture statistics and formants**: not detected. In voices, and in
  instruments with a separate resonating body such as bowed strings, a
  resonance stays fixed in frequency while the source moves; the current
  bodies transpose whole (`pitch_hz * ratio`).

### State leakage is a hypothesis with two links

Mapping life state onto sound has two links of different standing. *Life state →
excitation strength* is a chosen mapping, not a law. *Excitation strength →
spectral balance* is a covariation shared by many natural sources, and listeners
use it to judge playing effort apart from level (Fabiani & Friberg 2011). The
previous version's appeal to the Lombard effect was misplaced: that effect is a
vocal adjustment to background noise that depends on the noise spectrum, not a
rule from metabolic energy to brightness. `damping` stays, as the response
adopted for this body family, not as proof of evolutionary necessity.

### Degenerate winners

Once fitness integrates the whole body, a body can win by shedding its upper
partials or by going quiet. Compare bodies at controlled level, and in the
ecology charge emitted level through its cost. If silence or a high cut wins,
explain that behavior before calling it a timbre discovery.

## Core and body modules

Decided 2026-09-29. Synthesis methods must stay replaceable, so that other ways of
making sound can be tried without touching the ecology. The line runs here: from
the moment a body receives excitation until it radiates sound, the body is a
replaceable module; hearing that sound, evaluating it, and deciding when and how
strongly to excite belong to the core.

**Rule.** Every quantity the ecology evaluates is computed by the core from the
radiated sound. A body's self-report (partial ratios, spectral footprint, band
energy forecasts) is a fast path only, used when it matches the body's own render
in the conformance tests. A body cannot report its way to fitness, and a new body
with no self-model still works, slowly, on the render-derived path.

**Core.**
- The listener model (ledger §9.3.56 layers 1, 3, 4): front end, R/H/C, meter,
  habituation, `ListenerTwin`.
- Ecological decisions: when and how strongly to excite (phonation, and the
  mapping from life state to excitation strength), pitch search, cost measured
  from emitted sound, respawn and parent selection, one body per Voice.
- Sound transport: `ScheduleRenderer`, habitat/presentation routing, output
  processing (click prevention, output gain), per-source self-PCM capture, seed
  supply.
- The body module contract itself: the meaning of the inputs (excitation strength
  and time, continuous drive, stop, target pitch) and the output (an audio block),
  lifetime, resource bounds, determinism.
- The composer's founder vocabulary (`modes`, `brightness`, `spread`, `unison`,
  `motion`): shared by all bodies; each body maps it onto its own parameters.

**Body module.**
- Rendering: excitation to internal state to radiation, free response, the
  response to excitation strength (what `damping` does today), the mapping from
  the reference frequency to its own spectrum, and how fluctuation acts, using
  randomness supplied by the core.
- Genotype: traits in its own coordinates, mutation operators, the mapping from
  the founder vocabulary, and capture for heredity.
- Optional self-model: spectral footprint per pitch, band energy forecasts, own
  ratios for leave-one-out, each verified against the body's render.

**Placement of items that straddle the line.**
- ADSR: the core issues the excitation shape; the body owns its free response;
  the output envelope is core output processing.
- `damping`: life state to excitation strength is core; excitation strength to
  spectrum is the body's.
- Fluctuation: the random source (seed and advance) is core; its effect is the
  body's. The excitation/state contract settles the rest.
- Self-exclusion: per-source PCM subtraction is core and works for any body;
  ratio-based leave-one-out is a body fast path.
- Predictive terrain and energy forecasts: body fast paths; without them the core
  derives them from a representative render (`footprint.rs` already renders a
  `Tone` for this).
- Reference frequency of unpitched bodies (noise textures and the like): open,
  decided when such a body is first tried.

**Current coupling (2026-09-29).** The existing `SoundBody` trait and factory
registry cover only the Voice-side description; its per-sample `articulate_wave`
has no production caller. Sound is made through `BodySnapshot` (a closed record:
kind, fixed vocabulary, ratios), `Tone` and `AnyBackend` (a closed enum); the
modal body enters through the registry but renders through the resonator variant.
Ecology code reads body internals directly: kind branches in
`action_candidates/footprint.rs` and `action_candidates/energy.rs`, lanes and
ratios in `temporal_cognition/body.rs`, bank/sine/control forecasts in
`self_prediction`, `project_spectral_body` in `voice.rs` and `modal.rs`. Adding a
synthesis method today touches about ten files outside the body. The
[implementation plan](../superpowers/plans/2026-09-29-timbre-synthesis.md)
(Task 2, Phase 3) removes this coupling.

**Exposure.** A new body can be registered for research runs without any Rhai
surface. Reaching composers means only mapping the founder vocabulary; it adds no
new composer-facing parameter.

## Priorities

1. **State the listener model's assumptions**: H's integer prior and C as
   survival policy (this note).
2. **Complete self-exclusion and whole-body candidate evaluation separately**
   ([plan](body-aware-fitness-plan.md)). Self-exclusion covers the same Voice's
   ringing tail `Tone`s. Averaging C over partial positions is an approximation:
   H is relational, and several components supporting one root is not a sum of
   pointwise scores. Acceptance traces the causal path with a body-swap control
   (same fundamental, position and environment; only the body changes) from
   evaluation to energy and parent-selection probability, at controlled emitted
   level. After the excitation contract is implemented, this acceptance is run
   again before heredity.
3. **Body module contract and conformance harness**: semantics of
   `On / Off / Update`, ADSR moved out of timbre, one excitation semantics across
   bodies, continuity of fluctuation across a Voice's notes, and the module
   boundary above, with the three existing bodies migrated behind it. The
  [2026-09-29 contract draft](../superpowers/specs/2026-09-29-body-module-contract.md)
  proposes the behavior and conformance tests. Body-aware B1–B7 coordination and
  the initial [Claude independent review](../superpowers/specs/2026-09-29-body-module-contract-claude-review.md)
  are received; the review found conditional readiness for author policy decisions,
  not readiness for A4 adoption. The [response](../superpowers/specs/2026-09-29-body-module-contract-review-response.md)
  maps all fifteen findings. The revision has not been independently re-reviewed;
  author decisions and extra consumer reference/tolerance agreements remain
  pending. F2 representation and the I12b → I4 → body-aware F3 → timbre Phase 3
  integration gate remain. No migration has started, and Task 1 audition remains
  unperformed.
4. **Temporal features through existing paths**: with existing bodies' attack,
   decay and re-excitation only, build contrasts whose mean spectra match but
   whose temporal grouping differs. Find what the analysis loses and add only
   the mechanism that difference needs. No noise layer or spatialization first.
5. **Heredity of timbre**: reuse hereditary respawn and existing mutation
   (`jitter_cents` for ratios; brightness needs its own perturbation). Inherit
   persistent body traits, not ringing state or random sequences.
   `ParentCandidate` (`src/life/community.rs`) stores only id, frequency, energy
   and generation, so a genotype capture path is needed. Transmission
   (decided 2026-09-29): the child copies the body of the parent that respawn
   already selects, so pitch and timbre travel together and copying is biased
   toward success. Linkage is a model assumption, not a defect. Hitchhiking, a
   timbre spreading on its carrier's pitch fitness, is measured with a
   pitch-fixed body swap, a run with timbre evaluation disabled and several
   seeds, before considering copying from a successful non-parent, which is how
   instruments mostly spread. Heritability is granted per pathway (mutation
   direction, detector, behavior consumer), not per parameter. The ledger row
   "heredity of timbre" stays Horizon until this lands.
6. **Listener-side source attribution**: keep generated individuals and
   perceived sources distinct to the end, and keep ring-down time apart from
   memory retention. Grouping cues become ecological here. The order stays
   (decided 2026-09-29), with a gate per pathway: until attribution lands,
   nothing is heritable through grouping, although the same trait (a decay, say)
   may be heritable through the spectral pathway. Attribution comes from the
   listener-side work of the
   temporal DCC track; this design consumes it and builds no parallel model.

The first concrete target: a body with existing non-integer modes keeps its
re-excitation and tail, and its movement and survival change through the
relations of all its partials with others. Audition then establishes when it is
tracked as one continuing source and when it splits.

## Research question

Under a fixed listener model and survival policy, does the population re-find
harmonic spectra, pair stretched spectra with stretched intervals, or divide
between the two? The answer is this design's form of the Manifesto's question,
"why must it be this sound?", for timbre. Read it as C's verdict under this
model and as a test of the model, not as a finding about human preference.

## Deferred

- **Tension → brightness/jitter.** DCC pressure reaches only the pitch-search
  temperature, and render-side tension derives from NeuralRhythms. Calling a
  signal "tension" does not make it physiological arousal; it must pass the two
  conditions.
- **Learning the enculturation layer during a performance** (technote ledger
  §9.3.56). Not scheduled; see the ledger for when it is reconsidered.
- **Dynamic intonation**, partials pulled continuously toward terrain peaks.
  Material physics does not forbid it here, but crowding onto shared peaks can
  erase body diversity and trackability. First establish survival differences
  and generational change with fixed traits and state changes alone. The
  spawn-time `landscape_density_modes` / `landscape_peaks_modes` stay.

## Non-goals

- No new composer-facing timbre parameters. The vocabulary (`modes`,
  `brightness`, `spread`, `unison`, `motion`) is enough to begin with; that does
  not make it a complete map of auditory timbre.
- No presentation-only machinery (transient designers, noise layers,
  spatialization) ahead of an analysis path that detects it.
- No per-Voice timbre micromanagement from scenarios; heredity and state
  coupling are the ecology's job.
- No base-layer listener templates rewritten by the population's sound inside
  the instrument. This applies to layer 1 of the listener model; see
  [technote ledger §9.3.56](technote-ledger.md) for the layers and for
  in-performance learning of valuation as a deferred research line.

## Validation

Three questions are checked separately: did the synthesizer emit the intended
waveform; did the model detect the difference and return it to behavior; can a
person hear it. The machine answers the first two. Timbre itself is judged by
ear, which is why `examples/timbre_probes.rs` renders audition probes instead of
asserting. The ledger's resonator counterexamples are first-stage evidence, not
a reproduction of human grouping.

## References

- Carlyon, R. P. (1991). Discriminating between coherent and incoherent
  frequency modulation of complex tones. *JASA* 89, 329–340.
- Chowning, J. M. (1977). *Stria* (electronic composition).
- Chowning, J. M. (1980). Computer synthesis of the singing voice. In *Sound
  Generation in Winds, Strings, Computers*. Royal Swedish Academy of Music.
- Elhilali, M., Ma, L., Micheyl, C., Oxenham, A. J., & Shamma, S. A. (2009).
  Temporal coherence in the perceptual organization and cortical representation
  of auditory scenes. *Neuron* 61, 317–329.
- Fabiani, M., & Friberg, A. (2011). Influence of pitch, loudness, and timbre on
  the perception of instrument dynamics. *JASA* 130, EL193–EL199.
- McAdams, S. (1989). Segregation of concurrent sounds. I: Effects of frequency
  modulation coherence. *JASA* 86, 2148–2159.
- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception via
  statistics of the auditory periphery. *Neuron* 71, 926–940.
- McDermott, J. H., Schultz, A. F., Undurraga, E. A., & Godoy, R. A. (2016).
  Indifference to dissonance in native Amazonians reveals cultural variation in
  music perception. *Nature* 535, 547–550.
- McPherson, M. J., et al. (2020). Perceptual fusion of musical notes by native
  Amazonians suggests universal representations of musical intervals. *Nature
  Communications* 11, 2786.
- Norman-Haignere, S., Kanwisher, N., & McDermott, J. H. (2013). Cortical pitch
  regions in humans respond primarily to resolved harmonics and are located in
  specific tonotopic regions of anterior auditory cortex. *J. Neurosci.* 33,
  19451–19469.
- Pierce, J. R. (1966). Attaining consonance in arbitrary scales. *JASA* 40, 249.
- Sethares, W. A. (1993). Local consonance and the relationship between timbre
  and scale. *JASA* 94, 1218–1228.
- Sethares, W. A. (2005). *Tuning, Timbre, Spectrum, Scale* (2nd ed.). Springer.
- Sims, K. (1994). Evolving virtual creatures. *Proc. SIGGRAPH '94*, 15–22.
- Stockhausen, K. (1954). *Studie II* (electronic composition).
