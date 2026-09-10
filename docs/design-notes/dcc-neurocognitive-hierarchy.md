# DCC Neurocognitive Hierarchy

Status: design specification; reference algorithms selected, implementation and cognitive validation incomplete
Created: 2026-05-18
Revised: 2026-09-09
Scope: temporal DCC, from articulation and groove to the experienced whole piece

The task is to represent the auditory-cognitive mechanisms of musical time inside
Conchordal and let those representations shape generation. The existing harmonic
landscape is an input to this design; its roughness, harmonicity and consonance
kernels are not being redesigned here. Harmonic relationships unfolding in time,
including expectations relevant to phrasing, belong to the temporal task.

The canonical scope and implementation ledger is [technote §9.3.55](../../web/content/technote.md),
with a [Japanese account](../../web/content/technote.ja.md). This memo expands that
contract into a design specification and evaluation plan. It replaces the earlier
six-layer sketch; it does not certify a finished cognitive model. The [temporal-niche
design](rhythm-temporal-niche.md) records the implemented short-time participation
policy and its audition history. The [review record](dcc-hierarchy-review.md)
tracks actual Fable responses and the exact reviewed version separately from
implementation or cognitive acceptance.

## 1. Completion target

For every musical function below, the design must connect an observed cognitive
phenomenon to an internal representation, an update rule, interactions with other
functions, a consequence for generation, and a discriminating evaluation. Naming a
brain area, adding a history buffer, or improving waveform prediction does not
complete that connection.

The full target includes:

- articulation: attacks, continuation, release, overlap, gaps and gesture grouping;
- groove: contextual timing relationships and their invitation to participate;
- beat and meter: anticipated recurrence, subdivisions and grouping across beats;
- phrases: continuing units, boundaries and closure, including nonperiodic phrasing;
- repetition and variation: remembered relations across changed realizations;
- sections and form: contrast, return, development and the evolving context of the
  whole piece.

These functions overlap. A phrase may cross a measure boundary, several streams
may have different groupings, and variation can occur within a gesture or across
sections. The list is not a mandatory chain of increasingly slow oscillators.

Scenario continues to express the artist's direction. That does not remove the
need for the system to perceive, remember and act within long temporal structure.
An authored instruction to make room for a return is different from evidence that
a return has been heard. Neither the runtime clock nor the file ending supplies
that evidence.

## 2. Common principle and state contract

**Proposal: use interacting, uncertain representations of temporal relations as
the cognitive environment for local sound production.** Their common operation
is to retain evidence, anticipate continuations, revise relationships, and change
the opportunities available to a Voice. Each representation has its own dynamics.

The P/M/A contract in technote §9.3.5, "Contract for Perception, Memory and Action
under DCC", applies throughout. It is intentionally distinct from §9.3.55,
"Temporal DCC across the hierarchy and its connection to local generation": the
latter is the hierarchy and decision ledger referenced in the header. Both
sections exist; changes to the common P/M/A contract and findings about this
hierarchy have these different destinations.

- **P, auditory evidence:** continuous, timestamped sound features with observation
  coverage and uncertainty. Features and tentative groupings can coexist; a
  complete transcription or source separation is not a prerequisite.
- **M, retained relations and expectations:** order, grouping, recurrence,
  transformations and context, with uncertain retrieval. Distinguish recent
  acoustic state from memories that may be retrieved after an interruption.
- **A, situated action:** a Voice considers the audible consequences of its own
  possible actions, subject to its body and ecological disposition. Issued
  predictions, executed actions and observed outcomes remain distinct.

P/M/A are functions across timescales, not three serial processing modules or
three bands of brain activity. In particular, M must represent musical relations;
it cannot be replaced by a collection of residuals from waveform forecasts.

For a proposed relation state `z`, the shared inference contract can be written as:

```text
predicted_belief(z, t) = propagate(previous_belief, retained_context, elapsed_time)
updated_belief(z, t)   = condition(predicted_belief, newly_observed_sound)
retained_context      = retain_or_retrieve(evidence, relations, elapsed_time,
                                         intervening_information, attention)
action_consequences   = predict(sound | relation_beliefs, own_body, candidate_action)
```

This is the semantic contract, not an identified neural update law. Section 9
selects a concrete reference implementation. A probabilistic implementation must specify a joint or conditional model for
the reused evidence: passing the same acoustic frame through several layers must
not multiply it as independent observations. Uncalibrated scores remain scores.
Predictions can be revised for future decisions; forecasts already issued for
evaluation remain fixed. During missing input, propagate the state without an
observation update. Observed silence can constrain an expectation; an input gap
cannot establish that an expected event failed to occur.

Every relation hypothesis needs the following information. These are semantic
requirements, not new public Rust types or a generic inference framework.

| Information | Required meaning |
|---|---|
| Support | Which observed intervals and features support the relation, with gaps and analysis delay retained |
| Relation | Order, overlap, grouping, recurrence or transformation; alternatives may remain unresolved |
| Temporal extent | Observed start and uncertain continuation/end; physical time and relative position are both available |
| Persistence | What remains the same when local sound changes, and what evidence would revise that identity |
| Expectation | A future target and conditional prediction, including absence where observable |
| Context links | Which other relations affect interpretation or retrieval; their uncertainty and shared evidence |

Inferred event or motif handles may index memory internally. They are not note
symbols supplied by a score, and cannot be copied from generator Voice IDs or
Scenario labels. Representation is allowed; privileged symbolic instruction as
supposed auditory evidence is not.

## 3. Evidence and the constraints it actually supports

The entries below separate observations from design inferences. They replace the
old blanket ranking of whole layers as "strong" or "moderate". A neural correlate
supports a constraint on a model; it does not identify the model's data structure.

| Primary study and observation | Constraint adopted for this proposal | Boundary of the evidence |
|---|---|---|
| [Nozaradan et al. (2011)](https://doi.org/10.1523/JNEUROSCI.0411-11.2011): imagined binary/ternary meter changes EEG responses to a beat. | Keep metrical interpretation distinct from the acoustic pulse. | Imagery in this task does not establish a universal set of meters or a generation policy. |
| [Damsma et al. (2025)](https://doi.org/10.1093/cercor/bhaf258): an oscillator model and an evoked-response model can both reproduce tempo-dependent beat-frequency enhancement. | Compare mechanisms using more than spectral power or phase concentration. | A beat-frequency response alone does not identify intrinsic entrainment. |
| [Fujioka et al. (2012)](https://doi.org/10.1523/JNEUROSCI.4107-11.2012): beta-power dynamics during passive rhythm listening depend on regularity and timing. | Represent preparation and timing expectation separately from sounding events. | Beta power is not itself a groove score or a 20 Hz note clock. |
| [Witek et al. (2014)](https://doi.org/10.1371/journal.pone.0094446): intermediate syncopation in drum breaks elicited stronger reported pleasure and desire to move. | Evaluate groove through listening and participation, with timing complexity as one factor. | This task does not establish a universal optimum, a microtiming law, or a reward for every Voice. |
| [Neuhaus et al. (2006)](https://pubmed.ncbi.nlm.nih.gov/16513010/): phrase-related responses vary with pause, preceding tone duration, harmonic cues and expertise. | Preserve duration and release/gap relationships and combine them with context when testing grouping. | Phrase-boundary evidence does not identify a complete articulation model or make every gap a boundary. |
| [Sankaran et al. (2024)](https://doi.org/10.1126/sciadv.adk0010): cortical recordings distinguish pitch, pitch change and contextual expectation in Western musical phrases. | Preserve trajectory and contextual expectation alongside instantaneous pitch. | These results do not supply an algorithm for tracking overlapping streams or recognizing arbitrary variations. |
| [Teng et al. (2024)](https://doi.org/10.1523/JNEUROSCI.1331-23.2024): phrase-related gain modulation and predictive phase precession occur with Bach chorales after major temporal cues are removed. | Let phrase context affect short-time processing; test anticipation separately from reactions to boundaries. | The observed slow modulation does not prescribe one phrase oscillator, universal phrase length or cadence grammar. |
| [Hołubowska et al. (2026)](https://doi.org/10.1111/ejn.70481): behavioral and EEG measures track phrases with regular and irregular lengths in Bach-derived stimuli. | Include irregular phrase durations in evaluation; timing regularity cannot be the only segmentation cue. | The stimuli retain tonal structure and modified boundary cues; this is not evidence for unrestricted genre-independent segmentation. |
| [Harrison et al. (2020)](https://doi.org/10.1371/journal.pcbi.1008304): a memory-decay model accounts for effects of sequence length and speed in auditory pattern detection. | Separate elapsed time, intervening information and retrieval precision. | The symbolic input and fitted decay parameters are not a ready-made continuous-audio memory model. |
| [Bianco et al. (2020)](https://elifesciences.org/articles/56073): sparse recurrences of arbitrary tone patterns leave lasting implicit memory effects. | Allow retrieval beyond the recent acoustic buffer; compare recurrence after interruption with a novel pattern. | This detection task does not establish motif identity under musical variation or a specific memory graph. |
| [Farbood et al. (2015)](https://doi.org/10.3389/fnins.2015.00157): scrambling an extended musical excerpt at different structural scales changes fMRI response reliability in experienced pianists. | Evaluate long context with local material controlled, including measure-, phrase- and section-scale order. | Music-specific long-timescale evidence exists, but this study does not fix cognitive windows or a whole-piece generative model. |
| [Williams et al. (2022)](https://doi.org/10.1162/jocn_a_01815): neural event boundaries during music listening correspond to independently annotated meaningful changes. | Give section/event structure an internal representation and test it against listening evidence. | An HMM fitted to fMRI is an analysis method, not proof that an online auditory model or the brain uses that HMM. |
| [Hannon and Trehub (2005)](https://doi.org/10.1111/j.0956-7976.2005.00779.x): metrical discrimination differs across exposure groups and infancy/adulthood. | Keep metrical expectations adaptable to experience; test beyond simple isochronous groupings. | Existing 2/3/4 candidates are an engineering restriction, not a universal cognitive inventory. |

Articulation as an ongoing gesture and form as a memory-attention graph remain
explicit design hypotheses. The studies above constrain parts of those hypotheses;
they do not validate the entire proposed architecture. Likewise, the Manifesto's
delta/theta/alpha/beta allocation remains a revisable mechanism sketch.
Oscillatory signals, event segmentation and memory processes need not have
identical dynamics. The public
Japanese Manifesto currently describes four-band real-time extraction as an
implemented capability; that overstates the audited mechanism. In
`src/core/modulation.rs`, delta/theta are meter-derived bands while alpha/beta are
precision/error scalars, not four extracted neural bands. Route correction of this
public capability wording through technote §9.3.55 and M0's documentation
reconciliation item, preserving the broader commitment to neurocognitive grounding.
The corrected public claim must explicitly preserve these four numbered clauses:

1. The current implementation derives delta/theta bands from meter estimates and
   represents alpha/beta as precision/error scalars.
2. It does not extract four neural bands from audio or brain signals.
3. Representing interacting neurocognitive time structures is a design goal.
4. The four-band allocation remains a revisable proposal.

M0 checks the actual public wording against each numbered clause and records the
source/version comparison in
technote §9.3.55. The four clauses are normative content across languages. M0
registers the exact Japanese wording adopted in manifesto.ja.md alongside the
canonical English text and checks every clause for translation drift; the ledger
records both wordings and their source/version comparison. The check covers every present-tense temporal mechanism/capability sentence
throughout the Manifesto, including the enclosing ランドスケープ introduction's
claim that current audio constructs a two-axis internal landscape in real time,
and every such sentence in the 時間軸 section, including the cross-band assertion that sound agents
entrain to those periods to form organic rhythms, as well as every per-band
functional-role sentence, including delta governing
large-scale phrasing, alpha forming phrase-internal accents and beta controlling
microtiming/synchronization precision. For each sentence, either supply the audited
implemented mechanism or revise it explicitly to a proposed/design-goal role.
The named list also includes the 音響生命の創発 / 個体 sentence asserting that
each individual entrains to neural rhythms on the temporal axis. Sweep the
ポピュレーション bullet, the emergence section's closing claim about bottom-up
rhythm, the Scenario passage's rhythm-layer emphasis, and the closing passages
for other present-tense agent-behavior temporal claims. Give each such sentence
the same audited-mechanism or explicit-design-goal disposition and ledger entry;
equivalent claims outside ランドスケープ / 時間軸 cannot remain unaudited.
Correcting the extraction sentence while leaving unimplemented roles described as
current capabilities does not pass O02. Record those sentence-level comparisons
in the same ledger entry. A task entry alone does not complete this reconciliation.

## 4. Representations and their updates

### 4.1 Articulation and auditory gestures

Represent evolving sound trajectories: attack evidence, sustained change, release,
relative overlap and gaps, with uncertain grouping into gestures. Retain alternative
groupings when several components overlap. An onset-only event list is insufficient
for a continuous texture. Conversely, a spectral trajectory is not already a Voice.

The reference model uses continuous trajectories with competing
articulation-state assignments (§9.2). Its continuity and boundary scores use duration,
envelope and relative spectral motion, conditioned on phrase/context hypotheses.
The pause/duration evidence above motivates this comparison, not a fixed rule that
one envelope or a fixed number of milliseconds is one perceived gesture.

On the action side, keep the implemented distinction between intrinsic bodily
pace, participation period and sound duration. Extend the decision space to
**continue, release and leave a gap** only when their consequences can be observed,
alongside the existing onset/wait/skip choices. A phrase boundary can alter a
Voice's inclination to release without ending every other Voice.

### 4.2 Beat, meter and groove

Keep a belief over recurrence and relative position. When periodic support exists,
tempo and phase are compact coordinates; elapsed time and arrival predictions
remain usable when it does not. Metrical grouping adds relationships among
recurrences, with possible competing interpretations. It is not another label for
one maximally concentrated onset phase.

Compare the current oscillator/recurrence components with an arrival-based
alternative on omission and continuation tasks before replacing them. Separate
tracking tempo, inferring grouping and deciding where to sound. The current finite
subdivision/measure candidates are a baseline, not the completed grouping space.

Groove is a consequence of the relationship among expectation, timing, bodily
participation and experienced attraction. It is not an extra fast oscillator.
The candidate state retains **who follows which audible pattern, at what relative
time and with what variability**, allowing uncertainty about the pattern's source.
Phase concentration and predictive accuracy are diagnostics; neither is the groove
objective. Evaluate systematic offsets, expressive variability and complementary
participation separately from random jitter and exact unison.

### 4.3 Ordered relations, repetition and variation

Retain uncertain relations among sound trajectories: relative pitch changes,
relative timing, articulation shape and overlap, alongside absolute register,
speed and timbre. These dimensions must remain separable so that invariance can
be tested rather than assumed. No scalar similarity is sufficient by definition.

The reference representation is a bounded graph of observed spans and their
ordered relations, with explicit correspondence hypotheses between spans. A
transformed recurrence stores both the retained relationship and the estimated
change; it must not overwrite the earlier episode with the latest realization.
Partial matching, interruption and more than one concurrent stream are valid.

Update a correspondence using later sound that distinguishes it from a competing
pattern, including a pattern with similar energy statistics but a different order.
Retrieve a relation when supported by a cue and its context. Capacity limits may
prune candidates, but pruning is recorded as a computational limit, not human
forgetting. Retention and interference laws require separate task-based fitting.

These relations can change which continuation, articulation or response is
available to a Voice. Recognition of a recurrence does not instruct it to replay a
stored sequence. Local variation remains possible within a retained relationship.

### 4.4 Phrases and closure

Represent a phrase as a hypothesis about a continuing, internally organized span
and its relationships to other spans. Its local contents may change while it
continues. Use contextual continuation and boundary alternatives, with duration
as evidence rather than a timer that must expire.

The reference is a semi-Markov event-state model with explicit elapsed duration
and context-dependent transition scores (§9.4). Compare it against gap-only, fixed-duration and local-change
baselines. A normalized progress variable is optional only when a supported
continuation gives it meaning; the phrase's future end is not known in advance.

**Boundary, closure and change are separate targets.** A new phrase may interrupt
an unresolved one; a boundary may occur without silence; a timbre change may leave
the phrase intact. Closure requires evidence about a learned expectation becoming
satisfied, relinquished or otherwise resolved in context. The operational targets
and separate estimators are specified in §9.4 and §10; their fitted parameters and
perceptual validity remain empirical work.
The existing instantaneous consonance/tension values do not identify this state.

A boundary may change active context or strengthen a memory for the completed
span, while retaining unresolved relations and alternatives. It does not clear
all memory, reset all clocks, or force all voices to stop. An EOF supplies no
observed closure.

### 4.5 Sections and the experienced whole piece

Represent section context through distributions of phrase relationships,
participation patterns and recurrent material. Within-section motion is allowed;
a section is not one constant spectrum. Candidate context states can persist,
split or recur, and section boundaries need not coincide across all streams.

Extend the span/relationship graph to phrase and section episodes. Keep recurrence,
variation, contrast and unresolved continuation links, an active retrieval focus,
and uncertain expectations of return. This is the proposed form representation;
it is an engineering hypothesis constrained by long-context and memory evidence,
not an anatomical map. The same relation semantics apply at shorter scales, but
neither storage resolution nor transition dynamics must be identical.

The whole-piece state is the context accumulated **so far**. It may support a
return, a contrast or a sense of completion without knowing the eventual duration
or a canonical A/B/A template. A return after intervening material must differ
from a first occurrence, and a transformed return must retain the transformation.
Evaluation must separate these distinctions from the immediate sound being alike.

Scenario may shape ecological conditions and the artist's direction, including
opportunities for contrast or return. Those instructions are separate inputs on
the production side. Listener inference sees the presented sound and its history,
not the author's intended section labels or a prewritten form trajectory.

## 5. Cross-scale coupling and the generative landscape

The proposed hierarchy consists of interacting representations, not a pipeline
that finishes source separation, then meter, then phrases. Grouping and prediction
can inform each other while alternatives remain live. Meter and phrase organization
may cross; motif relationships may span several levels.

| Coupling | Evidence passed upward/across | Context returned to local processing or action | Discriminating comparison |
|---|---|---|---|
| Articulation ↔ phrase | Duration, overlap, release and continuation trajectories | Expected boundary or continuation changes candidate sustain/release consequences | Same onset pattern with different release/gap relations; same local ending in two preceding contexts |
| Beat/grouping ↔ gesture | Recurrence and accent relations with uncertainty | Relative timing expectations weight possible attacks without a global reset | Shared beat with complementary positions versus forced unison; phrase crossing a measure |
| Motif ↔ phrase | Ordered correspondences and partial matches | Retrieved continuation constrains what may follow or complete the phrase | Matched local suffix after a familiar or novel prefix; transformed recurrence versus reordered control |
| Phrase ↔ section | Continuing spans, boundaries and unresolved relations | Current section changes expected phrase continuation or contrast | Same phrase in a repeated context and at a context transition |
| Section ↔ whole-piece memory | Episode identity, recurrence and transformation evidence | Prior material changes anticipation of return and perceived completion | A return after intervening material versus a first occurrence with the same local audio |

Shared observation and relation summaries supply a **conditional environment for
action**. For a Voice and a prospective action/time, it should describe predicted
audibility/overlap, fit with or departure from a retained relation, support for
continuation or closure, and uncertainty. The existing harmonic terrain remains
another input. These quantities are not combined into an asserted universal
"musical goodness" scalar.

The local body and authored ecological disposition decide how to respond. A
Voice may support, delay, overlap, vary or decline an expected event. Section 9.6
specifies the local selection rule and its comparison. Metabolic value remains
on the existing path; a new cognitive reward requires a separate explicit design
and evidence. A perception model's likelihood does not by itself authorize a
survival reward. Consonance, synchronization and minimum prediction error are
not prescribed global endpoints.

Maintain causal separation in a proposed hop/update cycle:

1. Ingest each new audible interval once, with timestamp, coverage and delay.
2. Update tentative relations and retrieve prior context; retain alternative
   groupings and their uncertainty across scales.
3. Publish a bounded, timestamped environment summary. Contextual inference may
   revise earlier interpretations, but not what had actually been observed.
4. Each Voice compares its possible actions using its own state and accessible
   context. No section model dispatches a centrally scheduled note sequence.
5. Record the action that really executed; update outcome expectations only when
   the corresponding sound becomes observable.

Generator predictions may use known own-action intent; listener observations use
presentation audio. Sharing code does not make their evidence or states identical.
No explicit supporting pulse is introduced except in authored synchronization
demonstrations. Real biosignals are a later coupling extension, not a prerequisite
for modeling the auditory-cognitive environment.

## 6. Where lengths and parameters come from

| Quantity | Origin and identification requirement |
|---|---|
| Auditory integration and resolution | Constrain from the relevant perception task and observation model; retain analysis latency separately |
| Memory retention, interference and retrieval precision | Compare elapsed-time and intervening-material manipulations, including return after interruption; do not infer them from forecast count |
| Beat/grouping or phrase/section duration | Infer with uncertainty from sound and retained context; learned priors are conditional on experience, not universal seconds or fixed numbers of bars |
| Intrinsic pace, sound body and ecological preferences | Authored embodiment; an artistic choice does not become a neural constant |
| Sampling grid, history cells, candidate cap and compute budget | Engineering approximation; test sensitivity and expose truncation where it affects interpretation |

The current 0.125–16 s history-cell locations do not define a cognitive memory
lifespan, phrase length or maximum form length. Changing physical tempo need not
have the same effect as inserting additional intervening material. A slow section
state does not require a new infralow oscillator. Conversely, detecting a slow
neural modulation does not rule out oscillatory contributions; it constrains a
specific comparison rather than the choice of every layer's mathematics.

## 7. Current implementation boundary

| Function | Current production/research contribution | Still required by this design |
|---|---|---|
| Continuous evidence | Production Log2 spectral and band-energy histories; gap handling and own-sound separation on relevant paths; research waveform-state models | Evidence sufficient for persistent auditory relations in overlapping/continuous sound |
| Two-bus observation infrastructure | Both buses already instantiate the common analysis worker: habitat uses `AnalysisDelivery::Latest`, presentation `Ordered`; frame IDs, gap reset and SpectralHistory are present when enabled | Tap the new relation records on both existing workers, tag epochs/buses and enable temporal analysis independently of pitch coupling/report; no duplicate NSGT infrastructure is required |
| Articulation and participation | Intrinsic pace, participation period and duration separated; onset/wait/skip and observed outcome context partially connected | Cognitive gesture grouping; context-conditioned continuation/release |
| Beat, meter, groove | Existing beat/subdivision/measure candidates and optional accents; recurrence-based participation | Joint interpretation of grouping and context; groove-specific perceptual/participation validation |
| Ordered memory and variation | Interval/relationship/context prediction comparisons; partial production energy-history use | Identified retention/interference rules; correspondence through transformation and interruption |
| Phrases | Research change/expectation diagnostics | Online phrase identity, boundary/closure separation and a generative connection |
| Sections and whole form | Scenario supplies authored direction; conceptual model in this memo | Auditory section/context memory, return/variation relations and their local generative effects |

Author reports of naturalness, preserved flow and avoidance of onset collapse
apply to their recorded audition conditions. Recognition that one A/B condition
changed does not establish phrase perception. Recent waveform/regression work
provides numerical and observation-path evidence, not acceptance of these higher
functions. No whole temporal DCC model is implemented by this revision.
The presentation path is directly verifiable in `src/runtime/mod.rs`:
when `listener_forced` or nonzero pitch coupling enables listener analysis,
`wire_runtime` spawns `listener-analysis` with `AnalysisDelivery::Ordered`, and
`consume_listener_analysis_results` passes its frames to
`ListenerTwin::observe_presentation_landscape`, which retains their SpectralHistory.
The shared `src/core/analysis_worker.rs` supplies ordered IDs and gap resets on
both buses. The report/pitch-independent temporal enable remains required under
O21. These existing facts do not imply that either new relation instance
or its contextual action columns already exists.

## 8. Implementation sequence and acceptance

The next work must test the architecture across timescales. Do not require a
universally successful short-time acoustic predictor before studying phrasing,
memory or form, and do not replace those studies with another forecast-loss assay.

1. **Specify one complete relation experiment before adding models.** Use
   continuous or overlapping musical material with a repeated passage, a varied
   recurrence, an intervening contrasting passage and an unresolved/closing
   continuation. Specify which relations should be judged at gesture, grouping,
   phrase and whole-context levels. Matched controls reuse local audio while
   changing prior order, recurrence or release relationships. Scenario labels
   stay outside inference. Store numerical definitions, study-derived constraints
   and author questions separately; do not call the constructed labels perception.
2. **Implement a minimal passive path across the hierarchy.** Reuse current
   observations; represent tentative spans, order/correspondence, a phrase-context
   alternative and retrievable earlier material. Section/form support starts with
   testing the same material as novel versus returning after interruption. Keep
   beat/grouping evidence as a parallel input. Implement the reference relation
   score, duration model and retention rule in §9, then fit and compare them under
   §10. A failed comparison reopens that specific choice without removing the
   musical function from scope.
3. **Test the effect of longer context.** Compare full context, recent-context-only,
   order-discarded and no-retrieval variants on the same suffix. Observe changes
   in anticipation, grouping and recurrence/closure judgments separately from
   waveform accuracy. Include variable phrase lengths and metrical ambiguity.
4. **Connect one local action dimension at a time.** Start with an observable
   continuation/release or onset/wait decision conditioned by the supported
   relation. Compare context present/removed with the same body and stimulus
   history. Test whether removing a cross-scale connection removes the predicted
   behavior. Then expand the choices and recurrent context.
5. **Broaden musical scope and resource checks.** Test multiple seeds, tempi,
   textures, prolonged sound, nonmetrical passages and longer recurrences; retain
   uncertainty for unfamiliar idioms. Assess real-time cost after the bounded
   representation has a concrete consumer. Human judgments and source studies
   constrain cognitive claims; author audition determines artistic acceptance.

Acceptance is recorded separately for four claims: causal implementation
correctness, correspondence with the chosen cognitive task, audible change in the
intended relation, and artistic preference. Passing one does not imply the others.
Before promoting a component, name its task, input, predicted distinction,
competing model, falsifying result and intended generative consequence. If a
comparison fails, revise that connection rather than appending unrelated models.

The reference decisions below make the implementation path explicit. Unmeasured
parameters and unpassed comparisons remain delivery gates, rather than unspecified
architecture or evidence of completion. Replacing a reference estimator requires
the same input/output semantics, the same falsification controls, and an updated
decision record in the technote ledger.

## 9. Selected computational reference

This section selects the first implementation, rather than claiming that the
selected mathematics is uniquely implied by DCC. All musical functions in §1
remain required if a reference component fails. A replacement must meet the same
contract and outperform the relevant controls; it cannot redefine a phrase as a
slow energy envelope or a return as a low waveform error.

### 9.1 One causal conditional-state update

The observation unit is a newly available acoustic interval. Construct it in the
existing analysis worker from the same ordered NSGT hop and frame ID that advances
`SpectralHistory`, before latest-state delivery can coalesce frames. The compressed
history snapshot remains diagnostic in this reference, not a recoverable store of
raw past hops or an unspecified model input. Groove density uses observed accent
counts on the ordered record stream and only reuses the declared history-window
constants; it is not derived from spectral RMS history cells. Any future
snapshot-derived feature requires its formula and per-cell coverage mask in the
feature manifest and a new model version.
Reuse that path's support/coverage and reset decisions; do not introduce independent
gap bookkeeping in another acoustic buffer. Its record contains
sample support, availability time, coverage, Log2 spectral energy, envelope/flux,
and uncertain spectral-trajectory candidates. Preserve continuous features even
when no stable pitch or attack can be assigned. The existing R/H/C features may
enter as contextual evidence without changing their kernels.

The reference signal is one mono f32 sample stream per bus, matching the current
`render_and_route_audio` buffers. The device callback duplicates presentation mono
to its output channels; the current instrument has no separate spatialized bus.
For N actually observed sample times, `E_bus=sum_n y_bus[n]^2/N` and
`RMS_bus=sqrt(E_bus)`; N counts times, not interleaved device-channel samples.
Missing times are excluded with their separate coverage fraction; N=0 is unknown.
Every waveform energy/RMS quantity in this reference uses that convention before
its explicitly defined spectral/group assignment. A multichannel source file is
converted once to `y[n]=sum_(c=1..C) x_c[n]/C` (equal-gain arithmetic downmix, no
power compensation), and analysis and reference listening use that same y. This
permits channel cancellation and does not preserve the source's spatial evidence.
Record source C/order, the downmix matrix, subsequent fixed gain and any file-side
processing. Reference playback is mono or identical copies of y to the registered
device channels, never the original spatial mixture paired with downmix judgments.
O09 fixtures cover mono, identical stereo, asymmetric channels and antiphase
cancellation; verify the analysis samples equal the declared function of the
presented digital waveform and the energy formula remains invariant to duplicated
output channels. O06/O14 stimulus records retain these formats and hashes, device
mapping and gain/guard settings. Reference transfer trials require a transparent
output guard at the chosen level; nontransparent output processing is a separately
registered transfer condition, not waveform identity. The current listener tap is
before the device guard. A channel/downmix/processing convention change starts a
new observation epoch and model version with renewed feature/transfer validation;
this reference makes no spatial-hearing validation claim.

The canonical hop is `AnalysisStream::hop_samples()` divided by its actual runtime
sample rate. Current defaults are 512 samples at 48,000 Hz: 10.6667 ms, or 93.75
hops/s (`src/config.rs`; runtime may select the device rate). Thus a nominal 100 ms
worker cycle receives nine or ten hops, averaging 9.375. At the declared proposal
caps, ten hops require at most 655,360 local joint-log-potential evaluations per bus,
plus matching, cached hazard integrations and table construction. Enforce a
maximum of 256 scalar feature terms per local or shared log-potential evaluation
in the manifest, including any uncached head computations. Ten hops therefore
have at most 167,772,160 local terms and `10*32*256=81,920` shared terms:
167,854,080 passive inference terms per bus/cycle, alongside the separately
bounded proposal priorities, pair-grid updates, DP cells and preview-table terms.
The numerical reference rejects an oversized manifest; increasing the cap requires
a model/resource revision and a fresh complete 50 ms-budget benchmark. Record the actual
hop size/rate with the frozen model and benchmark; no hidden resampling clock is
introduced for cognition. Delivery bursts still process original IDs and support
durations in order, not a single coarse step. A changed analysis hop/rate starts
a new observation epoch and requires feature/parameter validation at that setting.
An analysis-configuration epoch change explicitly restarts cognition: release the
old live episode bank, group/context handles, pending commitments and private
references, then initialize empty/unknown under the new epoch. Record an operator-
configuration reset and its lost-state counts; it is neither cognitive forgetting
nor a musical boundary. Keep the Voice's bodily clock running and new contextual
pressure neutral. No old-epoch memory is retrieved, reinforced or silently
reinterpreted under new feature units. Temporal action stays disabled until that
setting's feature/parameter validation passes. Ordinary acquisition/delivery gaps
remain within an epoch and follow the preservation rules below; a gap alone must
not trigger this restart. Epoch isolation tests cover both cases.
Each bus's relation queue holds at most 32 complete observation records. A full
queue rejects new records without blocking audio, records their omitted frame/sample
ranges, and exposes those gaps when ordered processing resumes (also at EOF).
Higher hop rates or delivery bursts consume this same finite budget; backlog age
attenuates context, and overflow propagates missingness, never fabricated silence.
Such overload behavior does not count as meeting the ordinary-workload latency gate.
All fixed physical-time feature and coverage windows clip to the current
observation epoch's start: use `[max(epoch_start, now-window), now]`. Pre-epoch
time is not missing sound; gaps inside the clipped interval remain missing.
An empty interval is unsupported. Report nominal duration, actual clipped duration,
known duration and sample/event count; the feature manifest includes the clipped-
duration fraction for affected heads so an early short prefix is not silently
represented as a complete long window. The same rule covers detector/relative-
timing windows, density, ordinal probes and the 0.5 s action-coverage window.
Sliding windows advance in physical stream time even during missing input, not
only when known samples accrue. Required anchors, valid interval counts and
post-reset NSGT warmup remain separate conditions. Early probes are evaluated
with these clipped windows and reported as an early-prefix stratum; their windows
are not silently filled with zero or masked by fictitious pre-start silence.

A hypothesis contains compatible articulation assignments, recurrence/grouping
alternatives, active phrase/section contexts, and links to earlier episodes.
Different auditory groups may have overlapping spans. There is no requirement
that every gesture belongs to a detected beat or that phrase and measure
boundaries coincide. Graph handles refer to inferred acoustic spans; they do not
inherit Voice IDs. Generator self-action records remain on the generator side.

Use bounded beams of partial paths, retaining parent links inside a commitment
window. Never merge paths merely because their current labels match: pending
span assignments and episode links must also agree. Factor the retained
approximation as `q(context_path) * product_g q(group_path_g | context_path)`:
one bounded shared retrieval/form component and a local beam per
auditory group. A local path contains articulation, recurrence, phrase and local
section alternatives; section boundaries need not coincide across groups. Retain
an unknown local path for every group under every retained context, and an unknown
shared context. Local inference conditions on the preceding shared summary; the
new shared summary is computed once from the joint conditional features. Implement
the conditional feature log-potential as one shared term plus one local term per
group, conditional on the shared path; cross-group features use the preceding
summary, not a fresh local posterior. `f_shared` and `f_g` are additive log-scores:
a fully missing interval contributes zero log-potential, hence multiplicative
one, not annihilation of mass. Suppress registered missing-indicator terms too
for that fully missing interval; partial component missingness in an otherwise
observed interval retains its registered indicators.

The following is the complete reference normalization. `C=(c,c_next)` is a
shared parent/extension pair, and `E=(g_path,g_next)` ranges over all enumerated
local parent/extension pairs for group `g` under `C`. Each transition `T` is
normalized over the proposals of its own parent. `f_shared` and `f_g` partition
the single conditional feature log-potential for this new interval:

```text
A(C)       = log q_previous(c) + log T_shared(c_next | c) + f_shared(C)
B_g(E | C) = log q_previous(g_path | c) + log T_g(g_next | g_path, C) + f_g(E, C)
log Z_g(C) = logsumexp over ALL enumerated E of B_g(E | C)
M(C)       = A(C) + sum_g log Z_g(C)
q_new(C)   = exp(M(C) - logsumexp over ALL enumerated C of M(C))
q_new(E | C) = exp(B_g(E | C) - log Z_g(C))
```

Specify the reserved unknown transition at every fitting/inference stage. For
physical hop duration `dt`, use `eta_k = 1-exp(-dt/tau_k)` for component list k.
The initial engineering scales are 10 s for articulation, grouping and
correspondence, 30 s for phrase, and 120 s for section and shared-context lists.
Use that list's eta_k wherever eta appears below. Within each component list, normalize the admitted known raw scores to sum
one, give them total mass `1-eta`, and give its single unknown option mass `eta`.
If no known option is supported, unknown has mass one. A known phrase/section
stay uses its hazard survival; its exits use `(1-survival)*type_probability` before
this leak. Other known transition/admission scores are nonnegative normalized
scores (or softmaxed log scores). Unknown-parent recovery uses currently admitted
known options under the same rule; deduplicate unknown stay with the unknown
option. Apply this rule also to shared proposals, then form and renormalize the
local component product on the bounded tuple set below. The all-unknown tuple
therefore has a defined weight after both component-list normalization and
tuple-set renormalization, at every fitting/inference stage. A component marked unknown
masks its value features; only explicitly registered missing-indicator features
may contribute to its observation log-potential. An all-unknown proposal masks all
components. A missing acoustic interval still sets the entire observation
log-potential to zero. Freeze every list's scale in the numerical reference;
compare each independently at half/double, the former common-10-s variant and hop
subdivision. These are uncertainty-transition scales, not the fitted episodic
retention constants or neural timescales. Pruned mass is added after this
calculation, without applying another leak to that mass.

The development reference includes a 60 s sustained passage, 60 s of observed
quiet within a continuing section, and separate 2/8/32 s acquisition gaps followed
by the same diagnostic return cue. Record live section/context known and unknown
mass, committed-handle survival, retained identity and cue-driven readmission.
The numerical fixture supplies a uniquely matchable committed cue: after sufficient
cue support is observed, its retained handle must be eligible again in the first
matcher cycle, including recovery from an unknown parent. No label is injected to
make it win. Absence of sound does not renew observation support or force a section
exit. Human/context comparisons then determine whether the resulting posterior
retention and recovery fit the task; the fixture alone proves neither. Capacity
loss and missed readmission block freeze rather than being called cognitive forgetting.

Thus observation normalization is across parent/extension pairs, never separately
per parent. Compute every `Z_g` before local beam pruning. Retain the best 15
resolved local paths and put all other enumerated mass, including the explicit
unknown proposal, in the reserved unknown path. Its mass is their log-sum-exp,
not a discarded diagnostic; it contributes to `Z_g` and shared evidence on this
update. Prune shared pairs analogously to seven resolved paths plus their summed
unknown mass. A shared unknown stores unresolved local states. On later updates,
aggregated unknowns may stay unknown or admit new acoustically supported proposals;
the discarded identities cannot be recovered or reinforced as if still retained.
Report this loss of identity separately from preserved current-step mass. This
policy does not estimate unenumerated mass. Exhaustive small fixtures must test
both current partition equality before/after pruning and later interpretation bias
against an unpruned reference, including a diffuse local posterior under only one
context. Half/double beam tests reserve one unknown slot at every capacity.

Features include interactions such as release × phrase expectation. Children do
not send the same audio upward as additional independent observations. A revised
interpretation does not append another observation. Each acoustic interval has
one update identifier. Missing input executes transitions only (`f_shared=f_g=0`).
Neither a prediction request nor its later score updates this state; only new
observed sound does. Do not iterate new audio as fresh evidence between components.
Dropping residual
cross-group path dependence between updates is a declared assumed-density
approximation, checked against exhaustive enumeration on small fixtures.

These weights are model-relative conditional scores. They become reported
probabilities only after held-out calibration. They are not independent neural
posteriors. Candidate generation, feature scales and fitted weights are part of
the model being tested. Record the weight retained from *enumerated* proposals
when pruning; it says nothing about unenumerated interpretations. If evidence or
capacity cannot distinguish a relation, publish unknown/competing alternatives.

The reference observation clock follows analysis availability; the relation worker
may batch delivery but processes interval IDs in order. Features integrate over
physical support, rather than counting arbitrary report or prediction calls.
The working hypothesis scores are normalized after each update. External reports
include observation age, alternatives and approximation status, not just a
winning label. Generator and listener instances share algorithms, not evidence
or mutable state. No new public generic inference API is required.

Initialize with an empty episode bank and an unresolved ongoing mixture, with no
assumed beat, phrase start or section identity. Deterministic proposal order and
stable tie breaking make bounded inference reproducible. Proposals include staying
in the current interpretation and the supported local alternatives from §§9.2–9.5;
capacity cannot remove the explicit unknown alternative. A change to a proposal
inventory is a model revision, not a silent optimization.

Proposal composition itself is bounded, before the retained-beam limit. Each
shared parent proposes at most four children: stay, unknown, the strongest
supported retrieved context, and the strongest supported new/contrasting context.
Duplicates or unsupported children leave empty slots. For a local parent under
each shared extension, component lists have maxima 8 articulation/trajectory,
19 recurrence/grouping, 6 phrase, 5 section and 19 correspondence alternatives,
including their stay/unknown options. The articulation/trajectory list's reference
inventory is exactly attack, continuation, release, gap, and one unknown option.
For a known parent state, its same-state entry is stay with the base-rate survival
score; the other states use §9.2's competing-rate exit scores before unknown leak.
Stay is not an additional duplicate slot. For an unknown parent and newly observed
support, use uniform raw admission scores over admissible known states; observation
log-potentials then distinguish them. With no new support, an unknown parent has
no known-state admission. A new gap requires the registered observed-low-energy
gate: group RMS at most 1% of the global development 95th-percentile group RMS
reference, with a strictly positive reference floor of 1e-6 in full-scale amplitude.
Compare 0.5/2% thresholds and register that physical reference. Known gap stay may
persist through missing input as a prior; missing input cannot introduce a gap.
Other known-parent states retain their admitted transition scores even when
features are missing, under the missing-interval rule. Deduplicate by state ID,
with fixed ties attack, continuation, release, gap, unknown. Slots 6–8 are empty
capacity, not unnamed trajectory choices; filling them changes the model. Acoustic
ridge/group association is the separate fractional front end, not another latent
assignment silently hidden in this list. Record the inventory, scores and empty
slots in the proposal/feature manifest. The grouping list allows 16 fresh grouping
proposals plus current stay, known ungrouped and unknown; the correspondence list
allows 16 retrieved candidates plus current stay, known no-memory and unknown.
Current stay is outside each fresh inventory; deduplicate it if the same candidate
was retrieved/proposed. Known ungrouped/no-memory and epistemic unknown are distinct.
These 19-slot bounds therefore compose without compulsory truncation of a full
16-candidate inventory. No extra DTW is triggered for a cached stay; unsupported
cached features remain masked. Sort each by its transition/admission score,
breaking ties by fixed component IDs. Reserve stay/unknown before filling each
list with its highest-ranked admissible entries; any excluded supported entries
are reported component-proposal truncation. Compose tuples with the sum of component
log scores as a cheap priority, without evaluating the joint observation log-potential.
Reserve stay, unknown and the best legal tuple for each phrase/section stay-or-exit
combination. Fill up to 16 total joint tuples using best-first neighboring-index
enumeration, with at most 16 heap pops and five neighbors per pop. Duplicates do
not trigger an unlimited refill search. There are at most 86 cheap tuple-priority
evaluations per (local parent, shared parent/extension pair), because component
scores and reserved seeds depend on that shared extension. At most 32 shared
pairs, eight groups and 16 local parents give `32*8*16*86=352,256` cheap priority
evaluations per canonical hop and bus, or 3,522,560 per ten-hop cycle. Each priority
visits at most five component log scores (1,761,280 such visits per hop), separate
from full joint-feature evaluation. Include component-list construction/sorting,
heap work and these cheap priorities in the worker census and O04 preflight;
they are not cached once across different shared extensions. Evaluate the joint
log-potential only on the at most 16 admitted tuples per such parent/pair. The transition product is
renormalized on this declared proposal set; it is not the full Cartesian-product
prior. All legal phrase/section stay-or-exit combinations have a reserved proposal,
but not every within-category combination is enumerated.

Consequently each bus enumerates at most `8*4 = 32` shared pairs and
`32*8*16*16 = 65,536` local pairs per canonical hop, before retained-beam pruning.
"All enumerated" in the partition equations means this bounded set. Nonproposed
tuples have no computed mass and remain a separate proposal-coverage limitation.
Cache component features and matching results; do not run DTW or rebuild descriptor
histories per tuple. Record actual proposal counts and cache misses, test half/double
joint/shared and component-list proposal caps, and include this truncation in
exhaustive small fixtures.
The 50 ms budget in §9.6 covers all canonical hops accumulated in a nominal 100 ms
worker cycle, matching and table construction together. This operation bound is
not a measured timing proof: runtime acceptance must demonstrate the complete bounded workload
within that budget on the declared hardware before promotion. A budget miss is
an implementation/resource failure requiring revision, not permission to skip
observations while reporting a current summary.

All development standardization uses one frozen global mean and standard
deviation per component over the registered training corpus, including its
habitat/presentation routing and gain conditions. Use physical-support weights
for acoustic hop components and the registered record weights for body-prototype
descriptors; masked values supply no statistics. Store the actual mean, standard
deviation, numerical floor, record IDs and weights with that component's manifest
entry and model version. Do not recompute statistics per live bus, group or Voice.
This includes accent rise/flux, continuity gates, matching and prototype distances.
Per-bus/per-group normalization is an explicit development ablation requiring a
new fit, not a runtime convenience. Cross-bus transfer checks therefore measure
level/distribution differences rather than silently normalizing them away.

The conditional feature family consists of trajectory continuity/residuals,
articulation changes, recurrence/arrival residuals, grouping consistency, ordered
match residuals and transformations, elapsed-duration features, and interactions
with retrieved continuation. Standardize each observed component using development
statistics and carry a missing-component mask. Fit the joint weights by regularized
conditional log loss, marginalizing over hypotheses compatible with each partial
annotation. An acoustically controlled stimulus does not supply an annotation by
construction.
M0 registers the complete coordinate inventory before fitting. Freeze the formulas,
units and masks consumed by a stage before that stage starts: acoustic/match inputs
before stage 1, each head's inputs before its stage-2 substage, and the complete
ordered joint coordinate list before stage 3. Store these versioned entries in
`docs/roadmap/temporal-dcc/feature-manifest.json`, linked from the technote ledger.
Artifact status at this review: `docs/roadmap/temporal-dcc-completion.md` exists
as the separate delivery-plan document; its completion follows this memo's review.
`docs/roadmap/temporal-dcc/feature-manifest.json` is a planned artifact, not an
existing registry. M0 creates it before any fitting; missing required entries block
their stage. The delivery plan's M0 owns this registration artifact. Each entry gives a stable
ID, shared/local owner, exact formula, input provenance, unit, missing mask,
standardization, fitting stage and T-row comparison. Pairwise interactions require
explicit entries naming both operands; no automatic all-pairs expansion is allowed.
The emitted feature-vector length and order must equal the manifest. Adding,
removing or reordering a feature is a model revision with renewed fitting and
validation. Every sensitivity comparison registered anywhere in this memo also
has an O03/O10 entry naming the affected T rows, parameter/reference/variants,
paired material and split, evaluation metric with units, uncertainty calculation,
and a decision rule fixed before inspecting its outcomes. The rule specifies
thresholds for retaining the reference, adopting a variant as a new model version
with all affected refits/gates, or retaining a documented interpretation-limiting
sensitivity in the validity envelope. Capacity/resource metrics accompany, and
cannot replace, the affected cognitive/relational metric. A comparison without
that advance rule is incomplete; a favorable plot cannot authorize post-hoc
adoption. The artifact records the chosen finite feature inventory; its absence
blocks fitting rather than licensing implementation-specific coordinates. The boundary/closure training targets in §9.4 remain separate even
when their features interact in the joint state.

| State owner | Evidence and retained state | Consumer and boundary |
|---|---|---|
| Generator relation instance | Full observed habitat mixture; its own episode bank and context paths | Voices read its ecological context. Their own rendered sound is part of this shared heard history |
| Listener relation instance | Full observed presentation mixture; separate episode bank and context paths | Separate temporal-context snapshot and optional presentation-context action columns (§9.6). Hidden habitat sound and generator labels are excluded |
| Each Voice | Private executed-participation trace, own rendered-sound history and own-excluded local energy estimate | Combines the generator summary with bodily state and local overlap evidence; never conditions either shared bank with intention |

The two shared instances use the same algorithm but do not share hypothesis weights or
mutable banks. Each summary is tagged with its bus, epoch, observation support
and version. Private traces reference those tags and inferred episode handles,
not the reverse. A Voice's heard contribution can affect its later context: this
feedback is intentional, measured, and is not evidence of another agent replying.
Own-excluded energy remains a local overlap estimate, not a fabricated separated
auditory stream. If listener-context influence is enabled, its consequences stay
in separately tagged columns of the action vector; they are not multiplied with
habitat beliefs as independent evidence. Existing `ListenerState` beat/attention
and instantaneous stability/resolvability/tension outputs coexist with the new
temporal snapshot: current short-time consumers retain their current input, while
new temporal consumers read the tagged relation snapshot. Neither object silently
replaces or relabels the other's outputs.
The ordinal heads are fitted/calibrated from judgments of played single mixtures.
Using them for habitat action columns is a declared input-distribution transfer,
not evidence that habitat and presentation evoke identical judgments. Development
must include routing/gain conditions where those buses materially differ: present
each bus's offline waveform separately to independent, condition-blind listeners,
then check each head's loss and calibration against judgments of that waveform.
Matched complete bus histories must agree; differing histories are not expected
to have equal outputs. Until the transfer check passes, cognitive interpretation
and enabled ordinal action columns on habitat are restricted to matched-mixture
conditions; other transferred outputs remain diagnostic model scores. Listener
columns retain their own presentation-only validity boundary.

### 9.2 Trajectories, articulation and relative participation

Start from envelope trajectories and local ridges in Log2 spectral energy. Extend
ridges by continuity of frequency, slope and envelope; retain competing
associations at crossings or under masking. A ridge is an acoustic component,
not necessarily a fundamental or an individual source. Broad/noisy energy stays
in a residual trajectory, so pitched tracking is optional. Compatible groupings
are beam alternatives, including an unresolved mixture. Complete source
separation is not a gate for the rest of the architecture.
The reference uses at most seven ridge candidates plus a residual trajectory.
On each observed Log2-energy frame, detect strict-left/non-strict-right local
maxima (first point of a plateau), with energy at least 1% of the frame maximum
and prominence over the larger of the two-bin-left/right neighbors at least 10%
of that peak's energy. At an edge use the available neighbor; a zero-energy frame
has no peak. Keep the seven strongest by energy then bin ID. Assign each Log2 bin
hard to the nearest retained peak within two bins, with peak-ID ties, and to
residual otherwise. Thus `v_ib` is one for exactly one ridge/residual and zero
for the others; no bin energy is duplicated. Compare one/four-bin radii and
half/double prominence/relative-energy thresholds on development.

For each trajectory i before group association, compute its conserved assigned
energy `E_i=E_bus*sum_b(s_b*v_ib)/sum_b(s_b)` using §9.1's mono energy and the
hard per-bin memberships above. Apply the same zero-spectral-mass/residual rule
as E_gb below. Its envelope coordinate is `ell_i=0.5*log2(max(E_i,1e-12))`;
known zero energy is a floored observed value, while missing energy is masked.

For ridge continuity, use log2-frequency in octaves relative to 1 Hz. A candidate
link from a predecessor's last supported point (f0,t0) to the new point (f1,t1)
has secant slope `v_link=(f1-f0)/(t1-t0)` in octaves/s, only when these are adjacent
canonical hops with fully observed intervening support. Compare this candidate
slope to each supported predecessor slope and its frequency to `f0+v_old*(t1-t0)`;
the envelope residual is ell_new-ell_old. Standardize these three residuals with
their frozen development entries, not a per-link rescaling. After link selection,
store the selected secant as that link's next slope; two predecessor alternatives
retain two secants with their normalized continuity weights. No smoothing or
regression is implicit. Across a gap the new secant is masked; the old supported
slope may supply a frequency extrapolation, but the new point has no observed
slope until its next contiguous supported hop. Register this update, units and a
two/four-interval regression comparison in O03/O09/O10, including steady glide,
changing slope, competing predecessors and gaps. Predict a previous ridge's log2
frequency using its last supported slope. If it has only one supported point and no slope yet, use a
zero-slope persistence prediction for frequency while leaving the slope-residual
coordinate masked. This is an explicit initialization prior, not an observed zero
slope. Register it with the continuity constants and test birth followed by first
continuation, including competing predecessors. Compute RMS of globally standardized available frequency-
prediction, slope and log-envelope residuals; zero available coordinates means
unsupported. At each new peak, retain the two closest eligible predecessors
(distance <= 1), with weights proportional to exp(-distance squared), and stable
handle ties. If a predecessor has two slope predictions, use its smaller supported
distance (stable slope-ID tie) when ranking distinct predecessor handles. Resolve
all peak/predecessor links before handle assignment: a single predecessor retains
its handle only when it has one successor. A split gives every successor a fresh
handle linked to the parent, so simultaneous peaks never share one handle. Two
competing predecessors create a new provisional handle with both continuity links
and their separately stored slopes/weights. A peak with none creates a new ridge. Parent links provide
continuity alternatives, not extra current energy or recurrence credit. A ridge
without a supported continuation is dormant and retires after 0.25 s of observed
no-continuation support; missing input does not advance retirement. Retained slots
remain bounded, with oldest-dormant eviction recorded if needed; recovery after
retirement creates a new handle. Test crossings, split/merge, masking, missing
input and half/double distance/retirement constants. With seven peaks and seven
previous ridges carrying at most two slopes, there are at most 98 predecessor-
distance evaluations per hop; peak/bin passes remain linear in the Log2 bin count
and their actual time/storage are included in the worker census. The residual
trajectory may form an envelope-defined acoustic group with pitch/slope masked;
it is not identical to the later group layer's unassigned-mixture slot.

Group lifecycle belongs to this acoustic front end, not to Voice identity or the
section model. Use eight stable slots per bus, including one unresolved-mixture
slot; handles contain the bus epoch and a monotonically increasing generation so
slot reuse cannot inherit an old identity. Normalize each trajectory's nonnegative
association weights across eligible groups and residual, so ambiguous assignments
do not replicate its evidence mass. Eligibility uses development-standardized
frequency/slope/envelope continuity distance at most one, combined as the RMS
of available standardized residual coordinates, as in ridge/prototype matching;
absent pitch components are masked and no available coordinate means ineligible.
The reference for a resolved group is its last supported member-descriptor set,
not a frequency centroid. It contains at most eight trajectory descriptors, each
with its observation time, log2 frequency, up to two slopes, ell and masks, and
the normalized trajectory-to-group weight r_jg at that update. For a current
trajectory i and each saved member j, use the same frequency extrapolation,
candidate-secant/slope and ell residuals as ridge continuity; d_ij is the least
supported distance over j's at most two slopes. Missing pitch/slope leaves the
envelope comparison available. For each group define
`raw_ig=max_(j:r_jg>0 and d_ij<=1) r_jg*exp(-d_ij^2)`, with empty maximum zero.
The residual raw weight is the fixed `r0=exp(-1)`. Normalize once:
`a_ig=raw_ig/(r0+sum_h raw_ih)` and
`a_i,residual=r0/(r0+sum_h raw_ih)`. With no eligible resolved group a supported
trajectory is wholly residual. A missing current trajectory has unknown
association, not an observed full-residual assignment. The max keeps a group's
multiple harmonic members from diluting a good match by their count; it is an
engineering similarity, not a source posterior. Equal member scores use the
lowest member-generation/slope ID for reference provenance; equal group scores
remain fractional, not winner-take-all.

Compute these assignments against saved references before lifecycle decisions.
Afterward, refresh each nonsuperseded group with its current positively assigned
members/weights when supported; retain the prior reference if no current member
is supported. Birth/split/merge seed fresh groups with their exact triggering
members and r_jg=1 for the next hop; they do not reassign or credit the same hop
a second time. Superseded parents remain context references but are ineligible
for current acoustic assignment; dormant nonsuperseded groups retain their last
reference until observed retirement/cap eviction. Test this ordering explicitly.
At most 7*8 saved member descriptors of 128 bytes use 7,168 bytes per bus; the
uncached bound is `7*8*8*2=896` additional member-distance evaluations per hop:
seven resolved groups with saved references, eight saved member descriptors,
eight current trajectories including the residual trajectory, and two slopes.
Include maxima/normalization, separately charged from the 98 ridge comparisons in
O04. Reuse identical cached comparisons when possible, without assuming that all
groups' saved reference times coincide. Freeze the reference update, r0 and max
rule in O03/O09; compare r0/2 and 2*r0 under the existing association sensitivity
decision rules. Exact fixtures include no eligible group, one/two equally eligible
groups, a threshold-distance member, births from new bundles, superseded parents
and a gap. Check a_i sums, u_B on both sides of 0.5, and sum_(g,b) E_gb=E_bus;
ambiguous weights must change allocation without duplicating energy or support.

For every bundle/split/merge gate, the envelope series is the per-trajectory ell_i
defined above, before group assignment. Use ordinary Pearson correlation of the
equally weighted paired valid hop values: centered cross-product sum divided by
the square root of the two centered sum-of-squares products. Means and sums use
f64; zero variance in either series masks correlation. No log after correlation,
energy weighting or post-group envelope is substituted. Register this input and
statistic with the 0.8/0.2 gates, and compare linear-RMS input as a named variant
under O03/O10's existing window/threshold tests. Common-envelope correlation over
0.25 s of observed support proposes
bundles (at least 0.8 within a bundle); zero-variance or insufficiently observed
correlations are unresolved. All bundle/split/merge correlations use the preceding
0.25 s physical envelope window, clipped at the epoch start, with at least 90%
known paired support and at least eight paired hop values. Below either bound,
mask the proposal; do not substitute zero correlation. Test four/sixteen paired
values alongside the half/double duration windows. These are fixed engineering association gates, not
source-identity probabilities. The residual keeps temporal/envelope evidence even
when source association is unresolved, including noisy material.

Form bundles by deterministic complete-link agglomeration over the at most eight
current supported trajectory handles, including an eligible envelope-only residual
trajectory. Begin with singletons. A pair of bundles can merge only when every
cross-member correlation is valid and >=0.8; its merge score is their minimum.
Choose the greatest merge score, breaking exact ties by the lexicographically
ordered pair of sorted trajectory-generation member lists, then repeat. A masked
pair forbids that merge but is not evidence of anticorrelation. Stop when no pair
qualifies. The final disjoint bundles are not connected components of the threshold
graph: A–B and B–C support does not merge A–B–C when A–C fails. With eight members,
at most seven merges and 196 cached cross-correlation reads suffice for the
complete-link scans; include sorting/tie work in O04's front-end census.

A bundle's persistence identity is its exact sorted set of trajectory-generation
handles. Membership change, a retired/replaced handle or a missing/unsupported
condition resets its three-hop counter; no approximate overlap silently preserves
the count. Prior group handles in split/merge keys must also be unchanged. For
these proposals only, assign a trajectory a former resolved parent when its
previous-hop normalized association has a unique greatest entry in that resolved
group; ties or a residual maximum leave the parent unresolved. This proposal key
does not replace its fractional acoustic associations with a source-identity label.
Freeze those former parents when a new persistence key starts; later association
updates do not rewrite them during its three-hop test. A parent generation loss,
member-set change or failed required condition resets the key. Birth's u_B>=0.5
condition does not apply to split/merge persistence, which uses its own conditions.
A split requires two final bundles whose members all have the same former
resolved parent. Its key is that parent and the two sorted member sets; the key
must persist for three consecutive fully observed hops, with every cross-bundle
member-pair correlation valid and <=0.2 at each of those hop endpoints. Its score
is the maximum such correlation, using the trailing 0.25 s window above at each
endpoint. The birth-only u_B threshold is not a split condition.
A merge uses two former-parent subsets inside one final bundle; its key is the
unordered parent pair, canonically sorted by stable former-parent handle, with
each parent's exact member set kept attached to that handle. Swapping A/B is
the same key, not a second directed proposal; this is why the bound is C(7,2)=21.
Test swapped input order in O09. Use the minimum valid cross-pair
correlation, requiring >=0.8 at each of three consecutive fully observed
persistence-hop endpoints, with the same trailing 0.25 s window. Thus complete-link formation can already combine the
trajectories while the resolved-group merge waits for persistence; it does not
require two distinct final bundles that the algorithm would already have merged.
Birth keys use the final bundle's member set. At most eight birth, 28 split and
21 merge candidates exist per hop; keep bounded previous/current key counters.
For proposal ranking, support is `sum_i max_j a_ij` over its distinct valid
member trajectories, using current normalized associations including residual.
Rank by decreasing support, then kind (merge, split, birth), sorted former-parent
handles and sorted member handles. Process in that deterministic order, allowing each
trajectory or parent group to be consumed at most once per hop; reject/reset later
conflicting proposals. Freeze formation, keys, masked-pair treatment and counter
resets in O03/O09. Non-transitive correlations, equal merge scores, membership
churn, handle replacement, missing hops and conflicting merge/split proposals are
required lifecycle fixtures with the existing persistence/window sensitivities.

The residual slot is a real acoustic mixture with uncertain source association;
it is distinct from a beam's epistemic unknown alternative. Its participation is:

| Subsystem | Residual-mixture slot rule |
|---|---|
| Envelope/flux, accent/salience | Yes, from its own fractionally assigned energy; every record remains mixture-tagged |
| Pair-interval recurrence-period estimation | No; resolved-group proposals are never inferred from residual cross-source accents |
| Grouping and cyclic-word proposals/statistics | No; keep known ungrouped plus epistemic unknown and masked grouping features |
| Arrival hazard | Yes, predicts the next observed change of this residual mixture, not an individual source's beat |
| Inter-group timing pairs and within-group periodic timing history | No; neither target nor reference can be residual |
| Phrase/section local beams and span correspondence | Yes, collective residual-flow relationships with explicit mixture identity; no source-separated interpretation or transfer to a resolved group |
| Prototype association and Voice arrival-quantile source | No; only current resolved acoustic groups are eligible |

The seven resolved groups plus this residual still give eight local beams per
shared context, so the 65,536 local-pair and eight-group matcher/section bounds
include it. Inter-group timing instead has at most `7*6=42` ordered pairs with
two reference slots each, giving 84 inter-group histories, plus seven within-group periodic
histories: 91 histories and 11,648 records at 128 per history. An ordinal
probe includes residual assignment support under the same declared aggregation;
a first-event probe can concern a judged change of its collective flow. These
are falsifiable mixture interpretations, not hidden Voice labels. A noisy envelope
bundle can acquire a resolved group without pitch identification, so restricting
residual recurrence does not make pitched source separation a prerequisite.

For bundle B, define its per-hop unassigned fraction as
`u_B = sum_(i in B) a_i,residual / sum_(i in B) sum_j a_ij`, using only trajectories
with valid current association support; j includes every resolved group and the
unassigned-mixture slot. Thus the denominator is total association weight of the
bundle's own trajectories, not total bus energy or all residual trajectories.
A zero denominator supplies no birth proposal. A bundle with u_B at least 0.5
on three consecutive fully observed hops proposes a birth in a free resolved slot. Initialize its
local beam with unknown plus currently supported articulation/grouping proposals
and no inherited occurrence credit. Split and merge admission use their own
persistence-key conditions above. The three consecutive fully observed hops are a separate persistence
criterion, never the correlation-estimation window.
Process proposals by the support/kind/handle order above; unavailable slots leave evidence
in residual and record capacity limitation. Test half/double persistence/windows,
the 2 s supported-evidence retirement timer (1/4 s variants), and development
alternatives to distance/correlation/support thresholds. The small lifecycle
fixtures below include each retirement-timer variant and a gap that must not
advance it.

For a split or merge, create fresh group handles and retain the former contexts as
distinct reference proposals, not a mean of incompatible local paths. Pending
span credit stays in the original occurrence ledger; new continuation links may
reference the same immutable prefix support but cannot submit it again as newly
heard material. Future intervals receive normalized assignment weights once.
These topology changes censor unsupported track associations; they do not assert
a phrase boundary, closure, or an extra recurrence. Issued forecasts remain fixed.
Old slots become dormant. Masking and acquisition gaps preserve dormant handles
and elapsed-time uncertainty; missing input cannot advance a silence-retirement
timer. Two seconds of supported inactive/low-energy evidence retires a slot into
unresolved current state, while committed episode/context handles remain in memory.
If capacity forces earlier retirement, choose the dormant slot with the oldest
observed support and record computational loss. Recovery of a retired sound is
a new acoustic handle with possible retrieval of old episodes, not a restored
unobserved source identity. No lifecycle event resets a Voice's bodily clock.

The fixed-slot factorization represents uncertain activation and fractional
trajectory membership, not an exact posterior over variable-size partitions.
Correlations between alternative group counts/assignments are dropped as part of
the declared approximation; local beams cannot secretly reassign another group's
observations. Exhaustive small fixtures compare joint grouping alternatives with
this projection through birth, split, merge, masking, gap recovery and retirement,
including conservation of support/credit and effects on niche references and probes.

Each active group has attack, continuation, release and gap alternatives.
Transitions use newly observed envelope change, spectral motion and elapsed
state duration, together with phrase context. The acoustic trajectory remains
available alongside the discrete alternatives. A gap requires observed low-energy
support; missing samples cannot create one. A boundary may be revised as more
sound arrives, but its original availability and any issued forecast stay fixed.
The base articulation transition has fitted competing rates for each current
state s and each different destination r:
`lambda_sr = softplus(theta_sr · x)/1 s`, with total lambda=sum_r lambda_sr,
stay `exp(-lambda*dt)` and exit-to-r `(1-stay)*lambda_sr/lambda`. Its x contains an
intercept, observed log-envelope rise/decline, flux, spectral-ridge motion and
`log(1+state_elapsed_sec/1 s)`, with registered masks and global standardization.
Treat rates as constant within a canonical hop and test subdivision against a
finer reference. Fit/freeze theta in the first stage-2 substage on the T1 annotated
gesture-state intervals below, by regularized interval-censored transition-time
and destination-category log loss, marginalizing uncertain endpoints/assignments.
The named base-rate controls share this front end and the same targets, but have
fixed ordered inputs. `articulation_rate_onset_only` uses [intercept, admitted-
accent indicator, its missing indicator] (three coordinates). The accent value
is one only when the existing detector admits an accent at that update, known
zero when its complete detection support is available without admission, and
masked otherwise. It has no elapsed-time or continuous envelope/flux/ridge input.
`articulation_rate_envelope_change_only` uses [intercept, positive adjacent-hop
log-RMS rise, positive adjacent-hop log-RMS decline, rise mask, decline mask]
(five coordinates), without elapsed-time, flux or ridge motion. Decline is
max(0,ell_previous-ell_current), where ell=log2(max(RMS_g,1e-6)) is the group's
log-RMS from its conserved assigned energy; both differences require the existing valid
adjacent-hop/group-association support. Each value uses the full model's frozen
development scaling and mean-imputation/mask convention, with separate fitted
state-to-state rate coefficients. O03 freezes these IDs/layouts before the
articulation substage. They are controls of the articulation-rate input, not a
claim that the shared acoustic group front end itself consumes only onsets.
Phrase-context effects enter the stage-3 joint features, not this base fit, so
articulation fitting does not depend on a not-yet-fitted phrase head. Stage 1's
bootstrap sets each off-diagonal rate to 1/(3 s), with other coefficients zero;
this initialization is replaced by the fit, not asserted as a cognitive duration.
Stage-1 correspondence uses continuous envelope derivatives for articulation shape,
not these provisional state occupancies. Register the bootstrap, fitted parameters
and targets with the stage-freeze manifest.

A gesture can pass between resolved acoustic groups without an acoustic bundle
merge. Represent that possibility as a derived cross-group continuation link,
separate from source grouping and local phrase identity. On each 0.1 s relation
refresh, take each retained path's current non-gap run or, if currently in a gap,
its most recent non-gap run in the preceding 8 s. Retain uncertain/censored
endpoints. For a pair of resolved groups, order the runs by attack start. Propose
a handoff union when the earlier run has observed release support, the later run
has observed attack support, and the later attack starts no earlier than 2 s
before that release entry. The later attack must fall within the earlier run's
already observed continuation/release support, or no later than 1 s after its
observed low-energy/gap entry. The latter is the first observed gap-state entry
after release under the existing 1%-RMS gap gate, not release entry itself or a
predicted tail end. If that endpoint has not been observed, only supported overlap
can qualify; a missing interval cannot establish a gap. Equal starts give separate
gestures. The union retains member supports and the observed gap. Register the
2 s lead and 1 s post-completion allowance, comparing 1/4 s lead and 0.5/2 s gap
allowances plus the single-group-only control. The gap-separated fixture is a
positive admission case at observed gaps 0.25/0.75 s, including release durations
0.1/1/5 s; >1 s gaps test initial non-admission and its sensitivity. A listener-
supported one-gesture handoff missed by either time gate is a representation miss
with the same loss-floor/reporting rule as a larger union. It reopens M2's gesture
representation before T1 freeze (and M4 if phrase identity is affected); longer
gaps cannot be silently removed from a declared required stratum. Unknown
articulation or endpoints yield unresolved link support, not an inferred handoff.
These gates admit a hypothesis; they are not a rule that all such overlaps sound
like one gesture. Direction, articulation and phrase interactions can subsequently
change its support under the fitted local/context paths.

The reference gesture view is a finite mixture over eight single-group families
(including residual) and 21 unordered pairs of the seven resolved groups. A family's
raw prior weight is b_g for a single group and sqrt(b_g*b_h) for a pair.
Here b is a gesture-specific acoustic support fraction, distinct from the private
inventory's two-second a_g. Let W=[max(epoch_start,t-8 s),t]; for a current group
generation g use J_g=W intersect its lifetime so a new identity does not inherit
older slot support. At a hop with valid audio and association, define
`alpha_g=sum_i a_ig / sum_(all eight h) sum_i a_ih`, using the front end's already
normalized fractional trajectory-to-group associations a_ig, including its
residual trajectory. There is no additional amplitude or trajectory-salience
multiplier in this prior. Zero total association support is unsupported, not a
uniform assignment. Define `B_g=integral_Jg alpha_g(v) dv`
over valid hops and `b_g=B_g/sum_(all eight g) B_g`. Empty W or zero total B makes
the whole gesture view unknown. Residual participates in this normalization and
in its single family; pair priors use only the seven resolved groups' b values.

For each g, c_g is the valid audio-and-association physical duration in J_g divided
by duration(J_g), using the sample-support union; empty J_g gives c_g=0. Require
c_g>=0.9 for a supported family member and report lifetime/epoch clipping separately.
Normalize all raw single/pair priors before applying these masks or reading labels/
path compatibility. A single with masked g, or a pair with either member masked,
sends its entire normalized family mass to unresolved, without redistributing it
to other families. These priors use raw normalized acoustic support, with no
development z-standardization or stage-3 weighting. Register B/b, W/J, clipping,
normalization set, trajectory-weight source and masks in O03; fixtures include
empty/epoch-clipped windows, new generations, residual-only and masked families.
Compare pair-weight multipliers 0.5/2 around the reference one. Within a family and shared context C,
use q_g for singles and the factorized q_g*q_h for pairs. A qualifying pair emits
one union candidate; a resolved nonqualifying pair emits its two separate runs
with weights b_g/(b_g+b_h) and b_h/(b_g+b_h). A zero denominator has no family
weight. Unknown paths or an undecidable link preserve unresolved mass. Coalesce
identical member/run candidates by adding mass; normalize neither missing nor
nonmatching target mass away. Keep the top 16 candidates per shared context and
carry discarded enumerated mass as unknown. This is an explicit pairwise projection,
not an exact posterior over partitions into gestures or a new cross-group beam.

At most `8*21*16*16=43,008` path pairs plus 1,024 single paths are visited per
refresh; at most 87,040 raw candidate emissions precede coalescing/pruning. Budget
at most eight scalar feature terms per emission (696,320 per bus per 0.1 s cycle)
and 128 retained records at 96 bytes (12,288 bytes), reusing existing run/ancestry
references. Process shared contexts sequentially with a preallocated scratch array
of at most 10,880 raw records at 96 bytes (1,044,480 bytes per bus); sort/coalesce
in place before selecting the top 16, so repeated low-mass candidates cannot lose
their combined rank. Include sorting, this extra view work and scratch in the
complete worker timing/memory census. It does not enlarge the 65,536 per-hop local-state proposal bound.
For each resolved group publish incoming/outgoing union mass and unresolved support
from the preceding refresh. These lagged summaries enter registered stage-3
articulation/phrase features and the same candidate-preview convention; no extra
stage-2 head or reciprocal fitting loop is introduced. A committed link can occupy
an existing episode relation-edge slot, subject to that cap, but it adds no separate
occurrence or observation credit. Private Voice boundaries and clocks remain local.

T1's development fixtures include a two-group overlapping-release handoff, the
same attacks with unrelated sustained overlap, a gap-separated continuation,
ambiguous membership, and a gesture spanning three or more groups. The reference
view directly represents at most two-member unions; larger unions can only match
through the existing uncertain-span rule, and every unsupported larger gesture is
a representation miss with the loss floor, not a silently single-source annotation.
Report this capacity and miss rate before validation. If the declared T1 breadth
requires larger unions, revise the bounded view and refit before freeze; removing
handoff from T1 or declaring all overlap a handoff is not a remedy.

Retain the current recurrence candidates and meter as baselines. Add a competing
arrival model: a distribution over elapsed intervals to the next salient change,
conditioned on recent interval ratios and grouping. It continues to predict when
an expected event is omitted. Compare both mechanisms before choosing which
contributes to the conditional state. Neither supplies a mandatory onset clock.
Use a separately fitted softplus arrival hazard of the §9.4 form, with elapsed
time since the last observed salient change, recent interval ratios and grouping
as features. An omitted event advances elapsed time and survival; only an observed
change resets it. Gap survival is unobserved across missing input. Carry both a
no-unobserved-reset path and a reset-unknown alternative: at gap end, elapsed time
lies between zero and time since the last observed change, then both bounds grow
until another observed change resets them. Propagate the range of arrival/survival
predictions over that interval, not just its gap-inclusive upper endpoint; unknown
reset mass is not an observed event. This defines the arrival baseline independently
of the oscillator.
Let I0 be the most recent completed observed inter-accent interval since the last
acquisition gap, and I1,I2,I3 the three preceding such intervals. The arrival
feature layout is: intercept; `log(1+elapsed_since_observed_accent/1 s)`; the three
informative coordinates `log(Ij/I0)` for j=1,2,3; one missing indicator for each
of those three ratios; then exactly five grouping values and their five missing
indicators, for 18 total coordinates including intercept. The five values are
(1) log2(P_best/1 s), (2) its normalized recurrence-grid support, (3) cos(2*pi*phi),
(4) sin(2*pi*phi), and (5) a 0/1 indicator that a currently matched cyclic-word
proposal uses P_best, where `phi=frac(elapsed_since_observed_accent/P_best)`.
Cosine/sine preserve the circular neighborhood across phase zero; phi itself
is not another linear feature. P_best is the strongest
supported acoustic recurrence-grid peak, with smaller-period ties; use stage-1
admission evidence, never post-joint weights. Phase zero is the last actually
observed accent, not an inferred global beat. An absent period masks all five;
missing/cap-window-limited grouping search masks the word indicator rather than
assigning a credible zero. A completed supported search with no matching word
gives a known zero. All five values use the same frozen standardization/mask
rules as the other head inputs. The no-grouping-feature variant removes these
ten entries and is a registered T2 control; additions or changed coordinates
are model revisions, not manifest-only freedom. There is no I0/I0 coordinate. A ratio is masked if either of its two intervals is
absent; impute its value to the development mean before standardization. A gap-
crossing interval is excluded, not treated as an observed long wait. With no
completed interval the elapsed-time hazard can still run after its first valid
accent anchor, while all three ratios remain masked. Record the exact ordered
layout in the manifest; the two/eight-interval T2 variants have one/seven ratios
and 14/26 total coordinates with this same circular encoding. O03/O09/O10 include
phase 0.999/0.001 wrap fixtures and the old 16-coordinate raw-frac encoding as
a paired development control, with the advance T2 loss/calibration decision rule.
Changing the encoding requires the corresponding arrival and downstream refits.

For each hop, let `E_bus` be the mono mean-square energy defined in §9.1 and
`s_b` the nonnegative NSGT energy in Log2 bin b. The ridge front end supplies
nonnegative per-bin trajectory memberships `v_ib` normalized across trajectories
including residual. With the normalized group associations `a_ig` above, define
`E_gb = E_bus * s_b/sum_b(s_b) * sum_i(a_ig*v_ib)` and
`RMS_g = sqrt(sum_b E_gb)`. This conserves the bus energy across groups; it is an
RMS-equivalent assignment proxy, not a separated waveform. If spectral mass is
zero with positive sample energy, retain energy in residual and mask spectral
shape/flux. Known zero energy is zero in every group. Use
`L_gb = 0.5*log2(max(E_gb, 1e-12))`, in full-scale-squared units before the log,
and raw group flux `mean_b max(0, L_gb(now)-L_gb(previous))` over the unchanged
Log2 grid. Raw envelope rise is the positive adjacent-hop difference of
`log2(max(RMS_g, 1e-6))`. A missing/changed group association or hop masks the
corresponding difference; it cannot invent a new accent. Register these formulas,
floors, memberships and grid with the detector's frozen feature manifest.

Define accent evidence at a supported envelope/flux local maximum: its salience
uses two separately development-standardized components: the positive part of the
adjacent-hop log-envelope rise, and positive spectral flux. For each raw component
`x`, use `z=max(0,(x-mean_dev)/max(sd_dev,1e-6))`; salience is `(z_rise+z_flux)/2`.
An accent candidate exceeds one in this combined-mean scale;
its reference support weight is exactly `clamp(salience - 1, 0, 1)` because the
four-hop admission rule below requires complete acquisition coverage. A factor
for fractional known support is used only by explicitly registered variant
detectors, not this reference. Fractional reference accent weights therefore come
from salience, not partial acquisition coverage. Missing support adds no accent. The 90% coverage condition below
uses the union of actual observed samples, not overlapping NSGT-window counts.
Use these per-group RMS and flux values from adjacent fully observed hops. A three-hop local-maximum comparison detects the
middle hop and timestamps its availability at the third; do not compare across
a gap. Require a strict rise over the left neighbor and a value at least as large
as the right neighbor, so a plateau contributes at most its first peak. The
envelope-rise feature is the positive adjacent-hop log-RMS difference, with a
declared numerical floor for zero energy. Comparing saliences at hops n-1, n
and n+1 uses the four raw hops n-2 through n+1 because each salience includes a
difference. Their nonoverlapping canonical sample intervals form the accent's
acoustic support; its known fraction uses their sample union, not NSGT-window
multiplicity. The reference admits a peak only when all four raw hops and the
needed group associations are available, and both standardized rise and flux
components are computable at each of the three compared salience hops. A masked
component, including positive sample energy with zero spectral mass, makes that
detector comparison unsupported; it is not a zero or an available-component mean.
Report this detector-coverage loss separately from acoustic acquisition coverage.
An available-component-mean detector requires an explicit registered variant.
Hence acquisition coverage is one for an admitted peak; uncertain group assignment
still affects its energy and weight. Freeze this component-mask rule with the
detector entry.
This fixes the initial detector window and its extra availability delay.
Candidate recurrence support is weighted by both endpoint accent weights; zero
weight cannot admit a grouping. The detector, threshold and normalization are
frozen development choices; test thresholds, differencing/support windows and
normalization floors at half/double settings, including ambiguous fractional
assignments and the conservation formula, without interpreting an engineering threshold as a neural constant.

"Salient change" in the arrival model and nonperiodic anchor means exactly this
admitted accent candidate with positive support. Spectral-only positive flux can
qualify; a pure energy decline/release does not reset that arrival clock. Release
still enters articulation and phrase/closure features. At stream initialization,
or before the first actually observed qualifying change, arrival predictions and
nonperiodic anchors are unsupported; stream start is not an observed accent.
Freeze this event definition with the detector, including its sensitivity tests.

Estimate the grouping proposal periods directly from observed accent times,
independently of `meter_existing_234`, the existing meter control defined in §9.4.
Keep at most 128 accents per
group in the preceding 32 s, with observed timestamps and weights. Expire records
outside the physical window first; before inserting a new record into a full
bank, evict the oldest observed timestamp, with lowest stable accent ID for ties.
Never evict by salience weight. Retain the greatest timestamp removed by capacity
eviction for density-window masks; time expiry does not replace that value.
Register this order with the pair-grid/grouping manifest and test a saturated
bank with unequal weights and timestamp ties, exact pair removal and density
masking against an uncapped reference. On a log-period
grid from 0.125 to 4 s at 1/48-octave spacing, accumulate for each accent pair
`i<j` with at least 90% observed intervening support:
`w_i*w_j*max(0, 1-abs(log2((t_j-t_i)/period))/(1/24))`.
Normalize these nonnegative sums over the grid to define proposal support; an
all-zero sum supplies no period. All period consumers use one peak-selection
rule, not the eight largest individual bins. On the normalized linear log-period
grid, a peak is a positive equal-valued plateau strictly higher than each existing
neighbor outside the plateau; select its smallest-period bin. End plateaus use
their sole outside neighbor; an entirely uniform grid has no peak. Equality is
exact in f64. Rank peaks by support, then smaller period. Greedily retain up to
eight whose log2-period distance from every already retained peak is at least
1/24 octave. P_best is the first retained peak; grouping proposals and every
periodic-history pool use this identical cached inventory/support, including
the within-group pool. Absent peaks give unsupported P_best, not a forced period.
Register the plateau/edge/tie rule and separation in O03/O09/O10; compare 1/48
and 1/12 octave separations. Fixtures include one broad peak spread over several
bins, nearby competing peaks, a uniform grid, endpoints and the multiple-period
ladder from isochronous accents. This rule does not identify one ladder member
as the uniquely heard beat. At 241 bins there are at most 121 peak candidates.
O04 separately charges each hop's peak pass, bounded candidate sort and at most
121*8 separation checks per resolved group: at ten hops/seven groups, 16,870
bin visits, 8,470 sortable candidate entries and 67,760 spacing checks per bus,
in addition to the pair-grid work below. This bounded pair-interval estimator (at most
8,128 pairs per group) is an engineering recurrence proposal, not a meter verdict.
Freeze its range, resolution, accent cap and support window on development and
compare half/double settings; truncation is reported computational coverage, not
cognitive forgetting. The old metrical division bank supplies none of these
periods or weights.
Maintain the pair-grid sums incrementally in f64. On a new accent, add its at most
127 pairs with retained accents; each triangular kernel touches at most five of
the 241 grid bins. Cache each pair's eligible support and contributions at creation;
the 32-byte reference record uses two u16 accent-bank slot indices, a u16 base
bin, u16 support flags, five contiguous f32 contributions and four padding bytes.
Remove every incident pair before reusing an accent slot. Evaluate the kernel in
f64 once, round each contribution to f32, then add its exact f64 conversion to
the grid. Pad unused/out-of-grid contribution slots with zero. Removal subtracts
those identical converted cached values; rebuilding uses the same cache, not
newly rounded evaluations. This makes f32 contribution rounding a declared
approximation distinct from f64 summation/subtraction drift. O09/O10 compare both
with direct all-f64 kernel recomputation, including peak/admission changes near
ties and tiny weights; O04 includes the conversions and actual 32-byte layout.
Later gaps do not rewrite that already heard interval. On time/cap expiry remove
all pairs involving the expired accent once, subtracting their cached contributions.
A gap admits no accent, advances the physical expiry window, and permits new
cross-gap pairs only if their original intervening-coverage test passes. Cap
removal is computational expiry, separately counted. Rebuild sums from retained
pairs after each 1,024 admissions to bound subtraction drift (at most once in a
ten-hop cycle); compare with exact recomputation and 512/2,048-admission schedules.

At ten hops per cycle, conservatively allow ten new accents per resolved group:
1,270 inserted pairs and at most 8,128 initially retained plus 1,270 new pairs
removed. Thus incremental work is at most `(8128+2*1270)*5 = 53,340` bin additions/
subtractions per group/cycle. A periodic rebuild adds at most 40,640 bin additions
and 241 clears; normalization each hop adds at most 2,410 bin visits. The total
bound is 96,631 bin visits/group/cycle, or 676,417 across seven resolved groups;
residual has no estimator. These are operation counts, not measured CPU time,
and belong in the same 50 ms worker budget and benchmark. Cache at most 8,128
pair records of 32 bytes per group (endpoints, bin indices/contributions, support
flags), or 1,820,672 bytes for seven groups, plus fixed grid/accent arrays. Layout,
actual visits, rebuilds and expiry bursts are part of resource reporting.

Generate grouping proposals from the eight highest-supported recurrence periods
and observed accent intervals. For integer grouping length L in 2 through 16,
the proposed repetition duration is `D=L*P` for its anchoring period P. Require
two successive observed repetition durations D1,D2, each satisfying
`abs(Dj-D)/D <= 0.10`, with at least 90% observed intervening support. Evaluate
each repetition separately; errors in successive repetitions cannot cancel. Also propose cyclic interval words of
length 2–8, with relative steps in `{1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4}` and
total length at most 16 reference beats, after two matched repetitions within
the same per-repetition total-duration tolerance, now with `D=P*sum(word_steps)`.
A matched cyclic word also requires the declared ordered step sequence (up to its
explicit rotation alternative), with each observed step interval within 10% of
that step's `word_step*P`; equal total duration alone is insufficient. One reference
beat means the particular supported recurrence
period anchoring that proposal. Without a supported period, do not invent a beat
unit or quantized cyclic-word proposal: retain the raw ordered interval relations
for arrival prediction and nonmetrical matching instead.
Refresh the grouping inventory at the nominal 0.1 s stream-time cadence from the
current accent bank, not separately for every path/tuple. For each of the eight
periods, consider every retained accent as a start anchor, ordered by timestamp
then stable accent ID. For an integer L and each predicted boundary
`b_j=t0+j*L*P`, j=1,2, choose the admitted accent nearest in absolute time distance
to b_j among accents strictly after the previously selected boundary anchor
(t0 for j=1) and inside the two-sided window
`[b_j-0.10*L*P, b_j+0.10*L*P]`. An accent before the predicted boundary is eligible;
both predicted boundaries remain anchored at t0. The second tolerance is not doubled and
does not move with the chosen first boundary. Ties use the earlier timestamp/ID, anchors must be strictly increasing, and the
two resulting successive durations must each pass the separate per-repetition
test. Freeze both checks in the proposal manifest and vary their common 0.10
tolerance under the registered sensitivity comparison, retaining the distinction
between absolute search windows and successive-duration errors. Extra
intervening accents are allowed in this integer-group proposal; they do not become
unreported alternative boundaries. O09/O10 include a positive fixture with
P=0.5 s, L=4, t0=0 and candidate accents at 1.9/3.9 s: both anchors precede their
2/4 s predictions, D1=1.9 s and D2=2.0 s pass, and a one-sided search must fail
the fixture. Pin this alongside the existing absolute-window/per-duration tests.
For a cyclic word of length n, use the next 2n consecutive inter-accent intervals
without skipping. Quantize the first n intervals to their nearest allowed relative
step by absolute interval error (ties choose the smaller step), require the 10%
per-step test, and require the next n intervals to match that same ordered word
and the two total-duration tests. An extra accent changes this consecutive sequence
and may break a match; do not silently skip it. Considering every start anchor
already represents rotations. Deduplicate identical period/word/anchor proposals,
rank by summed endpoint support then stable IDs, and apply the 16-fresh cap.
Test a strict-no-extra-accent integer variant and an explicit one-skip word variant
on development, with separate approximation flags. The reference scans at most
`8*128*15=15,360` integer cases (two binary boundary searches of at most eight
comparisons each) and `8*128*7=7,168` word cases (at most 16 intervals and nine
step comparisons each), plus deterministic dedup/ranking per resolved group and
cycle. Include these counts, cache age and the actual grouping-refresh duration
in the 50 ms budget. A time jump refreshes once from the current bank and records
superseded refresh slots, rather than running an unbounded catch-up loop. Retain rotations as alternatives, cap proposals at 16 per
group by support, and keep ungrouped/unknown separately. Maintain continuous
timing residuals: these internal proposals never quantize actual sound or command
onsets. The inventory, admission thresholds and caps are declared engineering
choices, fitted or sensitivity-tested on development material before validation.
Existing candidates can survive an omission via the state transition. The current
2/3/4 inventory is a separate `grouping_inventory_234` proposal control (§9.4),
not the existing meter's period machinery. Unsupported, unfamiliar or nonmetrical
material may remain ungrouped; references can coexist across auditory groups.

These are maximum inventories, not an assertion that every period/word pair fits
the raw admission window. Let `W` be the span between the oldest and newest retained
accents, at most 32 s and possibly shorter after the 128-accent cap. Two nominal
repetitions of length `L` at period `P` require `2*L*P <= W`; the actual admission
test uses the observed repetition durations and the stated timing tolerance.
Thus at `P=4 s`, a full 32 s window supports at most four nominal reference beats
per repetition, whereas shorter periods can reach the maximum inventory. Report
window/cap-limited non-admission separately from lack of recurrence evidence.
T2 includes these boundary cases and half/double-window comparisons; an
unreachable candidate is computationally unsupported, not evidence of absent meter.

The private reference inventory is drawn from committed, still-retained episode
handles (§9.3) currently matched to heard spans in resolved habitat groups. It is
not a table of authored Voice IDs or intentions. The worker reuses each group's
existing 0.1 s matcher result and the committed-span/current-group association
from §9.6; it performs no extra Voice-specific DTW. For group g, let a_g be its
normalized observed assignment weight in the preceding 2 s (over resolved groups),
and s_eg its current cue-to-episode retrieval score. On that same window
`W=[max(epoch_start,t-2 s),t]`, define c_g as physical duration with both actually
observed audio and valid group-association support divided by duration(W).
Known inactivity with valid assignment counts as observed; missing audio or
unsupported association does not. An empty window gives c_g=0. This is not the
0.5 s action-coverage factor or the 0.1 s matcher cadence. Record W, numerator,
denominator and epoch clipping in the inventory manifest. Define
`z_eg = a_g*c_g*exp(s_eg)/(exp(bias)+sum_j exp(s_jg))` over that group's supported
retrieved episodes. Here bias is exactly the stage-1 fitted no-memory bias from
§9.3, reused frozen; the participation pathway introduces no new bias parameter.
Record that shared parameter reference in the inventory's manifest entry. The score compares the actually heard current group cue with
a stored episode using §9.3's normalized match/availability formula. The total z
across groups/episodes is at most one; no-memory, missing and ineligible support
remain unassigned. Stage 1 uses its frozen matcher/retention without stage-3 focus;
production may use the fitted common context, with this change covered by the
integrated-loop test, never a private trace's own success as its match evidence.

A reference key is `(bus_epoch, episode_handle, coordinate_family)` where family
is periodic phase or nonperiodic relative arrival. For each current group binding,
choose periodic only with its supported §9.2 periodic reference; otherwise choose
nonperiodic only with a valid observed anchor. Sum z over bindings with the same
key, retaining their normalized anchor/period alternatives (at most seven, one per
resolved group), rather than averaging phases. The group-generation handles are
current associations, not permanent parts of the remembered episode identity;
a returning episode can acquire a newly observed group association. Unsupported
anchors add unassigned mass. Rank keys by summed z then stable handle/family and
publish at most 16 for the whole habitat, reporting the discarded mass. The pool
examines at most 7*16=112 cached group/episode entries per worker cycle. Its records
have a 512-byte cap including the seven anchor alternatives: at most 8,192 bytes
per published inventory, separately accounted with immutable summaries.

At action issue, freeze these at most 16 reference keys, z weights, epoch, support
and anchor alternatives with the pending action. Reading them adds no memory.
Create a private trace only when that action has a confirmed actually rendered
onset or release with positive observed assignment; initialize its bins/strength
at zero before applying that first outcome. Let o be the known fraction of the
existing own-sound result interval for this executed event; failed-to-sound or
unconfirmed execution has o=0. Set `r_i=o*z_i` for each still-retained, same-epoch
reference with its frozen valid anchor, otherwise zero, and set
`r_unassigned=1-sum_i r_i`. Do not renormalize discarded/missing mass onto survivors.
The outcome is associated with its issue-time audible context, not reassigned by
a future cue or the timing bin it happened to favor. Integrate each r_i over that
reference's frozen anchor/outcome timing distribution; add the resulting mass to
its onset/release bins once. For another retained trace j, this outcome adds
`sum_(i!=j) r_i` interference; unassigned mass adds none. Retrieval alone and
unexecuted alternatives add neither reinforcement nor interference. Apply the
same rule in fixed-policy stage-1 runs and production, freezing all inventory,
assignment and result-support definitions before participation fitting. Test empty
inventory, first trace, competing references, cutoff mass, failed execution,
epoch/handle loss and a returned episode with a new acoustic association.

For a generator Voice, a temporal niche is a remembered relation between its
*executed* participation and an inferred audible pattern. Store pattern-match
weights and the distribution of relative onset/release timing, with the period
and its uncertainty when a periodic reference is supported. Without one, use
relative arrival time or overlap. Keep multiple offset modes instead of averaging
them into phase zero. Update from observed outcomes once; never from an intended
onset that failed to sound. The reference trace uses separate onset and release
histograms: 32 circular bins for periodic relative phase, or 32 linear bins from
zero to four intrinsic periods for nonperiodic relative arrival time, with explicit
out-of-range mass. For the latter, time zero is the most recent observed salient
change assigned to that pattern reference and available when the action is issued.
Freeze its timestamp alternatives, reference weights and the Voice's intrinsic
period at that issue; later reinterpretation cannot move the coordinate of an
already issued action. An absent or gap-obscured anchor leaves the trace unknown.
The nonperiodic bin coordinate is explicitly dimensionless,
`u=(outcome_time-anchor_time)/intrinsic_period_at_issue`, in [0,4]. Mixing outcomes
from different issue-time periods is intentional relative-pace transfer, not a
histogram of absolute seconds. Candidate lookup uses the current issue-time period.
Period-change comparisons test this transfer against resetting the trace and an
absolute-time control; its validity is not assumed from fixed-period fitting.
For both kinds of reference, mix the supported anchor/phase alternatives by their
weights and integrate the outcome-minus-anchor timing distribution into the bins
(wrap modulo one only for phase). Timestamp resolution within a known hop is
represented by a uniform interval; acquisition gaps are missing mass, not a
uniformly guessed anchor. This spreads mass without selecting an artificial mean
reference. All bins are retained; no circular-mean merge collapses
multiple modes. Keep at most 16 pattern-reference traces per Voice. Each bin's
occurrence contributions use the retention rule in §9.3 with participation-specific
parameters, then normalize for lookup; no observed support leaves an unknown trace.
For N=32 periodic bins with centers `(k+0.5)/N`, wrap the query phase phi into
[0,1), set `v=N*phi-0.5`, j=floor(v), and linearly interpolate the normalized
masses at j modulo N and (j+1) modulo N with fractional weight v-j. For N linear
bins on [0,4], centers are `4*(k+0.5)/N`; linearly interpolate adjacent centers
and clamp queries inside [0,4] beyond the first/last center to that endpoint
bin. A nonperiodic query outside [0,4] is unsupported for action pressure;
overflow remains a separate observed/assay outcome. Normalize including overflow
mass, then exclude its bin from interior lookup without renormalizing the
remaining bins. These lookups return interpolated bin mass, not a density in
seconds. Integrate a query timing distribution over this same lookup function
and keep unsupported query mass unassigned. Freeze these rules with N/range and
use the corresponding centers under the registered half/double-bin comparisons.
For a pattern-reference trace, interference is the summed assignment support of
newly observed own-action outcomes attributed to competing pattern references.
It is not opposition between neighboring timing bins. Outcomes assigned to the
same reference reinforce its supported bins; unexecuted candidates contribute
nothing. Fit participation-specific tau/kappa/strength in stage 1 from development runs'
actually observed own-action outcome timing, using held-out prequential timing-bin
log loss. Score each r_i-weighted observed bin distribution against the trace
prediction frozen before that outcome, averaging over assigned outcome mass.
A zero-mass initial trace uses a uniform 32-bin prior (33 including overflow for
nonperiodic timing) for this assay only; it remains unsupported for action pressure.
Report assigned/unassigned coverage alongside conditional bin loss, keeping the
same issue-time assignments for full and elapsed-time-only models. The matched
elapsed-time-only trace is the ablation. Freeze the body,
seeds and exposure policy for that comparison; it identifies a generator trace's
predictive utility, not human retention. Closed-loop behavior and groove judgments
still require their separate gates.
Evict the least available reference only for capacity, report it, and test
half/double bins, range and cap. This replaces the averaged energy-context template
only after the named T2/T3 local-generation comparisons pass all four applicable
gates in §10.3, including causal correctness, the task-specific cognitive criterion,
intended audible relations and author acceptance. Until then the current 2x3
template remains the production reference; failure reopens the trace design on
development. Record the tested conditions, comparison IDs and promotion decision
in technote §9.3.55; better timing-bin loss alone cannot authorize replacement.
O19's replacement is scoped to **local participation mode**: there the promoted
trace substitutes for the old 2x3 participation-memory contribution, rather than
adding a second copy of that contribution. Off and passive-observation modes
retain the unchanged 2x3 production behavior after O19 promotion. O19 does not
change the shipped default or O21's accepted off-mode baseline. Making the trace
the shipped default requires a separate authored promotion that versions the
default-mode contract and O21's expected-mode/baseline fixture together. Its new
accepted-baseline version cannot retrospectively turn an earlier mismatch into
a pass; explicit off continues to be checked against the retained legacy baseline.

The two relation instances also need auditory, not Voice-private, relative timing.
For each ordered pair of distinct acoustic group handles `(g, reference_g)`, retain
two reference slots: one reserved for a periodic hypothesis and one for the
nonperiodic median-interval reference. Only the seven resolved groups participate,
so there are at most 42 pairs and 84 reference histories per bus. The periodic
pool is the reference group's eight highest-supported recurrence-grid periods
from the independent accent estimator, with positive support and valid coverage.
Initially choose the strongest, using stable period-grid IDs for ties. Replace
an incumbent only if the same challenger has at least 1.25 times its support
continuously for 1 s of physical stream time with at least 90% observed coverage.
Reset this challenger timer on a failed comparison, changed challenger or gap.
An ineligible incumbent is masked immediately; retain its history dormant until
a supported replacement passes the rule, rather than presenting stale support.
If incumbent support is zero, any positive challenger satisfies the ratio but
still needs the dwell interval. A replacement starts a new periodic reference
version and records the discarded history; retrieval requests do not bypass it.
The nonperiodic slot is never filled by a second period. It is unsupported until
its anchor and minimum interval count qualify; an updated median freezes only
each new record's scale, without resetting that same dimensionless reference
family. Periodic and nonperiodic bins never share storage. Test 0.5/2 s dwell and
1.125/1.5 support ratios, including alternating near-tied periods and gaps. For
reference-count sensitivity, one slot means nonperiodic only; four mean nonperiodic
plus three periodic slots with the same selection/replacement rule. Freeze these
choices with the existing half/double history comparisons. At an observed accent in g, use the reference group's latest observed
accent no later than that accent's support time and available at this update.
For a supported periodic hypothesis, use 32 circular bins of
`frac((t_accent_g-t_ref_accent)/P_incumbent)`, with zero at the reference accent.
At each canonical update, snapshot every group's incumbent period/version and
eligibility before admitting that update's accent batch to any pair grid. Score
all inter-group records against that snapshot, using the latest reference accent
actually available in the batch at or before the target's time. A previously
unsupported period supplies no record. Then admit the batch to the grids and
apply incumbent/challenger updates; replacements or loss of support mask/retire
the old history under the existing rules. A current accent cannot select its own
period reference retroactively. O03/O09 pin this order and formula with phase
0/1 wrap, simultaneous accents, period replacement, unsupported incumbent and
permuted group-processing-order fixtures.
The nonperiodic slot uses 32 bins of `elapsed_from_reference / median(last_four_valid_reference_intervals)`
in [0,4], plus out-of-range mass. Require at least two completed, actually observed
reference intervals since the last gap; with two to four, take the median of those
available (the mean of the middle pair for even counts). With fewer than two the
scale is unsupported. Test minimum-count variants one/four and history caps two/eight
on development, freezing the selected rule with the timing-history configuration.
Without an anchor or valid scale, add no timing
sample and mark it unsupported. Mix timestamp/period uncertainty as in the private
trace, but obtain every value from observed acoustic groups, never Voice actions.

Each history keeps at most 128 weighted timing records in a sliding 8 s observed-
time window, with coordinates/reference versions frozen when observed. Remove
expired records from its histogram; missing input adds none and retains a coverage
mask. A reference-unit switch starts another reference history; cap eviction is
reported, not merged into a mean phase. On group retirement its short-window
histories expire; episodic relation memory remains separate. Normalize retained
mass for lookup. A mode is a positive-mass maximal equal-valued run higher than
each adjacent bin outside the run. Equality is exact in the normalized f64 bin
masses; no hidden contrast epsilon is used. Periodic runs/neighbors wrap modulo N;
a wrapped plateau contributes its first bin after the strictly lower predecessor.
Linear end runs compare their sole existing neighbor and contribute their first
bin in increasing order. An entirely uniform histogram has zero modes. Otherwise
each qualifying plateau contributes one mode, with the representative bin center
as its position; stable bin ID breaks subsequent equal-mass ranking ties.
Overflow is not a mode bin. Near-uniform unequal bins can have small extrema;
register uniform, plateau, wrapped-plateau, endpoint and small-perturbation fixtures
with the bin/smoothing sensitivity comparison. With N even, the maximum is N/2
modes (16 at N=32) for either family. Residual dispersion is the weighted squared
circular/linear distance to the nearest mode; with zero modes it is masked while
the supported zero mode count remains a valid feature.
No supported sample yields a missing feature, not zero dispersion. Cap-limited
window coverage also masks the feature below the declared 90% threshold. The
inter-group record count is 10,752 per bus; the seven within-group histories below
add 896, for 11,648 total. Each record uses at most 64 bytes, plus fixed
histograms/metadata; include it in resource reporting. Freeze window, reference
selection, coordinate units, masks and mode rule with T3; compare half/double
windows, bins, per-history records and references per pair. This is the public
source of the groove heads' inter-group offsets and persistent/residual variation.
Use one distance convention for these histories, the groove map and section
statistics: periodic distance is the shortest circular distance divided by 0.5
cycles; nonperiodic distance is absolute linear difference divided by four, with
no wrap. Coincident reference means zero in both coordinate families. Integer
periodic cycles coincide modulo one; a nonperiodic delay of one median interval
is a sequential relation, not declared simultaneous or modulo-aligned. Its positive
offset is descriptive, not a penalty or proof of poor coordination. Persistent
offset uses squared distance of each mode from zero; residual dispersion uses
squared distance of a record from its nearest mode. Register a nearest-integer
nonperiodic-distance control on development to test that reference choice, without
silently assuming a periodic beat for nonperiodic material.

Also keep one within-group periodic history for each resolved group, independently
of whether another group exists. It follows the same acoustic period pool,
incumbent/dwell/version rule, 32-bin circular histogram, 128-record/8 s bounds,
mode/dispersion and 90% support masks. At each observed accent, use that group's
strictly preceding observed accent and the incumbent period/version frozen at
that preceding accent, before observing the current outcome. Add the interval
phase `frac((t_current-t_previous)/P_frozen)`, weighted by the lesser of the two
accent support weights. Require a fully observed intervening interval and the
same currently supported period version; a missing interval, unavailable previous
period or changed version adds no record and remains unsupported. Freeze the new
previous-accent reference only after this scoring step. Zero represents an integer
multiple of the local period. This is an interval residual relative to an inferred
local recurrence, not a global beat phase or a recovery of absolute laid-back
placement without an audible reference. Timestamp uncertainty uses the existing
record-bin integration rule; no private action or authored grid enters it.

The within-group slot is not in competition with the two inter-group reference slots.
Use its two modes and the same 14-value/mask layout, with periodic flag one,
selected-history weight one, and inapplicable linear coordinates/overflow zero
when supported. It adds 28 coordinates per group, including masks. For section
statistics it joins that group's supported outgoing histories with the same
retained-weight times current-period-support weighting; a single resolved group
therefore has a defined timing source. Report within/inter-group support separately.
The seven histories add at most 896 records (57,344 bytes), plus their fixed bins,
references and previous-accent metadata, to O04's census. Register formulas,
causal reference freezing, first-accent/gap/version fixtures and half/double bins,
window and record-cap comparisons in O03/O09/O10. Compare the map with this slot
removed on the required T3 single-group stratum: systematic swing versus random
jitter matched for tempo, density and interval marginal distribution, plus a
straight reference. Keep the same groove/desire questions, uncertainty and
predeclared loss/calibration criteria. The histogram alone cannot identify interval
order; the existing ordered-word features supply that candidate distinction.
Failure on listener-supported distinctions reopens M2/M7's timing feature map
before stage-2 freeze, including both the head and section consumers. Do not drop
single-group material or count a two-stream success as this stratum's pass.

For the inter-group part of a group-local groove/desire head, use only outgoing
histories `(g, reference_g)`:
at most 12 at the default six other groups and two references. Rank supported
histories by retained record-weight sum times current reference support, using
stable reference-group handle then reference-family ID for ties. Nonperiodic
reference support is the fraction of its required interval history actually
observed; periodic support is its normalized recurrence-grid support. Retain the
top two histories for a fixed feature map. Within each, rank local-maximum bins
by normalized bin mass then stable bin ID; retain two modes. Each mode has four
value coordinates: cosine and sine of circular phase, linear position divided by
four, and normalized bin mass. In periodic mode the linear coordinate is a known
inapplicable zero; in nonperiodic mode cosine/sine are inapplicable zeros. A slot
also has six value coordinates: periodic/nonperiodic flag, normalized history
weight among the selected histories, normalized nearest-mode squared dispersion,
normalized mode count `log(1+count)/log(1+ceil(N/2))` (log(17) in the denominator
at N=32), overflow mass, and observed coverage. This count coordinate reaches one
at the declared maximum; freeze the N-dependent normalizer with bin variants.
Distances for dispersion divide circular distance by 0.5 cycles or linear distance
by four before squaring. The two mode slots and six other coordinates give 14
values per history, 28 total. Give every value a missing indicator: the whole map
has 56 coordinates. Absent histories/modes are masked, not credible zero offsets;
inapplicable coordinates are distinguished by the family flag. Mode count and
residual dispersion use all retained modes, so the two-mode truncation does not
claim unimodality. Freeze this map before the groove/desire stage-2 fit, with the
rest of the head's ordered manifest entries. Compare one/four history slots and
one/four modes on development, including rank ties, switching references and
multimodal histories. A residual-mixture group's map is entirely masked.

Groove is tested as experienced temporal participation (§10), not assigned to a
neural frequency or equated with a successful niche estimate. Add two separate
regularized ordinal heads: judged groove and judged desire to participate, each
on a five-level response scale. At a probe, ask about the preceding 8 s:
"How strongly did this timing make you want to move your body?" for the operational
groove target (none, slight, moderate, strong, very strong), and "How much would
you want to join this musical flow by making sound?" for participation (not at all,
a little, moderately, strongly, very strongly). These distinguish bodily movement
invitation from a desire to contribute sound; they do not exhaust the word groove's
meanings. Freeze wording, translated equivalents and anchors before collection.
The 56-coordinate inter-group map and 28-coordinate within-group periodic slot
together define the reference timing-residual, persistent-offset and variability
inputs. There is no other undeclared arrival/recurrence residual coordinate.
The remaining feature families are cyclic-word surprise/complexity, event density
and the explicitly defined acoustic grouping support below.
Listener heads use inferred auditory relations, never generator-private traces.
Persistent offsets come from retained histogram modes; residual dispersion around
them is a candidate distinction between systematic timing and random jitter,
whose limitations are tested with the required matched-marginal T3 controls. Density
uses each existing history window (0.125, 0.25, 0.5, 1, 2, 4, 8 and 16 s), with
counts divided by actually observed time. Each group also retains the most recent
accent timestamp removed by capacity eviction. If such an eviction intersects a
requested density window, mask that component and report cap-limited coverage;
do not divide a truncated count by the full observed duration. Expiry older than
the window does not invalidate it. Apply this rule to every retrospective accent-
count window, including continuation's 2 s density. Span/section cumulative event
statistics instead consume each admission once and retain their sums independently
of the accent bank; later bank eviction does not erase those already accumulated
counts. Test a dense fully observed stream exceeding 128 accents inside each
relevant window against an uncapped offline count, keeping acoustic and capacity
coverage distinct. Use the preceding 8 s of physical
support for residual, offset-histogram, word-complexity/surprise and grouping
features. Each feature has a known-support mask; less than 90% observed coverage
in its window masks that component rather than substituting observed zero.
Standardize observed components on development, with separate missing indicators,
and test half/double windows and development alternatives to coverage thresholds.
Predictions use the same whole-mixture probe aggregation as §9.4, with the 8 s
groove window in place of the closure window. These are feature-map windows,
not universal cognitive integration times or arbitrary report counts.
For cyclic words, define complexity as normalized symbol-frequency entropy
`H=-sum_b p_b*log(p_b)/log(9)` over the nine declared interval-step symbols, with
`0*log(0)=0`. Within each group's preceding 8 s probe window W, let a_h(t) be
the stage-1 acoustic admission support of each currently matched cyclic-word
hypothesis h, normalized over matched words at that t, and let p_hb be the
fraction of h's steps carrying symbol b. Identical word hypotheses are coalesced
before normalization. Define `p_b = integral_W [w(t)*sum_h a_h(t)*p_hb] dt /
integral_W w(t) dt`, where w is valid acoustic assignment support only where a
quantizable matched word exists. No matched word gives w=0, not a uniform word;
ungrouped/unquantizable/missing intervals are masked. Require the existing 90%
window support rule also for this valid word support; zero denominator is
unsupported. These weights are admission support, not stage-3 posterior weights.
Register this formula and its masks alongside H and H-squared. Include both H
and H-squared in the full head and in the otherwise
identical complexity-only control, allowing a nonmonotone response without assuming
its sign. Surprise is `-log((C_ab+0.5)/(sum_b C_ab+9*0.5))` for the newly observed
step b after step a. C is a 9-by-9 table of cumulative, fractional observed step-pair
counts per acoustic group generation, initialized at zero. Score before adding
the new pair; uncertain word assignments share normalized total update weight.
Use f64 for these long-lived counts outside the audio thread. These are background
transition statistics, distinct from episode availability; group retirement/epoch
reset ends this table, and retrieval alone never increments it. An ungrouped,
missing or unquantizable interval masks surprise/complexity rather than inventing
a step. Supported unseen pairs use the explicit half-count smoothing. Aggregate
these features over the declared 8 s probe window and compare half/double smoothing
and a decayed-count alternative on development. No index is claimed to be a
universal musical-complexity or syncopation measure.
Define instantaneous acoustic grouping support G_g at each completed grouping
refresh as the maximum, over currently admitted integer/cyclic-word proposals,
of the mean accent support weight over the distinct observed accents used by that
proposal's two repetitions. Integer proposals use their three boundary accents;
words use the 2n+1 consecutive accents. This is in [0,1]. A supported completed
search with no admitted grouping has G_g=0; an unsupported or cap/window-limited
absence is masked, not zero. A supported admitted proposal can supply G_g even
if longer, unenumerated groupings would require more history; this is support of
the admitted finite inventory, not proof about every possible grouping. Hold the
cached value/mask until the next registered refresh. The head's grouping feature
over its preceding 8 s window is `integral_valid alpha_g*G_g dt /
integral_valid alpha_g dt`, using the instantaneous association alpha_g above,
with the existing epoch clipping and 90% valid-support mask. Zero denominator is
unsupported. This definition also supplies the continuation
head's analogous two-second grouping feature. It uses stage-1 acoustic admission,
not the joint posterior or a fitted head output.

Both groove and desire heads use exactly the same ordered 109-coordinate reference
layout: intercept (1), the inter-group offset map (56, including masks), the
within-group periodic slot (28, including masks),
density values in ascending window order plus their masks (8+8), H/H-squared/
support-weighted mean observed step surprise over 8 s plus their masks (3+3), then
the grouping-support mean and its mask (1+1). Surprise is averaged with the same
fractional observed step-pair update weights defined above; no pair gives a mask,
not zero surprise. Each word feature retains its existing valid-word/window mask.
Freeze these ordered IDs, formulas and global development standardization in
O03 before fitting either head. The two heads have independent coefficients and
calibration; their input inventory is not chosen independently after inspecting
ratings. The named reduced-feature controls use subsets of this same layout.
Fit these heads in stage 2, with separate stage-4 calibration. Compare against
onset-concentration-only and complexity-only ordinal heads on the same judgments.
Failure to improve the registered held-out criterion falsifies the proposed T3
feature/model connection. A good timing forecast alone cannot pass it. The
participation head contributes a separate, optional contextual consequence in
§9.6; it is not an automatic survival reward or a universal musical objective.

### 9.3 Correspondence, episode memory and retrieval

A completed or provisionally continuing span stores time-indexed descriptors:
relative pitch motion where supported, envelope/articulation shape, relative
intervals, overlap, register and timbral profile. Store coverage and confidence
per component. Ordered span edges support partial and overlapping sequences;
absence of a pitch descriptor is not a zero pitch difference.

The reference extracts one raw descriptor sample per canonical audio hop per current
group, including residual, from that hop's actually observed §9.2 assigned energy
and spectrum. Its ten value coordinates, in order, are log2-frequency centroid,
log2-frequency spread, log2 RMS, positive adjacent-hop log-RMS rise, positive
adjacent-hop log-RMS decline, positive adjacent-hop log-spectral flux, assigned
energy share E_g/E_bus, and three centered spectral mass fractions. Centroid and
spread are the mean/sd of log2 bin frequency under E_gb in that hop; log2 RMS is
log2(max(sqrt(E_g),1e-6)). Rise/decline and flux use the existing valid adjacent-hop
comparisons, never inferred articulation labels. The three fractions sum E_gb in
log2-frequency-minus-centroid intervals (-infinity,-0.5), [-0.5,0.5], (0.5,infinity)
and divide by total group spectral energy. Zero spectral energy masks centroid,
spread and all three fractions; known zero bus energy gives known zero energy
share. These are a coarse observable timbral profile and envelope/overlap proxies,
not identified pitch, source identity or a fitted cognitive head. Relative pitch
motion and interval relations are obtained from this ordered centroid/time series.
All other unavailable coordinates are masked, not imputed as acoustic zero.

Use the original hop support and its endpoint as the raw sample time, retaining
the actual availability timestamp and full supporting-audio endpoint for causal
checks. A newly created span clips assignment weight to its observed support;
source-feature support may start earlier and remains separately recorded. Never
use evidence beyond the query/commit cutoff. A reported acquisition gap contributes
one all-masked interval record covering that gap, with no generated hop-by-hop
fill or new observation weight. New group generations do not inherit raw samples.
Create a descriptor knot from each nonoverlapping pair of canonical hops, using
the moment aggregation below. Align pairs from that group generation's first
hop; a span start/end or gap closes a partial block with its actual support.
Coincident pair and span endings are not duplicate knots. Coalesce contiguous
gap reports: a resumed hop can close an earlier partial block, append one masked
gap block and close the new observed block, at most three insertions per retained
span; ordinary hops close at most one. After beam selection, each retained span consumes each eligible raw record once;
cue prefixes and stored episodes use the same extraction and compression rules.
Physical span length is not fixed by 128 knots: compression, specified below,
keeps bounded summaries as a span continues. Commitment freezes the result.

Before stage-1 correspondence fitting, O03 fixes these coordinates, cadence,
provenance and global development means/sds (sd floor 1e-6 in each coordinate's
native units). Rebuilding knots after changing those statistics is mandatory.
Compare half/double cadence using one/four canonical hops per descriptor block
from the same raw samples; the canonical relation update remains 512/48,000 s.
Register the aggregation cost, masks and clipped boundary
blocks. Compare short and long sustains, rapid gestures, gaps and tempo-scaled
copies before any affected fit is frozen. This is a sampling/compression choice,
not a cognitive integration-time estimate.

Knot density after adaptive compression need not agree between a cue and an older
episode, even under affine time scaling. The matcher uses retained original times:
for each cue knot, map its time by the candidate tempo ratio and anchor offset,
then center the 33-reference-knot band on the nearest reference representative
time (earlier-time ties). It does not map cue index to reference index. Coarse
index pairing and interval-median tempo estimation still incur a declared density
approximation; there is no claim that compression makes them invariant. O10
includes identical/tempo-scaled audio at unequal span lengths and compression
levels, compared with uncompressed exhaustive matching, recording transformation
error, candidate misses, band-edge hits, T4/T6 losses and compression coverage.
Failure of a required relation comparison reopens M3's representation/index before
freeze; it cannot be excused as absent musical repetition. These cadence/density
comparisons use the existing advance retain/revise decision rules.

The reference matcher is causal, prefix-to-episode dynamic time warping with
explicit insertion/deletion costs. It compares observed prefixes only. Match
features separate order, relative pitch motion, time deformation, articulation
and timbre. Only transposition and tempo ratio are explicit transformation
coordinates in this reference. Timbre and articulation changes remain separate
component residuals, with their tolerance/scales identified by stage-1 correspondence
fitting; the matcher does not infer a timbre or articulation transformation operator.
T4's timbre/articulation variation tests correspondence retention under these
residual tolerances, including failures outside them. Human transformation tags
are targets for this distinction, not missing dimensions silently claimed as
estimated transformations. Estimate transposition and tempo ratio explicitly;
retain their residual error and the evidence for each. Their admissible search
ranges are recorded engineering bounds. A match outside those bounds is unknown,
not a proven novel motif. Order-shuffled and energy-matched controls prevent a
bag of local sounds from passing as remembered sequence structure. Normalize DTW
cost by the number of observed cue steps, include insertion/deletion penalties in
that numerator, and carry unmatched/missing coverage separately. Initial searches
allow transposition within ±2 log2 units and tempo ratio 1/4–4, using 1/16-log2
grids with local refinement to 1/64. Test half/double bounds and grid spacing;
boundary optima or unsupported transformations produce unknown rather than novelty.

O03 names `dtw_insertion_penalty` and `dtw_deletion_penalty`: positive cost per
unmatched cue/reference knot respectively, in the same dimensionless squared
development-standardized residual units as one supported matched step. The
simulation/reference start is 1 for each. Stage-1 correspondence fitting selects
each from {0.5,1,2} by the registered held-out correspondence loss, with smaller
total penalty then smaller insertion penalty breaking exact ties; freeze both
before the recognition fit. A missing acoustic value is not itself an insertion
or deletion. Register half/double selected-penalty comparisons, separately and
together, including normalized DTW loss, section exact/transformed/unmatched
classification at d=0.25/1, mid-episode retrieval miss rate and T4/T6 outcomes.
Apply the existing advance decision/refit rule and charge these fits to O12.

A correspondence edge links two episodes and carries a transformation vector
and its uncertainty. Matching a new realization never overwrites the earlier
episode. A repeated sound is not automatically the same motive: alternative
matches compete, and contextual continuation must distinguish them. A prefix can
retrieve several earlier continuations until subsequent sound resolves them.

The memory controls below have fixed operational meanings in T4/T6 and prospective
T7 comparisons. O03/O07 register their exact inputs, disabled feature IDs and
refit matrix before any margin is set; the descriptive short names elsewhere
refer to these variants, not controls to be invented after collection.

- `memory_no_retrieval` admits no stored-episode correspondence candidate to a
  relation state, reference inventory or action column. Keep known no-memory and
  epistemic unknown, the observed current-span state, and the same causal bank-
  writing/retention algorithms; removal of retrieval does not freeze elapsed time
  or delete heard observations. Writes follow this control's own causal state,
  not hidden full-model assignments. Mask episode-match, availability/focus,
  retrieved-position/continuation and episode-linked private-fit features and
  their joint interactions. Only its fitted no-memory bias supplies the known
  correspondence admission. The existing recognition link then has zero stored-
  return mass, recorded as a structural limitation. Also score the stronger
  `return_constant` control, `(n_yes+0.5)/(n_judged+1)` from training responses,
  on the same recognition targets; defeating a zero-return predictor alone does
  not pass the T4 recognition comparison.
- `memory_recent_only` retains the same bank/write rules but permits retrieval
  only when decision observation time minus the last observed reinforcement
  endpoint is <=16 s. Compare 8/32 s horizons. Apply this eligibility before
  candidate ranking and to every downstream retrieved/focus/continuation/private
  reference; an older handle cannot re-enter through an alias, cached stay or
  joint feature. Current observed-span evidence remains available. Report
  horizon-excluded mass/records separately from retrieval/capacity misses.
- `memory_time_only` keeps content matching, strength/saturation, competition
  and elapsed-time decay but sets the interference term to zero in availability;
  kappa is absent, not fitted and then ignored. Keep interference diagnostics
  separate from consumers, and remove all direct interference-derived joint
  features. This is the elapsed-time-only memory law, not `section_elapsed_only`.
- `memory_orderless` replaces ordered cue/anchor/DTW comparison with a fixed
  bag summary of the same observed descriptor-value coordinates: for each
  coordinate, its valid-knot mean and population standard deviation (one valid
  knot gives zero deviation; no valid knot masks both). Exclude timestamp/index,
  sequence-position and explicitly order-derived matching fields. Local acoustic
  values attached to a knot remain part of the bag; this is a content-order
  ablation, not removal of causal acquisition or local articulation evidence.
  Summarize the whole bounded cue and each stored episode, never an ordered
  first-eight-knot prefix. Estimate transposition as the difference of their
  supported log-frequency medians and tempo ratio from the difference of median
  log positive stored local-interval values; do not recompute intervals after
  permutation. Use the same transformation bounds, 1/16 rounding and up to four
  bracketing 1/64 refinements. Out-of-range/unidentified dimensions keep the same
  masks. For each transform, cost is the mean squared difference of the common
  globally scaled mean/deviation coordinates; if no coordinate is jointly valid,
  the cost is unsupported.
  Rank all episode bags by this cost, retain 16 with stable episode-ID ties and
  use exp(-cost) for interference under the same cache/support rule. No ordered
  anchor scan, DTW, path position or next-segment projection is consumed. Mask
  those fields, remembered-order transition/continuation links and their joint
  interactions; identity/availability and unordered section content masses remain.
  Refit the match/recognition mapping with these costs. The bag also loses detail
  beyond its first two marginal moments and does not preserve subsequence lookup;
  report those limitations. A claim specifically about order requires the
  registered matched-multiset/mean-deviation fixtures and controlled audio
  comparisons, not any full-versus-bag score advantage. Exact descriptor-row
  permutations must leave this control's cost unchanged while preserving masks,
  local values, availability and transforms.

T6's time-only/recent-context comparisons name `memory_time_only` and
`memory_recent_only` in addition to the separately specified `section_elapsed_only`
and `section_recent_only` heads. T7's prospective full-context comparisons use
these same memory variants and `memory_orderless`/`memory_no_retrieval`; its
retrospective completion-head controls remain separate. Under each whole-model
variant, replay/refit affected stage-1 correspondence/retention components (omit
unused parameters), then all downstream stage-2 heads whose inputs changed, and
stages 3/4. Hold unaffected acoustic extraction/scaling fixed. Frozen full-model
head weights cannot substitute for these refits. The variant mask list includes
all interactions that would restore removed information, including action columns;
keep target data, splits, body/candidate controls and calibration rules comparable.
Charge these refits in O12's finite matrix and compute envelope before collection.
O04 also censuses bag construction/refinement and caches separately: at most eight
queries, 256 episode bags and four transformations, each visiting at most 128*64
raw descriptor values plus 128 moment differences. Cached unchanged episode bags
may be reused only with identical support, scaling and transformation. No unbounded
search or free control computation is assumed. Register synthetic permutations,
horizon crossings and feature-leakage fixtures in O09/O10, with independent
return/continuation metrics and the existing advance decision rules.

Use separate availability and cue-match factors. For episode `i`:

```text
availability_i(t) = strength_i * exp(-elapsed_i / tau_time - interference_i / kappa)
log_availability_i = log(strength_i) - elapsed_i/tau_time - interference_i/kappa
retrieval_score_i = logaddexp(log_availability_i, log(epsilon_avail)) + w_match · match_features_i
epsilon_avail = 1e-300
```

`elapsed_i` is time since its last *observed* reinforcement. For each stored
episode i, a new span increments interference_i by its similarity to i's content,
weighted by the support assigning that span to a competing occurrence. Similarity
is `exp(-d_coarse_i)` under the cached coarse-anchor rule specified below, not
an uncomputed full-DTW cost for every stored episode. Support
assigning it to a recurrence of i instead reinforces i; it is not simultaneously
counted as full competing interference. Alternative assignments of one interval contribute
normalized total weight, not one full count each. Silence adds elapsed time but
no new content; an acquisition gap adds no observed interference. Under the
registered operating model, propagate unknown interference in `[0, r_max * gap_sec]`.
Set `r_max` from development to twice the largest one-second observed interference
increment per episode, with a floor of one weighted-span unit per second, freeze
it before validation and test half/double values. This is an explicit rate-envelope
assumption, not evidence about unheard content. If observed increments exceed it,
mark the envelope invalid and retain unknown rather than assert a narrow bound.
Report observed and bounded-unknown contributions separately. A short gap then
produces a proportionately short availability interval; abstain only where that
interval changes the supported interpretation. Each committed observed recurrence
restarts elapsed time and the interference interval for that reinforced episode,
including its unknown component; unrehearsed episodes retain their uncertainty.
No-memory/novel-context has a fitted constant bias in the same score competition;
its match features already carry the cue-length normalization above. For the
stage-1 recognition task, define
`p_yes = sum_i exp(retrieval_score_i) / (sum_i exp(retrieval_score_i) + exp(bias))`
over supported retrieved episodes, implemented with log-sum-exp. Do not divide
log availability or the resulting total score by cue length again. With no
supported retrieved episode this defined competition gives zero stored-return
mass; report retrieval/capacity misses separately and use the declared numerical
loss floor for a contrary known response. Freeze this link with the stage-1 fit;
uncertain availability yields probability bounds, not an unreported point estimate. Its interpretation is absence of a supported stored
match, not proof that the listener has never encountered the material.

Reinforcement adds the support of a newly observed recurrence, with a bounded
strength, once per supported episode occurrence. Retrieval alone does not count
as another heard recurrence. Identify the episode parameters with a human
prefix-listening return-recognition task: after a probe, each participant answers
whether heard material returns, with new-material foils and exact/transformed
returns. Manipulate elapsed time, intervening material and prior exposure count
independently. Fit positive `tau_time`, positive `kappa`, `strength_max >= 1` and
the no-memory bias to individual binary responses by regularized log loss, using
the total supported stored-match mass against the no-memory alternative as the
predicted yes probability. Keep initial occurrence strength at its assignment
support, without another free strength multiplier. Parameter uncertainty and
unidentifiable time/interference or saturation effects must be reported; a good
aggregate fit alone does not identify separate mechanisms. This targets human
return recognition under the registered task, not general neural retention.

The reference recognition question is: "During the final eight seconds, did any
musical material return from earlier in this performance, before those eight
seconds? Count a return with changed pitch, speed, timbre or articulation as yes
when you hear the same ordered gesture or phrase; a shared isolated tone or mood
alone does not count." The reference Japanese wording is:
「最後の8秒間に、それより前に聞いた音楽のまとまりが戻ってきたか。音高・速さ・音色・発音の仕方が変わっていても、
同じ順序を持つ身振りやフレーズが戻ったと聞こえれば『戻った』を選ぶ。単独の音や雰囲気が似ているだけでは戻ったと数えない。」
Options are yes/return (戻った), no return (戻っていない), and unable to judge
(判断できない). Map the first two to 1/0 binary targets; unable, skipped or invalid
answers provide no binary target but remain in assigned-trial and usable-response
denominators and unknown-rate reports. Do not recode transformed returns as no.

Each trial is one uninterrupted prefix of at least 16 s, with one 8 s target window
at its end. A registered candidate-return or matched novel-foil interval starts
at that window's beginning; stop at its end, without later audio or replay. A
longer realization may remain ongoing at the cut: judge heard material only,
record the right-censored realization and permit unable, never fill in its future.
Register earlier exposures, delay measured to target start, intervening content,
candidate/foil schedule and any truncation before collection; these are stimulus/
scoring metadata, not input labels. Each listener hears only one trial/prefix per
source family; the authored earlier exposures inside that prefix are the intended
exposure manipulation. Delivery failures are reported invalid trials, not no-return
answers; analysis missingness on otherwise heard sound remains model uncertainty.

For this identification assay, the target interval is a retrieval query. Make an
offline scoring copy of the bank immediately before target start, then advance
its elapsed time through the 8 s while matching the heard query. Only episodes
whose committed support ends before target start are eligible; no target-window
reinforcement, competing-content update or newly stored episode enters this copy.
This prevents the query from becoming evidence of its own prior occurrence.
The ordinary live bank still follows its usual observed-update rule; this explicit
assay copy is not a new runtime mode. The synthetic recovery forward model uses
the same copy/query rule. Freeze the English/Japanese wording, mapping, timing,
replay/censoring and exposure policy in O06/O11 before collection; any additional
translation preserves the same return criterion and passes its language pilot.

First fit acoustic feature/similarity scaling and `w_match` on separately annotated
heard-interval correspondence tasks within development; choose engineering scales
by held-out correspondence loss and freeze them before the recognition fit.
Correspondence annotation uses a separate participant cohort from return
recognition; nobody serves in both. Raters first hear the entire supplied prefix
uninterrupted, then may request at most one additional complete replay of that
same prefix. Afterward they navigate only a silent elapsed-time ruler: no audible
scrubbing, partial replay, waveform cue, source label or audio beyond the cut.
Log whether the extra complete replay occurred and the actual presented support;
freeze this policy with O06/O11. Those extra exposures belong to a correspondence
task, never to the retention fit or its exposure counts. Raters hear a prefix and mark pairs of contiguous gesture- or phrase-sized heard
spans that they judge related. They mark start/end times with 10 ms stored cursor
resolution and may supply uncertain endpoint intervals; this interface resolution
does not assert 10 ms perceptual precision. For each pair they select exact return,
transformed return, unrelated, or unresolved. A transformed return may be tagged
as pitch shift, speed change, timbre/articulation change, combination, or unspecified;
numeric transformation parameters and Scenario labels are not shown. In overlapping
material, raters may retain several possible span pairs or leave source association
unresolved. Preserve individual annotations and alternatives, including negative
pairs. Stage-1 correspondence fitting and the stage-3 interval-pair compatibility
rule use these same heard-span annotations; uncertain endpoints admit any pair
within the stated intervals, without turning uncertainty into exact agreement.
Those scales support discrimination, not a human forgetting-law claim. Generator
participation parameters instead target executed timing prediction (§9.2); they
are never substituted for the episode recognition fit. All remain separate from
acoustic integration. This factorization is a
falsifiable engineering memory model, not a transferred parameterization of
PPM-Decay or a claim about hippocampal mechanisms.

Initialize a stored episode's strength with its observed assignment support;
reinforcement is `min(strength_max, strength + new_occurrence_support)`. Use
f64 log-space availability and logaddexp for retrieval, never exp followed by
log, so numerical underflow cannot masquerade as forgetting. A zero-strength
episode is ineligible with score negative infinity; the 1e-300 floor applies only
to positive-strength numerical evaluation. Register epsilon_avail and its units
(dimensionless availability) in O03/O09, distinct from the 1e-12 energy/loss floors.
Compare 5e-301 and 2e-300 in the stage-1 recognition sensitivity entry, including
very long waits and the no-memory competition. This floor is numerical protection,
not an identified minimum psychological recall probability. Propagate both ends of the bounded
interference interval into availability. If the registered rate envelope is invalid,
the conservative fallback spans zero to the no-new-interference upper bound until
new evidence justifies a narrower state; do not label this fallback forgetting.

Commit memory with a declared 0.5 s observation-time lag after a candidate span's
end. Before commitment, its alternatives remain provisional and usable as current
context, but cannot add occurrence strength or interference. At commitment, write
fractional assignment support once per occurrence/support ID, integrated over
the retained path weights. Seal these writes: later reinterpretation adds revised
relation metadata, not retroactive reinforcement/interference changes. If later
support differs by more than 0.25, flag a commitment revision for the original
occurrence and include its rate in T4/T6. Lag and threshold are engineering
approximations tested at half/double values. A missing commitment interval does
not count as confirming evidence; commit only the available support, tagged with
unknown interference. Truncate older partial-path ancestry after sealing, while
retaining episode descriptors, provenance and committed writes.
Use the original observed occurrence time, not the later commitment time, when
computing elapsed retention. A delayed write does not make an old occurrence new.

The initial index is a bounded flat descriptor array, not a new search framework.
Index eight-step descriptors at every fourth retained knot, at most 32 anchors
per episode, including the episode beginning and later material. Score each
anchor's relative pitch, interval, envelope and timbre with masks and the declared
transformation bounds; select the 16 best episodes by their best anchor, recording
anchor position. The reference coarse fit pairs the first up to eight ordered
cue knots with the anchor's up to eight ordered knots, without DP. Preserve their
original positions and mask unsupported fields rather than skipping missing steps. Estimate log2
transposition by the median supported cue-minus-reference log-frequency difference,
and log2 tempo ratio by the median log2(reference interval/cue interval) over
positive, supported adjacent intervals. Higher tempo ratio means a faster cue;
the affine map into reference time uses that ratio. A missing transformation
dimension uses a zero log-transform with an explicit unidentified mask, not
positive evidence for an unchanged realization. Out-of-range estimates are flagged
unsupported under the existing bound rule.

At most eight pitch and seven interval samples determine these two medians (at
most 49 comparisons using bounded insertion sorts). Round the estimates to the
nearest 1/16-log2 grid point, with lower-grid ties, and evaluate at most eight
paired steps of at most 64 scalar descriptor/coverage fields: 512 field visits
per anchor score. Use the frozen scaled match residuals and normalize over valid
support; no valid coordinate makes the score unsupported. Each group query visits
at most `256*32=8,192` anchors, or 65,536 anchor scores per bus per 0.1 s cycle,
with at most 33,554,432 descriptor-field visits and 3,211,264 median comparisons
for that scan. No transformation-grid cross product runs over the whole bank.

For interference, the coarse cost of a supported anchor is the arithmetic mean
of squared, globally development-scaled available descriptor-value residuals over
the up to eight paired steps after the rounded transformation. Masks select valid
values and do not themselves add residual terms; no valid value is unsupported.
This is the nonnegative coarse cost used to rank anchors/episodes, distinct from
the fitted retrieval score and full-DTW cost. Cache each stored episode's minimum
supported anchor cost as d_coarse_i and exp(-d_coarse_i), tagged with episode
generation, cue occurrence/support ID, endpoint, masks and approximation flags.
Thus even an episode outside the 16 full-DTW candidates has a declared interference
similarity; no extra bank-wide DTW or transformation search is introduced.

At commitment, freeze the most recent completed coarse snapshot for that exact
cue occurrence whose observed prefix ends no later than the committed span and
within 0.1 s of its observed end. Reuse a prefix snapshot only under this declared
endpoint tolerance; compare 0.05/0.2 s on development. An evicted/unsupported
episode entry, wrong occurrence, superseded query or absent qualifying snapshot
does not mean zero interference: add the competing assignment's weight as the
upper endpoint of an unknown increment [0,weight], separately from acquisition-
gap uncertainty. A supported entry adds weight*exp(-d_coarse_i) once; recurrence
support retains its separate reinforcement rule. Only episodes with a first
observed-occurrence endpoint strictly earlier than this span's endpoint are
eligible: insertion/commit time cannot make an old occurrence new. An eligible
episode inserted after the cached snapshot has unknown similarity, not zero;
first occurrences at or after this endpoint are not retroactively interfered with
by it. Never
wait for a new match in the audio path or borrow a later occurrence's snapshot.
Freeze this exact rule for the O11 simulator, stage-1 fit, stage-3 replay and
runtime. O10 compares it with exhaustive full-DTW interference on small banks,
including similar non-top-16 content, unmatched/noisy cues and short/superseded
spans, reporting supported/unknown interference weight and memory-task effects.

Retain at most eight completed coarse snapshots per group, evicting the oldest
query endpoint then query ID; report snapshot loss. Across eight groups, 64
snapshots of 256 entries at at most 24 bytes (u64 generation, f64 cost and f64
similarity), plus 128 bytes of metadata including validity/approximation bitsets, use
at most 401,408 bytes per bus. The existing scan supplies all costs; O04 adds at
most 2,048 similarity-cache evaluations per cycle. At most one new ending can be
retained per local path/hop, so the 8*8*16 retained paths bound pending committed
endpoint records at 1,024 per hop; deduplicate aliases before writing. Charge
up to 10,240 endpoint writes and 2,621,440 cached episode-entry visits per ten-hop
cycle, including unknown/bank-generation checks, without claiming that all these
endpoints are independent observed occurrences. Exceeding the registered bound
is explicit computational loss, not an unbounded catch-up batch or cognitive
forgetting. Pending endpoint storage is independently capped at 65,536 records of
at most 128 bytes (8,388,608 bytes per bus), including lagged support/identity
references; overflow is reported unsealed computational loss. O04 includes this
layout/cap, oldest-endpoint/stable-ID processing order and the commit work.

For each selected episode's best anchor, form the at most four bracketing
1/64-log2 grid combinations around its unrounded two-dimensional estimate;
an unidentified dimension stays fixed/masked. Score these on the same eight-step
anchor and retain supported distinct transformations with score/grid-ID ties.
This refinement adds at most `8*16*4=512` anchor evaluations or 262,144 field
visits per bus per cycle before full DTW. Record anchor/refinement counts, bytes
read and mask-aware work separately from DP cells; include both stages in the
worker census and O04 preflight. Full matching uses subsequence DTW: prior material outside the
matched subsequence is not a deletion, while internal insertions/deletions are
penalized. This permits quotation from the middle of a stored span. Record excluded
candidates and cutoff ties, and report mid-episode retrieval miss rate against
exhaustive full-DTW fixtures. Test half/double anchor spacing and count. Compare with exhaustive full-DTW retrieval at the same
capacity on development fixtures. An empty/missed candidate set, ambiguous cutoff
or computational eviction is unknown/unretrieved, never positive novelty evidence.
Run matching nominally once per 0.1 s of newly observed time per represented group;
commitments enqueue work but do not bypass this cadence. For each of at most 16
episodes, keep at most four transformation hypotheses from the selected anchor's
bracketing-grid refinement above. Do not enumerate the full
transposition×tempo grid. Banded subsequence DTW permits at most 33 predecessor
reference knots per cue knot (16 on each side of the affine-time prediction);
boundary hits remain an approximation flag. With 128 cue knots, the per-group
pass is bounded by `16 * 4 * 128 * 33 = 270336` DP cells, or 2162688 for all eight
groups in one instance. Cache reusable prefix work without exceeding that bound.
Queue at most one latest query per group and report superseded queries, matcher
age and incomplete results. Test half/double cadence, transformation count and
band width against exhaustive short fixtures. The graph retains selected episodes'
order, transformation and context links.
Computational eviction has its own reason and
counter. It is never recorded as cognitive forgetting or negative recognition.
Current focus is the shared context-path marginal of §9.1, not an extra attention
variable. It retains at most eight paths including unknown and updates once per
new interval from retrieval support and section-transition evidence in the common
conditional log-potential. An episode's focus compatibility is the preceding shared
marginal weighted by its retained context-membership edges (each in [0,1]). Add
that compatibility as a joint-weighted retrieval feature; it uses the preceding
marginal, never the just-updated result from the same cue. Its coefficient belongs
to stage 3 and is zero in the stage-1 reference. Focus thus biases candidate
correspondences and anticipated continuations without reinforcing stored occurrence
strength. Test zero focus influence and half/double shared-context capacity in
T6/T7. Contrast and return reweight this same bounded state.

Derive context membership from sealed occurrence assignments. At an episode's
commit, let `r_eoc` be the joint assignment support of occurrence o to episode e
and context identity c, summed over the retained paths, and let `r_eo` include all
its context assignments, including unresolved. Maintain committed totals
`A_e += r_eo`, `U_ec += r_eoc`, and edge weight `U_ec/A_e` (zero if A_e is zero).
These are cumulative support totals, separate from capped occurrence strength and
availability. Stored resolved edge weights sum to at most one; unknown, evicted
and edge-cap-dropped memberships keep their share unresolved, never renormalized
onto surviving contexts. Later reinterpretation may annotate alternative relation
metadata and commitment-revision flags, but cannot rewrite these sealed totals;
only a new heard occurrence adds membership support. Test fractional assignment,
repeated occurrence, unknown mass, metadata-only revision and edge truncation in
the exhaustive focus fixtures as well as the handle cases below.

Membership edges refer to stable retained episode/context handles, not beam slot
numbers. A context identity uses its representative episode handle; a split path
that continues the same context keeps that handle, while a supported new context
uses a new observed episode. A return can reuse a still-retained handle. Pruning
a path transfers its focus mass to shared unknown; edges in the episode bank are
not reassigned to unrelated surviving paths. Unknown contributes zero differential
focus compatibility. If the representative episode is computationally evicted,
its affected memberships become unresolved and carry the eviction flag. Later
retrieval can restore focus on a retained identity, not reconstruct a discarded
path or credit. Include split, pruning, return and eviction cases in the small
exhaustive focus tests.

### 9.4 Phrase continuation, boundary and closure

For each active auditory group, keep a continuing-span hypothesis and alternatives
for its boundary, overlap with another span, or reinterpretation. A semi-Markov
transition uses elapsed duration and context:

```text
hazard(d, c) = softplus(a + b * log(1 + d / t_unit) + v · c) / t_unit
P(exit in dt) = 1 - exp(-integral_t_to_t+dt hazard(d(u), c(u)) du)
```

Set `t_unit = 1 s` for arrival, phrase and section hazards, with elapsed durations
expressed in seconds. This fixes the logarithmic duration feature's engineering
scale, not a phrase length or neural clock. Store it in the numerical reference
and feature manifest; changing it requires a new fit/version rather than reusing
coefficients and regularization at a different scale.

The reference phrase hazard and exit categorical each use the same ordered
26-coordinate layout, with separate coefficients: [intercept=1,
log(1+d/1 s), standardized v1..v12, missing indicators m1..m12]. A missing or
unsupported foreground start makes the whole duration-conditioned head unsupported;
it is not a fabricated age zero. Duration is unstandardized as in the equation.
The context c comprises the remaining 24 coordinates. The table fixes each raw v:

| IDs, in order | Raw value, unit, window and source |
|---|---|
| `phrase.attack`, `.continuation`, `.release`, `.gap` (v1–v4) | Indicators of the local path's current articulation state from the already frozen earlier stage-2 articulation substage; dimensionless, current completed update. Known state gives one 1 and three 0s; unknown state masks all four |
| `phrase.rise_250ms`, `.decline_250ms`, `.flux_250ms` (v5–v7) | Physical-support-weighted means of the §9.2 positive adjacent-hop log-RMS rise/decline and log-spectral flux over J=[max(epoch_start,group_generation_start,t-0.25 s),t]. Units are their existing raw log-change units; use valid adjacent comparisons only |
| `phrase.energy_share_250ms` (v8) | Physical-support-weighted mean of E_g/E_bus over J, dimensionless; observed zero bus energy contributes zero. This assigned-mixture coexistence proxy is not a source-count or overlap judgment |
| `phrase.grouping_2s` (v9) | The existing preceding-2-s acoustic grouping-admission mean G_g from §9.2, including admitted integer and cyclic-word proposals; dimensionless, with its declared epoch/group clipping and support mask |
| `phrase.ordered_match` (v10) | log(1+d_match) for this path's current retrieved episode, where d_match is its cached normalized §9.3 DTW cost. Dimensionless; latest completed query at the existing 0.1 s cadence, with original evidence endpoint. Missing/ineligible match or known no-memory masks this coordinate |
| `phrase.retrieval_support` (v11) | Sum of current stage-1 bootstrap correspondence mass assigned to retrieved episodes in this group's completed query. Dimensionless; retain known no-memory and unknown mass in the denominator, never renormalize them onto episodes. No observed query support masks the value |
| `phrase.retrieval_ambiguity` (v12) | For N positive episode masses, normalize just those masses to q_e and compute -sum(q_e*log(q_e))/log(N); N=1 gives zero. N=0 gives zero only for a known no-memory correspondence state and is otherwise masked; no observed query support is missing. Dimensionless, same query/cut as v11. This is retrieval ambiguity, not an asserted probability of an unresolved phrase |

For v5–v8 require at least 90% valid physical support in J; empty J is missing.
For v10–v12, current observation time minus the query's full supporting-audio
endpoint must be below 0.5 s, reusing the existing maximum observation-age limit;
otherwise mask all three. Include this consumer in the existing age-limit
sensitivity and stale-query fixtures. All other masks follow the table/source
rule, retaining actual support/age. Use
global training mean/sd per raw coordinate, sd floor 1e-6 in its native units;
available values become (v-mean)/sd, missing values become zero with m=1, and
available values have m=0. Masks are binary and unstandardized. These are inputs
from the earlier frozen acoustic/match/articulation stages, never a fitted phrase
or section output or a current stage-3 posterior. Cache group/window/query values
once per update/query and select the path's cached match; no extra DTW is invoked.
Predictive queries hold c at the last supported snapshot while d advances; they
do not fill future feature values from later audio. Fit the exit softmax from
this same vector at the exit time, with the same causal context convention.

M0 registers these exact ordered lists under O03 before O12's event-head
simulations or any boundary/type collection, rather than inventing their inputs.
Freeze them for their corresponding stage-2 substages and verify emitted length,
order, units, masks, clipping, cache age and standardization against the manifest.
O12's generating vectors use these same 26 inputs. A changed inventory is a model
revision that reruns the synthetic gate and affected fits. O03/O12 declare a
64-input cap specifically for each phrase hazard and exit-categorical head,
including intercept and masks. This is distinct from stage 3's N=64 free-coefficient
cap and does not constrain the separately declared 109-coordinate groove/desire
or 82-coordinate section layouts. These phrase heads' uncached work remains
within the 256-term evaluation cap. O09 includes known/unknown articulation, silent/empty J, partial
coverage, no-memory, ambiguous retrieval and stale-match cases. Estimate their
separate coefficients from training judgments. The exact
boundary event-time controls are:

| Control | Inputs and event-time law |
|---|---|
| `phrase_constant` | Intercept only, hazard softplus(a)/1 s |
| `phrase_fixed_duration` | One positive fitted duration D shared across spans; cumulative hazard `(d/D)^8` and survival exp(-(d/D)^8). The fixed shape concentrates exits around a context-independent duration without a zero-width point mass; compare shapes 4/16 on development |
| `phrase_gap_only` | Intercept, the current registered observed-low-energy indicator and log(1+duration of the current consecutively observed low-energy run/1 s), with masks; softplus linear hazard /1 s. Missing input masks the run duration and cannot extend a known gap |
| `phrase_local_change` | Intercept and preceding-0.25-s support-weighted means of positive log-RMS rise, positive log-RMS decline and positive spectral flux, with masks; softplus linear hazard /1 s. It receives no episode/context or phrase-age feature |
| `phrase_rhc_only` | Intercept and the latest supported R/H/C levels with masks and actual age, as in closure_rhc_only; softplus linear hazard /1 s |

All controls fit the same interval-censored event-time targets and frozen
acoustic/start-state support, with the same regularization selection, mixture-
first-event aggregation, unresolved-mass accounting and stage-4 cumulative-hazard
multiplier calibration. D uses a positive parameterization. Fixed-duration here
means one common duration law, not a claim of noiseless deterministic boundaries.
Head-only comparisons hold the exit-type categorical fixed; whole-model ablations
replace the hazard and mask joint features that would restore its excluded inputs,
then refit stages 3/4. T5 names phrase_gap_only as the primary boundary control and
requires the other four comparisons; closure_rhc_only remains the separate primary
closure control. Type, continuation and closure endpoints do not disappear when
an event-time control is introduced. Freeze all feature windows/masks and control
IDs before the corresponding stage-2 fit. Meter-relative duration is an additional feature only when its reference is
supported. The distribution has no predetermined end or obligatory bar count.
The same formulation can support a section transition with different features
and independently fitted parameters; sharing the function does not equate their
cognitive dynamics. Context is piecewise constant between canonical observed
hops; integrate the duration-dependent hazard with two-point Gauss–Legendre
quadrature per hop and test against step doubling/high-accuracy integration.
Batched delivery processes the original hops, not one coarse elapsed-time step.
Across a gap, propagate survival under the last known context as a prior and put
exit mass in an unknown-current-state alternative, which includes unobserved
multiple transitions. Do not fabricate observed boundaries or inactive spans.
Use bounded step doubling for the gap survival integral; if the error budget is
not met within 64 evaluations, report unresolved survival rather than block the
worker. Integration tolerances are frozen with the numerical reference.

For each local phrase state, the hazard gives total exit probability. Allocate
that mass by a separately fitted categorical softmax over new phrase, overlapping
span, reinterpretation and end-without-successor (inactive), using the exact
26-coordinate layout above. The stay probability is the survival term. Section
exit has its own hazard and categorical distribution over new context, supported
recurrence and contrast. Their conditional prior permits all four phrase/section
stay-or-exit combinations, with product transition weights before the common
observation update and the declared joint-proposal restriction (§9.1); neither
boundary forces the other. Renormalization on that restricted set may itself
introduce dependence, which belongs in the proposal-cap comparison. This conditional
independence is a testable reference prior. Joint conditional features may favor
co-occurrence or separation. Keep its distinction from the posterior dependence.
An inactive phrase exit requires observed low-energy/group-termination support,
with masking alternatives retained; missing input cannot assert it. Other groups
may continue. Section context may persist through silence without an active phrase;
EOF still supplies neither a section boundary nor closure evidence.

A local phrase component follows one foreground span, with a start time and its
elapsed-duration clock; the six proposal slots are stay, the four exits and
unknown, not six independently running spans. At an observed exit at hop endpoint t,
apply the following successor convention before the next hazard evaluation:

| Exit | Foreground successor and clock | Predecessor and unresolved expectations |
|---|---|---|
| New phrase | New span handle, start=t, elapsed=0 | End the previous grouping span at t, but do not declare closure; its unresolved continuation links remain |
| Overlap | New foreground span, start=t, elapsed=0 | Keep the older span as an open, right-censored continuation link with its original start and heard support; it has no second phrase-hazard clock |
| Reinterpretation | New interpretation handle for the same heard span, inheriting its original start and elapsed duration | Retain an alias to the previous interpretation and the same support/credit identity; do not count another occurrence or move an issued boundary |
| End without successor | Inactive foreground, no running foreground-duration clock | End the grouping span at t; retain unresolved expectation/closure eligibility in memory rather than inventing resolution |

Stay advances the foreground elapsed time in physical seconds. Inactive state
has no foreground exit hazard; a newly acoustically admitted span starts a new
foreground at its observed onset support, never at an unobserved point inside a
gap. Unknown-parent readmission preserves a supported span's inferred start/range;
without a supported start its duration remains unresolved. No unobserved exit
selects one of the four observed successor labels.

Use at most 16 continuation-link records per local path, each at most 128 bytes
(2,048 bytes/path, 2,097,152 bytes/bus at 1,024 paths), sharing existing descriptor
and occurrence handles rather than copying span audio. A link stores original
start, last heard endpoint, right-censor flag, expected-segment handle, unresolved
support and its closure eligibility. New observed correspondence/release evidence
may update it, with the same normalized interval-assignment/once-only credit rules;
merely switching foreground supplies no completion or reinforcement. It remains
available to unresolved-expectation/closure features without another live phrase
hazard. Seal only actually supported prefix relations under §9.3; an open link
cannot commit an invented end. Before capacity loss, preserve an existing committed
reference when available; evict the least-supported link (oldest-support tie),
report unresolved-link loss, and never call it cognitive closure. Register the
layout/cap and include it in the worker memory/copy census. Evaluate candidate
successor changes through parent references; copy link arrays only for retained
paths. At the reference bound this is at most 2,097,152 bytes per hop and
20,971,520 bytes per ten-hop cycle per bus, in addition to the section-record
copies. Do not clone a link array for every pre-pruning tuple.

Small exhaustive fixtures and T5 target compatibility cover all four exits,
overlap followed by an older span's observed release, repeated reinterpretation,
inactivity/reentry, gap propagation and link eviction. A type label constrains
both the exit and this successor state/clock, with unjudged types marginalized.
No simultaneous second foreground hazard is inferred from an overlap label; test
this bounded continuation-link approximation against an explicit two-span offline
reference on the overlapping cases before the T5 freeze.

The reference phrase foreground is group-local. The T1 handoff-union view does
not silently turn it into a phrase spanning acoustic groups: hocket, antiphony
and timbral handoffs can carry one perceived phrase beyond that representation.
A source becoming inactive does not itself force an immediate phrase exit, but
surviving local clocks alone do not represent cross-group phrase identity.
Register a dedicated T5 development stratum of these handoffs, with matched
within-group continuations and genuine phrase-boundary controls. Preserve listener-
marked continuing spans and their uncertainty. After the same single prefix
hearing, ask: "In the last eight seconds, did one phrase continue while the sound
carrying it changed?" Offer yes/no/unable to judge; a yes response marks the
continuing interval and uncertain endpoints on the same 10 ms silent ruler,
without replay. Report all assigned, yes/no and unjudged counts. Register the
required controlled cases and annotation-reliability criterion with O06/O07 before
collection, not by selecting favorable listener responses afterward. If a supported continuing span
requires successive groups and no retained group-local phrase hypothesis covers
it, classify it as a phrase-representation miss, score its required compatibility
at the declared loss floor, and report its rate over all judged continuations in
that stratum separately from boundary timing error and unknown coverage. Do not
reinterpret it as a listener boundary to fit the model. O10 requires this report
before T5 freeze; a required, reliably judged handoff that the bounded inventory
cannot represent reopens M4's representation and resource/fitting gates. It cannot
be removed from the planned T5 breadth or absorbed into ordinary aggregate loss.
This is a declared limitation of the initial reference, not a completed cross-
strand phrase mechanism; the delivery plan must resolve a demonstrated miss before
claiming T5 completion.

Define a mixture boundary as the first judged grouping transition in any audible
strand within the elicitation window. The instruction explicitly says to consider
all simultaneous strands and mark the earliest such change, with its exit type;
it does not require every strand to end. For each shared context C, marginalize
local paths' first-exit laws before combining groups. At the scoring snapshot,
let `s_g(t|C)` be the mass of paths known at the window start that still have no
first exit and no unresolved history through t; let `f_gk(t|C)` be their density
of a first exit of type k before any loss of required history. A known inactive
group has survival one and density zero. A group's unresolved path mass supplies
neither a fictitious survival nor an observed event. Under the declared conditional
factorization, using one fixed snapshot of the context/local history weights:

```text
S_mix(t)   = sum_C q(C) * product_g s_g(t|C)
f_mix,k(t) = sum_C q(C) * sum_g [f_gk(t|C) * product_(j!=g) s_j(t|C)]
R          = sum_C q(C) * product_g s_g(0|C)
E_mix(t)   = sum_k integral_[0,t] f_mix,k(v) dv
U_mid(t)   = R - S_mix(t) - E_mix(t)
U_event(t) = (1-R) + U_mid(t)
```

Thus `E_mix(t)+S_mix(t)+U_mid(t)=R`, and adding initial unresolved mass gives
total one. U_mid is nonnegative mid-window unresolved accrual: support lost
before a known first event cannot be recovered by admitting a later current
state. When a first event is already resolved, later support loss does not erase
it unless its scoring history itself is irrecoverably pruned/lost. Score a known
interval with its unconditional event mass, and no boundary with S_mix(end), so
abstention is penalized, not conditioned away. Report R, U_mid and U_event
separately; all-unresolved predictions exert no action pressure. This conservative
all-groups support rule is an aggregation hypothesis to compare with an acoustic-
salience group-selection control on development. Group lifecycle changes do not
invent exits: an unsupported association makes the affected required history
unresolved. Shared unknown contributes unresolved mass. Apply this construction
separately to phrase and section exits; their group events need not coincide.

For stage 2, a path's raw first-exit law is its hazard/survival law on the frozen
causal feature sequence, with the cumulative-hazard calibration multiplier one
until stage 4. The eta uncertainty leak is not a physical exit rate and is not
part of that law. Actual missing support and known-history loss are separately
accounted in s/f and U_mid; in their absence E_mix+S_mix=R. For stage-2 annotation
scoring use the frozen out-of-fold earlier-stage weights at the annotation-window
end. For stage 3, use retained context and conditional local-history weights at
that same window-end cut from the complete causal replay, with its retained
first-event time/type register and known/unresolved history flags. Its eta/pruning
loss before a resolved first event enters U_mid. R is the known-at-window-start
mass under that chosen scoring snapshot, not the mass of an earlier forecast.
These are retrospective heard-prefix annotation scores. An actually issued
forecast instead freezes its issue-time weights and cannot be reweighted by the
window-end posterior; record the snapshot kind/time with every comparison.

The offline scoring register survives ancestry sealing and records start-known,
resolved first-event, resolved no-event or unresolved-history status. It is a
fitting/evaluation instrument, not a new observation/model feature. Finite-hop
calculations apply all same-hop history-loss masks before exits, conservatively
treating an inseparable loss/event tie as unresolved. Among resolved simultaneous
exits, give the event to the smallest stable group handle: for group g's hop-h
first-exit mass, multiply other groups' same-hop survival after their exit test
for j<g and survival after loss masks but before their exit test for j>g. This
partitions event ties once. Test factorized and enumerated small joint histories,
including initial unknown, eta, loss before/after the first event, different
snapshot weights, ties, overlap, no event, group lifecycle and type labels. Freeze
mapping, questions, discretization and type loss before stage 2. Event-time
calibration scales each raw cumulative hazard before this mixture operation;
it never removes separately reported unresolved mass.

A boundary target is this listener-judged mixture transition with an uncertain
interval. A section target similarly asks listeners for a larger contextual/event
boundary, and for new, returning or contrasting context, using heard material
rather than composer labels. Phrase and section judgments remain separate and may
disagree. Fit both event-time heads by interval-censored negative log likelihood:
`-log integral_[left,right] p(boundary_time) dt`; a fully observed no-boundary window
uses negative log survival. Preserve each annotator's interval and censoring,
including right censoring, under the registered annotation protocol. A genuinely
unjudged window supplies no target. The reference elicitation is offline interval
marking after one uninterrupted hearing of a prefix, not real-time keypress timing.
On a silent elapsed-time ruler with 10 ms cursor resolution, the listener marks
the uncertain interval of the first phrase boundary in the last 8 s, or first
section boundary in the last 32 s (clipped to the heard prefix). They may instead
mark no boundary or unable to judge. No audio beyond the prefix or replay is
provided. The marked interval is the censoring interval; cursor resolution supplies
no invented motor-lag correction. Other boundaries in that window are not targets
of this first-event trial. Such retrospective boundary marking does not by itself
validate online neural response latency.
After marking a phrase grouping change, ask for one corresponding interpretation:
(1) the previous unit ended and a new unit followed; (2) a new unit appeared while
the previous unit continued; (3) earlier material was regrouped without a clearly
new unit; or (4) the unit ended with no heard successor. "Unable to distinguish"
leaves type unjudged while retaining the boundary interval. For regrouping, mark
when the change of interpretation became evident, not a retroactively moved earlier
onset. These are the reference human targets for new phrase, overlap,
reinterpretation and inactive exit. Fit the exit categorical conditional on the
marked interval, integrating its type probability against the already fitted event-
time density in that interval and normalizing by interval event mass. Preserve
individual disagreement. Section exit types use the corresponding new/return/contrast
question with the same conditional type loss. Its reference marking question is:
"Looking back over the last 32 seconds you heard (or the whole heard prefix if
shorter), mark the earliest interval when any audible strand changed from one
larger musical context to another. Mark no boundary or unable to judge if that
fits your hearing." The type question is: "Which interpretation was strongest:
a new context without a marked contrast, a return to an earlier heard context,
or a contrasting departure from the preceding context?" Offer unable to distinguish
when no single interpretation is strongest; then retain the interval and marginalize
type. These map to new/return/contrast, not Scenario section labels. Freeze the
wording, translations and options with the phrase instruments before collection.
In stage 3, a known type restricts
compatible exits; an unjudged type marginalizes them. Compare the typed phrase
model with an undifferentiated-exit model on held-out T1/T5 judgments and local
consequences; unreliable type identification remains a failed/uncertain mechanism,
not permission to invent labels from gaps or the Scenario. Continuation is a separate five-level judgment
answer to: "Considering the simultaneous sounds together, how likely is this
ongoing musical flow to continue over the next second?" Use very unlikely,
unlikely, uncertain, likely and very likely. Do not instruct listeners to select
a single lead unit. Freeze this wording and translations with the whole-mixture
aggregation; disagreement about that aggregation remains part of its evaluation. Its regularized
ordinal head and log loss are distinct from event-time survival. The 1 s probe
horizon is an engineering task definition; compare 0.5/2 s on development.
Closure is a separate five-level judgment with the complete ordered anchor list:
not at all completed/resolved, slightly completed/resolved, partially completed/
resolved, mostly completed/resolved, fully completed/resolved. Freeze all five
labels and their translations; the third is the midpoint.
It has its own ordinal coefficients and log loss. Train the
closure estimator on ordered-context and anticipated-continuation features,
including proxies for fulfilled, interrupted and relinquished expectations.
Use a fixed stage-1 feature map: normalized match residual against the retrieved
next segment at its alignment-predicted time; best competing-content score minus
that continuation's score; elapsed time since its last observed reinforcement;
the decrease in its log availability since the cue; remaining unmatched expected
segment support; observed release/continuation occupancy; and the existing R/H/C
levels. Elapsed time uses seconds and the fitted episode time scale, not a phrase
length. Each cue's prediction and availability baseline are fixed when issued;
score it against newly observed support, never a retrospectively improved forecast.
Aggregate within each path over the probe's preceding 2 s using physical support;
current path weights marginalize the resulting head predictions, not its input
features (§10.1). Standardize on development and carry component missing masks: missing
sound or an unsupported expected segment adds no fulfillment, contradiction or
abandonment evidence. Unreinforced elapsed time/availability decline is only an
abandonment proxy. These are computable inputs, not the human target labels.
Harmonic and release cues are inputs; none defines closure alone.
The named closure controls are `closure_rhc_only`, an ordinal head with only
current supported R/H/C levels and their missing masks, and `closure_gap_energy_2s`,
with preceding-2-s known-gap fraction, mean RMS and endpoint log-RMS slope/masks.
Current R/H/C means the latest complete observed frame at issue time, with its
actual age; the control receives no episode, phrase or availability feature.
Both use the full head's training judgments, L2 selection, acoustic/path aggregation,
fixed-marginal backoff and stage-4 temperature convention. T5 registers
`closure_rhc_only` as its primary closure control and requires the gap/energy
comparison too; boundary and continuation endpoints remain separately required.
“Relinquished” is identified through listener judgments of an abandoned expectation,
not by declaring every long gap resolved. Disagreement remains a distribution.

The reference closure estimator is regularized ordinal regression over those
features; grouping boundary detection uses the duration transition above. Their
interaction is a conditional feature in §9.1, not an identity between outputs.
A high boundary score can coexist with unresolved expectation, and high closure
can coexist with overlap. A completed span is eligible for memory reinforcement;
unresolved relations survive context changes. EOF is censoring, not a target label.

The continuation ordinal head extends the closure feature map with four declared
inputs: the already fitted arrival probability of a change in the next 1 s,
phrase-hazard survival over that second, accent density over the preceding 2 s,
and mean acoustic grouping-admission support over those 2 s. Use the stage-1
assignment/feature support and stage-2 out-of-fold hazards, with their masks;
no stage-3 posterior average becomes an input. Hazard fits precede this ordinal
fit, so there is no reciprocal training loop. Its named controls are the original
closure-map-only head, a phrase-survival-only ordinal control whose survival input
is the multiplier-one frozen stage-2 hazard and whose ordinal output has its own
stage-4 calibration (`control.phrase_survival_only`), and a
recent-gap/energy-only ordinal head (2 s known-gap fraction, mean RMS and endpoint
log-RMS slope with masks). All use the same judgments, regularization selection,
calibration split and held-out log loss. Failure against these controls reopens
continuation features before stage-2 freeze; a boundary score alone cannot pass. For both ordinal heads the normalized expected rating is the mean
category (0–4) divided by four; this is not a probability of a boundary-free second.
Hazard-derived survival remains a separate event-time forecast and diagnostic.

Probes rate the whole presented mixture, rather than a hidden generator Voice or
an automatically selected lead group. Freeze this aggregation for all ordinal
heads. In the probe window (2 s for continuation/closure, 8 s for groove/desire),
let each retained group's nonnegative weight be its stage-1 acoustic assignment
support integrated over observed time. Normalize these weights over groups with
support. First marginalize each group's path-local category predictions over its
conditional paths, then mix groups by those fixed acoustic weights and shared
contexts by their path weights. Let s_supported be the total supported head
mass in that path/group/context mixture, and e_window the observed physical
sample-support union divided by the duration of the epoch-clipped probe window.
Define `r=e_window*s_supported` explicitly (both factors in [0,1]); an empty window
has e_window=0. Normalize only the supported mixture to a category distribution
p_s; never normalize unknown support into p_s. Report both factors separately.
Let p_0 be that head's empirical training-category marginal with a half-count per
category, fixed from stage-2 training responses and recorded with the model.
The single calibration/backoff convention is
`p_report = r*temperature(p_s,T) + (1-r)*p_0`. Temperature is applied only to the
supported, already path/group-marginalized distribution; p_0 is already a category
probability in the reported space and is never temperature-transformed. Stages
2/3 use T=1. Stage 4 fits T by log loss of the final p_report on all calibration
responses, including backoff cases. An unsupported local head contributes to
1-r; no supported group yields p_0 alone. Report missing/unsupported
mass as unknown, separately from this explicitly prior-based prediction. All-
unknown predictions supply no action pressure. This aggregation can be evaluated
with the factorized beams; it does not enumerate the cross-group Cartesian product.

Use this same convention for stage-2 ordinal loss, stage-3 label compatibility
(T=1 in both) and stage-4 calibration. Only the supported marginal changes under
the fitted temperature; missing support and the recorded p_0 do not. The stage-3 compatibility is the matching
weighted sum of per-group category probabilities before path marginalization;
linearity permits the factorized calculation. Backoff contributes its frozen
category probability, never compatibility one for every known label. Include
all probe responses in scoring, even when the model backs off. Record ordinal
behavior in overlap and silence explicitly; averaging competing group expectations
is a falsifiable aggregation choice, not a claim about attention's neural rule.

Explain before listening that playback stops are experimental prefix cuts, not
musical endings; ratings concern the heard material before that cut. A stopped
prefix cannot supply an inferred closure observation to the model.

Fix probe scheduling before data collection. After the first 8 s of a passage,
partition eligible time into 16 s strata; retain a final partial stratum only if
it is at least 8 s long. Select at most eight strata uniformly without replacement
using a registered seed. Within each, draw the probe time uniformly between four
seconds after its start and four seconds before its end (a single midpoint for an
8 s stratum). This gives at least 8 s between adjacent probe times. Randomly assign
one of the declared judgment tasks per trial, balanced across listeners/passages.
Each listener hears at most one prefix/probe of a given passage, so a probe never
interrupts a later rated continuation or adds hidden repeated exposure. Event-time
marking and ordinal tasks use separate trials. Short passages may supply fewer
probes. Event-locked tests of a designed return or ending are an explicitly
separate stratum, reported separately and not silently pooled into random-probe
training/calibration. Full-piece retrospective assays also retain their own label.

T2 additionally registers an omission-expectancy task in the separate event-locked
stratum. At a prefix cut, ask: "Before the final quarter-second you just heard,
how strongly did you expect a new sound during that quarter-second?" Use none,
slight, moderate, strong and very strong. Explain the 0.25 s target interval with
practice examples; do not show a beat label, metronome or expected-event marker.
Probe an expected omitted event, its sounded counterpart, an unexpected insertion
and matched nonmetrical controls, balanced across participants; one prefix per
passage/participant still applies. The model issues its target-window forecast
at `cut-0.25 s`, using offline causal replay at that exact stream time for the
primary T2 cognitive assay. Ignore wall-clock queue/CPU delay, but consume only
canonical records whose complete raw-audio support ends at or before the issue
time `cut-0.25 s`, not the later heard-prefix cut. This applies to every underlying
NSGT/raw hop as well as each derived timestamp. The accent detector needs its
full four-hop evidence, including right context, so its last admissible accent
is correspondingly earlier than the issue boundary; a recent accent timestamp
alone does not authorize its later raw support. Respect the model's declared
observation-time update cadence. A partial future hop cannot be borrowed. Arrival
and both baselines use this identical support cutoff and replay rule. O09/O18
register a leakage fixture that alters audio only in [cut-0.25 s,cut] and verifies
bit-identical issued forecasts and their incorporated-support records.
For each issued forecast record the latest audio support actually incorporated,
matcher/reference support and effective evidence horizon relative to issue time.
Report them in the 0.125/0.5 s window comparisons too. A separately labeled runtime
T2 replay through actual publication/consumption with measured delays is required
in M8 and the action gates; ideal offline success does not certify live timing.
The listener hears to the
cut before rating; this is a retrospective judgment of expectation, not a direct
measure of neural prediction latency. Actual absence is not an expected-event
training label. Freeze wording, translations, target placement and 0.125/0.5 s
window sensitivity before collection.

Use held-out five-category log loss on these ratings as T2's primary cognitive
metric. Fit a separate scoring-only ordinal link in stage 2 for each compared
anticipation model, with the same regularization and calibration procedure. Its
single observed predictor is that model's frozen target-window expectation score:
arrival and `grouping_inventory_234` use their respective conditional probability
of at least one change; `oscillator_beat_only` and `meter_existing_234` use their
maximum normalized predicted pulse strength
on 16 equal midpoint times in the window, projected from the issue-time phase,
period and strength without resets from later sound. For a resolved path mixture,
average the score with issue-time weights, and then use the same fixed acoustic
assignment weights over groups from the preceding 2 s. Mix the ordinal link's
category distribution with the frozen training-marginal distribution according to
its issue-time supported mass; all unsupported predictions remain reported as
such and stay in scoring. Freeze each baseline's pulse function and normalization
in M0's numerical reference and compare 8/32 midpoint times. The link is an
assay predictor, not a new generator reward or an observed-event target. Contrast
the full recurrence/arrival model against each baseline; M0 names its primary
control and margin before validation. Omission survival in model traces alone
cannot pass T2; grouping and complementary-action comparisons remain required.

T1 adds a gesture/state annotation trial on the same prefix-only, no-replay
interface. Ask: "In the last eight seconds, mark the start and end of the clearest
musical gesture you heard. Sounds that belong together may share one gesture even
when they overlap." Allow uncertain endpoints, already ongoing at the window start,
still ongoing at the cut, and unable to identify a gesture. Within that selected
span and any immediately following heard gap, ask participants to mark only the
portions they can identify as attack, continuation, release or gap; unmarked
portions are unjudged, not labeled gap. Then ask: "Did another distinct musical
gesture overlap the one you marked?" with yes, no and unable to judge. Raters see
a plain elapsed-time ruler, not inferred group IDs, Voice labels or a score.
Store endpoints at 10 ms cursor resolution without claiming that perceptual
precision. Freeze wording, translations, instructions and censoring before collection.

Map the selected gesture to the §9.2 single-run or handoff-union candidates using
at least 0.5 temporal support IoU, admitting uncertain endpoints and censoring.
For a union, compute IoU from the union of its supported intervals, including its
observed linking gap, not from either member alone. Use the frozen family/path
mixture above, never hidden source labels. State-interval likelihood marginalizes
member assignments with their acoustic support restricted to each marked interval;
unmarked member intervals remain unjudged. A known gesture with no compatible
single/union candidate is a representation miss under §10.1, including divided
support that defeats the IoU rule.
For overlap, let G be the selected candidate's member set (one or two groups).
Conditional on its enumerated paths and C, let R_j be the other group's resolved
span mass and n_j its resolved mass without a gesture intersecting the marked
support. Let u_j=1-R_j; resolved overlap in group j has mass R_j-n_j.
Known overlap mass is `1-product_(j not in G)(n_j+u_j)`, known no-overlap mass is
`product_(j not in G) n_j`, and the remainder
`product_(j not in G)(n_j+u_j)-product_(j not in G)n_j` is unresolved. One certain
intersecting gesture establishes overlap even when another group is unknown. If a nonselected member's path was already enumerated
by a separate-run pair branch, use that conditioned path's mass, not another q_j
factor. Marginalize candidate families, their paths and C; use the supported
two-category distribution and unresolved mass with §9.4's calibration/backoff convention. Unknown groups cannot
be silently counted as known absence. Compare this
calculation with joint enumeration on small fixtures, including a certain
intersecting gesture beside a fully unknown group. Joint enumeration of the same
factorized model is authoritative: disagreement is a formula/implementation error,
not permission to relabel definite overlap unknown. T1's primary passive
metric is held-out overlap/grouping-response log loss; state-transition loss and
marked-span/censoring agreement are additional required endpoints and supply the
base articulation fit. A boundary-only or correspondence-only success cannot
replace these grouping judgments.

T2 also has a grouping judgment, separate from omission expectancy. Ask: "At the
end of what you heard, what repeating grouping was clearest?" Offer equal groups
of 2 through 16 reference beats, an equal group larger than 16 (with optional
integer count entry), an unequal repeating group, a recurring pulse
without a larger grouping, no recurring reference, and unable to judge. Practice
examples explain the vocabulary without adding an audible metronome to study
trials. The participant selects the strongest interpretation and may record a
second simultaneous grouping. Register that second answer as diagnostic-only in
O06/O07 before collection: retain its counts, uncertainty and examples, but give
it no confirmatory Gate-2 score or adoption role. Within one acoustic group,
conflicting concurrent groupings are represented only as posterior alternatives,
apart from the two nested levels within an individual grouping hypothesis; the
second answer probes that restriction. A recurring mismatch may motivate a new
development model/task version, not post-hoc confirmatory scoring. Different
acoustic groups may still have their own concurrent groupings. Also ask them to
mark at least four consecutive remembered reference beats on the final-16-s
silent ruler (clipped to the heard prefix, 10 ms resolution, no replay), or unable
to mark. Sixteen seconds accommodates three spacings at the reference 4 s maximum
period; the strongest-grouping question still concerns the ending interpretation.
This secondary task
is offered for all recurring-reference answers, including unequal and pulse-only.
Define P_response as the median of the marked consecutive spacings; retain all
marks and their range. It estimates the listener's chosen tactus for scoring only
and never enters audio inference, proposal selection or generation.

Freeze a hierarchical categorical score. First map hypotheses to four grouping
kinds: equal group, unequal group, recurring pulse without a larger group, or no
recurring reference. Integer hypotheses and uniform cyclic words map to equal;
nonuniform cyclic words to unequal; known ungrouped with supported recurrence to
pulse-only; arrival-only nonperiodic interpretation to no-reference. Cap/window-
limited or unknown hypotheses remain unresolved, not no-reference evidence. Mix
kinds with fixed observed group weights and the declared temperature/backoff rule.
Every judged answer supplies this kind loss, including an equal-group answer with
unusable tactus marks; unable to judge has no target and its rate is reported.

For an equal-group answer with valid marks, also score count conditional on that
kind and P_response. For integer (L,P), grouping extent H=L*P; for a uniform cyclic
word of n steps of length d, H=n*d*P. Map H to the nearest integer k to H/P_response
with k>=2 only if `abs(H-k*P_response) <= 0.1*k*P_response`, with lower-integer ties.
Counts 2..16 have individual bins; k>16 enters the separately reportable above-16
bin, so smaller listener tactus levels still have a truthful response. Otherwise
retain mass in a distinct level-mismatch failure bucket. Thus 3*P and 6*(P/2) map to
the same count. The bucket is a model failure outcome with no human option; never
renormalize it away when scoring a known count. This is conditional evaluation of
perceived grouping extent, not a claim that listeners share one universal tactus.
Grouping phase/boundary agreement remains a separate timing endpoint.

Marginalize the supported equal-group hypotheses into this count distribution;
apply its own stage-4 temperature and mix unsupported mass with the empirical
training marginal over the sixteen answer bins (fifteen counts 2..16 and above-16),
half-count smoothed and extended with zero in the failure bucket. Record optional
raw counts above 16 diagnostically; this head identifies that aggregate, not the
exact larger count. The integer inventory explicitly supplies every extent 2..16
in its anchoring-period units, including 13; uniform-word coverage is not used as
a substitute. The response tactus can change the resulting count, so an above-16
percept is not itself a level-mismatch error. O03/O09/O10 register integer maxima
8/16/32 and the prior 2..12 restriction as comparisons, including an equal-13
fixture with response tactus equal to the anchoring period and the same extent
heard at a finer tactus. Record inventory, window and cap non-admission separately;
the broader integer search does not remove the 16-fresh-proposal cap. Use the ordinary fixed prior for the four-kind
head separately. Stage 2/3 temperatures are one. A missing P_response or spacing
range exceeding 20% of its median leaves only the count target unjudged; retain
its kind loss, raw count, mark coverage and inconsistency rate. The joint trial
loss is kind negative log probability plus, when judged, conditional-count
negative log probability. Report both components and count coverage separately.
Both full and restricted controls use the same marks, conditional scoring and
calibration protocol, without access to the marks during inference. Register
5/20% count-mapping tolerances, 10/40% mark-consistency thresholds, and strict raw-
count scoring as development comparisons; report half/double tactus mismatches.

The controls have separate IDs and scoring maps, fixed in O03/O06 before collection:

- `meter_existing_234` runs the audited `core::meter::MeterNetwork` and its existing
  observed-audio drive adapter once per bus, with default `MeterShaping` and the same
  causal prefix. It uses the network's beat/subdivision/measure frequencies and
  phases, never the new pair-interval grid. Let c_b be beat confidence, c_m measure
  confidence when measure_ratio is 2, 3 or 4 (zero otherwise), and c_s subdivision
  confidence when subdivision_ratio is 2, 3 or 4 (zero otherwise), all clamped to
  [0,1]. Its measure-equal, subdivision-equal, pulse-only and no-reference masses
  are respectively c_b*c_m, c_b*(1-c_m)*c_s,
  c_b*(1-c_m)*(1-c_s), and 1-c_b; they sum to one. The first two contribute to
  equal-group kind, with extents H=measure_ratio/beat.freq_hz and
  H=1/beat.freq_hz respectively. The latter extent contains subdivision_ratio
  units of 1/subdivision.freq_hz. It supplies no unequal-group hypothesis.
  These are registered scoring weights, not asserted neural probabilities.
  Nonmetrical sound can retain spurious periodic weight; declining confidence
  moves it toward no-reference. Missing observation support or invalid frequencies
  are unsupported, not no-reference evidence. Its omission-strength predictor is
  the maximum, over the beat and detected subdivision/measure bands, of
  confidence_band*(1+cos(phase_band+2*pi*freq_band*u))/2 at each of the existing
  sixteen future midpoints. Freeze all band values at the issue-time cutoff.
- `oscillator_beat_only` runs that same network/drive but reads only its beat output.
  Its omission predictor is the beat term of the preceding formula. Its grouping
  map is pulse-only mass c_b and no-reference mass 1-c_b, with no equal/unequal
  extent; it follows the same unsupported-input rule.
- `grouping_inventory_234` retains the new independent 241-period grid and full
  proposal/admission rules, restricting only integer L to 2, 3 or 4 and disabling
  cyclic-word proposals. It retains known ungrouped, arrival-only/no-reference and
  unknown hypotheses; cap/window failures never become no-reference. Integer
  extents are H=L*P, in that proposal's grid-period units. There is no unequal-group
  hypothesis. Refit its affected stage-2 arrival/scoring components and stages 3/4;
  its omission predictor uses its own fitted arrival probability, not the old
  oscillator pulse. Unaffected data, acoustic front end and body controls stay fixed.

All three use the same listener P_response marks, kind-plus-conditional-count loss,
failure bucket, calibration and unknown/prior-backoff protocol as the full model.
If a control supplies no equal extent, its conditional-count prediction is the
same training-marginal backoff; do not drop a judged equal-count loss. The two
network controls provide one bus-level distribution with common observed-prefix
coverage/backoff, not invented per-group oscillator instances. The full model and
new inventory control retain their declared observed-group aggregation. Record
the drive-adapter code hash, period provenance and these maps in O03/O06. O09
fixtures cover a uniform pulse, accent-free subdivisions, measure accents,
nonmetrical sound, missing input, zero confidence and different response tactus.

This grouping endpoint must beat `grouping_inventory_234` and `meter_existing_234` as well
as the separate primary omission criterion; phase concentration alone does not
score it. M0 freezes wording/translations, the kind/count factorization, tactus
marking/equivalence, the separately named `t2_count_validation_coverage_min`
derived only from an adequate pilot and frozen before validation, and the optional
second-group analysis before validation. Inadequate count coverage is an unpassed
endpoint, not permission to claim grouping from the coarser kind loss alone.

### 9.5 Sections, return and whole-piece context

A per-group local section hypothesis maintains a 39-component descriptor since its candidate
start: five correspondence-edge masses (exact recurrence, transformed recurrence,
contrast, unmatched material, unresolved correspondence/continuation), their 5×5 ordered
transition masses, four articulation occupancy fractions, and five participation
statistics (event density, supported-periodic fraction, persistent-offset dispersion,
residual timing dispersion, overlap fraction). Normalize masses with their observed
support, retain missingness, and keep physical seconds rather than report counts.
The edge categories here are stage-1 acoustic/correspondence features, not already
validated perceptual labels. The fifth includes pending retrieved continuation
and unsupported correspondence; unmatched is a bounded-search result, not a claim
that the sound has never been heard.

For the first 30 coordinates, use each completed foreground phrase/span record
on that local path, with its §9.3 correspondence assignment and observed support.
Read the cached normalized nonnegative DTW match cost d, estimated log2 frequency
shift delta_f and log2 tempo ratio delta_v, their masks/bound flags, and the
retrieval status. Do not run another matcher or read the section head being fitted.
The category order is exact, transformed, contrast, unmatched, unresolved. For
each record u and its assigned correspondence alternative, define a one-hot e_u:

- Unknown/pruned assignment, pending-but-unheard continuation, empty/missed
  candidate set, ambiguous cutoff, missing required cost or a transformation bound
  hit gives unresolved. These conditions cannot supply positive unmatched mass.
- A supported assigned match with d<=1 is exact when d<=0.25 and both identified
  transformation magnitudes are <=1/48 log2 unit. It is transformed when any of
  those three exactness tests is known to fail. If none fails but a required
  transformation is unidentified, the exact/transformed distinction is unresolved.
  A transformed classification from acoustic residual includes tolerated timbre/
  articulation variation; it does not invent their transformation operators.
- Otherwise, a known no-memory assignment or a supported match with d>1 can
  classify nonrecurrence only after a covered, completed, nonempty search. Compare
  the six observable ending descriptors of u and the immediately preceding
  completed foreground span, using §9.6's identical two-second extraction and
  frozen development scaling, and RMS over their common valid coordinates.
  Departure >1 gives contrast; departure <=1 gives unmatched. If a predecessor
  exists but has no common valid coordinate, give unresolved. With no preceding
  span, a supported nonrecurrence result is unmatched. This means no accepted
  match in the declared searched inventory; index limits remain reported.

Freeze these cutoffs in O03 before the section stage-2 fit and compare half/double
match acceptance (0.5/2), residual exactness (0.125/0.5), transformation tolerance
(1/96 and 1/24 log2 unit), and departure (0.5/2) under O10's advance decision rules.
The six-descriptor formula/scaling is prepared before stage 1; section features
reuse it without requiring medoid fitting, table construction or M6 action use.
For a completed span u, the ending descriptor uses t equal to that span's last
observed assigned-support endpoint, not its later commitment/availability time.
Its exact window is
`J_u=[max(epoch_start,group_generation_start,span_start,t-2 s),t]`, using the
generation that carries the ending support. Apply §9.6's same six coordinate
formulas, global development scaling and per-coordinate 90% known-support rule
over this clipped window; an empty/unsupported window is masked. No predecessor
audio is borrowed merely to fill two seconds. A continued prefix crossing a
generation change therefore uses only the ending generation's supported segment;
the linked older prefix is not silently mixed into this descriptor. Extract/store
it from the original support when the span ends and preserve it with that record,
rather than recomputing it from a later live-group state. O03 fixes this window
and provenance for both the new span and its predecessor. O09/O10 fixtures use
consecutive sub-two-second spans in one generation, a generation-crossing prefix,
gaps and an empty ending window. The departure-threshold sensitivity comparison
also contrasts this clipping with an otherwise identical window that omits only
span_start (retaining epoch/generation bounds), measuring contrast/unmatched
assignments and the registered section-task metric under the advance decision rule.

Let w_u be the span's integral of valid acoustic assignment support in seconds,
multiplied once by its declared fractional span membership. Normalize alternative
assignments of the same observed occurrence under the existing commitment rule;
reinterpreted aliases, retrieval-only reads and repeated scoring add no new w_u.
The five masses are `M_c=sum_u w_u*e_u,c / sum_u w_u`. Unresolved e_u contributes
to the fifth mass, not to a fabricated exact/nonmatch label. No positive observed
span weight masks all five; do not create a record from an unheard interval.

Pair consecutive completed foreground span records on that path in observation
order (support-end timestamp, then start timestamp, then stable occurrence-ID
ties). Do not skip an unresolved record to connect its neighbors. Use
`w_uv=min(w_u,w_v)` and the outer product `e_u,a*e_v,b`. The 25 coordinates,
row-major in the category order above, are
`T_ab=sum_valid_adjacent(u,v) w_uv*e_u,a*e_v,b / sum_valid_adjacent(u,v) w_uv`.
Normalize over all 25 cells together, not separately per row. An adjacent observed
record pair is valid only if its ordering and intervening audio/association support
are known with at least 90% physical coverage; otherwise record its excluded
weight. If valid pair weight is below 90% of all adjacent observed-record pair
weight, mask the transition block. With no positive valid pair denominator it is
also masked. This is coverage of observed record adjacencies, not an estimated
count of unheard spans. Unknown correspondence with known adjacency still supplies
the unresolved row/column. Include ties, overlap/censoring, new section starts,
first/single spans, missing intervals, ambiguous retrieval, nonuniform membership
and alias deduplication in O10's exact-sum fixtures.

Cumulative masses use records assigned since that section candidate's start;
the recent descriptor uses its last four completed records and only adjacent
pairs within that ring. Carry the same numerators, denominators and excluded
weights; no pair crosses a new section start or a dropped ring boundary. Store
these statistics in the bounded section records and include the ending descriptor/
predecessor state in the declared 640-byte layout check and O04 census. These
rules define the 30 coordinates without adding a latent graph or 25 extra heads.

Define the nine articulation/participation coordinates on a span or the section's
cumulative support as follows. Let `w_g(t)` be stage-1 valid acoustic assignment
support for that group, including known inactivity; it is not energy magnitude.
`D_g = integral w_g(t) dt` over newly observed physical support. A component's
missing fraction records unsupported input separately; below 90% valid support
mask it. For ratios with no valid denominator use a missing value, not zero.

| Coordinate | Formula and provenance |
|---|---|
| Four articulation occupancies | For each attack/continuation/release/gap state s, integrate `w_g(t)*1[path_state=s]` over time when articulation is known, divided by the total known-articulation weighted duration. They sum to one there; unknown state duration is missing. These are path-local assignments, not a new posterior average |
| Event density | Sum the group's admitted accent weights in that support divided by D_g, in events per known weighted second; count each accent once |
| Supported-periodic fraction | Integrate `w_g(t)*1[at least one integer-grouping or cyclic-word proposal is acoustically admitted]` divided by D_g. Include both uniform and nonuniform cyclic words: periodic refers to recurrence of the whole word, not equality of its internal intervals. P_best alone, or a surviving path candidate without current acoustic admission, is insufficient. Known ungrouped support contributes zero; residual or unsupported grouping input is masked |
| Persistent-offset dispersion | At each new hop, for all supported outgoing and within-group periodic timing histories, average mode distance squared from the coincident-reference position, weighting modes by normalized peak-bin mass and histories by the ranking support defined in §9.2. Normalize circular distance by 0.5 cycles and linear distance by four; then integrate this value with w_g over its valid duration. If no mode/history is available, mark missing |
| Residual timing dispersion | The same support-weighted history average of record distance squared to its nearest retained mode, with the same coordinate normalization; integrate over valid physical duration. This distinguishes a persistent offset from jitter around it |
| Overlap fraction | For a resolved g, integrate `w_g(t)*1[E_g/E_bus >= 0.01 and sum_(j in resolved, j!=g) E_j/E_bus >= 0.01]` divided by D_g. The sum includes only the other resolved groups; residual energy remains in E_bus but cannot by itself satisfy coexistence. The residual slot's overlap coordinate is masked. Known zero bus energy gives zero for a resolved g; missing assignments are masked. This measures coexistence of assigned resolved activity, not separated physical sources or a Voice count. Test 0.005/0.02 energy-share thresholds and the explicitly named residual-included alternative |

O03's section-feature manifest pins that periodic-proposal union. O10 includes
integer-only, uniform-word-only, nonuniform-word-only, P_best-only and masked
cases; the nonuniform-word-only case contributes one on its valid support.
O03/O09/O10 also pin the overlap coordinate's resolved-only index set. Overlap fixtures include one
resolved group with large residual/noise mass, two resolved groups, ambiguous
fractional assignments and a residual-only mixture. Compare the residual-included
variant on the same section-task material with the registered threshold/decision
rule. Neither assigned-energy proxy is the T1 overlap-judgment likelihood, and
unresolved source content is not asserted absent when the resolved-only value is zero.

Weighted averages divide by the sum of valid weights; ties/modes use §9.2's rules.
Store numerator/denominator sufficient statistics, not all past timing records.
The timing quantities sample only a new observed hop's current history summary;
they are contextual duration statistics, not repeated independent observations of
its older accents. Apply the same definitions to the four-span ring, preserving
each component's valid-duration weight. Freeze these formulas and provenance
before the section-hazard stage-2 fit. Carry component validity masks alongside
the 39 descriptor values and 82 head coordinates. For this reference section head,
a masked scalar is imputed to its frozen development mean (zero after
standardization), with no extra coefficient-bearing coordinate; the existing
unobserved-support fraction is the explicit missingness predictor. Do not silently
add another 82 learned indicator weights. Register and compare this imputation
against an explicitly larger indicator-head variant on development.

Accumulate weighted sufficient statistics once per new supported phrase/span
assignment; update articulation/time integrals each hop. Provisional membership
uses path-local statistics and is sealed by the §9.3 policy. Alongside the cumulative
39-vector, maintain the same descriptor over the last four completed supported
phrase spans, with fractional assignment weights and a bounded ring. This recent
contrast allows internal development without erasing the ongoing section. Test
recent windows of two/eight spans and cumulative-only as ablations; a span-count
window is an engineering summary, not a section's prescribed length.

The section hazard and exit-category feature vector contains the cumulative
39-vector, recent-minus-cumulative 39-vector, log elapsed seconds, fraction of
unobserved support, best length-normalized retrieval match, and retrieval ambiguity:
82 components, standardized on development. For the last two coordinates reuse
the group's existing §9.3 matcher query at its 0.1 s cadence and 16-episode bound;
no section-specific matching pass is added. The cue is the latest ongoing acoustic
span prefix selected by greatest valid acoustic assignment support integrated
over `[max(epoch_start,t-0.5 s,span_start),t]`, at matcher observation endpoint t.
Use the same stage-1 acoustic support weights as the span descriptor, in weighted
seconds, before joint reweighting; this selects a cue and grants no new memory
credit. Break ties by later last-observed-support endpoint, then later span start,
then lower stable occurrence ID. A candidate with zero supported integral is
ineligible. If no ongoing candidate qualifies, use the supported committed span
with the latest observed end, then latest start, then lowest occurrence ID; if
none qualifies, mask both retrieval features. The selected query still uses its
whole bounded prefix descriptor, not only this selection window. Register the
0.5 s selector in the section-feature manifest and compare 0.25/1 s windows;
O09/O10 includes competing ongoing spans of different ages/support, ties and
missing input, with exactly the same cue passed to the reused matcher. It is neither the cumulative 39-vector nor the four-span
ring. For each distinct eligible episode take its best supported transformation's
acoustic score `m_i=w_match · match_features_i`, already cue-step normalized by §9.3;
do not include log availability or normalize by section age. Best match is max m_i;
ambiguity is the best-minus-second-best m gap (smaller means more ambiguous),
masked if fewer than two episodes qualify. No match masks both. Preserve matcher
support/age and truncation flags, freeze these coordinates before the section
stage-2 fit, and compare with an exhaustive same-cue fixture. It contains no stage-3 posterior or
closure-head output. The categorical has new-context, recurrence and contrast
outputs. These are frozen stage-2 inputs; their shared/local-context interactions
belong to stage 3. Bounded descriptors are copied with hypotheses rather than
retaining an unbounded section-long sample list. This allows internal development
while maintaining contextual identity. Propose stay, novel
context, transformed recurrence and contrast using the graph in §9.3. Use the
section hazard in §9.4 and the common conditional update, without reading Scenario
labels. A texture change is evidence for a possible transition, not a section by
itself. Competing section boundaries can coexist across auditory groups.

Name two section-head controls before M0 collection. `section_elapsed_only` uses
only intercept and `log(1+section_elapsed_sec/1 s)` in its softplus hazard and
three-category exit softmax. `section_recent_only` uses the recent four-span
39-vector, that log-elapsed value and unobserved-support fraction (41 inputs),
with the same mean imputation; it receives no cumulative descriptor, retrieval
score or ambiguity. Fit both on the same interval/type targets with the full
head's regularization, mixture-first-event aggregation and stage-4 calibration.
These are fitted head controls, not outputs read from the full section posterior.
For whole-model ablations, replace the corresponding head and mask joint features
that would reintroduce excluded section inputs, then refit stages 3/4. T6 names
`section_recent_only` as the primary section boundary/type control and requires
the elapsed-only comparison as well; its separate return-recognition endpoint
still tests retrieval against no-retrieval/recent-only memory controls.

The cumulative descriptor and four-span ring belong only to local section paths.
Shared context paths keep episode/context handles and weights; their features read
the preceding local summaries, not a second mutable copy of section statistics.
Use a fixed record budget of at most 128 f32 statistic slots plus 16 u64 bookkeeping
slots (640 bytes) for each cumulative/ring entry, including support/weight data.
Five entries therefore cost at most 3,200 bytes per retained local path. The default
`8 contexts * 8 groups * 16 local paths` permits at most 3,276,800 bytes per bus
for these records, excluding separately reported graph/ancestry storage. Compute
candidate updates as scratch deltas against immutable parent records; materialize
copies only for paths surviving both shared and local pruning, not all 65,536
enumerated pairs. At ten hops per cycle the conservative copy bound is 32,768,000
bytes per bus per cycle. Pending ancestry retains compact support/assignment links,
not another full descriptor per historical hop. Include actual descriptor bytes,
copy traffic, other state storage and half/double capacities in the runtime budget.

Whole-piece state is the episode graph plus current retrieval focus and unresolved
continuation links. It accumulates only what has been heard so far. A return edge
requires correspondence to retained earlier material and distinguishes exact
recurrence from a transformed one. After an intervening context, a returning
prefix can reactivate its earlier continuation; a first occurrence has no such
support. Compare these cases with identical local audio and all other accessible
recent evidence matched. Long form is therefore a distinct test of retrieval and
context, not a large numeric time constant.

A form summary exposes supported contrast, recurrence, transformation, development
and unresolved relations. It neither predicts the composer's entire future score
nor dispatches section changes. Expected return and experienced completion may
remain uncertain or absent. An open ending is valid; agreement with the artist's
intended closure is measured, not imposed by the runtime ending.

T7 includes a separately labeled retrospective whole-piece completion instrument.
After one uninterrupted hearing of the complete registered performance, with no
replay, ask: "Considering the whole musical course you just heard, how complete
or resolved does it feel?" Use the same five fully enumerated completion anchors
as §9.4. Administer this after the heard ending without a reaction-time target;
record response latency, but do not treat it as neural closure latency. Explain
that playback stops are observation endpoints. Include shaped endings, unresolved
open endings and interrupted controls; a low completion rating of an open ending
is a valid target, not an automatic artistic failure. This stratum does not reuse
or pool the short-prefix ratings as if they were whole-piece judgments.

Fit a separate scoring-only ordinal completion head after the other stage-2 heads
are frozen. Its four scalar inputs are: the current observed-prefix closure
head's expected rating; the continuation head's expected rating; the active
retrieved continuations' unmatched-expected-support fraction (assignment-weighted
by stage-1 retrieval weights, using §9.4's already defined remaining-support proxy);
and the section head's conditional return-type probability given an exit. Use
out-of-fold supported lower-head outputs before temperature or backoff, aggregated
with the fixed acoustic weights and carrying support separately. This scoring-only
whole-mixture head is downstream of those predictions, never a new local inference
factor or an input back to them. Do not add the post-stage-3 posterior itself as
a feature. Include one missing
indicator per scalar, eight coordinates in total; no supported retrieved
continuation masks that proxy. Record the exact four inputs before this fit.
Use the ordinary §9.4 supported-mixture/backoff/temperature convention, with the
whole-piece head's own coefficients, training marginal and stage-4 temperature.
Impute each missing scalar to its frozen development mean with its indicator set.
The head's support is the arithmetic mean of the four input support fractions;
this declared availability proxy permits a partially supported judgment, with
all-missing support zero. Report the four fractions and their mean, and compare
this rule with a minimum-support gate on development before freezing it. The snapshot is the last
causally incorporated audio state before EOF; do not append silence, reinforce
memory, reset context or feed an EOF feature to predict the rating. The same
feature extraction can run at an arbitrary prefix and cannot detect a file end.

Compare with closure-head-only, recent-gap/energy-only (the §9.4 2 s features),
and elapsed-time-only ordinal controls on the same whole-piece ratings and splits.
Held-out ordinal log-loss improvement is the primary T7 completion metric; M0
selects its primary named control, margin and power alongside the prospective
return/whole-context comparisons, which remain separate required endpoints.
A failure cannot be hidden by pooling it with successful short-prefix closure.
This retrospective head is an acceptance instrument, not a centralized stop or
completion reward. T7's local generative influence still passes through retained
context and the action dimensions below. Record its fit and scoring cost separately
from the ordinary live worker; it adds no undeclared per-cell head computation.

### 9.6 Local generation from contextual opportunities

Expose a timestamped environment summary to each Voice: relevant relation and
context alternatives, expected arrival/continuation, boundary/closure support,
retrieval support and coverage. Selection uses the Voice's existing body state.
Keep intrinsic pace and sound duration separate from participation adaptation.

Candidate actions comprise onset now/after a delay, wait, skip, continue, release,
and leave a gap. This contextual search selects only class and timing. Pitch, timbre and envelope-
shape variation stays on the existing seeded body/flow/controller path; the one
body recipe supplied for an opportunity is fixed across all 112 previews. No
extra pitch/articulation-variant candidates multiply that bound. Class-dependent
onset, continuation, release and gap timing still change articulation as declared
below. Preserve the existing variation draws and schedule in on/off comparisons;
no stored score is replayed. Evaluate a bounded
preview of each action against the same immutable observed context. Known own
intent may enter this preview, but synthetic preview audio is not listener evidence
and never reinforces memory. Predict only to the supported horizon; longer
expectations supply conditional constraints, not invented detailed waveforms.

The seven class previews have explicit meanings, using the actual body envelope:

| Class | Hypothetical extension and scheduling meaning |
|---|---|
| Onset now | Excitation/attack at decision time, with the body's ensuing trajectory |
| Delayed onset | No new excitation until the candidate time, then the same onset prototype |
| Wait | No added excitation or forced release; retain any existing body's natural trajectory and reconsider the pending opportunity at the candidate time |
| Skip | The same no-added-sound extension for this preview, but consume the current opportunity without excitation and defer to the next intrinsic opportunity |
| Continue | Follow the active body's existing planned trajectory without a retrigger or early release |
| Release | Start the body's release at the candidate time and retain its actual projected tail |
| Gap | Release if active, then withhold new excitation for one current intrinsic period; preserve the release tail rather than replacing it with invented zero energy |

Wait and skip can therefore have identical acoustic consequences while differing
in opportunity bookkeeping and bodily cost. The gap is a hypothetical abstention,
not a new observation of silence; it may fail to become quiet within the supported
horizon. Body legality and rate constraints still apply.

Classify the body default from the existing planned bodily transition with new
context pressures disabled: a due excitation/release takes its corresponding class
in the body's existing event order; an otherwise sounding body continues at zero
delay; an otherwise silent body uses onset at its next intrinsic opportunity
(now or delayed), or waits until reevaluation if ecology permits no excitation.
Do not create an otherwise forbidden event. Store this mapping with the model
version and keep it identical across ablations. Zero new context returns this
default directly; a supported-context utility tie also prefers the default before
the stable per-Voice tie order for other candidates.

For Voice `i`, the reference decision is:

```text
utility_i(action) = disposition_i · consequences(action, context)
                    - bodily_cost_i(action)
```

The consequence vector keeps audibility/overlap, relation support or departure,
continuation, closure, participation invitation and uncertainty separate. Dispositions are authored signed
preferences, allowing support, contrast, overlap or withdrawal. They are not
learned human reward constants. Preserve the current bodily timing cost and onset
rate limits. Freeze the following scaling with the fitted/calibrated model version;
dispositions refer to that version, not to arbitrary newly fitted raw scores:

| Action consequence | Unit and reference scaling |
|---|---|
| Relation support/departure | Signed `tanh(score_difference / scale)` in [-1,1]; scale is the development interquartile range of nonzero differences, floored at 1e-6 for numerical protection |
| Private participation fit | Difference in normalized timing-bin mass at the candidate versus body-default relative position, in [-1,1]; a learned Voice trace, not a human probability |
| Continuation and closure | Difference from the body-default candidate's calibrated expected ordinal rating divided by four, each in [-1,1]; report the absolute normalized rating [0,1] separately, without relabeling it event-time survival |
| Participation invitation | Difference in calibrated expected desire-to-participate rating divided by its four-step range, in [-1,1]; optional authored coefficient |
| Overlap | Sum of envelope-weighted `own * external / (own + external + epsilon)` divided by the corresponding own footprint energy, in [0,1]; each numerator term is at most own energy, and zero/unknown remain separate |
| Audibility proxy | Own-energy-weighted mean of own/(own+external+epsilon) on the same 16-point time footprint as overlap, in [0,1]; an energy proxy until perceptual validation |
| Uncertainty | For reporting, `U = u_a + (1-u_a)*H_a/log(N_a)` in [0,1] on the per-cell alternatives defined below; entropy is zero for N_a <= 1, and all-unknown U=1. For action, use `U(candidate)-U(body_default)` only when both projections have support; otherwise zero |
| Listener contextual consequences | Separately bus-tagged copies of relation, continuation, closure, participation invitation and uncertainty differences, using the same units and body-default references as above. Each has its own optional authored coefficient, initially zero; no listener copy of a private Voice trace |

Every difference-based column requires support on both the candidate and body-
default side for the same coordinate/reference, including relation support,
private fit, ordinal ratings, uncertainty and listener copies. Exactly one
supported side makes that component unsupported and its action contribution zero;
do not substitute a fabricated zero rating/bin mass or a one-sided score. When
mixing private references, use only their paired supported contributions under
the same frozen reference weights; retain other mass as unassigned without
renormalizing it onto paired survivors. Apply the corresponding mask to tabulated
head differences. A known zero consequence, such as supported silence, is valid
support and differs from unknown. Other independently supported columns remain
eligible. Register one-sided cases in both directions, partially paired references,
out-of-range nonperiodic timing and routed-away listener sound in the candidate
fixtures, alongside the both-supported/all-unknown/body-default cases.

The Voice's local own-excluded forecast supplies the external energy for both
overlap and audibility, in linear full-scale-squared audio units. The shared
table's bus-mixture energy is not used as a fallback or added to that forecast;
it includes the Voice's heard contribution and cannot be labeled own-excluded.
Convert the local forecast to the ratios above before the authored dot product. Scores are
not combined with unscaled energy. Listener columns project the candidate through
its actual presentation routing and gain; a Voice with no presentation contribution
has neutral listener consequences. Incompatible projections stay unknown. These
columns consume the listener relation instance, while existing `ListenerState`
consumers retain their short-time inputs. Before calibration, probability-based
action columns remain disabled. The body-default candidate and normalized squared timing
displacement are fixed in each ablation. Refitting requires a new model/scaling
version and rechecking disposition semantics, not silent reinterpretation. Select deterministic maximum utility, with a stable per-Voice seeded tie breaker.
Existing body/flow variation and bounded pitch/articulation variants provide local
variation; preserve their seeds and random-draw schedule in context ablations.
Do not add a hidden softmax temperature or unlogged random action selector. A
Voice-specific tie breaker avoids identical empty-evidence choices becoming a
centralized clock. Every unsupported signed contextual column contributes zero,
including the action uncertainty difference; absolute unknownness is still reported
as one. An all-unknown state therefore supplies no action pressure. Known local
energy ratios and the existing body/ecological path remain usable independently.
Low confidence attenuates only the new context contribution, including uncertainty
and optional listener columns, not the Voice's whole ability to sound.

For O21, the new author-facing temporal control surface is limited to: temporal
mode (off, passive observation, local participation), with shipped reference
default **off**; the signed gains for the
explicitly enumerated consequence columns above; and the separate optional
listener-column gains, initially zero. There is no additional action-class-prior
control or additive class bias: all seven actions use the stated consequence/
bodily-cost rule. Existing body/ecological dispositions remain on their own path. Defaults expose
only the mode and an authored disposition; advanced setters compose the same
listed fields rather than inventing more musical controls. Register their exact
Rhai/config identifier spellings and enumerations at M8 against real consumers;
the default mode semantics are already fixed here. Opt-in is explicit, and
omitting the new setting must preserve existing instrument/2x3 participation
behavior bit-for-bit under the same seed, body and input. O21 tests omitted mode
against explicit off and the versioned accepted legacy baseline, including after
O19's local-participation-only replacement. A later default change is a separate
authored promotion that versions the expected default mode and its accepted-baseline
fixture together, not an M8 naming decision. Existing pitch, body,
routing and Scenario controls remain available. Beam widths, detector constants,
fit parameters and resource limits remain explicit research/model configuration,
not extra musical sliders; changing them follows the model-version/refit contract.
Report settings cannot enable, disable or otherwise change temporal behavior.

The reference temporal mode is fixed for one performance and selected before
runtime wiring; mid-performance mode mutations are rejected without changing
workers, memory or body state. A different mode requires a new performance with
fresh relation-instance/epoch identities, empty episode banks, empty new private
reference traces and no inherited tables. Mode is independent of the existing
pitch-DCC enable and report settings:

| Mode | Relation processing and new state | Generation |
|---|---|---|
| Off | No new temporal relation workers, queues, banks, tables or private-reference learning | Existing instrument/body and production participation behavior only |
| Passive observation | Both relation workers, banks, tables and observed-outcome private traces run from empty state | Every new temporal/private/listener action column is disabled; existing body-default behavior remains |
| Local participation | The same processing/state as passive mode | Eligible new columns may influence local choices under all support/age/legality gates |

The off and passive rows preserve the legacy 2x3 participation behavior even after
O19 passes. Only the local-participation row substitutes the promoted trace; passive
learning alone never applies that trace to generation. The separate default-promotion
and baseline-version rules above apply to any later change of the shipped mode.

Existing analysis or ListenerState work required for pitch DCC/reporting may still
run in off mode; it is not a hidden temporal bank. Register a mode-isolation fixture
covering fresh starts in each mode, a rejected live mutation, report/pitch toggles,
and separate performances with the same input. Context-on/off pressure experiments
use the explicit assay intervention on otherwise identical state, not an
undocumented live-mode switch. Analysis-configuration epoch restart within a
performance keeps the previously declared rule and does not change its mode.

Here **air-gap has the repository's specific meaning**: the conchordal instrument
never writes audio to disk in any build profile. Its CLI has no WAV/output-path
option and its entrypoint never calls run_render; offline recording belongs to
the separate conchordal-render binary. M8 inspects the instrument's CLI, entrypoint
and reachable output paths and records this boundary, following AGENTS.md.
This is distinct from the already required evidence-isolation contract: Scenario
labels, authored intentions, Voice IDs and private participation traces cannot
enter either relation instance's observation path. That separate contract is
verified by typed ownership and registered label/route leakage fixtures. Do not
rename data-isolation testing as the audio air-gap or add recording to the instrument.

The worker publishes shared consequence tables, separately tagged by bus, for
at most eight body-prototype clusters fitted on development onset/envelope/spectral
and rate descriptors. These are shared tables, not one graph preview per Voice.
The body and acoustic-group sides use the same six observable descriptor values,
in order: spectral log2-frequency centroid, log2-frequency spread, log2 RMS,
mean positive log-envelope rise, mean positive log-spectral flux, and
log(1+weighted_accent_count/observed_seconds*1 s). For a live group, extract them
from J=[max(epoch_start,group_generation_start,t-2 s),t]. Spectral centroid and
spread are the mean and standard deviation of log2 bin frequency under E_gb
pooled over valid physical hop support; zero spectral mass masks both. Window RMS
is the square root of its time-mean assigned energy and log2 RMS uses the existing
1e-6 amplitude floor. Rise/flux are physical-time means of the already defined
raw positive adjacent-hop components on valid comparisons. Accent count sums the
registered detector's weights; its density uses the same coverage/cap-eviction
rule as other density windows. No fundamental-pitch estimate or Voice rate/ID
is substituted for an absent observable coordinate.

Each coordinate needs 90% valid support in J, or is masked; empty J masks all.
Report epoch/lifetime clipping. Obtain body-record descriptors with this identical
extraction from their actually rendered development audio/observed accents, and
freeze one development mean/sd per descriptor coordinate over those registered
body records (including routing/gain variants). Apply these same statistics and
the 1e-6 sd floor to body, prototype and live-group descriptors. These are that
descriptor's global training statistics, not live per-group restandardization.
Use RMS distance only over coordinates available on both sides; no common
coordinate means ineligible. Equal distances prefer more common coordinates,
then the stable source/record ID for bodies or group-generation handle for groups;
near-equal values are not silently rounded into ties. The group pool excludes
residual. Cache its six values/masks once per worker cycle, reusing the group
energy/envelope/accent accumulators; at most 8*7*6=336 coordinate comparisons per
bus/cycle assign eight prototypes. Include extraction/window-storage work in O04,
not an uncounted per-prototype spectral rescan. Register the ordered descriptor,
window, masks, scaling-record hashes, distance/gate and ties in O03/O14 with the
body/model version. Masked pitch-like spectra, envelope-only groups, ties,
partially shared masks and no eligible group are required projection fixtures.

For an arbitrary live body, the source is its own actually emitted audio
contribution after the actual bus routing/gain and before mixture summation and
guard processing, separately for each bus. Apply the same spectral/envelope/accent
extractor and six formulas to this isolated contribution, with the whole spectrum
as its energy assignment; do not infer or borrow an auditory-group identity.
Its causal window is J_body=[max(epoch_start, Voice_birth,
body_model_generation_start, t-2 s), t]. Use the same frozen development scales,
90% per-coordinate coverage, masks and zero-spectral-mass rule. Known silence is
observed support; an empty window is unknown. Publish at most every 0.1 s with
original audio support, Voice/body-generation and bus tags. Apply the existing
observation-age and coverage gates to this descriptor as well as to its table;
missing, stale or incompatible descriptors leave the new contextual columns
unknown and the existing body behavior available. A young Voice uses its clipped
window, without a mandatory two-second warmup or an ancestor's descriptor being
presented as its own measurement.

A replacement or inherited/mutated body recipe starts a new body-model generation
and clears only this descriptor history, never its bodily clock. Ordinary
oscillator/envelope evolution and registered continuous controller/interpolation
trajectories stay within that generation. O14 records their supported projection
range; a structural recipe change outside it invalidates the old descriptor until
new actual sound supplies support. An action candidate outside that registered
range is unknown even if the current descriptor is near a medoid. Development
records and actual-Voice offline comparisons use this same causal window, including
young-body clipping. Counterfactual renders remain fidelity targets, not live
observations. O09/O14 fixtures include unregistered, newborn, inherited/mutated,
interpolated, silent, rerouted and stale bodies.

This is a private body-characterization path: isolated contributions and Voice IDs
cannot enter either passive relation instance. It runs in passive-observation and
local-participation modes and is absent in off mode. Capture uses preallocated
rings and a separate background extractor, with no spectral analysis or allocation
in the audio callback. At the canonical 48 kHz configuration, the maximum two-second
raw rings for 64 Voices and two buses occupy 64*2*96,000*4 = 49,152,000 bytes;
capture copies at most 64*2*512*4 = 262,144 bytes per hop. Descriptor records have
a 128-byte bound per Voice/bus, totaling 16,384 bytes. O04 separately inventories
the extractor's analysis state, spectral windows, queues and scratch; these raw
ring bounds are not its total memory cost. Its work joins the concurrent preflight:
the private extractor must meet p99 <=40 ms per 50 ms cycle, and capture is charged
to the existing combined hop budget. An overrun masks lost support rather than
republishing old sound as current. This path writes no audio to disk.

Novel and evolving/mutated populations are inside O17's 4/16/64-Voice envelope.
Report compatible and unknown fractions over all decisions, including rejected
bodies, and apply the same action-fidelity, freshness and required context-effect
gates to these strata. An all-unknown novel-body population cannot pass the
context-effect gate. A failure reopens M6/M8's descriptor, projection or resource
design; excluding those bodies after observing failure cannot complete O17.

Fit them with k-medoids (PAM) on the development-standardized descriptors, using
actual observed body configurations as medoids. Deterministic farthest-first
initialization starts at the lexicographically smallest registered source/record
ID; subsequent ties use that same ID. Apply the best objective-decreasing medoid
swap until improvement is at most 1e-6 or 100 passes; record the stop. Store medoid
IDs, body recipes, descriptors, feature scaling, distance rule and assignment ties
with the fitted/calibrated model version. The mismatch threshold below freezes
with them. Any prototype/scaling/threshold change creates a new model version and
repeats the shared-versus-actual-Voice action-rank comparison before promotion.
Use standardized root-mean-square descriptor distance at most 0.25 for a compatible
projection; otherwise the new contextual columns are unknown. This is an explicit
prototype-mismatch approximation: compare shared-prototype and actual per-Voice
projected scores offline, report action-rank disagreements, and test half/double
prototype count and mismatch threshold before promotion. A population expansion
does not expand the worker's prototype table count. For each prototype:
for each of the seven action classes, 32 future-offset cells carry relation
support/departure, continuation, closure, participation invitation and uncertainty,
plus projected bus-mixture energy and supported duration bounds. Bus-mixture
energy is a contextual descriptor, not a Voice-specific external-energy estimate. Construct cells by extending the frozen
context with each action class's descriptor prototype (an onset, sustained
trajectory, release or gap), evaluating its conditional feature-score difference,
and averaging over retained alternatives. Tables store absolute relation scores,
ordinal distributions/ratings and uncertainty, not a medoid's pre-subtracted
consequences. Each Voice looks up its actual body-default candidate once and forms
the declared candidate-minus-default differences before scaling. Unsupported
candidate or default context gives a neutral signed column. No preview is committed
as evidence.
A cell has at most 128 shared-context/selected-group-local-path alternatives.
Start from their issue-time joint weights `q(C)*q(local|C)`. The cell's unknown
mass `u_a` is the original shared/local unknown mass plus the weight of resolved
alternatives for which that action's feature projection is unsupported. It cannot
redeem original unknown mass using a hypothetical sound. For the remaining N_a
resolved alternatives, normalize `q(C)*q(local|C)*exp(delta_log_score_a)` to sum
one, calling the result p_a; their total retained mass stays `1-u_a`.
`H_a=-sum p_a*log(p_a)` uses natural logs; divide by `log(N_a)` only for N_a>1.
This is entropy of represented context/path interpretations conditional on the
candidate, not entropy of ordinal rating categories. Use this hypothetical mixture
for projected heads and uncertainty; the passive state is unchanged. Test equal,
concentrated, single-resolved, all-unknown and action-incompatible cells. Keep
u_a separate from the observed-coverage/backoff factor below.
For each prototype and bus, select the currently supported resolved acoustic group closest
to the six-coordinate descriptor just defined, using that window/mask/standardization
and exact distance-tie rule. Apply the same distance<=0.25 compatibility gate;
the stable-group handle is the last tie breaker after common-coordinate count. The reference
preview changes only that group's path-local features with the hypothetical action
extension; all other group features and all observed probe-aggregation weights
remain frozen. It adds no extra pseudo-group. An unmatched prototype has unknown
contextual columns.
The preview has a separate hypothetical-support channel. Every feature input is
tagged observed, projected-from-the-candidate, or unsupported. On scratch summaries
only, evaluate the same feature formulas with observed and supported hypothetical
values usable by a head; never label the latter observed in P/M or fitting data.
The manifest lists this projection policy for every consumed feature before the
first table is built. The reference permits candidate-projected envelope, duration,
articulation/overlap occupancy and candidate accent timing against already supported
references. A scratch timing histogram may add projected candidate accents while
its observed reference history remains fixed. Ordered-match residuals may compare
the candidate descriptor with a retained expected segment; its expected timing and
memory availability come from the issue-time state advanced without reinforcement.
Current R/H/C and unmodified background descriptors are held at their issue-time
values and tagged as that approximation. No new acoustic grouping, external accent,
episode commitment, recurrence credit or future retrieved identity is fabricated.
If a formula needs another unsupported future input, its value stays masked.

For a cell at a future offset, its head window ends at that offset. Feature values
use the available observed prefix and this scratch hypothetical extension inside
the window; the hypothetical-valid fraction is reported separately. The observed-
coverage factor `e_issue` and acoustic group mixing weights are frozen from the
head's ordinary issue-time window (2/8 s), common to candidate and body default.
Projected category predictions use the hypothetical alternative mixture above.
Collect its supported head mass s_a (excluding u_a/inapplicable heads) and normalize
its supported path/group marginal to p_s,a. Using exactly §9.4's convention, the
cell distribution is `r_a*temperature(p_s,a,T) + (1-r_a)*p_0`, with
`r_a=e_issue*s_a`. The same frozen p_0 and head temperature apply to candidates
and body default; no temperature is applied to backoff or before path/group mixing.
Do not recompute e_issue as the proportion of a shifted future window already
heard: that would mechanically erase every candidate's supported future effect.
Existing unknown mass stays unknown; entirely unsupported projections give the
same backoff as default and no signed action pressure. Age/coverage attenuation
at actual consumption still uses real issue support, never hypothetical support.

Ordinary §9.4 reports and stage-2/3/4 fitting consume observed prefixes only. A
preview is a conditional consequence estimate using a transferred feature map,
not a report that future sound was heard or a newly calibrated perceptual fact.
Freeze this policy and test offset-dependent effects, masks and default differences
in the offline shared-prototype versus actual-Voice comparison, with rendered
counterfactual continuations reanalysed to test forecast/rank error and head
calibration under this transfer. Enable ordinal consequence columns only after
that declared transfer check passes on development and its frozen held-out
comparison. Correct algebra alone cannot establish future consequence accuracy. Rendering is
an offline assay, not work in the instrument's hop path.
A uniform grid of 32 cells, including offsets zero and 4 s from the table's observed
support end, is an engineering bound; unsupported cells are unknown
and cannot be extrapolated as detailed future sound. Context lookup uses the
nearest whole cell by absolute target time, with an exact midpoint tie choosing
the earlier cell. Copy that cell's category distributions, support, uncertainty
and provenance together; do not interpolate them or mix an unknown neighbor with
a known one. An unsupported selected cell or out-of-horizon target gives neutral
contextual columns and reported unknown; do not search farther for a convenient
supported cell. Local energy-window and private timing-bin interpolation keep
their separately defined rules. Freeze this whole-cell rule with the body/model
version and compare direct per-Voice projection/rank error, including transitions
to unknown cells, at half/double table resolution and horizon. A per-Voice body projection scales the prototype to its supported
pitch/envelope/rate range; an incompatible projection contributes unknown context.

At a decision, construct at most 16 timing candidates deterministically. Let `P`
be the Voice's current intrinsic period, and `H` the smaller of `2*P` and the usable
remainder of the 4 s table horizon, measured from the decision time. Include 12
uniform offsets `k*H/11` for `k=0..11`, the unchanged body-default next opportunity,
and the supported arrival distribution's 0.25/0.5/0.75 quantiles conditional on no
observed arrival yet.
Quantile construction uses the stage-4 calibrated arrival hazard, as recorded
per consumer in §10.1; an uncalibrated model provides no calibrated quantile claim.
Select that arrival model from habitat only: use the currently supported group
of the Voice's highest normalized-match-weight pattern reference. Map that
reference to a current acoustic group using committed spans assigned to the
reference in the preceding 2 s of physical stream time: sum each span's sealed,
fractional acoustic assignment support by current epoch/generation handle and
choose the largest positive sum. Old retired handles cannot match reused slots;
a retrieved old episode needs a newly heard committed span to associate it with
a current group. Ties use stable episode handle, then group handle. A private
reference with no such current association supplies no quantiles; do not silently
switch it to an unrelated group. With no supported private reference at all, use the
currently resolved group with greatest stage-1 acoustic assignment support in
the current 2 s summary, with the same stable tie rule. As in §9.2's residual
participation table, residual is ineligible for both this fallback and private-
reference mapping. Residual-only support gives no quantile candidates; the body
default remains available.
This is one selected group, not an undeclared mixture across all references.
Compare this rule with assignment-weighted cross-group arrival mixing on development;
freeze the source rule, 1/4 s association-window comparison, and timing-candidate
and half/double-grid comparisons. For a mixture of resolved arrival alternatives, form their
mass-weighted conditional CDF and invert it. Any nonzero reset-unknown interval
mass makes all three quantile candidates unsupported until an observed change
resolves that reset uncertainty; no midpoint or guessed distribution replaces the
range. A quantile not reached within the supported horizon is also unsupported.
Discard duplicate offsets and unsupported/out-of-range
quantiles; do not refill them randomly. The body-default opportunity is always
available, even beyond `H`, with neutral contextual columns where unsupported.
Without a usable table, retain that default and use `H=min(2*P,4 s)` for bodily
alternatives with neutral new context. Filter these times by each class's existing
body/rate legality with the following fixed class-to-time mapping. Onset-now uses
zero; delayed onset uses positive offsets. Wait uses positive offsets that are
legal reconsideration times for a pending opportunity, including the 50 ms
decision-spacing constraint; its candidate time is that reconsideration time,
never an immediate zero-delay search loop. Skip uses zero only when the current
due intrinsic opportunity remains pending and unconsumed; it cannot target a
future opportunity. Continue uses zero and requires an active body. Release uses
zero or positive offsets at which the existing body projection is still active.
Gap uses zero or positive offsets: the candidate time starts release if then
active, or begins abstention if already inactive, and withholds new excitation
until candidate time plus P frozen at the decision. Its natural release tail is
retained. All classes obey the existing bodily constraints, and the unchanged
body-default mapping above remains available. O03/O09/O21 register this mapping
with fixtures for due versus reconsideration-only skip, wait spacing, active/
inactive gap, delayed release after natural body expiry and the frozen withholding
endpoint. Filtering adds no times or classes beyond the 112-preview upper bound.
No filtering resets intrinsic
phase. In a context-pressure on/off comparison, both branches receive the same
candidate set generated from the common passive summary; a candidate-proposal
ablation is a separate comparison. Test half/double horizon multiplier and timing
resolution on development, retaining the body default in every comparison.

Contextual decisions occur at a due intrinsic opportunity or an explicitly
scheduled reconsideration; a newly published passive summary may request one
reconsideration of a pending/sounding action. Coalesce simultaneous triggers.
A Voice may run at most one contextual decision in any 50 ms of physical stream
time (20/s); the next eligible time is at least 50 ms after its last decision.
Candidate execution and table lookup do not recursively trigger another search.
Pending release/continue changes obey the same limit. Existing bodily events run
at their own legal times; if a trigger cannot be evaluated before its opportunity,
use the unchanged body default with neutral new context for that opportunity,
without resetting its clock or postponing an existing bodily release. Report
rate-limited contextual opportunities separately from unsupported evidence.

The registered ordinary temporal workload is at most 64 simultaneously living
Voices, evaluated at 4/16/64; this does not impose an instrument population limit.
The hop worker admits at most 64 contextual decisions per canonical audio hop and
128 in any half-open rolling 100 ms window: at most 7,168 candidate evaluations
per hop and 14,336 per 100 ms, each with the 16-point/16-trace bounds below. Store
these counters and scheduling scratch outside the audio callback. Process eligible
Voices in a rotating stable-ID order. If either shared budget is exhausted, use
body default with neutral new context and record computational truncation; no
unbounded catch-up queue is allowed. Above 64 Voices, report operation outside
the registered acceptance envelope even if this same cap preserves audio service.
Test 25/100 ms per-Voice limits, half/double shared budgets and clustered pending
opportunities on development. Normal-workload latency/audio gates and T-row action
comparisons include this decision schedule, actual rates and fallback fractions;
a per-decision bound alone does not establish hop-deadline compliance.
At the reference 512-sample/48-kHz hop, allocate at most 2 ms of hop-path CPU time
to all new temporal decisions combined, with 20% headroom: their measured p99
must be <=1.6 ms. Concurrently require total instrument-hop processing p99 <=80%
of the 10.6667 ms hop (8.5333 ms), with zero underruns/callback errors. Measure the
combined hop directly; adding separate p99 measurements cannot prove the total.
These CPU limits are not an extra end-to-end latency allowance: decision work is
inside the existing 25 ms publication/consumer contribution. O04 tests this slice
before collection and M8 repeats it with the complete implementation. A different
rate/hop requires a newly registered CPU allocation and the same headroom rule.

Each Voice then considers at most seven classes at those 16 timing candidates.
It selects one complete future-table cell, combines the existing own-excluded energy
forecast with a 16-point body-envelope footprint for overlap, and computes bodily
cost and the authored dot product. It also evaluates at most 16 private pattern
traces at the candidate's actual relative phase/arrival time, using their published
reference uncertainty and fixed-bin interpolation, combining supported reference
differences with their normalized match weights. This directly connects the
multimodal niche trace to action; the coarser future tables do not quantize that
relative position. A missing/evicted reference contributes no private-fit pressure.
The footprint covers the candidate's projected body-envelope support intersected
with the common absolute forecast horizon ending at the table limit (at most 4 s)
and the supported local own-excluded energy horizon. No table energy extends this
local support. The two energy ratios use this same supported intersection and
report any excluded envelope mass/tail.
Use 16 uniform midpoint samples on that supported interval, with equal integration
weights equal to interval duration divided by 16. Evaluate the actual body envelope,
including a delayed onset and release tail; times before excitation contribute zero
only when the body actually predicts silence. Own and external values are linear
full-scale-squared energy, aligned to absolute sample time. For the local external
forecast, linearly interpolate between adjacent supported energy-window centers;
inside a supported end window beyond its center, hold that window's value only
until its declared edge. Never interpolate across an unsupported window or
extrapolate past the supported edges. Use the same external values for overlap
and audibility. With O_j and X_j the own/external energies and equal integration
weights d_j, define `audibility = sum_j d_j*O_j*[O_j/(O_j+X_j+epsilon)] /
sum_j d_j*O_j`. Both ratios use epsilon=1e-12 in full-scale-squared energy units
inside O_j+X_j+epsilon and the same positive own-integral denominator; no
additional spectral-band weighting or decomposition enters this reference proxy.
Freeze window-center/edge and body-envelope mappings with the body/model version; test them against the existing
window lookup and dense offline reference. A zero own integral
means zero overlap only for a known-silent preview; audibility then is inapplicable.
Insufficient support or an unresolved tail beyond the horizon is reported, not
invented silence. Freeze support construction and quadrature with the body/model
version and compare 8/32-point footprints, including short attacks and long tails,
against dense offline integration and action-rank results before promotion.
The bound is 112 candidate evaluations, each with fixed table access, at most 16
footprint points and 16 private-trace lookups, each mixing at most seven frozen
anchor alternatives (at most 112 anchor evaluations per candidate); no graph traversal,
rendered candidate waveform, DTW or allocation occurs in the hop path. Continue
and release use the actual active body/envelope state. Worker table construction
is separately timed and bounded; stale cells lose context influence, not the
ability of the body to continue sounding.
Rebuild once per nominal 100 ms relation-worker cycle from one immutable state
version. Each bus builds at most `8*7*32 = 1,792` cells, averaging at most eight
shared contexts and sixteen local paths of the selected group: at most 229,376
projected-path evaluations per cycle. The frozen feature manifest permits at most
256 scalar feature terms per such evaluation (including the ordinal heads), giving
at most 58,720,256 scalar term evaluations per bus/cycle, plus fixed reductions.
Cache unchanged features and group mixtures; do not enumerate other groups' joint
paths. Record actual rebuild duration/count, scalar terms and skipped/incomplete
rebuilds alongside inference and DP counts. Increasing these limits requires a
new model/resource version; an unfinished table cannot be published as current.
Look up absolute target sample intervals;
reading a summary never resets its forecast origin. Each column retains both its
latest integrated acoustic support and any match-query support it consumes. Define
its live evidence age from the older of those support ends to decision time;
retrieving a historic episode does not make its original occurrence time the
live-update timestamp. A new cheap frame cannot conceal a stale matcher. Age
influence is 1 through 0.25 s,
then ramps linearly to 0 at 0.5 s. Coverage influence is 0 through 50% observed
support in the preceding 0.5 s, ramps to 1 at 90%, and remains 1 above it. Multiply
these two bounded factors with every new context contribution, including the
uncertainty difference; listener columns use their own bus's age and coverage.
The private participation-fit column also receives age/coverage attenuation,
applied per current reference binding before its existing weighted aggregation.
For a binding, use the older of (a) the published inventory's latest integrated
habitat support used for that binding's assignment and anchor/period-validity
check and (b) its cue-match query's raw-support endpoint. Age is decision time
minus that endpoint, with the same 0.25/0.5 s ramp. Use the published habitat
coverage in `[max(epoch_start,t_audio-0.5 s),t_audio]`, where t_audio is (a),
as actual observed sample-support union divided by this window's duration, with
the same 50/90% coverage ramp; an empty window gives zero. Publication time,
the original remembered episode time, the anchor event's original time and the
Voice's last executed outcome are not this live-evidence timestamp. An old anchor
can remain supported by a fresh acoustic validity check; an invalid/lost anchor
contributes zero regardless of a recent publication.
Multiply each already-weighted candidate-minus-default trace difference by these
two factors and its current binding-validity indicator once, using identical
factors for candidate and default. Retain unassigned/attenuated weight without
renormalizing it onto other references; this is the private column's sole age/
coverage attenuation, not followed by a second shared-table age multiplication.
Trace content/availability still follows its learned retention/interference law.
A long-idle Voice can therefore use retained trace structure when a fresh, valid
auditory binding returns. Store the support timestamps/flags within the inventory
layout limit and register this rule in O03/O09/O13/O15. Fixtures cover fresh and
stale inventory, fresh publication with a stale matcher, a long-idle Voice with
a retained trace, a lost anchor, partial coverage and a half-strength gate applied
once. Test half/double age/window thresholds and development alternatives to coverage thresholds; they
are engineering latency/availability choices, not cognitive retention constants.

M0 creates the planned `docs/roadmap/temporal-dcc/hardware-baseline.json` before
resource benchmarking. It records CPU model and usable physical/logical cores,
thread affinity and any VM/cgroup quota, RAM, OS/kernel/architecture, compiler and
release flags, power profile, audio device/rate/hop/buffer settings, bus and device
channel counts/order, downmix/output mapping, gain/guard convention, and the complete
ordinary workload. M8 reports reference that immutable baseline and its hash;
changing hardware starts a separately named operating-envelope result. Require
20% headroom inside the 50 ms processing slice: each bus's complete worker-cycle
processing p99 must be <=40 ms while both buses and the instrument run, in addition
to the measured end-to-end and audio-deadline criteria. Record actual maxima and
fallback fractions; a count bound alone does not establish this headroom.

Before stage-1 human data collection, M0 must pass an early numerical workload
gate and save `docs/roadmap/temporal-dcc/feasibility-preflight.json`. Build a bounded
synthetic workload from the selected reference operations: all canonical-hop
conditional terms, per-parent/shared-pair component lists and the 352,256 cheap
tuple-priority evaluations per hop, gesture-view emissions/coalescing, section/
continuation-link copies, table cells, the full anchor scan and its bounded
coarse-fit/refinement work, matcher DP and pair-grid expiry/rebuild. Include the
grouping-refresh scan explicitly: up to 15,360 integer and 7,168 word cases per
resolved group per cycle, at most 107,520 and 50,176 respectively across seven
resolved groups per bus, including search/step comparisons and dedup/ranking.
Run the cap-saturated 64-Voice decision slice concurrently: 7,168 candidate
evaluations at a maximal hop and 14,336 per rolling 100 ms, with table reads,
16-point footprint quadrature and up to 112 private-anchor lookups per candidate.
Use legal clustered schedules attaining these maxima, respecting each Voice's
50 ms interval, and the actual body-default, support and fallback paths. Require
the separate <=1.6 ms decision p99 and <=8.5333 ms combined-hop p99 at the reference
configuration, alongside both relation-worker budgets. Use nonzero varied inputs and the intended layouts, so a compiler cannot replace it with a constant
or a mask-only no-op. Run both bus workloads concurrently with the ordinary
instrument/analysis baseline on the declared hardware. After 60 s warmup, measure
10 min (6,000 cycles) for the cap-saturated case; require each worker's processing
p99 <=40 ms. Record maxima, RSS, operation counts and checksums against numerical
reference outputs. Missing optimized kernels are implemented or conservatively
charged before this gate; an empty stub is not a feasibility test.

Also report expected-case counts/costs from registered existing development renders
and synthetic 25/50/90% valid-feature coverage with two/four/seven resolved groups,
including silence, dense accents, retrieval refresh and long-form record copying.
Distinguish mask-skipped scalar terms from enumerated candidates and DP work that
still runs. Estimate wall/core cost per complete replay from measured mask-aware
rates and actual corpus durations, alongside the cap-saturated upper bound.
A miss invokes the revision order below before human collection or fitting
investment. This preflight qualifies the numerical design, not full integration;
M8 still repeats the complete implemented pipeline and end-to-end acceptance.

On a development benchmark miss, use this registered revision order:

1. Reuse caches/scratch and optimize numerically equivalent work first; repeat
   numerical and complete-workload timing checks, with no automatic model refit.
2. Try local joint-tuple cap 16→8 (never below the six reserved seeds), then local
   retained paths 16→8, then shared retained contexts 8→4, preserving unknown and
   all functional proposal categories. Each retained change is a new bounded model:
   replay/refit all four stages, rerun M5 sizing and relevant T-row comparisons.
3. Try table cells 32→16, then body prototypes 8→4. Refit medoids for the latter,
   repeat actual-Voice/counterfactual projection, transfer and all affected action/
   closed-loop gates. P/M parameters stay fixed only if their inputs are unchanged;
   changed inputs reopen their earliest affected fit and every downstream stage.
4. Try the scalar-term cap 256→128 only by a registered reduction of optional
   joint interactions, retaining every required function and its observation map.
   Refit stages 3/4 and retest all T rows; changes to an earlier feature map require
   that stage and all downstream refits. If required terms do not fit, reject this
   reduction rather than remove a temporal function.

5. For the Voice-decision slice, try footprint points 16→8, then timing candidates
   16→8 (four uniform offsets k*H/3 for k=0..3, body default and the same three
   supported arrival quantiles, with existing dedup/legality), then private traces
   16→8 using the existing availability eviction rule. Keep all seven action
   classes and the unchanged body default. Compare dense-footprint numerical
   error, action ranks and matched-candidate on/off behavior before all affected
   T-row/closed-loop gates. Quadrature/timing changes require new projection and
   action calibration checks, not a P/M refit when its inputs are unchanged.
   A trace-cap change refits the stage-1 private-trace model and repeats T2/T3
   and every trace-consuming action comparison; changed cognitive inputs reopen
   their stage and downstream fits. Charge the reduced cap's lost coverage.
6. For matching, try anchors per episode 32→16 by taking every eighth retained
   knot from the beginning instead of every fourth, transformation candidates
   4→2 by the same coarse-score/stable-grid ranking, then the DTW band 33→17
   knots (eight on each side of the same affine-time prediction). Preserve ordered matching, both transformation dimensions and
   mid-episode entry. Compare candidate recall and transformed/order-control
   correspondence loss against the original and exhaustive/roomy references;
   rerun M5 long-return capacity checks. Each retained change alters retrieval
   features: refit stage 1 and all downstream stages, and retest T4–T7 plus every
   earlier T row consuming retrieved context. Failed recall is not novelty.
7. For the gesture view only, try at most eight resolved local paths per group
   instead of the full local beam (highest conditional mass, stable-ID ties),
   then retained gesture tuples per context 16→8. Keep all single/pair families;
   excluded local/tuple mass enters the view's unknown mass without survivor
   renormalization, while the source local beam is unchanged. Compare the view
   with joint enumeration and the full-view T1 handoff/overlap targets; refit the
   stage-3/4 consumers and repeat T1/T5 plus every affected action gate. If an
   earlier-stage input also changes, reopen that stage and downstream fits.
8. For recurrence/pair-grid work, try the accent bank 128→64 (same 32 s window),
   then log-period grid spacing 1/48→1/24 octave while retaining the range and
   1/24-octave kernel half-width. Recompute incremental/rebuild and grouping costs;
   preserve missing/cap-limited support rather than shortening the denominator.
   Compare period/phase/arrival stability, omission/grouping/drift and groove
   judgments against the reference and named controls. Changed acoustic admission
   features require all four stages to be replayed/refitted; retest T1–T7 and M5
   because accents also feed gesture, span and section statistics.

Register every step's variant, metric, fidelity threshold, refit scope and charged
cost in O03/O04/O10 before preflight. A step with no measured work in the failing
slice can be recorded as inapplicable; do not change unrelated model capacity to
pretend to fix that slice. Keep a reduction only after its development fidelity
checks pass; otherwise revert it and try the next. Combine retained reductions in this order until both budget
and functional conditions pass. Record every accepted/rejected step, then freeze
one model/resource version before validation. If no permitted configuration passes,
revise the algorithm or declared hardware on development and repeat these gates;
do not silently extend freshness windows, waive headroom or excuse a validation
miss. This is a resource revision path, not permission to shrink T1–T7's scope.

The initial end-to-end target is p99 live evidence age at action consumption at
most 0.25 s under the registered ordinary workload. Budget 75 ms for acoustic
acquisition/analysis availability, 100 ms for relation/matcher scheduling, 50 ms
for canonical-hop inference, matcher computation and table construction together,
and 25 ms for publication/consumer scheduling. The 100 ms line is waiting/cadence,
not additional matcher CPU time. Use one relation worker per bus. These latency
budgets apply to each instance; the ordinary workload runs both habitat and
presentation instances simultaneously, alongside the instrument. Measure each
bus and their shared CPU/memory contention; running one alone cannot pass the
two-instance ordinary-workload gate.
These are engineering budgets to measure, not current performance claims; measure
the end-to-end distribution directly rather than adding marginal p99 values.
The matcher reference cadence is 0.1 s (§9.3). Every T1–T7 local-action comparison
reports component latency, end-to-end p50/p95/p99, and the fractions of decisions
at full, partial and zero age/coverage influence. A missed budget fails integration
readiness even if a score-only experiment passes. Revise the workload, cadence or
declared budget/ramp together on development, then refreeze before comparison;
do not silently extend stale forecasts to conceal latency.

Enable one consequence/action dimension per causal comparison before combining
them. Record the selected action, its issue-time forecast, actual execution, and
later acoustic support. Outcome learning uses observed trajectories and timing;
unexecuted candidates are not rewarded. The observed result updates contextual
expectations, while authored preferences remain distinct. This is not proof that
a local intervention caused every subsequent change in the whole mixture.

The existing harmonic/metabolic path is preserved. A proposed survival reward
from cognitive likelihood is outside this reference and cannot enter silently.
Scenario changes ecological dispositions and opportunity, not inferred listener
labels. Higher context affects the action landscape; it does not send a global
stop, reset or note sequence.

Voice death clears that Voice's pending action outcomes and private participation
trace; it does not erase the shared heard episode bank. A new Voice starts with
an empty private trace and can access the current shared environment. Inherited
body/disposition parameters are generation inputs, not memories claimed to have
been heard by the newborn Voice. Ending a Scenario censors outstanding forecasts;
it does not add a closure observation or an artificial final memory reinforcement.

## 10. Identification, acceptance and completion contract

### 10.1 Parameters and approximation policy

Separate trained model parameters, authored dispositions and compute limits in
configuration and reports. Fit conditional-score weights and transition/closure
coefficients on development material; use regularization selected within that
material. Preserve the distribution of listener annotations instead of assuming
one constructed score is perceptual truth. Fit memory parameters using both
elapsed-time and intervening-material manipulations. Report parameter uncertainty
and unsuccessful fits. No preferred numeric value is claimed before that study.

Every five-level head uses L2-regularized cumulative-logit proportional-odds
regression: groove, desire to participate, continuation, closure, T2's omission-
expectancy scoring link, retrospective whole-piece completion and their ordinal
controls. For Y in {0,1,2,3,4}, `P(Y<=k|x)=sigmoid(alpha_k-beta·x)`, k=0..3,
with ordered cutpoints alpha_0<alpha_1<alpha_2<alpha_3. Adjacent cumulative
differences give the five category probabilities (outer bounds zero and one).
Keep the documented constant input slots for layout consistency but fix their
beta coefficient to zero; the free cutpoints supply the intercept/location.
Fit weighted mean negative log loss plus `lambda*sum(beta_j^2)/2` over the
nonconstant coefficients, without penalizing cutpoints. Select lambda from the
finite development grid registered before fitting, using the existing grouped
splits; record its values, weighting, solver and constrained-cutpoint convention
in O03 before the first ordinal substage. All category probabilities then use
the already specified path/group aggregation, temperature/backoff and expected-
rating convention. An adjacent-category or other link is a registered model
revision with affected refits/comparisons, not an interchangeable implementation.

Before stage-1 human collection, O11/O13 require a synthetic parameter-recovery
gate at the planned participant/target or fixed-policy-run size. M0 registers the
exact factorial timing, intervening-content and exposure allocation, censoring,
assignment/missingness mechanisms, simulation seeds, fitting/regularization
procedure and nuisance-feature ranges before running it. Use independently varied
time/content/exposure conditions, including repeated exposures reaching saturation;
simulate responses/outcome bins from the same forward model that will be fitted.
Initial nuisance scales are frozen simulation-only engineering values, not human
estimates; rerun with the fitted acoustic/match scales before collecting memory
responses. This avoids needing human memory results to design their own gate.

The initial engineering design envelope uses all combinations of tau_time in
{2,20,200,1200} seconds, kappa in {1,4,16} weighted interference units and
strength_max in {1.5,3,6}; the episode model also crosses no-memory bias in
{-2,0,2}. These are experiment-design stress points, not cognitive estimates.
For each grid point simulate 100 independent datasets at the planned allocation,
fit with the declared procedure, and require for every positive parameter median
absolute log(est/true) <= ln(1.25) and its 90th percentile <= ln(2). For bias the
corresponding absolute-error limits are 0.25 and 0.75. Failed fits count as infinite
error, not excluded draws. Apply the positive-parameter rule separately to the
participation trace's conditional-bin model, using its actual fractional-credit,
normalization and overflow law. Retain all recovery distributions and joint
time/interference/saturation error plots, not just pooled prediction loss.
Recovery must pass each registered nuisance/missingness condition and grid point;
a parameter canceled by normalization or pinned only by a prior cannot pass as
identified. A failed gate revises the task/exposure/sample allocation or an
unidentifiable parameterization, then repeats the registered simulation; it does
not relax these tolerances after seeing failures. Any changed design envelope
is an explicit new study version, not deletion of difficult cells. Simulation
costs are declared before execution, separately from the stage-3 compute envelope.
This gate blocks collection/fit investment under O11/O13; passing only establishes
recoverability under the stated simulator. Human fits still report uncertainty,
model mismatch and failed identification without claiming neural validation.

Before phrase/section boundary and type annotation collection, extend the synthetic
gate to their stage-2 fits through the full mixture-first-event forward map.
These coefficients are predictive parameters of the declared representation, not
separately identified neural rates. Register generating coefficient/feature vectors
covering low/medium/high event rates and balanced/imbalanced exit types, single-
strand and two/four/eight-group textures, the planned interval-censoring allocation,
and 0/10/30% unresolved support. Freeze the exact vectors, support-loss mechanisms
and all planned target counts with O05/O06/O12; missing entries block collection.
Simulate 100 datasets per registered condition, fit with the declared stage-2
procedure and compare predicted unconditional mixture event-time CDFs and event-
type probabilities with their known generating values on an independent synthetic
probe grid. CDF error is RMS over the registered 10 ms time grid in the 8 s/32 s
windows; require its median across fits <=0.05 and 90th percentile <=0.10 in each
condition. For the first-event type distribution, including no event/unresolved,
require total-variation error median <=0.10 and 90th percentile <=0.20. Include
single-strand probes, so a stable mixture alone cannot hide unconstrained per-group
predictions. Failed fits count as infinite error. The grid, tolerances and cost
are fixed before simulation, independently of human outcomes. Failure revises
annotation allocation/censoring, single-strand anchor material, feature/parameter
restrictions or the model before collecting these annotations; repeat the gate
without relaxing thresholds after failure. Record nonunique coefficients even
when predictive stability passes. O12 stores this gate alongside O11/O13 recovery;
it establishes simulator stability, not a human event-segmentation mechanism.

Fit in four frozen stages, using grouped out-of-fold predictions within
development to avoid teacher-forced context at the next stage:

1. Fit acoustic/match feature scaling and correspondence parameters on annotated
   heard-interval tasks; freeze before fitting episode retention, interference,
   strength saturation and no-memory bias to human return-recognition responses
   under the independent time/content/exposure manipulations in §9.3.
   Separately fit participation-trace parameters on observed outcome timing from
   fixed-policy development runs (§9.2). Freeze. Before joint weights exist, use
   admission-support-proportional weights for known proposals and the fixed
   physical-time unknown leak of §9.1; zero total known support yields unknown. These are explicit bootstrap weights, not calibrated
   cognitive probabilities.
2. Using stage-1 out-of-fold path-local features, first fit/freeze the base
   articulation competing-rate parameters on the T1 state-interval transition-time
   and destination loss, with onset-only and envelope-change-only controls. Replay
   out-of-fold articulation features, then fit and freeze arrival/phrase
   hazards on event-time loss (phrase/section use the mixture-first-event mapping
   in §9.4; arrival retains its per-group observed accent targets), then phrase
   exit categories and continuation,
   closure, groove and participation heads on their conditional-type/ordinal losses.
   Freeze these, replay out-of-fold to construct the phrase-dependent 82-component
   section features, then fit/freeze the section hazard before its exit categorical. The first substage does not consume a
   fitted section-head output; section-to-phrase influence enters the joint
   conditional features in stage 3. No reciprocal within-stage fitting is implied.
3. Fit only the joint conditional weights of §9.1 using the frozen stage-1/2
   predictors and out-of-fold outputs; minimize annotated conditional log loss
   over retained compatible paths. This is the declared bounded-model objective,
   not exact likelihood of an unbounded graph. No joint alternation is used.
4. On the separate development calibration split, fit one positive temperature
   per reported categorical/ordinal head by log loss. After path and, where
   applicable, group marginalization, apply `temperature(p_s,T)(k) =
   p_s(k)^(1/T)/sum_j p_s(j)^(1/T)` to the supported distribution only, then mix
   with the unchanged empirical training marginal using §9.4's supported mass.
   Optimize loss of that final mixed distribution on all responses, not a
   supported-only subset. Stages 2/3 use this same order with T=1; §9.6 preview
   cells use the fitted T with their explicitly hypothetical supported mixture.
   Preserve ordinal order and separately report unknown/backoff mass. Neither
   individual paths nor p_0 are temperature-transformed. The frozen calibration
   record includes this order, p_0 counts and the support rule.
   Record prequential log loss, Brier score and ten equal-frequency-bin reliability
   error, with passage/participant-level uncertainty and unknown/abstention rate.
   For event-time heads, fit a positive scalar multiplier on cumulative hazard
   using interval-censored log loss on that same calibration split; derive both
   boundary probabilities and event-time survival from the calibrated hazard,
   using §9.4's first-event aggregation after each local hazard is scaled.
   Arrival receives its own positive cumulative-hazard multiplier, fitted on
   observed-accent event-time targets. Freeze the consumer split: T2's scoring-link
   predictor and the continuation head's arrival input use the unscaled frozen
   stage-2 hazard (multiplier one), preserving their fitted feature distributions.
   The continuation head's phrase-survival input and the phrase-survival-only
   ordinal control also use multiplier-one frozen stage-2 phrase hazards. Record
   consumer IDs `continuation.phrase_survival_input` and
   `control.phrase_survival_only` explicitly with the calibration record; their
   ordinal output calibration remains separate. Feeding scaled phrase survival
   to either is a model revision requiring its stage-2 refit and affected checks;
   reported arrival probabilities and §9.6 timing quantiles use the stage-4 scaled
   hazard. T2's ordinal link has its own temperature, not an implicit substituted
   hazard. Record both versions/consumer IDs and apply the same scaled quantile
   source to both context-on/off candidate sets. Recalibration changes those
   candidate times and requires repeating the registered action comparison.
   This survival does not replace the separately calibrated continuation rating.
   Check event-window reliability and censoring-aware loss as well as ordinal
   calibration. Set acceptance tolerances and confidence thresholds before validation.

The stage-3 joint inventory has a hard initial maximum N=64 free coefficients,
separate from the 256-scalar-term runtime cap. The feature manifest declares N
and rejects a larger inventory before fitting. Register nested 64/32/16-coordinate
variants in advance, retaining at least one free coordinate for each of the five
§5 cross-scale links and every required relation function; only optional joint
interactions may be removed in declared feature-ID order. Each smaller variant is
a model revision, not silent coordinate screening. If the required inventory does
not fit a variant, reject that variant. Earlier-stage heads are outside this N.

M0 records available development compute with the hardware baseline: the initial
stage-3 envelope is at most 168 elapsed hours and 2,048 CPU-core hours, including
all three order fits, the delayed-schedule comparison and final scoring replays.
The manifest also registers a finite stage-3 fit matrix of at most 64 jobs,
including every L2 candidate, fold, N variant, coordinate-order fit and delayed-
schedule comparison; no hyperparameter search or sensitivity refit is uncounted.
The four required order/schedule fits at one N consume up to `4*(20N+2)` replays.
Across full-replay jobs, the conservative bound is `sum_f (20*N_f+2)` full
replays, including each initial baseline and final scoring; reduced-cost jobs
below have their separately charged probe/sweep replay counts. Benchmark a complete
corpus replay at the actual long-form lengths and declared worker allocation;
record wall cost C_w and core cost C_c, including I/O, using the largest observed
cost across fit configurations. Report measured expected mask-aware work as well
as cap-saturated cost. Require the sum of each job's charged replay costs to fit
both compute envelopes before starting. More than 64 required jobs revises
the registered fitting plan before execution; it does not authorize hidden search.
Corpus units may run independently; sequential coordinate dependencies remain serial.
First optimize equivalent replay/cache work, then try the registered smaller N
variants, refitting stages 3/4 and retesting all T rows. A changed earlier feature
map reopens its stage and all downstream fits. If none fits both function and
compute conditions, revise the fitting algorithm or declare a new compute envelope
on development and repeat feasibility checks before validation. An interrupted
run or an affordable but functionally incomplete variant is not a completed fit.

Stage 3 uses a deterministic derivative-free reference fit, so weight-dependent
beam membership is part of the evaluated objective rather than an unspecified
gradient. Initialize joint weights at zero; visit coordinates in the registered
feature order and try plus/minus a common step, initially one standardized
coefficient unit. For every trial vector, replay each complete training prefix
from the same empty state with frozen stage-1/2 parameters, recomputing proposals,
pruning and commitment. Minimize mean annotated log loss plus the development-
selected L2 penalty. Accept a trial only if that full-replay objective improves
by more than 1e-6. Halve the step after a sweep with no acceptance; stop below
1e-3 only after at least three complete sweeps, or after ten complete sweeps.
For N registered joint coordinates, each sweep evaluates both directions from the
same current coordinate baseline, exactly 2N trial vectors, before accepting the
better improvement at that coordinate. The maximum budget is therefore 20N trial
vectors, plus the initial baseline replay; do not truncate a sweep at an unrelated
500-evaluation cap. Report N, complete sweeps, per-coordinate probe counts and any
unprobed coordinates. An interrupted partial sweep is not a completed fit. Compare
registered, reversed and one preregistered seeded-permutation coordinate order on
development at the same sweep budget, then freeze the chosen order and report
order dependence. Retain the best fully evaluated vector and report the stop. This finite-budget fit is not a claim
of a global optimum. Cached audio features may be reused; weight-dependent states
may not. On development compare it with an explicitly delayed schedule that
holds the last replay's paths/features/priors fixed for four trial vectors before
re-inference. Score both final vectors by a fresh complete causal replay and
report objective, held-out loss, evaluation count and wall time; never report the
delayed schedule's surrogate as the actual bounded-model loss. Freeze the chosen
schedule, coordinate order, tolerances and evaluation limit with the model version.

Pre-register `stratified_probe_full_sweep` as the concrete reduced-cost alternative
before feasibility measurement. It replaces full-corpus coordinate probes, not
final acceptance or the target material. Select nested 1/8, 1/4, 1/2 and full
sets of complete development passages using a fixed random order within T-row,
source-family, duration and texture strata. Assign each passage one primary
selection stratum in advance; secondary T labels remain analysis labels, not
duplicate sampling entries. At each fraction use ceil(fraction*n_s) passages in
stratum s, at least one, including a complete 30 min T7 passage; record realized
rather than nominal fractions and require every T row to have scored targets.
Replay each selected passage from its true start with all prior context. Its
selection probability is pi_i=m_s/n_s. Define the surrogate data loss as
`sum_selected_i [sum_targets_in_i loss/pi_i] / N_full_targets`, plus the same L2
penalty; per-T checks use that row's full target count. Freeze target/censoring
eligibility before model comparison. This fixed denominator avoids a changing
sample composition silently changing the objective's normalization.
No short window is initialized as though it had already heard the omitted prefix.
A pilot comparison against the full protocol on the same smaller complete corpus
qualifies this schedule; include those jobs in the fitting matrix and budget.
The registered/reversed/seeded-order jobs and delayed-schedule comparison may all
use this reduced protocol and must do so if their full-protocol costs fail the
pre-start compute envelope. Use the same nested passage sets, sweep/step rules and
fidelity gates across those comparisons. For the delayed variant only the selected-
corpus probes hold their last inferred states for four trials; full baseline,
sweep acceptance and final scoring still re-infer all states. Report order/schedule
dependence as conditional on this reduced protocol; it is not evidence of full-
protocol order independence. Charge the smaller-corpus full qualification fits
and every comparison job, including worst permitted corpus enlargement.

For this protocol only, both directions of each coordinate are scored on the
selected corpus and accepted provisionally within a sweep. At every complete
sweep end, replay the full corpus at the proposed vector before accepting any
change to the last full-accepted vector. Require true full-objective improvement
above 1e-6 and surrogate fidelity: absolute difference between the sub/full
objective *changes from that sweep's baseline* <= epsilon, both overall and for
each represented T row. Set epsilon before fitting to the smaller of 0.001 nats
per annotated target and 10% of the smallest registered Gate-2 meaningful margin;
use the same L2 penalty in both objectives. Register the exact weighting and
report every checked drift, not just the final one. The bound applies at checked
sweep endpoints; it makes no guarantee about unaccepted coordinate probes.

If full loss fails to improve, roll back and halve the step. If fidelity fails,
roll back and enlarge to the next nested corpus, then repeat with a fresh sweep
within the same finite budget. Keep at most ten attempted sweeps (20N probe
vectors) and require at least three complete fidelity-passing sweeps before a fit
can finish. Charge at most 12 full-corpus replays per job (initial baseline, ten
sweep checks and final scoring), plus 20N selected-corpus probes at the largest
permitted fraction. Include the worst permitted enlargement in wall/core estimates;
if the next size cannot fit the remaining declared budget, the fit is incomplete
and the algorithm/compute declaration must be revised before another run.
Full-corpus baseline, every accepted sweep and final scoring remain exact causal
replays with the frozen bounded model. Compare final full and per-T losses, drift,
evaluation counts and wall/core time; no reduced-corpus loss becomes a held-out
validation claim. All four acceptance gates and long-form material stay unchanged.

Stage-2 heads consume path-local feature values, not stage-3 marginal averages:
grouping support comes from acoustic admission support, word surprise from the
stage-1 weighted word counts, and retrieval features from stage-1 correspondence
and availability. At inference evaluate these same fixed feature maps per path,
then marginalize head predictions using the current path weights. The input
definition does not switch to post-joint support as a feature. After stage 3, run
the complete causal loop on held-out development and check per-head loss and
feature ranges against the frozen stage-2 reference; do not assume bootstrap
weights match the deployed state distribution. Failure reopens that component's
training design on development; stage 4 calibrates the final integrated loop.
No unrecorded post-validation refit or automatic alternating optimizer is permitted.

Stage-3 annotation compatibility is target-specific. For a phrase/section boundary,
use §9.4's factorized mixture-first-event mass and associated earliest-exit type
in the declared elicitation window. Expand the annotated interval by one observation
hop for timestamp resolution; a no-boundary target uses the product of local
no-transition masses. Unknown history never counts as known survival. For an episode correspondence, compare the labeled
heard interval pair with the path's pair (at least 0.5 support intersection-over-union
per interval); ambiguous simultaneous groupings retain their labeled alternatives,
not generator IDs. For ordinal targets, use the frozen head's category probability
as the path's compatibility weight through the probe aggregation in §9.4, aligned
to the stated endpoint and the head's 2 s or 8 s feature window; do not create a
hard closure label on every path. Each likelihood
marginal is the sum of path weight times target compatibility. Preserve conflicting
annotator targets separately; a genuinely unjudged target gives all paths weight
one and no learning signal. A known target does not treat the internal unknown path
as automatically correct. Record support-overlap and alignment tolerances and test
them on development; fix the annotation/probe protocol before validation. Prospective
probes use heard prefixes only; retrospective full-piece judgments are labeled as
such and cannot validate a claim about issue-time anticipation by themselves.
If a known target has no compatible retained path, record a representation/capacity
miss and its loss; do not relabel it unknown or discard the trial. Use a 1e-12
likelihood floor only to keep numerical reporting finite. Revise proposals on
development if needed; validation misses remain failures of the frozen model.

Freeze the selected feature scaling, model parameters, calibration and evaluation
criteria before validation. Partition by complete source passage/performer and
participant, not adjacent frames. Transformations of one source stay in the same
split. Familiarization may update online state under the declared rule; it cannot
refit validation criteria or provide hidden section labels. Calibrate confidence
on a separate development split; until then publish model scores and unknowns.

Initial compute budgets are engineering starting points: at most 8 auditory
groups, 8 shared-context paths and 16 local paths per group per shared context
(including reserved unknown paths), 8 acoustic trajectory candidates per frame,
256 retained episodes and 16 retrieval candidates per decision. Unsupported excess
groups remain in the unresolved mixture, with a truncation flag. Allocate the
same local cap to each represented group; do not let a strong group consume
another group's unknown slot. Compare half and double each limit, separately, before
realtime promotion. These numbers do not bound musical duration. Exceeding them
must expose pruning/eviction and widen uncertainty rather than silently assert
novelty or forgetting.
Before freezing a model for long-form validation, M5 performs a required capacity-
sizing study on the registered long-form development material. Measure new-episode
commitments separately from reinforcement/metadata writes, per-bus commitment-rate
profiles, occupancy, retained-edge pressure and eviction age. Project occupancy
over the registered 30 min performance and 2/10 min return delays using the observed
rate profiles, then replay actual complete development performances with the exact
proposed eviction/index policy. Compare with a roomy offline bank retaining all
that finite development input and exhaustive same-cue retrieval. Every designed
development return supported by this roomy reference must remain accessible until
its return cue at the published bank/edge/index limits; identify loss to bank
capacity, edge cap and candidate/index cutoff separately from fitted availability.
If the check fails, increase/revise the declared capacity or index before validation,
repeat long-return and runtime/memory checks, and record the choice and evidence
in technote §9.3.55. A commitment-rate extrapolation alone cannot pass this gate.
M5 completion and M9 validation entry require this sizing record; unforeseen
validation misses still fail the frozen model and are not excused afterward. A longer-duration evaluation may require a larger or
indexed memory; accepted limits belong to the declared operating envelope.

Bound each episode to 128 descriptor knots and 16 relation edges. A provisional
knot is a contiguous block of original raw records. For each of the ten value
coordinates store valid observed duration W_j, its weighted mean mu_j and weighted
within-block squared error S_j, in f64. A raw observed hop has S_j=0; a missing
coordinate has W_j=0. Merging blocks a,b uses W=W_a+W_b, the weighted mean and
`S=S_a+S_b+W_a*W_b/W*(mu_a-mu_b)^2`; if either weight is zero, retain the other
block's statistics, and if both are zero keep the coordinate masked. W is measured
in observed seconds; missing intervals contribute no W. Component coverage is W
divided by the block's physical duration, and coverage below 0.9 masks its match
value. Keep original support endpoints, the observed-duration-weighted mean of
raw endpoint times as representative time (support midpoint if all missing), and
an OR-combined acquisition-gap flag. Adjacent-interval tempo evidence is unsupported
when either block carries a gap; no interpolated unheard timing is inferred.

When appending would produce 129 knots, merge the eligible adjacent pair with the
smallest `sum_j (S_merged,j-S_a,j-S_b,j)/sd_j^2`, using O03's frozen global scales.
Protect the first and newest knot, so at most 126 interior pairs are considered;
break ties by earliest original support start. Use one bounded insertion scratch
record, not a growing array. The accumulated reconstruction error is
`sum_blocks,j S_j/sd_j^2`, in observed seconds times squared standardized error.
It is the integrated error of piecewise-constant reconstruction of the original
known coordinate samples in exact arithmetic, not a maximum pointwise error or
a timing-error bound. Store the f64 reconstruction-error statistic and measure
its rounding discrepancy against uncompressed direct f64 reconstruction in O09;
do not label the floating-point statistic a proven numerical upper bound.
Record compression/coverage and lost gap location
separately. A merged block may span missing time without claiming it observed.
Freeze the descriptor at commitment. Keep the highest-supported edges and flag dropped
alternatives. Active-span duration can keep increasing without growing storage;
detail lost to compression is reported as approximation, not auditory forgetting.
For allocation accounting, cap each descriptor knot at 320 bytes: at most 256
numeric bytes: ten f64 W/mean/error triples (240 bytes), representative time and
its total observed weight (16 bytes), plus 64 bytes of timing/provenance/mask
metadata. No extra per-knot heap arrays are allowed. These ten values
and their derived coverage/timing fields remain within the matcher's 64-field
visit bound. An anchor references its
eight knots rather than copying them: at most 96 bytes for indices and cached
scalars. Each relation edge, including context-membership totals or transformation
residuals, has at most 128 bytes; all edge kinds share the 16-edge cap. Reserve
256 bytes per episode for handles, availability/support totals, masks and other
metadata. Thus one episode uses at most `128*320 + 32*96 + 16*128 + 256 = 46,336`
bytes and the 256-episode bank at most 11,862,016 bytes per bus, including its
anchor index and edges. Record these layout limits in the numerical reference;
implementation asserts actual layouts/capacities and accounts separately for
active provisional paths, DP scratch, tables, queues and allocator overhead in
the total worker report. No hidden per-record heap storage escapes this census.
Freeze the descriptor coordinate inventory within these slots before stage-1
fitting; a larger inventory revises the memory/model budget. O04 includes shared
raw extraction once per group/hop: at most two Log2-bin passes per group for
centroid/spread and centered fractions (16*n_bins visits per bus/hop), reusing
the existing envelope/flux values. Compact only retained span states. The
uncached ordinary-hop bound is 1,024 retained local paths times 126 pairs times
ten coordinates: 1,290,240 merge-priority coordinate evaluations, plus moment/
metadata updates and any parent-block copies. Gap resumption with three insertions
raises that bound to 3,870,720. At most 1,024 full 128*320-byte span copies would
read/write 41,943,040 bytes per insertion round in each direction, or 125,829,120
at three rounds; sharing/caching must be measured, not assumed free. Include this work
and provisional storage in the same two-bus preflight and resource-revision rule.
Validate half/double
knot/edge caps and compare against uncompressed short fixtures, reporting the
corresponding byte totals and both simultaneously running buses.
Evict the episode with the lowest upper availability bound when the bank is full,
breaking ties by oldest committed occurrence; references to it become explicit
unretrieved links. Long-return tests must pass at the published capacity.

Audio callbacks do not perform graph inference or allocate new per-action graphs.
Run relation inference in a bounded worker, publish immutable summaries, and let
the existing hop path read the latest available summary. A late summary cannot
retroactively change a sound. Age/coverage is checked in the generator, with the
neutral-context fallback above. Test whether this asynchronous delay preserves
observable anticipatory effects, rather than promising them by architecture alone.

### 10.2 Required comparisons

Every row needs an offline causal test, a passive perceptual comparison and a
local-action comparison. Some share material, but no row is discharged by another
row's success. Acoustic controls establish constructed differences; independent
listener judgments establish the chosen perceptual targets. Author audition
establishes artistic acceptance for the declared conditions.

| ID | Function and controlled manipulation | Required distinction and competing explanation | Local generative consequence |
|---|---|---|---|
| T1 | Same attacks, different continuation/release/overlap; mixtures with ambiguous components | Gesture grouping follows supported articulation relations; compare onset-only and envelope-change-only models | Context changes continue/release/gap decisions while preserving independent Voice boundaries |
| T2 | Omitted events, complementary offsets, changing grouping, nonmetrical material, phrase across measure | Anticipation survives an omission where listeners expect it; grouping is distinct from phase concentration; compare oscillator and arrival baselines | Voices can share a reference with distinct participation positions; unsupported meter does not force periodic sound |
| T3 | Systematic offsets, matched random jitter and unison, with tempo/density controlled; single-group swing versus matched interval-marginal jitter and a straight reference | Separate expected timing, reported groove and desire to participate; predict listener judgments beyond onset concentration | Authored complementary roles persist and adapt without a mandatory common onset or added pulse carrier |
| T4 | Exact recurrence, transposition/tempo/timbre variation, reordered control and interruption | Correspondence retains order and identified transformations; compare recent-only, orderless and no-retrieval variants | Retrieved relation changes a local response without compulsory replay |
| T5 | Identical local ending after continuing, closing or interrupted contexts; variable phrase length and no-gap boundaries | Distinguish boundary, closure and change: §9.4's phrase_constant, phrase_fixed_duration, phrase_gap_only, phrase_local_change and phrase_rhc_only event-time controls, plus closure_rhc_only and closure_gap_energy_2s ordinal controls | Same local sound/body admits different supported continuation/release choices because of prior context |
| T6 | Same local passage as first occurrence and return after contrast; transformed return; development within a section | Retrieve earlier context and preserve section identity despite local motion; compare time-only and recent-context models | Return/contrast context changes local participation without Scenario labels entering perception |
| T7 | Longer evolving pieces, multiple returns, unresolved endings and overlapping groupings | Whole-context predictions depend on remembered order and return, not total elapsed time, EOF or one slow envelope | Multiple action dimensions coexist, with no forced final unison, consonance or silence |

For each comparison, record input provenance, allowed prior information, target
availability, competing model, expected direction, and a falsifying result before
running it. Use prequential loss and calibration for anticipations, boundary timing
with listener uncertainty, ordinal agreement for closure/groove judgments, and
correspondence accuracy/unknown rate for recurrence. Evaluate false-positive
recognition and failure to distinguish controlled contrasts as well as successes.
Do not pool incompatible measurements into one “DCC score”.

The initial registered breadth includes short/long and pitched/noisy sounds,
sparse/overlapping textures, more than one metrical idiom, nonmetrical passages,
and return delays beyond the current 16 s history cells. T2/T3 also include gradual
tempo change: 10 s at a starting period in 0.5–1.2 s, a 30 s log-linear ramp with
`d(ln BPM)/dt` equal to ±0.0025, ±0.01 or ±0.02 per second, then 10 s at the ending
tempo; ln is the natural logarithm. The extreme endpoint periods are
`0.5*exp(-0.02*30)=0.2744 s` and `1.2*exp(0.02*30)=2.1865 s`, inside the
0.125–4 s reference grid. Add a 30 s smooth reversal with
`d(ln BPM)/dt = 0.02*sin(2*pi*t/(30 s))/s`, using
the same 10 s pre/post holds, and compare constant tempo at matched mean rate and a
stepped change with the same endpoints. Compared variants hear the same realized
audio/tempo-density trajectory; the prescribed ramp is stimulus/scoring metadata,
never an observation label or a new pulse carrier.

Under drift compare full arrival/recurrence anticipation with
`oscillator_beat_only`, `meter_existing_234` and `grouping_inventory_234` under
§9.4's T2 omission/grouping maps. Retain these controls for T3 timing/groove:
the inventory variant refits the affected groove/desire heads using their same
109-coordinate layout and its restricted proposals. For the network variants,
fit separate groove/desire cumulative-logit heads to the causal issue-time
network summaries. The beat-only values are log2(beat.freq_hz/1 Hz) and beat
confidence; the existing-meter variant appends subdivision confidence, measure
confidence, subdivision_ratio/4 and measure_ratio/4, with undetected ratios and
their associated confidences known zero. Invalid frequency or unavailable audio
support masks the affected values. Apply the same frozen development scaling,
one missing indicator per value, intercept, fit and ordinal calibration rules:
five total inputs for beat-only and thirteen for existing-meter. These controls
use the same judged prefix, targets and unknown/backoff rules, not Scenario tempo
labels. O03/O07 register these layouts and the required loss comparisons before
collection. The other T3 onset-concentration/complexity and timing-order controls
remain required.
Measure tracking lag, relative-position persistence, unknown/admission rate and
cap loss under the declared 10% repetition tolerance, 32 s bank and 1 s reference
dwell, including their registered sensitivity variants. Test private-trace transfer
with gradually changing bodily/participation periods as well as stepped changes,
retaining issue-time period anchors and the existing credit rule. A good fixed-
period fit is not a pass for drift. Register the natural-log base, realized rates,
reversal function and paired density controls before collection; an expected offset drifting with
a shared reference is distinct from all voices collapsing onto one onset.
A long-form gate must
include a complete evolving performance, not only concatenated short scoring
windows. Validation material and its target duration are fixed in the delivery
plan before data collection; capability claims name the tested envelope.

Before any human pilot or confirmatory collection, M0/O05 records the study's
ethics approval reference or the responsible institution's documented determination
of the applicable review route, consent text in each study language, voluntary
withdrawal procedure, compensation, and response-data handling protocol. The latter
specifies pseudonymous participant IDs, separately access-controlled contact/payment
records, access roles, retention/deletion periods and the de-identification rules
for sharing individual ratings and experience/language strata. Preserve the
within-participant linkage needed by the registered analysis without publishing
identifiable combinations. Missing approvals/determinations or consent/data-handling
entries block all human collection, including pilots; this is an execution
prerequisite, not an assumption that every jurisdiction uses the same review route.

Before any instrument pilot, O06 registers its procedural-comprehension and
usable-response criteria. Use at least 12 pilot listeners per study language,
independent of confirmatory listeners. Each instrument has two procedural practice
checks about the requested time window/response operation (not a prescribed
musical judgment); at least 90% of assigned pilot listeners must pass both.
The following initial usable-response minima apply on advance-designated clear
practice/control trials, with all assigned trials in the denominator. Ambiguous
study material remains separately reported and is not removed to pass these gates.

| Instrument | Minimum usable response on its designated clear controls |
|---|---|
| Heard-span correspondence and return recognition | 70% valid alignment/category or return/no-return judgments, respectively |
| T1 gesture/state/overlap | 70% valid gesture-span marks, 70% judged state portions on selected spans, and 70% yes/no overlap answers; each endpoint assessed separately |
| T2 omission and other five-level prefix ratings (closure, continuation, groove, desire) | 80% valid anchored ratings for each head; an uncertain middle rating remains valid |
| T2 grouping/tactus | 70% judged kind responses and, among clear equal-group trials, 70% consistent usable tactus/count targets under §9.4 |
| Phrase/section boundaries and exit types | 70% valid interval/no-boundary answers for each scale; among judged-boundary trials, 60% judged types |
| T5 cross-group phrase handoff | 70% judged yes/no answers (unable is unjudged); among all yes answers, 70% valid continuing-interval marks; both assessed on designated clear controls |
| T7 retrospective whole-piece completion | 80% valid anchored ratings after complete hearing |
| Gate-3 relation descriptions | 70% selection of one of the three descriptions rather than cannot-tell on clear controls; this does not require choosing the experimenter's preferred description |

A valid mark lies within the heard support with ordered endpoints and respects
the declared censoring rules; a valid categorical/ordinal answer uses a listed
option. Minimum percentages assess whether the instrument can supply its targets,
not agreement with the model or an intended aesthetic. Record every field's
missing/invalid/unjudged rate, including denominators and language/experience
strata; do not exclude failed-comprehension listeners to improve these pilot rates.
Here `t2_count_pilot_readiness_min=0.70` is the instrument-readiness gate on
advance-designated clear equal-group pilot trials, with all assigned such trials
in its denominator. O06 may register a stricter readiness value before the pilot.
It is distinct from §9.4's `t2_count_validation_coverage_min`, which gates usable
count coverage in the predeclared validation count-assay stratum. O06/O07 register
that stratum, its all-assigned-trial denominator and the derivation procedure
before the pilot; only after the pilot passes readiness can its representative
task responses supply the numerical validation constant. Freeze that value,
language/experience handling and required response count before validation; an
unidentified value blocks validation, and a failed validation cannot lower it.
Record both IDs, origins, denominators and freezes separately. Ambiguous or
unjudged answers stay in the coverage denominator, not silently excluded. Gate-3 validation still
includes cannot-tell in every choice denominator, regardless of this readiness check.

If any required endpoint or comprehension rate fails, revise instructions, practice
or the instrument/material's clarity, version the change and re-pilot with a new
development cohort under preregistered criteria before freezing that instrument.
Do not coach musical answers, drop the temporal function or lower a threshold
after seeing the same pilot to declare it adequate. O07 cannot set a function's
Gate-2 meaningful margin from an inadequate instrument; collect an adequate pilot
first. Even an adequate pilot may leave a margin unidentified, which remains a
blocking uncertainty rather than permission to select a favorable validation score.

Gate-2 registration belongs to M0 of
`docs/roadmap/temporal-dcc-completion.md`: each T row fixes its primary cognitive
metric, one named control, a minimal meaningful margin, clustering units and
sample-size simulation before confirmatory data collection. The reference primary
metric is held-out mean log-loss improvement (interval-censored event-time loss,
categorical correspondence loss or ordinal-response loss as appropriate). Require
an improvement at least the registered margin and a passage/participant-clustered
95% interval excluding zero; report unknown/backoff responses in the score.
Set the numeric margin from the development pilot's response distributions and a
stated task-relevant distinction, then simulate a fixed sample size with at least
80% power for that margin. Freeze methodology before the pilot and numbers before
validation. No validation-driven margin or recruitment stopping is permitted.
Other T-row descriptive metrics do not silently replace this primary criterion.

M0 also owns O03/O07's `docs/roadmap/temporal-dcc/control-comparisons.json`
registration for **every required non-primary control comparison**, before its
corresponding collection or component freeze. Each entry identifies the T row and
endpoint, full/control IDs and refitted versions, fixed evaluation stratum and
sample/denominator, exact loss/statistic, clustering and interval method, threshold,
and failure owner. Any requirement to compare, beat or fail against a control must
have an entry with a declared inferential role; it cannot become optional at a
component freeze.

For required superiority comparisons, the reference rule is delta = held-out mean
endpoint loss(control) minus loss(full), retaining all declared unknown/backoff
responses. Use the endpoint's own event-time, categorical or ordinal loss, not an
easier metric from the same T row. Pass only when the lower endpoint of a
passage/participant-clustered 95% confidence interval is greater than zero. The
primary meaningful margin is not automatically imposed on other endpoints; any
additional positive margin must be registered separately. Register interval
construction, clustering and sample-size/power treatment with the primary protocol:
method before pilot, data-dependent constants before validation. All required
comparisons must pass together; selecting whichever control was beaten cannot
complete the endpoint.

This includes T5's additional phrase-boundary and closure controls, the three
continuation controls, both T2 grouping controls in §9.4, section elapsed/recent
controls, memory variants for T4/T6/T7, and other named superiority requirements.
M0 enumerates and checks the complete list before the affected freezes. Failure
reopens the named M2–M7 representation, head or control fit and blocks its freeze
and T-row completion; recording a limitation alone does not pass. Explicit
numerical-equivalence, sensitivity and descriptive diagnostics retain their own
registered rules and roles. No required superiority test may be reclassified as
descriptive or have its threshold relaxed after its results are known.

### 10.3 Four independent gates

1. **Causal implementation:** no future access or label leakage; observation,
   missingness, intent and outcome separated; equivalent delivery partitions agree
   at common observation times; bounded resources and known numerical error.
2. **Cognitive task correspondence:** on held-out material and listener judgments,
   the specified relation model beats its relevant simpler controls, including
   ablations, by the preregistered practically meaningful criterion. Uncertain
   targets stay uncertain. A numerical assay alone cannot pass this gate.
3. **Audible relational effect:** context present/removed, with identical prior
   sound and body, causes the predicted change; removing the corresponding
   cross-scale connection removes it. Listening identifies the intended relation,
   not merely that two recordings differ.
4. **Artistic acceptance:** the author accepts the flow and intended relation in
   the declared conditions, with regression against previously accepted samples.
   Preference is not inferred from a diagnostic being measurable.

Gate 3 uses a registered condition-blind relation-identification task. Construct
matched context-present/removed pairs with identical model prehistory and body;
randomly assign each listener only one member of a given pair to avoid hearing
one branch before judging the other. After the stated prefix/continuation, choose
among three balanced plain-language relation descriptions plus “cannot tell”.
Descriptions identify the T-row's relation (for example transformed return versus
literal repetition versus new material), not the model name or whether the sound
is preferable. M0 registers the exact three descriptions, their translations and
the intended directional contrast for every T row before any study listening;
randomize their display order. The following canonical examples make the required
instruments concrete. In each, the first description is the intended effect of
that assay; different authored contrasts require their own advance registration,
not relabeling after answers are seen.

| Row | Intended description | First foil | Second foil |
|---|---|---|---|
| T1 | The parts pass a gesture through overlapping releases | The parts stop and start together | Long sounds overlap without a shared gesture |
| T2 | The parts keep distinct positions around an anticipated recurrence | The parts converge on the same repeated onset | Timing changes without a recurring reference |
| T3 | Parts keep different timing positions and fit their entries together | Parts align their entries on the same timing position | Parts shift their entries without a persistent timing relation |
| T4 | Earlier material returns with a changed realization | Earlier material repeats without a heard transformation | Material arrives without a heard return |
| T5 | The local ending resolves an earlier expectation | The local ending interrupts an ongoing expectation | The local ending continues the ongoing expectation |
| T6 | A returning section brings back an earlier way of participating | A new section brings a different way of participating | The parts continue their current way through the change |
| T7 | Earlier returns alter later participation across the piece | The same local response recurs regardless of the earlier course | Participation changes without a heard relation to earlier material |

For T3, hold tempo and density comparable while asking whether distinct positions
fit together, converge, or vary without a stable relation. This is separate from
the groove/desire rating task. Pilot comprehension and balance wording without
mentioning a model, condition name or preferred aesthetic; freeze wording changes
before validation. Every row also offers the same explicit cannot-tell response.

Use independent validation listeners who did not fit or calibrate the model; record
musical experience and represented idioms. Start with at least 48 listeners and
increase the preregistered number if development-based clustered simulation needs
more to detect a 0.15 difference in intended-description choice at 80% power.
Balance conditions by passage and participant; never stop recruitment based on a
favorable interim result. Use all assigned responses, including “cannot tell”,
in every choice-rate denominator. For context-on trials define the foil null as
`p_null = (p_foil_1+p_foil_2)/2` on that same denominator; require
`p_intended-p_null > 0` with a passage/participant-clustered 95% interval excluding
zero. Here p_intended is the observed intended-description choice rate; p_null is
the mean foil rate, not an assignment replacing p_intended. Also require `p_intended_on-p_intended_off >= 0.15`, with
its own passage/participant-clustered 95% interval excluding zero, again on all
responses in each arm. There is no 1/3 or 1/4 uniform-choice assumption. Report
abstention separately, never remove it or treat it as a wrong-description vote.
The sample-size simulation uses these two joint success conditions and the
registered abstention behavior. These are declared engineering
acceptance criteria, not universal perceptual constants. A reversed, negligible
or uncertain contrast fails this gate. Author preference is still gate 4. Long-form
trials supply the full preceding context; prefix-only anticipation and retrospective
whole-piece judgments remain separate endpoints.

### 10.4 Delivery obligation index

The delivery plan must contain each stable obligation ID below, linking its memo
section to the owning milestone, planned artifact and blocking condition. Creating
the plan completes this index mapping, not the future artifacts or gates. M0 owns
registration; later milestones own implementation and evidence. A missing ID blocks
plan completion. Changes to this table require the plan's matching row to change.
The milestone table in the [delivery plan](../roadmap/temporal-dcc-completion.md)
is the authoritative definition of M0–M9. Each M-ID referenced here must resolve
to exactly one milestone there. Renumbering or changing a milestone's scope in
either document requires a matching revision in the other, including its owners,
blocking and reopening assignments; plan completion checks this bidirectionally.
The plan's pending completion status does not license silent milestone changes.

| ID | Memo source and required obligation | Owner | Blocks until recorded/passed |
|---|---|---|---|
| O01 | §7–8: hash the actual worktree, settings, existing assays and accepted audition conditions | M0 | Comparable implementation baseline |
| O02 | §3: all present-tense temporal capabilities throughout the Manifesto, including the landscape introduction, 時間軸 mechanisms and 個体/創発 entrainment; sweep other agent-behavior claims, four numbered clauses and source ledger | M0 | Public-document reconciliation |
| O03 | §9.1: feature/proposal manifest, formulas, units, masks, scaling, inventories, freezes and sensitivity metrics/decision rules | M0; each fitting stage | Any fit with missing or changed inputs |
| O04 | §9.1, §9.6: hardware/channel baseline and complete tuple/grouping/timing-history/worker/Voice-decision numerical preflight with CPU headroom | M0 | Stage-1 human collection and fitting investment |
| O05 | §10.2–3: controlled material, idioms/textures, 20–90 s/3–5 min/30 min lengths, 2/10 min returns, source/participant splits, ethics/consent/compensation and response-data protocol | M0 | All human collection including pilots; approval/determination and protocol must precede collection |
| O06 | §9.2–5, §10.1–2: instruments, translations, uncertainty/mappings, pilot adequacy, stimulus channel/analysis mapping, event-head synthetic gate and diagnostic-only secondary grouping | M0 | Corresponding collection; inadequate pilots require revision/re-pilot before freeze |
| O07 | §10.2: per-T primary metrics/controls and every required non-primary control statistic, interval, threshold and failure owner; practical margins from adequate instruments, clustering and power simulation | M0 | Affected component freezes and Gate-2 study; method before pilot, numbers before validation, no margin from an inadequate pilot |
| O08 | §10.3: relation descriptions/foils, cannot-tell handling, joint success rule and fixed recruitment | M0, M7–M9 | Gate-3 validation |
| O09 | §9.1–6: numerical reference and fixtures for support/gaps, analysis-configuration epoch restart, label/route/private-state isolation, normalization, rates, quadrature, pulse and candidate evaluation | M0–M2 | Feature/forecast interpretation or fitting; evidence/epoch isolation must pass |
| O10 | §9.1–5, §10.1: exhaustive fixtures, cross-strand phrase misses and cap/window/threshold/clock/handoff/retention sensitivities with advance decision rules | M2–M5 | Model freeze for each affected T row |
| O11 | §9.3, §10.1: pre-collection synthetic recovery, heard correspondence and independent time/content/exposure memory identification | M0, M3 | Stage-1 collection; episode parameters and long-context claims |
| O12 | §10.1: pre-collection event-head predictive stability, finite fit/compute, full/reduced protocols, OOF, calibration and consumer split | M0, M2–M7 | Boundary/type collection; integrated freeze and calibrated action columns |
| O13 | §9.2, §10.1: pre-collection synthetic trace recovery, fixed-policy executed-outcome fit and credit/coverage fixtures | M0, M3, M6 | Stage-1 collection/trace fitting; private action use and template promotion |
| O14 | §9.1, §9.6: bus transfer, arbitrary live bodies' own-audio descriptors, prototype/actual-Voice projection and rendered counterfactual fidelity | M6, M8 | Ordinal generation columns |
| O15 | §9.6, §10.3: distinct identical-input on/off and changed-prefix state-intervention experiments, fixed candidate pools for pressure ablations | M6–M7 | Causal relational-effect claim |
| O16 | §10.1: complete long-form M5 capacity sizing, roomy-bank comparison, retention/index/edge revisions | M5 | M5 completion and M9 validation entry |
| O17 | §9.6: complete two-bus 4/16/64-Voice resource, freshness, headroom, audio, fallback and recovery report, including novel/evolving/mutated bodies and private characterization | M8 | Runtime promotion |
| O18 | §9.4, §9.6: T2 replay through actual publication/consumption with measured timing | M8 | Live anticipation/action acceptance |
| O19 | §9.2, §10.3: T2/T3 four-gate decision for replacing the 2x3 template only in local participation; off/passive retain the unchanged legacy behavior | M7–M9 | Local-participation template replacement; no default or O21 baseline change |
| O20 | §5, §10.2–3: all T1–T7 four gates, five cross-scale ablations, full-piece/return and accepted-sample regressions | M9 | Temporal-DCC implementation completion |
| O21 | §8–9.6: narrow API with shipped default off, versioned legacy baseline retained after O19, separately authored/versioned default promotion, mode-isolation fixtures, report-independent enable, air-gap and document alignment | M8–M9 | Ordinary-use/documentation completion; expected-mode/baseline and mode-isolation fixtures must pass |
| O22 | §7, §10.3: exact review/version record, declared validity envelope, limits and canonical technote status | Plan completion; M9 implementation update | Honest distinction of design, implemented and empirically accepted states |

A design specification is complete when each T1–T7 function has a representation,
update and parameter-identification rule, cross-scale connection, local consumer,
and falsifying comparison, and review finds no remaining actionable specification
defect. It is not a completed instrument or an experimentally validated brain
model. Implementation completion additionally requires all four gates for each
function, the integration and resource envelope, and an honest record of limits.
A failed cognitive or artistic gate triggers revision and retest of that function;
it is not waived because the lower-level predictor, code tests or reviewer passed.
