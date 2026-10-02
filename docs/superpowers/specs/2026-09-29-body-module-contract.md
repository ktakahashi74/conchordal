# Body module contract

Date: 2026-09-29. **Revised contract with partial author policy adoption;
initial independent review received, revision not independently re-reviewed,
full A4 adoption and Phase 3 implementation pending.**
This is Task 2 of the [timbre implementation
plan](../plans/2026-09-29-timbre-synthesis.md). It does not authorize Phase 3.
The [author decision record](2026-09-29-body-policy-author-decisions.md) adopts
A1, A2's Off coefficient policy with 0.5 s T60 as a prototype value, and A3's
cost-payer policy. Other numerical proposals below remain unadopted and are not
amendments to an existing acquisition registration.

The [Claude review](2026-09-29-body-module-contract-claude-review.md) found the
draft conditionally usable for author policy decisions, but not ready for A4
adoption. The [response and decision table](2026-09-29-body-module-contract-review-response.md)
disposes of all fifteen findings. The later author decision adopts only the
listed policies; unresolved details remain recommendations.

## Purpose and authority

A body receives excitation, maintains state, and radiates audio. The core hears
and evaluates that audio and decides the next excitation. Adding a body must not
require a new body-kind branch in ecology code. A self-model is optional and is
admitted separately for each consumer and supported condition only after comparison
with the same body's render.

The adopted boundaries are [Timbre: core and body
modules](../../design-notes/timbre.md#core-and-body-modules), plan decisions D6–D10,
and [listener-model layers, ledger
§9.3.56](../../design-notes/technote-ledger.md#9356-layers-of-the-listener-model-what-is-fixed-and-who-sets-the-rest).
The composer sets listener valuation/priors, founder bodies/placements, and macro
form. This contract adds no composer-facing parameter. The founder vocabulary
remains `modes`, `brightness`, `spread`, `unison`, and `motion`. Mutation and
capture interfaces do not enable heredity: Phase 5's observability and causal-path
gates still apply, separately for each pathway.

**Representative render is authoritative.** The [body-aware current
plan](../../design-notes/body-aware-fitness-plan.md) retains the F2 comparison
limits: maximum score error 0.025, maximum level error 0.0125, and no strict rank
reversal for pairs whose reference score gap is at least 0.1. [Direct model
v1](../../design-notes/body-fitness-direct-model-results-20260929.md) failed
157/728 candidates, with maximum score error 0.447470 and three reversals;
its resource screen passed 0/16. [Partial-group
v2](../../design-notes/body-fitness-partial-groups-results-20260929.md) failed
159/728, with maximum score error 0.397169, maximum level error 0.126823, and
three reversals; its resource screen passed 10/20. None of those failures is
resolved by writing this contract or redefining the reference.

The [2026-09-30 feasibility assessment](../../../target/body-fitness-feasibility-20260930/decision-v1.md)
does not recommend extending the current interval route toward production.
Conditional kernel/history reuse leaves the candidate-by-72-frame work and its
large band/bound traversal intact; no timing or speedup was established. The
shared self-model output remains render-derived processed f32 density plus
body/observation identity, with the existing current-field integration. An
inexpensive producer preserving phase, time, support and frontend response has
not been established. No cache/packing adapter, reference change, F2 admission
or render-only Phase 3 entry follows from this assessment.

## What the current implementation actually provides

These observations concern the shared main checkout inspected on 2026-09-29,
not a claim that isolated body-aware implementations have landed there.

| Location | Observed behavior and consequence |
|---|---|
| `src/life/sound_body.rs` | `SoundBody` and its factory registry describe the Voice-side body. The `articulate_wave` API is not the scheduled renderer boundary. A registry extension alone cannot supply a new rendering algorithm. |
| `src/life/sound/events.rs`, `sound/any_backend.rs` | `BodySnapshot` closes over three kinds; `AnyBackend` closes over oscillator/resonator. `DriveMode::None` makes Sine ignore drive; the Tone wrapper adds `sine_impulse_boost`. |
| `src/life/sound/tone.rs` | A Tone owns backend state, envelope, smoothing and randomness. `note_off` changes the output envelope. `render_tick` shapes drive and also multiplies output by the same articulation/envelope path. The new separation changes this behavior. |
| `src/life/schedule_renderer.rs` | `(source_id, tone_id)` identifies a sounding Tone. Mixing follows deterministic key order. `fork_source` clones retained Tones for offline forward rendering. A new On allocates a Tone; previous Tones may still sound. |
| `src/life/action_candidates/footprint.rs` | This is a **temporal** representative footprint: 16 energy windows, peak normalization, and a four-second cap. `compute` constructs a Tone but obtains energies through `ToneEnergy` and coherent `project_window`, not by rendering PCM. It is distinct from F2's 72-frame **spectral** footprint. |
| `src/life/action_candidates/energy.rs`, `self_prediction/` | Forecasts compose sine/bank/control descriptions, retained Tones, routes and intervention windows. Unknown, known zero and unsupported control already have distinct meanings. |
| `src/temporal_cognition/body.rs` | The actual path is `src/temporal_cognition/body.rs`, not under `src/life/`. Acoustic descriptors already come from private routed PCM and NSGT. Recipe identity and prediction inputs still inspect body details. |
| `src/life/voice.rs`, `modal.rs`, `pitch_core.rs` | Spectral projection reads body-specific internals; approximate LOO uses supplied ratios but generic index weights. The current `ExactScan` removes one analysis bin, not the source's complete PCM and tails. Neither name establishes whole-source exclusion. |

The current temporal footprint and F2 spectral footprint must not share a cache
entry merely because both are called a footprint. Their horizons, front ends,
normalizations and uses differ.

## Fifteen decisions and their tests

“Chosen” in this table means the draft's recommendation. Initial review, revision
and pending author approval are recorded separately at the end.

| Task 2 question | Chosen behavior | Required conformance test |
|---|---|---|
| 1. On / Off / Update | On opens an excitation handle and injects its declared kick; Off closes future drive at its effective sample; Update changes only declared targets. Free state, output silence, and disposal are separate. | Sample-exact command table below, including duplicate On, future Off, same-tick order, and an Off with a nonzero tail. |
| 2. Decaying / self-sustaining bodies | Both have zero input after Off and a finite free response. Sine/Harmonic become driven oscillators with a decaying amplitude state; one kick cannot sustain them indefinitely. Modal keeps its own free-response law. | Impulse-only, finite drive, driven steady state, Off, and eventual disposal for every body, including zero excitation. |
| 3. Excitation units | Nonnegative dimensionless drive level specified at sample times plus a dimensionless instantaneous kick amplitude; time in sample ticks, pitch in Hz. Transfer functions and supported sample-rate domains remain body-specific. These inputs are not measured acoustic energy. | Changing drive changes radiation; every declared rate and cross-rate invariant passes its own drive/kick reference. Undeclared rates are unsupported. No Sine gain-boost exception. |
| 4. Re-excitation | Preserve existing state. Separate Tone states may implement superposition only for a declared linear, fixed-coefficient response under identical output processing. No generic state merging. | One linear state with two kicks versus two summed response states; a nonlinear/damping negative control must not qualify for merging. |
| 5. Sounding pitch/coefficient change | Pitch Update transfers state of addressed open handles only, through the core pitch ramp. Closed tails retain their last coefficients. Structural genotype changes apply to future On. See A1 and the lowering rules below. | Active-handle/tail selection, pitch-ramp continuity, unsupported retuning, new-On versus re-kick lowering, simultaneous old/new tails and complete source subtraction. |
| 6. Fluctuation | The core owns a source-generation modulation clock and distinct Tone noise substreams; the module owns their effect. The logical clock continues through silence at bounded cost. Off ends fresh noise/modulation; the proposed free-tail policy freezes its last effective frequency offset and spectral balance. | Split-block equality, bounded silent-gap advance, independent noise, overlapping On, coefficient continuity at Off, tail evolution and checkpoint/replay equality. |
| 7. ADSR | ADSR shapes excitation only. Core output gain and a separately declared click guard remain outside the body. Ordinary release must not multiply a tail by the old ADSR output release. | Two release settings alter drive termination but do not erase an already free response; constant post-gain scales PCM and measured cost consistently. |
| 8. Lifetime/resources | The renderer owns sounding state after Voice death. Fixed capacity, deterministic admission and finite tail bounds; no allocation in admitted block rendering or pool-backed Tone creation. | Repeated On/Off and Voice turnover at capacity, longest tails, full queues, rejected On, invalid module output, exact disposal identity, allocator instrumentation. |
| 9. Regression scope | Preserve routing, determinism, identity, unknown handling, scan invariants and air-gap behavior. Re-register expected waveform changes from excitation/decay, not old audio goldens. | Existing invariant suites below plus Phase 2/F2–F4 re-acceptance on the new renderer. |
| 10. Three faces | Module-native immutable genotype, bounded renderer state, optional purpose-specific self-model; the core never consumes native lanes or coefficients. | A registry body with no self-model renders and receives offline core-derived evaluation without ecology changes; genotype capture/restore is exact. |
| 11. Core guarantees | Deterministic input clock and random supply; checked pitch/drive; caller-owned buffers; bounded admission; routes, gain and source PCM capture outside the module. | Identical scheduled input through live and representative entry points, all bus combinations, unsupported capability without fabricated evidence. |
| 12. Dispatch | Built-ins use static variants inside one module-dispatch boundary; research bodies use a registry factory and an object-safe block renderer. No virtual call per sample. | Register a fourth body without adding consumer branches; verify allocation and dispatch counts at Tone creation and block rendering. |
| 13. Reference frequency | Commanded Hz is a transposition anchor. It need not be the lowest partial, strongest partial, or perceived pitch; modules disclose the mapping in their schema. Unpitched semantics remain unscheduled. | Known inharmonic ratio set, below/above-band anchors, Nyquist behavior, pitch change, and no inferred fundamental substituted by the core. |
| 14. Fallback/conformance | Use core render-derived evidence per consumer; fast-path support is conditional on purpose, state, horizon and version. The table below supplies the reference and tolerance. | Disable each self-model independently and compare admitted fast output with its own reference; stale/unknown/silent are distinct. |
| 15. Body-aware coordination | Preserve an equivalent representative-render request and provenance, with one shared F2/self-model representation. No closed-kind migration in the current F2 diagnostic. | Old-version fixture reproduction before migration; new-version reference generation and re-acceptance after migration; no mixed-version cache hit. |

## Excitation, state and output

### Command table

Commands use absolute integer sample ticks. The scheduler imposes a stable order
for equal **effective action ticks**: close old excitation, apply target changes,
then accept new On. An Off request starts its release ramp at the request tick;
closure is scheduled at the ramp's end. An immediate Off uses the same tick for
both. Thus a pitch Update at the closure tick cannot retune that closed tail,
whereas an Update during its release ramp can still address its open handle.
An already closed handle's Off is idempotent. A repeated On with the same
source-generation/tone identity is a duplicate, not another kick. Late command
handling remains a scheduler concern and is recorded; a module cannot backdate it.

| Command | Input | Persistent body state | Radiated output and lifetime |
|---|---|---|---|
| On for a fresh tone identity | Begin its bounded excitation program; apply kick once at the onset tick. | Start from the declared initial state and seed. Existing Tone states remain owned by the same source. | Radiation follows the module transfer function. A silent kick/drive is allowed and is known silence only after successful rendering. |
| Further excitation of an open retained handle | Add the declared kick and/or update that handle's drive; use the explicit re-kick operation, not duplicate On. | Preserve previous phase, amplitude, resonator memory and body-local drive response. | Never reset the state just to realize a new attack. Closed handles cannot be reopened implicitly. |
| Off | At the effective Off tick, set this handle's input to zero and cancel future kicks/updates for it. A release-shaped input ramp, if requested, precedes this tick. | Keep the entire free state. Do not reset it or draw new stochastic excitation for it. | Continue the free response until its declared termination condition. Off does not mean silent or disposed. |
| Update: amplitude/continuous drive | Change the excitation target through its declared ramp. Core output gain has a separately identified control, never an alias for drive. | Keep state; no implicit kick. | Reflect the body's response to the new drive. An amplitude-zero update does not delete stored energy. |
| Update: reference pitch | Address only open excitation handles of this source generation, preserving oscillator phase or equivalent resonator state coordinates through the pitch ramp. | Retune those Tones in place. Closed/Off tails are excluded, even when their source ID matches. | Unsupported live retuning is rejected before any addressed state changes; no reset, crossfade or silent deferral. A future On may use a separately accepted pitch. |
| Structural body replacement | Install a new immutable genotype generation for subsequent On. | Old Tones retain their original genotype, coefficients and ownership until disposal. | New and old generation radiation may overlap. This is not monophony and is included in source cost and exclusion. |

An input release duration describes how long drive takes to fall to zero. The
excitation-stop timestamp is the end of that ramp; reports also retain the request
timestamp. This avoids calling an interval with nonzero drive “free response.”
Scenario duration and existing release events must be lowered to these two
timestamps explicitly during Phase 3.

The revised A1 recommendation lowers an ordinary new phonation after closure to
fresh On at the current target pitch; its preceding Off tail remains at its last
effective pitch. A pitch change within an open phonation lowers to Update of all
its open handles. An explicit repeated excitation within that open phonation
lowers to re-kick of the named handle, with no new Tone slot. A closed handle is
never reused to conceal capacity pressure. Source-level Update validates every
addressed handle's capability first; any unsupported handle rejects that Update
without partial application, and the accepted source target stays unchanged.
The rejection is reported to the controller. A later new-phonation pitch request
is independently checked for fresh-On support; it is not a queued old Update.
This policy bounds state growth from Updates, not from ordinary new phonations,
and permits old/new pitches to overlap across phonations. A1 must accept that
consequence or explicitly choose a source-wide retuning/crossfade amendment.

### Chosen initial body family

For Sine/Harmonic, introduce a real amplitude state with a finite free decay. The
proposed first implementation uses a 0.5-second amplitude T60 at zero drive,
independent of frequency. A kick adds amplitude state; continuous drive replenishes
it using `a[n+1] = r*a[n] + (1-r)*drive[n] + kick[n]`, with
`r = 10^(-3/(sample_rate*T60))`, before radiating that sample. The admitted drive
and kick ranges are finite and recorded in the body capability domain; this law
does not license arbitrary overflowing inputs. Thus the steady response depends
on drive level rather than the number of samples per second. The author has
adopted a Sine/Harmonic amplitude T60 of 0.5 s as a prototype value and the
policy of freezing motion offset and spectral balance at effective Off. The T60
remains an initial non-heritable body setting, not material physics or final
listening acceptance. The recurrence, drive/kick domain, module rates and noise
transfer, random generator, resource capacities and full A4 adoption remain
unadopted unless separately recorded. Harmonic's existing excitation-dependent
spectral damping remains a body response, with its existing floor applying to
relative spectral balance, not an indefinitely audible carrier amplitude.
The proposed 0.5 s T60 implies an amplitude time constant of about 72 ms;
a sharp onset therefore depends on the kick, not on continuous drive alone.

Modal retains its declared mode-specific free decays and noisy drive transfer.
The body generates its excitation waveform from the common amplitude input and
core-supplied random values. Sine's deterministic carrier, Harmonic's drive law
and Modal's noisy excitation need not produce equal RMS for equal input. There is
no automatic equal-loudness normalization in the runtime. Controlled-level
comparisons calibrate emitted PCM as a separate experimental condition.

Each module declares supported sample rates and its rate-specific normalization
of continuous drive and discrete kicks. The oscillator recurrence above gives
one such law, not a proof for Modal. Modal must separately state whether noise
input preserves a physical-time power convention and how a kick's impulse/state
increment scales with rate; tests compare those declared quantities and decay in
seconds. Equal random sample streams or equal instantaneous input numbers alone
do not establish cross-rate energy invariance. No declared/tested domain means
unsupported at that rate. A2 leaves the concrete Modal normalization and admitted
rate set for a written module specification before implementation/acquisition.

The old `energy` name on an onset kick must not imply joules or measured radiated
energy. Proposed A3 accounting uses the source's post-route habitat PCM, summing
its retained Tones before measuring cost; presentation-only radiation is excluded.
While the Voice lives, this measured cost is charged to its source generation.
After death, the tail's acoustic cost remains recorded against the retired source
generation, but no living parent's or child's energy is debited. The tail still
changes the habitat and consumes renderer resources. Presentation-only processing
cannot reduce habitat cost. Habitat cost attribution is already adopted: charge
the living owning Voice; record a dead tail against its retired source generation
without transferring cost to a parent or child. The conversion coefficient,
renderer and capture budgets and capacities remain unapproved. Charging a
descendant or reserving the full tail cost before death is a separate amendment.
A pre-admission forecast may reserve a budget; it is identified as predicted and
does not overwrite the render-derived observation.

### Pitch policy and source continuity

Three alternatives have different meanings:

| Alternative | Benefit | Cost |
|---|---|---|
| Transfer state in place | Continuous moving source; bounded state count under repeated pitch Updates. | Each body must define a stable transfer; arbitrary coefficient changes need not support it. |
| Retain old coefficients and start a new state | Preserves every old free response. | Frequent pitch updates accumulate old-pitch sound and work; one Voice becomes a cloud of pitches. |
| Crossfade old/new states | General bounded transition between algorithms. | Alters the old tail through output processing and can change interference and measured fitness. |

**Recommendation:** state transfer for the open handles addressed by pitch Update,
with phase-preserving oscillator state and preserved resonator memory; structural
genotype replacement is a generation boundary affecting future On, whose previous
tails remain. These are deliberately different operations. There is no promise
of exact waveform continuity for arbitrary filter-coefficient changes. The admitted
pitch range/ramp must pass finite-output and transition tests; unsupported dynamic
retuning is not silently implemented as resetting or crossfading a research body.
Choosing general in-performance crossfade or immediate structural mutation instead
would require an explicit amendment. The author adopted this A1 policy in the
[decision record](2026-09-29-body-policy-author-decisions.md); implementation and
transition checks remain pending.

Separate per-Tone states are retained initially. An algebraic merge optimization
requires linearity with unchanged coefficients, identical modulation, identical
post-gain and no different per-Tone release multiplier. It also requires the same
state coordinates/phase basis, an exact sum of initial/current states and inputs,
and preservation of every handle's independent future control and random stream.
Random initial phases are not assumed equal: without an exact state mapping and
summation proof, such Tones cannot merge. No merge optimization is required by
this contract. Harmonic's nonlinear drive
response does not qualify automatically. Off closes only its own excitation; it
cannot release every Tone of the source. Physical source continuity and one-pitch
monophony are not interchangeable acceptance criteria.

A fresh On remains a new excitation channel/Tone. Re-kicking an open retained
Tone uses the renderer's impulse operation and preserves that state; it
is distinct from duplicating On. For nonlinear bodies, the chosen per-Tone channel
composition is part of the module's declared transfer law. It must not be described
as exactly re-exciting one shared nonlinear resonator. A future module needing
shared nonlinear state across channels must declare and test that composition
before admission; it cannot acquire it through an unverified merge optimization.

### Randomness and fluctuation

The source-generation seed is derived from the run seed and stable source identity.
Use distinct keyed domains for common source modulation, per-Tone initialization,
and per-Tone noisy excitation. The proposed Modal noise streams are independent
between Tone identities; only the declared modulation driver is shared. Keys bind
source generation, Tone identity where applicable, stream purpose and sample
clock. Splitting blocks or changing visitation order cannot draw different values.
Merge cannot replace independent excitation with a common stream.

The common fluctuation driver has one logical value per source sample, not one per
rendered Tone; its clock persists across silence. During an interval with no
driven handle, advancement must have a declared bound independent of the number
of skipped samples (for example a counter-addressable process or exact bounded
jump). A stateful filtered-noise generator needs such a declared continuation
method; skipping its hidden state update is not equivalent. Charge any per-source
resume work to the admitted block budget. No silent per-sample loop over every
live source is permitted by this proposal. Source retirement retains enough
identity to render its already sounding tails, but creates no continuing drive.
New source generations never inherit the live RNG state as genotype.

The body applies shared fluctuation to its driven components according to its own
documented mapping. Existing Harmonic motion uses a shared Hz offset, not identical
cents on every partial; a replacement must name any change. Revised A2 recommends
freezing the last effective motion offset and excitation-dependent spectral-balance
coefficients at effective Off, while keeping phase/amplitude and the declared free
decay law. Fresh random excitation and modulation then stop. With a release ramp,
the frozen values are those reached at its endpoint; Off adds no further
coefficient step. The tail may retain a detuned frequency until disposal.
Alternatives are immediate return to nominal/free coefficients (a frequency or
spectral step), or a separately timed coefficient ramp (continued deterministic
tail evolution requiring a registered duration). Effective-Off coefficient freeze
is already adopted. Immediate reset or a post-Off coefficient-return ramp would
amend that choice and require an explicit decision. The concrete source/Tone
random generator and supported rate law remain unadopted. Test frequency/phase, balance and
free decay immediately before/after Off, including motion extrema, zero drive
and Harmonic damping. The external clock may continue for other Tones without
perturbing this closed state.

Task 1 decides the presence/level of irregular fluctuation by audition only.
This contract does not select that level or replace `motion` with Task 1's stimuli.
Habitat output must reproduce under the same implementation, seed, command stream
and sample rate, including block partitions. No cross-platform bit-identity claim
is made without a separate test.

## Three module faces

The following signatures express ownership and data flow, not a request to add all
the displayed names as public wrapper types. Use existing structs where they fit;
keep crate-private items unless an actual external test/research client needs them.

```text
genotype face (setup / generation boundary):
  construct(founder_vocabulary, construction_context, rng) -> immutable genotype
  capture(genotype, caller_owned_bytes) -> canonical bytes + schema/version
  restore(canonical_bytes, schema/version) -> validated immutable genotype
  mutate(parent_genotype, allowed_pathways, rng, caller_owned_storage) -> child genotype

renderer face (prepared state; bounded block path):
  prepare(genotype, sample_rate, limits, caller_owned_storage) -> state or unsupported
  start(state, reference_hz, initial_seed) -> initialized state
  render(state, drive_amplitudes, kicks, pitch_ramp, random_span, out_audio) -> status
  copy_state_into(state, prepared_destination) -> copied state or unsupported
  free_response_status(state) -> active or finished
  reset(state) -> reusable empty state

optional self-model face (no ecological decision inside the module):
  support(purpose, frozen_request) -> supported domain or unsupported reason
  predict(purpose, frozen_request, core_analysis_context, caller_owned_output)
      -> evidence + support interval + model/version, or unsupported reason
```

Construction context includes actual pitch, sample rate and the immutable
environment needed by a candidate-dependent founder constructor. Canonical capture
includes the resolved traits, registry module identifier and schema/version, not
only the original founder controls. Mutation is deterministic and in range; an
unobserved pathway is frozen. Capture excludes phase, live drive, RNG cursor,
metabolic energy and listener state. Reproduction copies the parent chosen by the
existing respawn mechanism (D2), not an independently preferred body.

For live prediction, `frozen_request` contains a copied render state and explicit
future controls/random clock. For representative prediction it contains a genotype
and declared initialization/excitation schedule. These purposes cannot be
substituted for one another. The copied state operation uses storage prepared
outside the hop path; offline construction may allocate. A live body that cannot
supply a bounded state copy has no live forecast capability and still supports
offline representative rendering.

The self-model may internally use oscillator lanes, resonators, direct partials or
partial groups. Those are private module machinery. Core-facing evidence has one
meaning per purpose: audio-derived spectral mass on the declared Log2Space,
windowed mean-square energy with explicit sample windows, or supported spectral
positions/weights for a verified exclusion approximation. No R/H/C score, fitness,
parent probability or fabricated listener attribution comes from the module.
Core analysis transforms are used for listening weights/normalization; any shortcut
through them is tested against the complete render-and-analysis path.

F2 uses the same spectral-evidence representation as this self-model face; it does
not receive a second timbre-only partial model. Dense `_scan` output has exactly
`space.n_bins()` entries and calls `assert_scan_len_named` at boundaries. A sparse
representation names its coordinates explicitly, is not called a scan, and is
evaluated with the registered interpolation/readout semantics. Candidate-to-scan
bin location always uses the Log2Space log2 mapping (repository F4); linear-Hz
array indexing is forbidden. Hz-linear interpolation weights between those
located bin centers are allowed only when explicitly registered as the readout
semantics. The historical diagnostic below is not a waiver for future interfaces.

## Core guarantees, dispatch and bounds

The core validates finite sample rate/pitch/control inputs before admission,
provides stable sample clocks and deterministic random streams, and supplies
buffers of the admitted sizes. The module completely writes its output slice and
performs no bus routing, global gain, disk I/O, global RNG access, registry lookup,
lock acquisition or heap allocation during block rendering. Output gain, click
prevention, route changes, summation and per-source PCM capture remain core code.
The capture point includes the gain actually sent to the relevant bus. Mixed
output is accumulated in stable source/tone order.

Built-in Sine/Harmonic/Modal renderers use static dispatch behind a single
extensible dispatch boundary. That boundary may have a dynamic research variant;
it replaces the closed `BodySnapshot`/`AnyBackend` pair rather than moving an
exhaustive kind match into every consumer. A trait object is allowed for a registry
factory, module-native genotype and a research renderer called once per block.
Within a block, built-in loops remain monomorphic. Ecology reads capabilities and
evidence, never a module ID to choose its scoring formula.

**Proposed resource policy, pending the existing A3 decision:** retain the
registered workload, including 64 live Voices and ordinary birth opportunities.
Total source-generation capacity remains undetermined until retained retired
generations are included in the bound. With all 64 live generations sounding,
one death with a retained tail and one replacement birth require 65 generations.
The earlier 64-total proposal is therefore withheld, without reducing live load
or delaying birth. The proposed 16 Tone states per source also remain unproven;
count open handles and retained tails together. The proposed 576 lanes per
built-in state (64 partials × 9 unison) and 16 queued target updates per state
remain proposals, not measured deadline guarantees. For the update queue,
same-target updates at the same tick coalesce, otherwise overflow rejects the new
command with a recorded reason. Tone creation on the worker takes a prepared pool
slot and performs zero allocations; research state/genotype allocation occurs at
registration/preparation, with explicit byte counts. These limits are proposed
admission bounds, not an assertion that this maximum load meets the audio deadline.

Admission must also satisfy a measured per-block work budget for the configured
sample rate/hop and concurrent load. Count pre-grouping lanes, state copying,
candidate-dependent construction, source capture, and live prediction as well as
the final sparse groups. The Phase 3 registration must allocate a numeric renderer
share of the existing full-hop deadline before any timing acquisition; this draft
does not invent available milliseconds from the F2 local screen. Offline runs may
support slower bodies with explicitly recorded limits. A body without a certified
fast self-model is offline-first; its expensive fallback never blocks ordinary
same-hop birth or quietly delays Voice participation.

The proposed 16-state bound needs an admission envelope, not only a pool size.
Phase 3 registration must bind the A1 lowering rule, maximum simultaneous open
handles, onset burst/rate, maximum open duration and maximum retained free-tail
duration under admitted gain/drive. At every tick, open handles plus retained
closed tails must fit the chosen state capacity. Retired sources consume the same
generation pool and must be included in its capacity calculation. A conservative
check is burst allowance plus arrival rate times (maximum open duration + maximum
tail retention), rounded up, with explicit boundary handling. Here arrival rate
must bound state-creating On arrivals in every interval, including bursts; a mean
onset rate is insufficient. Where open duration is unbounded, bound simultaneous
open handles separately instead of shortening their lifetime to fit the pool.
A measured burst trace must also pass. T60 alone is not tail retention: a
unit-amplitude 1e-6 cutoff takes two
T60 intervals even before gain/overlap. No current result proves this envelope
for long Modal decays. A3 must choose a compatible declared envelope, larger
preallocated limits with measured work/memory, or an explicit different lowering
policy; ordinary onset refusal remains failure, not successful backpressure.

Free state is disposable only when the module's tested residual-output bound falls
below amplitude 1e-6 at the maximum admitted post-gain, or at a declared hard free
tail cap. Proposed cap: 60 seconds after the last excitation; the final 5 ms use
an explicit core retirement fade. There is no fade on ordinary Off. Cap truncation
is recorded, included in all representative references/energy accounting, and
invalidates an untruncated forecast. Future excitation handles are canceled on
Voice death; existing free state keeps source/generation/route attribution until
disposal. Empty instantaneous output alone is not a disposal criterion.

At capacity, reject the newest state-creating On before it changes state, record
the refusal, and preserve already sounding state. No silent tail stealing or
allocation fallback. A refused registered ordinary birth/onset is a failed resource
condition, not successful service. Invalid/non-finite module output retires that
state with an explicit error and unknown forecast status; it is not evidence of a
quiet, low-cost body. No instrument disk-write path is introduced.

## Render references and self-model conformance

### Reference request and evidence identity

Preserve an equivalent entry point to constructing and rendering a representative
Tone: an actual resolved genotype, anchor Hz, actual source identity/seed policy,
sample rate, onset/hold/stop schedule, drive/kick, smoothing, modulation, output
gain, route/capture point, analysis configuration and finite horizon are explicit
inputs. The live and representative entry points use the same module renderer and
core output chain. A representative request initializes declared state; a frozen
live request copies the real state. Neither reads a future actual birth clock as
though it were already known.

An identity includes module/schema/renderer version, genotype bytes, source and
body generation where relevant, all effective input fields, listener-analysis
version, purpose and horizon. Hash immutable data during preparation, not every
hop. A prediction includes issue/availability/valid-until times and completeness.
Missing, stale, unsupported, truncated and known silent are distinct. Zero mass
is known silence only after a supported computation; unsupported must not normalize
to a uniform, apparently valid body.

The preserved F2 reference's weak-peak readout places subjective mass at selected
local-maximum `bin_idx`, while `u_erb` may describe a redistributed centroid. The
[anchor-readout diagnostic](../../design-notes/body-fitness-anchor-readout-diagnostic-20260929.md)
changes readout alone with group power, selection, timing and motion held fixed.
Its 728 saved conditions and old comparison limits remain frozen. This contract
neither adds full-partial-support preservation as a new requirement of that unit
nor claims that restoring anchors resolves interference, windows, time averaging
or the failed resource screen.

### Per-consumer reference, fallback and gate

All new tolerances below are proposals for registration before Phase 3 acquisition.
Existing stronger local tests remain. F2's existing thresholds are unchanged.
Use exact known/unknown/support-mask agreement before any numerical comparison.
For mean-square energies define the proposed error criterion as
`abs(predicted - rendered) <= 1e-10 + 0.001 * abs(rendered)` per window;
record both absolute and relative errors. Exact-zero fixtures must remain zero.
Here `rendered` means a **fixed-randomness realization**, conditional on the
request's complete random/state identity. It is not an ensemble expectation.
The revised draft recommends this reference for the rows below. An analytic
expected energy for noisy Modal cannot be silently compared as if it predicts
that realization. An expectation-based consumer needs its own purpose, seed
ensemble, estimator/uncertainty, error rule and behavioral meaning agreed by its
owner and the author under B5. No such ensemble or tolerance is registered here;
until then that shortcut is unsupported. This may rule out an otherwise useful
analytic shortcut and is an explicit pending decision, not an accuracy failure
to be remedied by increasing tolerance after acquisition.

| Consumer / purpose | Render-derived fallback | Fast-path conformance gate and consequence |
|---|---|---|
| F2 whole-body candidate spectral mass, shared with the optional spectral self-model | Fixed registered render realization: render the registered representative schedule through the pinned front end, then core weighting/peak extraction and normalization. Keep the current 72-frame reference as its own version. | Every registered candidate: score error ≤0.025, level error ≤0.0125, zero strict reversals at reference gap ≥0.1. Preserve mass/distribution diagnostics and resource gates. No v1/v2 production admission by this document. |
| `action_candidates/footprint.rs`: representative temporal footprint | Proposed fixed realization: render the declared representative Tone and integrate squared emitted samples in the exact 16 windows, with four-second cap/truncation metadata; normalize by peak only afterward. | Proposed energy criterion in every window and maximum absolute error ≤0.001 in normalized power; exact silence/support/truncation agreement. Existing Sine `<1e-3` relative-energy test remains. An unverified bank forecast falls back offline or returns unsupported. |
| `action_candidates/energy.rs`: intervention/retained-source energy | Proposed conditional fixed realization: copy all routed states and random streams of the source, apply the explicitly frozen Continue/Off/On/Gap alternative, sum PCM, then integrate each requested window. Use one shared external context. | Proposed energy criterion per window, exact window/route/support agreement, plus preservation of the existing Sine carrier absolute-error `<0.002` test. Include constructive/destructive overlap. Incoherent energy cannot pass as a coherent mean when cross terms matter. |
| `src/temporal_cognition/body.rs`: acoustic descriptors and body identity | Actual observed realization: keep the current routed PCM → NSGT → features path. Replace introspection-based recipe identity with canonical module capture. | This observed path has no analytic substitute in Phase 3: same PCM/config must produce identical descriptors and masks. Any later predictive descriptor shortcut compares each raw coordinate under a separately registered tolerance; until then it is unsupported, not an observed descriptor. |
| `self_prediction`: future energy/control/carrier and descriptor forecasts | Proposed conditional fixed realization: fork the complete frozen source state, commands, routes and random clock; render its declared horizon. Derive energy/descriptors using the same core code as observations. | Energy uses the proposed window gate above; timing, masks, owner generation and frozen issue support match exactly. Descriptor shortcuts remain disabled without their own registered per-coordinate gate. Learned residual models remain labeled predictions and do not redefine the physical render reference. |
| `voice.rs` / `modal.rs`: `project_spectral_body` and predictive terrain | Proposed fixed realization: analyze a representative or frozen-state render, according to the declared prediction purpose, on the same Log2Space. Preserve emitted amplitude/mass before normalization. | Proposed normalized spectral-mass L1 ≤0.01 and total mass relative error ≤0.01 above the declared silence floor, plus the F2 downstream score/level/rank gates when used for ecological comparison. Amplitude projections cannot be compared directly with power or subjective mass. Unsupported projection is unavailable prediction, not an all-zero terrain. |
| `pitch_core.rs`: ratio LOO | Actual observed realization: subtract the complete source's aligned habitat PCM, including old Tone tails, from the habitat input and rerun core analysis; use the body-aware F1 source-exclusion path. | Supplied ratios alone are not a certificate. A ratio/weight shortcut must pass the proposed spectral L1/mass gates and downstream F2 score/level/rank gates on the resulting source-excluded terrain, including overlap/unison/tails. If it fails, use source-PCM exclusion where admitted; otherwise mark exclusion unavailable. Do not substitute a generic harmonic series or call one-bin `ExactScan` whole-source removal. |

The F2 score limits come from the existing registered comparison, not a proof of
musical acceptability or correct parent selection. The additional spectral/energy
limits proposed here are engineering gates, not perceptual thresholds. Body-aware
and I-series owners must resolve their scope before registration. Passing one
consumer's gate does not enable another consumer, and passing a numerical gate
does not pass the resource, ordinary-runtime, listening or author-approval gates.

A declared capability domain records sample rates, pitch range, genotype bounds,
drive/control regimes, initialization/live-state conditions, horizon and analysis
version. Conformance exercises interior/boundary cases and adversarial examples;
it is not a proof over arbitrary continuous parameters. The runtime checks the
declared domain. Any excluded condition produces explicit unsupported evidence.
It does not try all-candidate representative rendering synchronously in the hop.

## Tests and expected changes

Phase 3 adds `tests/body_conformance.rs` for the public/cross-module contract and
module-local tests for private state transfer and native genotype validation.
An external test may justify exposing a narrow testable registry/render entry
point; it does not justify exposing native lanes or the entire sound module tree.

The harness covers every registered body, including one research fixture without
a self-model. It exercises zero input, impulse, finite and continuous drive, Off,
re-excitation, dynamic pitch, structural generations, silence gaps, death/tails,
Nyquist/analysis edges, maximum declared genotype/state sizes, capture/restore,
mutation bounds, source-order independence and fixed-seed block partitioning.
Mutable fixtures compare both fresh and nonzero retained states. Genotype round
trips compare canonical bytes and resulting seeded render; copying live state must
reproduce the next samples and must not advance the original RNG.

The following existing contracts should pass without weakened assertions:

- `tests/log2space_scan_invariants.rs`: Log2Space alignment and boundary failures.
- `tests/render_binary.rs`: valid offline output, zero-amplitude handling, report
  transparency, reserved source IDs, actual route capture, descriptor transparency,
  and `body_footprint_renders_repeat_exactly` for identical new-version runs.
- `schedule_renderer` observer/candidate tests: enabling diagnostics does not
  change audio, retained ownership, learning, or command scheduling.
- `action_candidates` and `self_prediction` tests for complete retained inventory,
  unknown versus zero, generation mismatch, unsupported control and no double
  training; ordinary Sine carrier comparison keeps its existing error bound.
- Oscillator tests for finite output, seeded repeatability, nominal-pitch Nyquist
  masks, silent padding and optimized/basic kernel agreement under identical inputs.
- `tests/sample_seed_policy.rs` and instrument/offline-render separation.

Tests encoding the old contract must change deliberately: ADSR-output-release
assertions in `sound/tone.rs`, body-envelope lowering in
`tests/body_envelope_propagates_to_tone_adsr.rs`, indefinite oscillator ringing,
Sine impulse boost, per-Tone fluctuation restart, closed snapshot construction and
generic harmonic LOO fallback. Retain their intent where applicable, replacing
obsolete assertions with input-release/free-tail, driven-level, continuous source
clock and evidence-availability assertions. List changed tests and their old/new
meaning in the Phase 3 review; do not merely regenerate expected outputs.

Expected changed audio includes Sine/Harmonic impulse decay, driven level, ordinary
Off tails, interference between tails, and fluctuation trajectories across notes.
Changed fitness can consequently change pitch, energy and reproduction. Existing
étude WAV identity is not required. Old captures and failed F2 records remain
immutable. New renderer references get a new version and are not used to make an
old comparison pass. `examples/timbre_probes.rs` supplies inspection/audition
material; no listening result is claimed by the harness.

After integration, repeat Phase 2's body-swap, controlled emitted-level, silence
and high-cut controls inside F2–F4; trace evaluation → energy → parent-selection
probability. A changed selected parent is not mandatory. Source exclusion,
ordinary same-hop Voice creation, full-hop resource behavior and device/author
acceptance remain separate gates. Source changes require the repository's full
test record and relevant standard checks; no code/test/render acquisition belongs
to this document-only Task 2 unit.

## Points to settle with body-aware

Body-aware response recorded on 2026-09-29. “Confirmed” below means that the
boundary already follows the cited F1/F2 contracts or the adopted integration
order. It does not approve this draft's new policies, select a fast model, or
establish runtime acceptance. This is a joint-design response by a second Astra
agent, not the independent Claude review required before author approval.

| ID | Answer | Confirmed boundary | Still open |
|---|---|---|---|
| B1 | Share the core-facing request and evidence identity; keep native partial groups private to a body's optional self-model. | Bind the actual resolved body and renderer inputs; the core owns the audio-to-evidence transform and ecological evaluation. | The admitted fast representation and concrete interface are not selected. |
| B2 | Pin the old selected-bin readout and the diagnostic's distinct partial-frequency readout explicitly. | Frozen references, interpolation, weighting and temporal order stay versioned; historical failures remain. | Anchor readback passes but old-render comparison still fails (93/728); whole-suite regression has an unresolved asynchronous preparation failure. No runtime/resource acceptance follows. |
| B3 | No supported runtime domain has passed the complete registered error-and-cost gate. | Keep the whole registered domain, including 576 lanes and candidate-dependent body preparation. | A representation meeting that domain and the complete birth/hop budget remains unestablished. |
| B4 | Use the actual child's resolved inputs and identity; keep the canonical numerical render offline. | Preserve scheduled, unexpected and respawn same-hop Voice creation targets; record acoustic onset separately. | An admitted fast path satisfying those targets remains unimplemented/unaccepted. |
| B5 | Retain existing F2 tolerances. Do not approve the additional consumer tolerances through F2 agreement. | Each new consumer purpose needs its own registration before acquisition. | Extra energy, mass, spectral and temporal limits require their consumer owners and author decision. |
| B6 | Whole-source, route-correct PCM subtraction before analysis is the generic observed-audio LOO boundary. | Preserve identity, alignment, analysis history and retained-tail ownership. | Evidence covers bounded existing implementations, not every future module or ordinary-runtime generation transition. |
| B7 | Preserve I12b closure/base freeze → I4 wiring → body-aware F3 → timbre Phase 3, then Phase 2 re-acceptance before Phase 5. | Preserve old manifests and version every changed renderer reference. | The concrete integration base, assignments and new acquisition registrations await the orchestrator's integration point. |

### B1 — Shared request, identity and ownership

F2 agrees with the request boundary, not with treating a ratio list as sufficient
evidence. Include module/schema/renderer version, actual resolved genotype,
candidate Hz, sample rate, excitation/hold/stop, gain, modulation and smoothing,
route/capture point, seed inputs, representative initial state and horizon, and
analysis configuration/version. Bind source identity, generation and actual seed
frame where relevant. Distinguish a fresh representative state from a snapshot
of a continuing source; include the evidence purpose and support/availability
when a result is delivered asynchronously. A future interface must preserve
these distinctions even if some fields are in enclosing request objects.

The body radiates PCM and may supply a conforming fast self-model. The core owns
routing, the capture point, analysis, Log2Space-aligned evidence and F2 readout. F2 consumes
mass-weighted environment values; energy, footprint and temporal consumers use
their own declared purposes. For F2, preserve both subjective density and its
ERB-integrated mass, or an explicitly equivalent bin-mass representation, with
readout coordinates and analysis identity. Equal vector length does not prove
equal space/epoch. The environment also needs its Landscape/epoch/space identity;
it is not determined by a genotype hash. No new common storage format, cache or
fast formula is approved here.

Evidence: [direct-model contract §§1–3](../../design-notes/body-fitness-direct-model-contract-20260929.md),
[representative-render/F1 reference](../../design-notes/body-aware-fitness-reference.md),
and [runtime restart §§1–4](../../design-notes/body-fitness-runtime-restart-20260929.md).

### B2 — Frozen old readout versus the anchor approximation

The old reference renders 72 PCM hops from a fresh declared Tone/analysis state.
Each hop sums complex NSGT contributions before squaring and applies band
smoothing. After peak selection and power redistribution, subjective mass is
placed at the selected local-maximum **bin**, with that bin's A-weighting. The
power exponent precedes frontend smoothing and the 72-output average. F2 then
forms the mass-weighted mean of `C_eff` and applies the sigmoid once.

The diagnostic changes only v2's group readout from ERB centroid to selected
**partial frequency**. It retains v2 group powers, selection, decay averaging,
72 midpoint motion samples (one point without motion), Hz-linear interpolation
between adjacent Log2Space bin centers, projected-bin A-weighting and the
nominal-out-of-band pre-exclusion. A partial anchor is not the old NSGT peak bin;
the intervention does not reproduce coherent interference, window lobes or the
old temporal order. It adds no full-partial-support requirement.

Evidence: [readout audit](../../../target/body-fitness-support-20260929/reference-semantics-audit.md),
[frozen diagnostic contract](../../design-notes/body-fitness-anchor-readout-diagnostic-20260929.md)
and [freeze manifest](../../../target/body-fitness-support-20260929/anchor-contract-freeze-v1.json).
The frozen old reference is `semantic-v1.jsonl`, SHA-256
`fb604ba3e83ad598a4ed9c4ed741cee44b6da33435cfdf86e6ee81dead3711e1`;
frozen v2 is `semantic-v2.jsonl`, SHA-256
`f895342626cc19463e7722fdd41915e0f42344e9fba6b3b01b9d85b087607d09`.
The frozen semantic `landscape_peaks` rows contain actual Harmonic bodies and
must not be replaced with Modal bodies. Retain [v2's 159/728 semantic failures
and 10/20 cost failures](../../design-notes/body-fitness-partial-groups-results-20260929.md).
The saved [independent checker result](../../../target/body-fitness-support-20260929/anchor-v1-independent.json)
and [coordinator reduction](../../../target/body-fitness-support-20260929/coordinator-reduction.json)
now agree: all 728 candidates pass replay/control and are interpretable; 93/728
fail the old-render semantic comparison (85 recovered from v2, 19 new failures).
Maximum score/level errors are 0.572653651/0.267519325, with 4 strict reversals
among 345 eligible pairs and no new ties. Thus readback passes while the
old-render gate fails. The old maximum score-error example improves from
0.397168636 to 0.075178325, still above its limit; the new maximum is
`harmonic_spread`, base 440 Hz, +24 cents, in the `sine_440` environment.
The diagnostic does not establish a sufficient replacement for v2. Independent
checking reconstructs saved groups/motion, not PCM/NSGT or group formation.
Three full Rust runs failed the same existing asynchronous preparation test:
two without progress instrumentation and one with it. The isolated test passed.
After the final instrumentation cleanup, focused checks passed but the full
suite was not repeated on that final source; the whole-suite failure is unresolved.
See the [diagnostic result](../../design-notes/body-fitness-anchor-readout-results-20260929.md)
for the bounded follow-up and current validation state. Resource and runtime
acceptance remain absent. In any later diagnostic, uninterpretable
rows must remain visible but outside an interpretable-only semantic denominator.

### B3 — Whole-domain admission and complete cost

The registered bounds are B=2048 bins, N=2048 candidates, J=256 additional Hz,
K=64 ratios, U=9 active unison voices and L=576 lanes, with targets of 16 births
in one hop and 32 coexisting Voices. Candidate-dependent Landscape patterns
must use the actual child ID, community seed, seed frame, environment and spec
at each candidate; a 440 Hz snapshot reused everywhere is a different experiment.

The initial screen includes preparation and scratch allocation, 690 candidates
and 21 additional Hz, cold single-body and cold 16-body requests: **registered
upper limits**, not measured maxima, of 1 ms and 8 ms respectively. The 8 ms
screen is a design allowance, not a measurement of remaining hop time. It does
not establish the maximum B/N/J case, 32-Voice full-hop cost or device acceptance.
Candidate body/pattern generation,
environment preparation, existing audio/ecology work and source-removal analysis
must also be accounted for at integration. v1/v2 establish no admitted subset
merely because some small cases pass. Invalid, unsupported, zero-mass, capacity
and deadline failures remain explicit; no truncation, stale value, point fallback
or delayed birth may turn them into success. Source: [registered domain and
failure contract §§3–4](../../design-notes/body-fitness-direct-model-contract-20260929.md)
and [v2 results](../../design-notes/body-fitness-partial-groups-results-20260929.md).
No tuning, threshold change or resource remeasurement is authorized by this answer.

### B4 — Child identity and the two clocks

Actual child preparation must use the same resolved genotype/Recipe, candidate
Hz and body seed inputs as the eventual child. Preparation, validation, selection
and commit are distinct; a failed child must not partially consume its
ID/counter/Population state. Earlier successful births in a sequential batch
are not rolled back by a later failure. An uncommitted seed frame must not be
silently replaced by the later lifecycle start time.

Scheduled Spawn must select/create in its existing action-processing hop;
unexpected Spawn in its accepted/processed hop without requiring a cache hit;
respawn in the death/cleanup hop using current surviving parents, updated energy,
environment and occupancy. Respawn cleanup follows phonation collection, so the
child's ordinary phonation opportunity starts next hop or later. Record Voice
creation, first nonzero sound, birth-hop own PCM and subsequent SourceRemoved
receipt separately. The 48 kHz/512-sample hop is 10.667 ms, not a birth-only budget.
A future-child inventory with invented identity or clock is not valid. Fully
fixed scheduled children may permit advance preparation, but that is no general
solution for unexpected requests, death or environment-dependent bodies.
Evidence: [runtime restart §§1–3 and preserved time4 limitations](../../design-notes/body-fitness-runtime-restart-20260929.md).
No fast path has yet established these runtime guarantees.

### B5 — Existing values versus proposed consumer limits

F2 keeps maximum score error 0.025, maximum level error 0.0125, and zero strict
rank reversals for old score gaps of at least 0.1; ties remain separate. These
are engineering comparison gates, not perceptual thresholds. F2 distribution
L1 and mass differences remain diagnostics in the old-render comparison; they
must not acquire a new pass/fail limit retrospectively. The independent-formula
and diagnostic replay/control tolerances serve different checks and stay as
registered. Evidence: [direct-model §4](../../design-notes/body-fitness-direct-model-contract-20260929.md)
and [diagnostic §§3–4](../../design-notes/body-fitness-anchor-readout-diagnostic-20260929.md).

This draft's proposed per-window energy bound `1e-10 + 0.001 * abs(render)`,
normalized temporal-power limit 0.001, spectral normalized-L1 limit 0.01 and
mass-relative limit 0.01 are **not accepted by this response**. Neither their
numerical adequacy nor their applicability follows from F2. Before acquisition,
consumer owners must specify units, windows, silence/support handling, realization
versus expectation, random-state/ensemble identity, render reference and behavioral
consequence, then agree limits with the author. The per-consumer table proposes
fixed realizations; it does not approve that choice on the owners' behalf. T1/T2
owners must settle temporal windows/descriptors and Phase 4 controls: attack,
decay and re-excitation; mean spectrum; level, onset density, modulation and
meter; human, detector and behavioral differences reported separately. No
replacement numbers are proposed here. Evidence: [timbre plan Phases 2 and 4](../plans/2026-09-29-timbre-synthesis.md).

### B6 — Whole-source LOO, route, generation and tails

F1 captures every Tone belonging to the target source at ScheduleRenderer's
post-route habitat PCM point, subtracts that sum from the aligned habitat mix
**before NSGT**, and preserves the matching analysis history/support. Habitat-off
or presentation-only contributions are not subtracted from habitat audio.
A Tone retains its captured route; a later Tone's route does not rewrite an
old tail. Retired-source tails remain audible in the mix and count as other
sound for a new source. Source identity includes ID, generation and birth sample;
slot reuse must not transfer old-source exclusion ownership to a new generation.
This is an observed-audio path, not the fresh-state representative prediction
used to score a prospective child.

Evidence has several distinct scopes. [F1 reference and v5 results](../../design-notes/body-aware-fitness-reference.md)
([results](../../design-notes/body-aware-fitness-results-20260926.md)) cover existing
Sine/Harmonic/Modal, overlapping Tones, routes, release/retirement tails and
mid-hop glide. F3a in that results document adds a four-source processor:
synthetic PCM checks birth/retirement/slot reuse; separate actual-Tone comparisons
check four-source analysis. [F3e Phase 1](../../design-notes/body-aware-fitness-f3e-phase1-results-20260926.md)
checks generation/body-generation and other delivery invalidations with an
independent fixture ledger, not arbitrary normal-runtime transitions.
[Runtime observation](../../design-notes/body-fitness-runtime-observation-results-20260927.md)
adds real capture/delivery in an opt-in offline-render observer, bounded to four
sources. [Runtime lifecycle observations](../../design-notes/body-fitness-runtime-lifecycle-results-20260927.md)
check birth/retirement identity and receipt timing; they do not prove a live
retired acoustic tail, a body-generation change or same-ID generation reuse.

The PCM boundary is suitable for future modules whose entire routed contribution
is captured and aligned, including all retained tails. Existing evidence does
not certify those modules or their ordinary-runtime lifecycle behavior. New
modules and renderer changes need conformance at that boundary; generic ratios,
one-bin subtraction and representative forecasts do not establish whole-source
exclusion. No resource-unbounded fallback is admitted by this architectural answer.

### B7 — Integration baseline and re-acceptance

The adopted order is I12b closure and base freeze, I4 wiring, body-aware F3,
then timbre Phase 3; Task 2 approval and an agreed F2 representation remain
entry conditions. Request/identity agreement in B1 is necessary but does not
select that production representation. A representation decision must identify
the common evidence schema and readout, concrete producer/self-model, supported
registered domain, pinned reference/version, validation status and complete cost
route. Runtime use still requires the registered accuracy, resource and same-hop
gates; no representation has established them. This is not a claim that a
whole-domain fast model is feasible, nor authorization to reduce that domain.

Until those decisions/gates are met, ordinary runtime keeps its current default
point-evaluation behavior with body-aware fitness OFF. A failed or unsupported
opt-in body-aware request must not silently become a successful point evaluation.
Slow representative rendering remains offline; it does not delay ordinary births.
The Claude review's suggestion to remove fast-F2 dependency from Phase 3 is **not
adopted by this revision**. An author could explicitly amend the F2 entry condition
to allow render-only module migration while body-aware runtime consumers remain
disabled, but that amendment must name the reduced exit and later acceptance
work. It would still retain I12b → I4 → body-aware F3 → timbre Phase 3 integration
order. In its absence the existing gate stays unresolved; B1 alone or a render-only
fallback does not satisfy it. Existing permission for an isolated prototype is
not permission for main integration, and this document task starts neither.

The orchestrator serializes integration and records the exact
base and workstream owners there. Historical F1/F2 base `06a4772` and the frozen
v1/v2 manifests identify old evidence; none is declared the next integration
base. Evidence: [timbre plan dependencies, Phases 2–5 and coordination](../plans/2026-09-29-timbre-synthesis.md).

Under the changed renderer, register new F2 representative PCM/NSGT references
and self-model comparisons using the actual body/excitation/decay, full candidate
and selected-Hz preparation, and the complete applicable cost gates. Recheck F1
whole-source subtraction and tail/route/generation boundaries as a prerequisite
to F3 delivery and consumer checks. F3 must preserve identity, support/freshness,
invalid/unknown handling and the actual evaluation/action path. F4 must preserve
updated-energy parent selection, scored-body/actual-child identity and the
same-hop creation/receipt distinctions above; old delayed-birth success is not
new timing acceptance.

Phase 2 re-acceptance within F2–F4 must include body-swap at fixed fundamental,
position and environment; controlled emitted level; silence and high-cut benefit
and cost; and the trace evaluation → energy → parent-selection probability.
A changed selected parent in one run is not required. Keep fixed-seed regression
and multi-seed research evidence separate. Complete this re-acceptance before
Phase 5, along with its F4/F5 entry conditions. Do not overwrite old captures or
use a new renderer to relabel an old failure. New registrations must be written
before acquisition; this answer does not initiate them or approve A1–A3.

These seven responses were included in Claude's initial independent contract
review. They establish existing F2 boundaries and evidence limits, not adoption.
Author/owner decisions remain the extra consumer tolerances and the draft's
A1–A3 policies; the fast
representation and whole-domain feasibility also remain unresolved engineering
work. Review may assess these open items, but author adoption and Phase 3 entry
cannot be inferred from this response.

## Remaining author decisions and review record

Current adoption: A1 and A3 cost attribution are adopted; A2's Off behavior is
selected and its 0.5 s T60 is a prototype value. The table retains the full
decision scope so that pending capacity, module details and A4 are not confused
with these adopted policies. See the [author record](2026-09-29-body-policy-author-decisions.md).

| ID | Decision needed | Draft recommendation |
|---|---|---|
| A1 | Pitch scope, lowering and unsupported operation | Retune open handles only; preserve closed tails. New phonation after closure uses fresh On, repeated excitation of an open handle uses re-kick. Reject unsupported live retuning atomically, with no implicit deferral/reset; future On gets a separate pitch request. Structural genotype affects future On. Accept cross-phonation pitch overlap, or explicitly amend this policy. |
| A2 | Drive/free decay, Off coefficients, noise and rate domain | Propose 0.5 s free T60 for Sine/Harmonic; freeze last motion offset and spectral balance at effective Off. Alternatives: immediate return or a separately registered coefficient ramp. Shared source modulation and independent Tone noise; bounded silent-clock advancement. Each module declares/tests rate normalization, including Modal. Task 1 audition remains separate. |
| A3 | Capacity, lowering envelope, retirement and cost payer | Retain registered load including 64 live Voices and ordinary births. Total source capacity awaits a bound including retired tails; the earlier 64-total proposal is withheld. The proposed 16 states per source remains unproven. Keep 576 lanes per built-in state, queue 16, 1e-6 residual threshold, 60 s free-tail cap and 5 ms retirement fade as unadopted proposals. No ordinary onset refusal accepted as success. Determine capacity and lowering without silently reducing load or delaying birth, and register a numeric renderer budget before measurement. Charge post-route habitat PCM to the live owner; keep dead-tail cost on its retired generation, without debiting parent/child. |
| B5 decision | Additional consumer limits and stochastic reference | Proposed fixed realizations and energy `1e-10 + 0.001*abs(render)`, normalized temporal power 0.001, spectral L1/mass relative 0.01. Consumer owner plus author must approve purpose/reference and numbers; an ensemble expectation requires a distinct registration. Existing F2 limits remain fixed. |
| F2 entry decision | What satisfies the Phase 3 dependency | Retain the existing unresolved production-representation gate and integration order in B7. Request identity alone is insufficient. Render-only integration with disabled consumers is an explicit possible amendment, not an adopted consequence of the review. No global fast-model feasibility claim. |
| A4 | Adoption of the completed contract | Requires recorded A1–A3 decisions, B1–B7 disposition, owner/author agreement on extra reference/tolerance choices, explicit F2 entry interpretation, and resolution of the independent review. This revised draft has not been independently re-reviewed. Contract adoption is separate from runtime admission and Phase 3 integration. |

Review status:

- Draft/revision: this document; static source/document inspection only. The
  [response](2026-09-29-body-module-contract-review-response.md) records all
  fifteen dispositions and the remaining author choices.
- Independent review: **received, 2026-09-29**, from actual model
  `claude-opus-5-5[1m]`, using the author-authorized fixed six-document packet,
  with no tools or linked-document/source access. Verdict: conditionally usable
  for author policy decisions, **not ready for A4**. The
  [original review](2026-09-29-body-module-contract-claude-review.md) is unchanged
  (SHA-256 `7288f5374d888cb8d5e6d46b4bf47abe0db79a1339d5da8122c1bf2fbe8af805`).
  Its numeric/code/resource statements are document comparisons, not independent
  implementation or measurement verification. The revised draft has **not** been
  sent for another independent review.
- Review routing: Task 2's explicit other-model requirement and the author's
  specific Claude-send approval apply here. The 2026-09-30 execution instruction
  in `AGENTS.md` makes Sol 6.1 primary and reserves Astra for necessary exceptions;
  it does not replace Task 2's other-model review requirement. Astra disposition is not a substitute
  for independent review or author adoption. A second Claude review is not an
  automatic new gate: any further review scope is determined by the changes and
  unresolved concerns. A4 remains unready because A3 capacity, remaining module
  details, B5 and the F2 entry decision remain open, not merely because the
  revision has not been re-reviewed.
  The previous send approval covered the fixed old packet, not this revision.
- Review history: an earlier local attempt failed OAuth refresh; automatic
  approval review rejected a retry before the destination/payload authorization.
  The author subsequently authorized the fixed packet, and review was received.
  Those historical failures are no longer the current review status.
- Body-aware responses: **recorded, 2026-09-29** in B1–B7; existing boundaries confirmed, fast-path admission and additional consumer tolerances unresolved. Joint design only; not independent review.
- Author approval: **partial policies adopted** in the later
  [decision record](2026-09-29-body-policy-author-decisions.md): A1, A2 Off policy
  with prototype T60, and A3 cost attribution. Capacity and full A4 adoption
  remain pending. The old six-document send approval was separate. No listening
  or subjective acceptance is claimed.
- Phase 3 implementation/integration: **not started by this task**. Entry requires
  approval and the unresolved F2 representation gate in B7; serialized integration
  preserves I12b closure/base freeze → I4 wiring → body-aware F3 → timbre Phase 3.
- Verification for this draft: internal cross-reference/coverage checks only;
  no cargo, tests, rendering, resource acquisition, source migration or commit.
