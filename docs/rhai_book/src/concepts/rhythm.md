# Rhythm: Evidence and Participation

A beat is a recurrent temporal reference inferred from sound. Sharing that
reference does not require every Voice to attack at the same phase.

## Timing relationships

A preset defines the relationship its timing policy keeps. An opportunity is
distinct from an attack: lifecycle, phonation gates, finite duration, rest and
articulation can prevent an onset or extend a living Tone.

| Preset | Keeps | May change | Does not guarantee | Meaning of `entrainment(x)` |
| --- | --- | --- | --- | --- |
| `entrained()` | The intrinsic bodily rate, fixed during a Voice's lifetime | The participation period follows supported recurrence weighted by proximity to the bodily scale. Costs select a time near due, a wait up to one participation period, or a skip | Equal onset intervals, permanently different onset rates between Voices, or alignment with a shared beat phase | Influence of acoustic context on period adaptation and timing costs; omitted value `0.5` |
| `flow()` | The intrinsic scale of bounded, irregular renewal draws; no period adaptation to heard recurrence | Renewal intervals vary; acoustic costs can select a different time, wait, or skip | A regular onset sequence, a beat, or an exact mean interval equal to the intrinsic period | Influence of acoustic context on timing costs, not renewal-scale adaptation; omitted value `0.05` |
| `steady()` | The intrinsic period and later private due-time grid, with sample rounding | Context selects sound or skip at due; execution constraints can remove onsets | An attack count, a pitch outcome, or following another Voice's time | Influence of acoustic context on sound-or-skip costs; omitted value `0.5`. It does not move due or adapt the period |
| `metric()` | Attraction to the acoustically confirmed production beat, with the configured beat-phase offset; independent initial phase before confirmation | Effective rate and phase follow the beat and its confidence. Before confirmation the clock free-runs; execution and articulation can withhold attacks | That a beat will be confirmed, exact phase coincidence, or independent noninteger tempo relations | Confidence-weighted attraction of rate and phase to the beat; omitted value `0.95` |

`entrained()`, `steady()` and `flow()` use local acoustic participation, while
`metric()` uses beat synchronization. Setting `entrainment(0)` removes these
acoustic timing influences; flow still draws irregular renewal intervals.
The production meter has one beat reference and integer-ratio subdivision and
measure layers; it does not implement a network of independent noninteger tempi.

Entrained period updates preserve the remaining fraction of a future,
uncommitted cycle. New acoustic context does not move a committed onset time.

For entrained and flow timing, a causal observer reads past habitat audio. Each
Voice also retains the sound it rendered, so the energy attributed to itself can
be excluded from a predicted competitor. The Voice compares sounding near its due
time, waiting up to a participation cycle, and omitting the current event. Overlap is
integrated across an estimate of the planned hold and ADSR release. Successive
voluntary omissions become more costly; an emitted onset resets that count.
Only an emitted onset creates a pending context-memory update, which is applied
when its matching actual audio is available. Waiting, skipping and an unexecuted
plan do not count as an onset.
Hold lengths remain tied to the intrinsic period, so cadence adaptation does not
stretch each sound. The overlap preview uses the same hold duration.
The three-band energy proxy is an engineering approximation, not a model of
auditory masking or consonant fusion.

## Continuous audio recurrence

The top-level TOML key `temporal_meter_t2 = true` lets the production meter infer
recurrence from continuous habitat audio. Place it before any TOML section.
The default `false` uses emitted onset events. This setting is independent of
`[temporal_t2]`, which controls group accents and arrival prediction.

Continuous audio proposes recurring intervals; subsequent sound supports or
weakens them. Habitat spectral changes and actual onsets supply phase and
intensity. Confidence starts at zero and expresses evidence for recurrence,
not a probability that beat phase is correct. Missing audio contributes no
evidence. `metric()` follows the resulting confidence-weighted beat.
The ListenerTwin continues to estimate its own beat from presentation events.
Offline rendering uses deterministic observation cutoffs; playback uses the
observations available at each update.

## Keeping private opportunities: `steady()`

Missed or illegal steady opportunities advance by whole intrinsic periods rather
than shifting the grid. The initial sustained Hold onset remains immediate,
before that grid's later opportunities. At due time, the Voice compares sounding against skipping without moving the opportunity.

```rhai
let voices = place(harmonic().steady().sustain().intrinsic_rate(2.0)
    .seek_consonance().pitch_apply_mode("gate_snap"), line(220.0, 330.0).count(2));
wait(6.0);
release(voices);
```

The last timing preset wins, so `flow().steady()` selects steady and
`steady().flow()` selects flow; duration still follows its separate
rules. Timing presets are initial-only. Either order with explicit `sustain()`
keeps a Tone through skipped opportunities. Glide, GateSnap, Rearticulate, its
movement boundary, `tie` (omitted zero), `articulation_persistence` and finite
duration with `rest` retain their composition and priority. Even Glide
adds no adoption-time candidate in this preset. Live pitch-mode changes retain
the Tone without adding an onset. Replacements inherit the template's policy
and rate, with fresh phase, due, plan and performance state.

## Director-level terrain

The director shapes the rhythmic terrain, symmetric to the consonance-field
operations. These operations shape the production meter used by metric timing. They do not
set the bodily rate or recurrence candidates of entrained/steady/flow participants:

- `meter_stability(value)` — attractor depth in `[0,1]`, default `0` (neutral):
  how readily a pulse forms and how strongly its reading favors a preferred
  tempo. The tempo-preference weight equals this depth. The preferred center is
  the center of `temporal_basin`, or 2 Hz in the default band. It only deepens
  the basin for a *real* periodicity; it never fabricates a beat from non-metric
  input. This is a composer-set cultural prior of the modeled listener.
- `temporal_basin(min_hz, max_hz)` — a continuous tempo region for the emergent
  beat (the time-axis analogue of `consonance(min, max)`). It seeds the beat at
  the center and confines adaptation to the band. At stability `0`, it adds no
  center preference or restoring pull. Higher stability strengthens both the
  preferred reading and confidence-weighted center restoring. It shapes the
  terrain; it does not place a beat, and it never forces a measure. The same
  sanitized band center also sets the intrinsic starting rate of `metric()`
  Voices, independently of stability. Without a basin, that rate remains 2 Hz.

The basin is one initial condition for the whole scene. The final declaration
in the script applies to every metric Population, through `place()`. A declaration after a Population or after `wait()` still replaces
that same scene value; it does not change the basin during performance.

A body's initial tempo is distinct from a beat confirmed by the listener.
Metric Voices start with independent phases. Until sound corroborates a beat,
confidence and shared-beat attraction remain zero. After confirmation, the
coupling follows the heard beat. Respawn starts from the Population's
resolved template; it does not inherit the parent's current clock rate.
Entrained/steady/flow bodily priors and explicit `pulse(rate_hz)` are unaffected.

From a cold start, a regular 4 Hz input in the default band is read as a 4 Hz
beat at stability `0`, and as duple subdivision of a 2 Hz beat at stability `1`.
Depth also retains its effects on forcing and evidence presence; it is not an
independent control of tempo weighting. Confidence remains zero until the first
interval hypothesis is corroborated. Chance corroboration in irregular input
can still produce temporary rises afterward.

## Per-voice presets and modifiers

The Tier-1 presets take **no rate argument**: `metric()`, `entrained()`, `steady()`, and
`flow()`. Configure them on a `PopulationSpec` before calling `place()`. Their timing relationships and coupling defaults are listed above;
`intrinsic_rate(hz)` sets the bodily scale of participating presets.

Do not confuse `.entrained()` with `brain("entrain")`. The preset controls
onset timing; the brain selects the voice's articulation life and metabolism.
They are independent and may be used together. See
[Population — A Persistent Unit of Voices](voice_life.md).

Per-voice modifiers refine these intentions:

- `entrainment(strength)` — influence in `[0,1]`, with the per-preset meanings
  and omission values in the table. Within a selected preset, it changes
  that influence rather than selecting another preset.
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — the voice's
  metrical job. `accent` emits a stronger onset that drives the shared meter
  harder, so a recurring downbeat can seed an emergent measure.
- `microtiming(amount)` — a signed metric beat-phase offset in `[-0.5, 0.5]`.
  A value of `0.5` targets the half-beat. It does not assign fixed positions to
  entrained/steady/flow participants.
- `measure_accent(amount)` — weak onset emphasis following the detected measure,
  in `[0,1]` (default `0`). The onset strength multiplier is
  `1 + 0.35 * amount * confidence * cos(phase)`. The phase origin follows observed
  accents, and no detected measure means no emphasis. It changes strength, not
  onset scheduling, and works independently of `rhythm_role`.

```rhai
meter_stability(0.85);     // attractor depth: how readily a pulse forms
temporal_basin(1.8, 2.2);   // tempo region the emergent beat gravitates toward

let beat = harmonic()
    .metric()
    .rhythm_role("accent")  // a strong onset that drives the shared beat
    .cycles(2);

let entrained = harmonic()
    .entrained()
    .cycles(2);

let drift = harmonic()
    .flow()
    .cycles(1);

let offbeat = harmonic()
    .metric()
    .microtiming(0.5)       // a half-beat offset reads as syncopation
    .cycles(2);

place(beat, at(110.0));
place(entrained, consonance(110.0).peak().count(3));
place(drift, consonance(300.0, 1200.0).count(4));
place(offbeat, at(220.0));
wait(12.0);
```

Except for explicit sustain plus participation described below, calls on the same axis are last-write-wins: the last timing mode and the last
duration mode determine the final behavior. `cycles(n)` counts the intrinsic
bodily period for entrained/steady/flow and the effective beat period for metric timing;
flow retains its threefold hold multiplier. Modifiers are remembered and
applied when their matching preset is selected, so `entrainment(0.8).metric()`
and `metric().entrainment(0.8)` are equivalent. The same applies to
`duration_range(...).adaptive_duration()` and the reverse order.

## Setting the body's intrinsic rate

`intrinsic_rate(hz)` sets the founder Voices' bodily scale for `entrained()`, `steady()` or
`flow()`. Its period is `1/hz`. This is a body initial condition, not an
onset-interval command or a common beat. Entrained timing adapts its participation
period to heard recurrence without changing the intrinsic value. Steady keeps fixed opportunities; flow uses that
value for irregular renewal. All can omit events, and entrained/flow can wait;
different intrinsic rates do not guarantee permanently different onset rates.

Set it on a PopulationSpec before `place()`. Either order with the
participation preset has the same meaning. It cannot be changed during performance.
Replacements read the Population template, with fresh initial due and performance
state rather than a parent's adapted period, plan or articulation policy.

The same bodily period sets finite note lengths (`cycles`, `duration_range`),
`rest(periods)` and the representative hold used for overlap prediction. Changing
the rate therefore changes note lengths and rests in seconds together. Flow keeps
its threefold hold multiplier; ADSR release remains in seconds. Explicit sustained
notes without a finite duration stay alive until release.

```rhai
let slow = harmonic().sustain().entrained().intrinsic_rate(0.5)
    .seek_consonance().pitch_apply_mode("gate_snap");
let quick = harmonic().intrinsic_rate(2.0).sustain().flow()
    .seek_consonance().pitch_apply_mode("gate_snap");
let a = place(slow, at(220.0));
let b = place(quick, at(330.0));
wait(6.0);
release(a);
release(b);
```

These rates are example inputs, not recommended values. A final entrained/steady/flow
specification accepts finite values in `0.01..=200` Hz. Values outside that range
or nonfinite values are script errors before placement. The range protects
execution; it is not a musically recommended range. Flow's renewal interval has
separate limits of `0.035..=4` seconds, so extreme rates do not imply
proportionally scaled intervals. Other final timing forms, including metric,
pulse, repeat and once, warn and ignore this modifier even if its value is invalid.
Omission uses 2 Hz; metric uses the scene's basin.

`rhythm_freq(hz)` sets the Entrain articulation core oscillator and does not set
this participation scale. `proposal_interval(seconds)` controls pitch proposals;
it does not command an onset or change the body's intrinsic rate.

## Sustained sound with participation opportunities

Explicit `sustain()` plus `entrained()`, `steady()` or `flow()` keeps a Tone sounding while
its Voice continues to consider participation opportunities. Either builder order
has the same meaning: `sustain().entrained()` and `entrained().sustain()` are equivalent.
At an opportunity, choosing to sound releases the old Tone and attacks a new Tone
at the body's pitch at that sample. Waiting or omitting an opportunity retains the
ongoing Tone, so omission does not create a gap. An pitch can be re-attacked.

```rhai
let p = place(
    harmonic().sustain().entrained().seek_consonance().glide(0.03),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
release(p);
```

With the default `"glide"` mode, a newly adopted pitch target also offers its sample as a candidate for an undecided,
legal participation opportunity. The timing, acoustic-context and overlap
costs choose among this candidate and the candidates; ties retain the
candidate, and a committed plan is not moved. If the adoption is not
selected for an attack, movement glides. An adoption and clock onset at the same
sample produce one attack. A clock onset during a glide uses the current body
pitch; it does not jump ahead to the unfinished target. Release, death and removal
cancel pending requests. Emitted attacks enter the usual energy and acoustic paths once.

This uses an estimate of the note's finite footprint for participation prediction;
it does not compare an exact rendering of continuing against release plus a new attack.
It does not add a phrase-boundary detector or a pitch-interval threshold.

For this combination the omitted pitch application mode resolves to `"glide"`.
Explicit `"gate_snap"` with free pitch instead holds one adopted target until the
Voice chooses to sound at a participation opportunity. The old Tone then releases
and a new Tone attacks at that target. A later adoption replaces the pending target.
Waiting and omission retain the sounding pitch and the pending target; movement
does not glide. With tie omitted or zero, an opportunity without a pending target re-attacks the same
pitch. GateSnap offers no extra candidate at the adoption sample and uses the
participation costs and skip rule. At the same sample, adoption precedes
the opportunity; a future adoption cannot affect an earlier onset. Release, death
and removal cancel the pending target.

```rhai
let p = place(
    harmonic().sustain().flow().seek_consonance().pitch_apply_mode("gate_snap"),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
p.pitch_apply_mode("glide");
wait(3.0);
release(p);
```

Live Glide ↔ GateSnap ↔ Rearticulate switching keeps the sounding Tone and emits no extra
attack. It cancels unrendered requests and the pending attack target. On returning
to Glide, movement follows the controller's retained target at the next control
update. GateSnap and Rearticulate wait for subsequent adoptions. Calling the same mode again keeps
its pending state. Replacement Voices inherit the mode, starting with no pending
performance state. Timing presets remain initial-only.

With `pitch_apply_mode("rearticulate")`, `rearticulate_above(cents)` combines
sliding and deferred jumps. A small adoption cancels old pending pitch and starts
gliding; a large one stops sliding and defers the target to the next sounded
opportunity. With tie omitted or zero, an opportunity without a target attacks at current body pitch.
Waiting and skipping do not stop glides. The boundary uses the last actual
onset pitch, updated by every attack, and does not suppress clock attacks. Zero or
omission matches GateSnap. Unlike Hold without participation, Rearticulate here
attacks at a sounded opportunity rather than at adoption. Boundary changes preserve
the selected behavior until the next adoption. Mode changes clear unrendered
requests, pending pitch and the selected glide without changing the Tone or the
controller's target. Builder order does not change this form; replacements inherit
controls, not pending pitch, selected glide or last-onset memory.

The implicit WhileAlive default and `while_alive()` alone do not select this form.
`sustain()` alone, `entrained()` alone and `flow()` alone keep their own timing and duration rules. With `pulse()` or `repeat()`, `sustain()` still uses the
last-write-wins rule. Explicit `sustain()` plus `metric()` selects sustained
synchronization as described below. Without `rest()`, a later finite duration via
`cycles()` or `adaptive_duration()` replaces the sustained form. With `rest()`,
finite note length and sustained participation are retained independently, and
each note releases before a pause measured from actual tail completion. See
[Cutting notes and leaving a rest](consonance.md#cutting-notes-and-leaving-a-rest).

## Retaining an articulation policy

Within free-pitch sustained participation, `articulation_persistence(p)` lets a Voice
keep a private articulation policy across several pitch adoptions and participation
opportunities. It requires `sustain()` plus `metric()`, `entrained()`, `steady()` or `flow()`,
`pitch_apply_mode("rearticulate")`, and a positive `rearticulate_above(cents)`.
The value is set before placement; it is not a live Population method.

```rhai
let p = place(
    harmonic().sustain().entrained().seek_consonance()
        .pitch_apply_mode("rearticulate").rearticulate_above(50)
        .articulation_persistence(0.75),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
release(p);
```

The boundary and persistence in this example are comparison inputs, not recommended
values. The private policy starts in **Base**, using the boundary and
participation rules. **Connect** slides adopted targets of either size while a Tone
lives, and intentionally withholds opportunities that the participation policy
selected to sound. It extends the current Tone rather than recording a cost skip.
It creates no new Tone, onset recharge, or onset-derived context learning.

Small adoptions and cost skips propose Connect; large adoptions and newly selected
sounded opportunities propose Base. For `0 < p < 1`, a proposal against the current
policy keeps it with probability `p` and changes it with probability `1-p`.
Matching proposals and waiting hops do not draw again. A future planned opportunity
keeps its decision until execution. Entering Connect cancels pending pitch; the
last actual onset remains the reference, and cancelled adoptions never replay.

Persistence is **policy inertia, not a legato proportion**. It acts on Base as well
as Connect. Raising `p` need not increase Connect time or decrease the number of
attacks monotonically. Duration follows events, not a fixed number of
seconds or beats. These retained intervals are not tied to audible cadence,
contour, or rests, and do not infer phrase boundaries.

The default `0` and endpoint `1` both use the path without a private
policy or extra random draws; at `1`, birth in Base never changes. Finite inputs
clamp to `[0,1]`; positive infinity becomes `1`, negative infinity and NaN become
`0`. An unsupported final Voice specification warns and ignores the modifier,
regardless of builder order. Replacements inherit the template value and start
in Base with fresh random and performance state. A live mode change resets Base;
repeating the same mode or changing only the boundary retains the policy.
While the boundary is zero, the private path is suspended.

Finite Off, `rest()`, tail completion, gaps, release, death and removal take priority.
When no Tone lives, Connect allows the next legal onset, including the initial
onset and return from a rest. The policy survives individual notes and rests; it
does not decide whether each note should sustain or release.

## Extending opportunities without a new pitch destination

`tie(p)` sets the probability of extending a living Tone instead of re-attacking
at a selected participation opportunity without a pending target. It is an
initial composer parameter for free-pitch `sustain()` with `metric()`, `entrained()`, `steady()` or
`flow()` and explicit `gate_snap` or `rearticulate`, including a zero boundary.
Omitted tie resolves to `1` for sustained synchronization and to `0` for sustained
entrained/steady/flow participation, where the Voice can skip opportunities by cost.
Explicit `tie(0)` keeps pending-free attacks, `tie(1)` always extends eligible
Tones, and intermediate values mix the two per opportunity. Fixed pitch and Glide
remain outside tie and re-attack each metric crossing. These choices do not claim
a musical optimum.

A pending target still jumps and attacks. Small adoptions below a positive
boundary already glide without pending pitch, so those gliding Tones can extend
too: the condition is not exact pitch equality. GateSnap makes small and even
zero-distance adoptions pending, so they still attack. Tie reduces attacks to a
thinned sequence of opportunities; it neither moves the opportunities
nor changes pitch proposals, costs or skip. It is not an overall legato amount,
a phrase switch, or a response to heard phrase boundaries.

```rhai
let voices = place(harmonic().sustain().entrained()
    .pitch_apply_mode("rearticulate").rearticulate_above(50).tie(0.5),
    line(220.0, 330.0).count(2));
wait(3.0);
release(voices);
```

The boundary and intermediate probability in this example illustrate composition,
not recommended settings. Extension creates no new Tone, onset metabolism or
onset-context learning; the opportunity is consumed, without cost-skip learning
or later replay. With `articulation_persistence`, Connect withholds first without
sampling tie. In Base, pending targets attack; only pending-free opportunities
sample tie. Tie does not change Base/Connect proposals or retention. Connect's
sliding of large moves and withholding of attacks is a separate policy.

Initial onsets and legal onsets after finite Off, tail completion and `rest()`
are never blocked when no Tone lives. Release, death, removal and rest legality
take priority. The parameter cannot change during performance. Switching to
Glide suspends it; returning to an applicable mode resumes it. Builder order
is independent at `place()`; replacements inherit the template value
with fresh random and performance state. Finite values clamp to `[0,1]`, positive
infinity becomes `1`, negative infinity and NaN become `0`; unsupported final
forms warn and ignore the value. Zero has no tie state or draws, one has no draws,
and intermediate values use a separate individual stream at execution only.


## Explicit when/duration (Tier 2)

Below the presets sit explicit controls: `once()`, `pulse(rate_hz)`,
`while_alive()`, `cycles(n)`, and `adaptive_duration()` (with
`duration_range`, `duration_curve`, and `shorten_on_drop` for tuning).

## Timing as survival

`rhythm_freq(hz)` sets the Entrain articulation oscillator.
`rhythm_coupling_vitality(lambda_v, v_floor)` scales its influence by vitality.
`rhythm_reward(rho_t, "attack_phase_match")` rewards phase-matched autonomous
attacks only with `brain("entrain")` and continuous Hold phonation.
Finite gated notes and sustained coupled opportunities use ordinary consonance
recharge; an explicit reward there produces a warning and has no effect.

```rhai
let living = harmonic().brain("entrain").sustain()
    .endurance(8.0).recovery(4.0)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");
place(living, at(165.0));
wait(8.0);
```

The body's period sets finite note lengths and rests. Participating timing can
adapt its cadence without stretching that period; flow multiplies finite holds
by three. `rhythm_freq()` controls the articulation oscillator independently of
`intrinsic_rate()`, which sets participating Voices' bodily scale.

## Scaffolds (research controls)

The scaffold functions impose an *external* pulse for comparison assays:

```rhai
scaffold_off();
scaffold_shared(2.0);
scaffold_scrambled(2.0, 17);
```

They are useful for demos and assays. They are not the rhythm-composition
abstraction: use them only when explicit synchronization is essential to the demo or assay.
A dedicated beat carrier is not required for entrained/steady/flow participation.

## Sustained synchronization

Explicit `sustain()` plus `metric()` keeps a live Tone and uses the
coupling clock's integer crossings as sounding opportunities. Either builder order
has the same meaning. The shared beat is confirmed from sound; before confirmation
the Voice free-runs with its basin-centered rate and independent phase.
This combination does not place a beat from outside.

The first Hold onset is immediate. Later opportunities occur on every crossing.
With free-pitch GateSnap/Rearticulate, the default attacks at a beat with a pending
adopted destination and extends the living Tone on other beats, just like `tie(1)`.
Write `tie(0)` to re-attack on every legal beat. Fixed pitch and Glide still
re-attack current body pitch at each crossing; Glide moves between attacks.
GateSnap retains the latest adopted target for the next legal crossing. Rearticulate with boundary zero does the
same; a positive boundary cancels pending pitch and glides small adoptions, while
larger adoptions remain pending. Same-sample adoption precedes the crossing.

```rhai
let wire = harmonic().brain("drone").sustain().metric().seek_consonance()
    .pitch_apply_mode("rearticulate");
let voices = place(wire, at(660.0));
wait(4.0);
release(voices);
wait(2.0);
```

For free-pitch GateSnap/Rearticulate, an explicit `tie(p)` uses its written
probability on pending-free crossings. The omitted probability is resolved before
placement and also applies while live mode switching enters the eligible range.
Sustained entrained/steady/flow retains omission at zero; its cost policy can skip
opportunities. No live tie setter is added.
`articulation_persistence` retains Base/Connect as in participation, but only
adoptions and sounding crossings offer proposals: synchronization has no cost
skip or planned waiting. `rest()` combines explicit finite duration with tail
completion and a gap; the clock keeps advancing and omitted crossings never replay.
Role, measure emphasis and microtiming keep their meanings. Fixed pitch
is accepted just as in sustained participation; tie and persistence remain limited
to free pitch. Merely setting entrainment or a role does not select this combination.
