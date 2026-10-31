# Indefinite performance: design

Date: 2026-10-31. Source revision: `a6b197d5`. Status: revision 9. The independent
review (Astra) passed at round 8, and round 9 confirmed revision 9 with no remaining
findings. Implementation status is recorded in §0.

This design lets Conchordal perform a piece with no planned end. It covers the Rhai
API, the scenario IR, the runtime, the GUI and CLI, and the long-run defects that a
performance lasting days would expose.

## 0. Authority and accepted decisions

Implementation status (2026-10-31): IP-U1–U5 are integrated, including the U5 review fixes and U2 numerical optimization; V5 hardware recovery, V6 endurance (V6a/V6b), V7 long-run estimator precision and V8 realtime-load acceptance remain unacquired.

- Manifesto (`web/content/manifesto.md:106`): alongside finite works, Conchordal
  envisages an acoustic ecosystem running without a predetermined end.
- DCC hierarchy §1.1 (`docs/design-notes/dcc-neurocognitive-hierarchy.md:59-63`) and
  the ledger entry "Core cognitive mechanisms and long-term listener research (adopted
  2026-09-13)": indefinitely running use is a design target. Causal state updates need
  no planned duration, EOF, whole-piece completion or storage of all past events.
  Retrospective whole-piece measures belong to an explicitly bounded interval. Runtime
  and recovery acceptance for this use is unverified.
- Ledger I10 contract, row "Memory and continuing operation"
  (`docs/design-notes/technote-ledger.md:4461`): bound parameters and pending windows
  rather than accumulating a piece-wide event list; process windows without EOF.
- Author decisions (2026-10-30 and 2026-10-31):
  - A1. Two stages. Stage 1 adds an open end. Stage 2 evaluates the script
    incrementally during the performance.
  - A2. The open end is a scene-level declaration, `endless()`.
  - A3. Runtime absolute time is the existing `u64` sample tick. Script and IR time is
    `f64` seconds, converted to ticks at one boundary.
  - A4. Every other recommendation in the preceding analysis is adopted.

## 1. Goals and non-goals

Goals:

- G1. A scenario may declare that it has no planned end. The performance then
  continues until the performer ends it, `--duration` elapses, or the process stops.
- G2 (stage 2). The authored macro form may itself be unbounded: Rhai `loop` and
  `while` with `wait()` run during the performance.
- G3. A performance can run for days. Retained state does not grow with elapsed time,
  the default path has no time-dependent numerical degradation, and a recoverable
  audio-device event does not end the performance.
- G4. Finite pieces keep their semantics: the same events, in the same order, with the
  same end. Float-level differences are allowed where this design names them (§3.3,
  §6.1, §6.2, §7.1). No contract requires bit-identical waveforms across revisions
  (CLAUDE.md compatibility policy); tests that pin exact values change only for the
  named differences.

Non-goals:

- Scripts reading ecology state during the performance. Reading it would break the
  determinism argument of §5.4 and needs a separate design of the composer's layer.
- Script-level random numbers (`rand()` and similar), and concurrent unbounded lines
  (for example `every(period, f)`). Both can be added later on top of stage 2.
- Seek, timeline scrubbing, playlists, re-seeded repetition of finite pieces and
  automatic restart after a crash.
- A forgetting or retention constant for the cumulative estimators of §7. DCC §6
  requires retention to come from task comparisons, not from this design.
- Long-term listener memory (I9 and related research extensions).

## 2. Current behaviour

Verified at `a6b197d5`.

### 2.1 Script evaluation and the IR

- `ScriptHost::load_script_with_warnings` (`src/scripting/engine.rs:1982-2000`)
  evaluates the whole script, then `ScriptContext::finish`
  (`src/scripting/context.rs:400-427`) emits pending actions and appends
  `Action::Finish` at `max(cursor, last release tail)`. A script that never advances
  the cursor gets a duration of 0.01 s (`finalize_duration`, `context.rs:299-307`).
- `wait(sec)` casts the Rhai `FLOAT` (f64) to f32 (`engine.rs:934-951`,
  `context.rs:715-718`). The cursor, `TimedEvent.time`, `SceneMarker.time` and
  `Scenario.duration_sec` are f32 (`src/scenario.rs:67,101,107`).
- `parallel` rewinds the cursor to the block start for each branch, so events are
  emitted out of time order (`engine.rs:1122-1156`). It is the only construct that
  moves the cursor backwards. The Conductor sorts by `(time, order)`.
- `ScriptContext::finish` rewrites `base_rate_hz` of every already-emitted
  Synchronized `Spawn` from the final `temporal_basin` (`context.rs:404-418`). The
  last basin declaration therefore applies to all metric founders, regardless of
  where it appears (test `metric_founders_read_the_final_scene_basin_through_create_and_place`,
  `src/scripting/tests/spec.rs:2003-2037`).
- `temporal_mode` must be declared at top level before any event; the guard tests
  `scenario.events.is_empty()` (`engine.rs:963-993`).
- Scope frames (`section`, `play`, `parallel` branches) record every population and
  Flora created inside them (`ScopeFrame`, `context.rs:41`, pushes at `:172` and in
  `create_population`) and release the live ones on exit. `pop_scope` expects every
  scoped Flora entry to exist (`context.rs:278`).
- Warnings are formatted strings with IDs and are appended to `ScriptWarnings`
  (`context.rs:37`).
- rhai 1.27 is built without `sync` (`Cargo.toml:80`), so `Engine`, `Dynamic`,
  `FnPtr` and `AST` are not `Send`. No `on_progress` hook is installed.
  `EvalAltResult::ErrorTerminated` is not catchable by script `try`/`catch`
  (rhai `src/types/error.rs:355-360`, `src/eval/stmt.rs:748`).
- `validate_scenario` (`src/runtime/scenario_load.rs:43-101`) requires events, a
  `Finish`, `duration_sec > 0`, a duration not before the last event, and strictly
  increasing event order.
- The largest bundled scenario holds 115 events and 172 actions
  (`tests/scripts/listener_contour.rhai`; counted with `--compile-only` over
  `samples/`, `samples/research/` and `tests/scripts/`).

### 2.2 Wiring pre-scans and load order

`wire_runtime` (`src/runtime/wiring.rs:345-420`) scans the complete event list. The
scenario is loaded before the audio device is opened, and the runtime sample rate is
the device rate (`wiring.rs:493,530-544`).

- Voice IDs. The script assigns voice IDs in emission order
  (`context.rs:600-651`). Runtime respawn IDs start after the largest script ID
  (`reserve_runtime_ids_through`, `src/life/community.rs:281-326`). ID values change
  sound: they enter the Voice RNG seed (`src/life/voice.rs:402`), articulation
  construction (`voice.rs:425`) and crowding pairwise signs (`community.rs:868,906`).
  `tests/render_binary.rs:549-666` pins this.
- LOO observers. `loo_populations` lists populations whose spawn spec or any later
  patch enables self-exclusion (`wiring.rs:371-404`). Capture must start at birth: the
  analysis observer clones the shared stream before the birth hop
  (`src/core/analysis_worker.rs:126-159`), and test
  `pcm_loo_late_scenario_patch_keeps_history_from_birth`
  (`src/runtime/tests.rs:696-824`) requires a late enabling patch to keep history from
  birth.
- Flora. Any `PlaceFlora` enables shared-stream resource power at run start
  (`wiring.rs:278-285`). A metric Flora placement creates the shared Flora meter, which
  observes from run start (`wiring.rs:405-410`, `docs/design-notes/flora.md:513-516`,
  test `flora_meter_is_reserved_only_for_metric_and_observes_before_placement`).
- Scene-level values read at wiring: `seed`, `output_gain`, `temporal_mode`,
  `scaffold`, `control_update_mode`, `meter_shaping`.

### 2.3 Conductor and the end of a run

- `Conductor` holds all events in a sorted `VecDeque` (`src/life/conductor.rs`).
  `is_done()` means "queue empty".
- The first hop with `is_done()` sets `scenario_end_tick`, clears all Voices, releases
  Flora and shuts the renderer down (`src/runtime/phases.rs:137-144`). Phonation
  batches stop (`phases.rs:103-117`), respawn stops (`cleanup_dead(scenario_finished)`,
  `src/life/community/respawn.rs:597-600`) and the monitor warns "Scenario done but
  audio active" (`src/runtime/worker.rs:763`).
- `Action::Finish` sets `abort_requested` (`community/actions.rs:12-14`). The worker
  finishes in the same hop without Flora, or once the renderer is idle and Flora are
  gone (`worker.rs:700-714`).
- The GUI Exit button, window close and Ctrl-C all set the stop flag; there is no
  graceful end. Offline render has no duration argument and stops only at `Finish`
  (`src/bin/render.rs:16-40`). Headless runs block in `join_runtime`
  (`wiring.rs:644-657`).

### 2.4 Absolute time

`Timebase::tick_to_sec` returns `t as f32 / fs` (`src/core/timebase.rs:10`). The f32
second has an ulp of 0.24 ms at 1 h, 7.8 ms at 18.2 h, 15.6 ms (more than one 10.67 ms
hop at 48 kHz / 512) from 36.4 h, and 31 ms from 72.8 h. Users of absolute f32 time:

- Event dispatch compares f32 seconds (`conductor.rs:79`, `phases.rs:13-17`).
- ListenerTwin computes `dt` as a difference of absolute f32 seconds
  (`src/listener_twin/mod.rs:120-124`). From 36.4 h some hops get `dt = 0`. The effect
  is limited to listener beat, measure and attention (UI and report); DCC reads only
  tension.
- Scaffold Shared phase is `(time_sec * freq).rem_euclid(1.0)` in f32
  (`src/life/report.rs:1223`, via `analysis_sync.rs:200-222`). Opt-in; it changes
  generation.
- Report and UI times, runtime-event times (`community.rs:297-300`) and
  `LifeAccumulator.elapsed_sec` (f32; stops growing after about 9 h,
  `src/life/telemetry.rs:47-49`).

### 2.5 Other long-run degradation on the default path

- `MeterNetwork.beat_cycles: f32` (`src/core/meter.rs:370`) accumulates beats without
  wrapping (`:553,:693,:699`). At 2 Hz the per-hop increment is biased by +0.7 % at
  1 h, +9.9 % at 10 h and −27 % at 24 h, and the sum freezes after 49.7–99.4 h. It
  feeds measure phase (`:753-759`, `:850-853`), which reaches generation through
  `NeuralRhythms` (`src/core/modulation.rs:68-79`) when `measure_accent` is set, and
  the report.
- `Recurrence` accumulates `time += dt` and counts `steps` since construction
  (`meter.rs:186-188`). Its roundoff tolerance `4γ|time|/window` with `γ ≈ steps·ε`
  (`meter.rs:264-267`) grows with the square of run time: about 0.01–0.05 at 24 h and
  0.5–2.4 at 7 days. Coverage lies in [0, 1], so after a few days every period
  candidate ties and the shortest wins. The continuous (T2) variant already uses `u64`
  ticks and is unaffected.
- `HistoryEnergyPrediction` (`src/core/history_prediction.rs`) keeps the inverse of
  `G = 0.01I + Σxxᵀ` by Sherman–Morrison updates (`:223-241`) over 57 features, one of
  which is the constant 1 and eight of which are coverages near 1. In the resulting
  near-singular direction the inverse has eigenvalue near 1/0.01, while the other
  eigenvalues fall as 1/N. Rounding of order `N·ε·100` in the recursion overtakes the
  1/N eigenvalues near N ≈ 10⁷ updates, about a day at one update per hop.

### 2.6 Termination and growth

- Any cpal error callback ends the run: `callback_errors_total > 0` sets `exiting`
  (`worker.rs:262-268`), and `finish_run` records any error count as a failure
  (`worker.rs:349-355`). On ALSA an xrun calls the error callback and then recovers, so
  one xrun ends the performance today. There is no stream rebuild path
  (`src/audio/output.rs:150-247`).
- `Community::populations` (`community.rs:97`) gains one entry per placed population
  and never removes it. With crowding enabled, every substep copies all of them into
  `advance_scratch.population_visibility` (`community.rs:779-789`).
- `--report` keeps `rhythm_onsets`, `rhythm_observations` (about 24 MB/h) and
  `listener_beat_confidence` for end-of-run summaries (`report.rs:18-20,626,683,694`).
  Two summaries, the 1/f slopes and the late-span mean, need the whole series.
- `--profile` fails after 100 000 hops, about 17.8 min (`src/runtime_profile.rs:9,215`).
- Offline render writes WAV through hound, whose data size is `u32`: about 12.4 h at
  16-bit mono 48 kHz. In a release build the size wraps and the header is corrupted.
- Hop-path log lines are written synchronously to stdout (`src/main.rs:12-18`). The
  "Cleaned up N dead voices" line appears once per hop with a death
  (`community.rs:1113-1126`); its rate is set by the ecology.

### 2.7 Verified bounded

The three-way audit found every other long-lived container bounded by live counts,
rings or time windows. This includes:

- `Community::voices`, `death_observed` and Flora
- renderer tones and capture slots
- LOO and Flora observers in the analysis worker, and spectral history
- temporal-cognition captures, contexts and ledgers, and ListenerTwin contours
- the GUI histories (5 s windows) and egui plots
- all production channels, which are bounded

`phonation_clock.phase` is f64 with rounding at crossings, and timing counters are
`u64`. The LOO recycle timeout (`phases.rs:418-430`) is not a long-run risk:
`wait_for_analysis` already waits without limit, so the packet has returned before the
receive except during an early exit.

## 3. Time representation (both stages)

### 3.1 Runtime clock

The worker clock remains `frame_idx: u64` and `Tick = u64`. Every absolute comparison
uses ticks. `Timebase::tick_to_sec` returns `f64`. Changing its return type makes the
compiler list every call site, which is the audit: each site either keeps ticks, uses
f64 seconds, or converts a difference of ticks to an f32 interval.

At 48 kHz, `u64` ticks cover about 3.8·10¹⁴ s. f64 seconds stay finer than one tick for
about 9·10¹⁰ s.

### 3.2 Script and IR

- `ScriptContext.cursor: f64`; `wait()` no longer casts to f32.
- `TimedEvent.time`, `SceneMarker.time` and the finite duration are `f64` seconds.
  The script never needs the sample rate, which is known only after the device opens
  (§2.2).
- Ordering stays `(time_sec, order)`, as today.
- Conversion to ticks happens in one place, the runtime's reading of score items
  (§5.5): `tick = round(sec · fs)` (`Timebase::sec_to_tick`, changed to take f64).
  Rounding is monotone, so ordering by seconds and comparing by ticks agree. Summing
  f64 seconds in the cursor and rounding once avoids the one-directional drift of
  rounding each `wait` to ticks (for example `wait(60.0/97.0)` at 48 kHz drifts 0.28
  tick per beat, about 0.8 s per day).

### 3.3 Dispatch

At a hop starting at tick `a`, the runtime dispatches every score item with
`round(time · fs) ≤ a`: the current rule, evaluated exactly. Items due in the same hop
keep their `(time, order)` order. An event that lay within f32 rounding of a hop start
can move by one hop. This changes the waveform of some existing pieces at sample
level; no contract requires their old waveform.

### 3.4 Local quantities

Intervals, durations, phases and ages stay f32 (CLAUDE.md DSP policy). `hop_sec: f32`
is derived once from `hop / fs`.

### 3.5 Sites converted

- ListenerTwin: `dt` from the difference of ticks, `(tick - prev_tick) as f32 / fs`.
- Scaffold Shared phase: `((tick as f64) · freq / fs).fract()`. The phase error stays
  below 10⁻⁶ cycle while `tick·freq/fs < 2³²`, about 68 years at 2 Hz.
- `Recurrence`: §6.2.
- UI and report times (`SimulationMeta.time_sec`, ListenerState time, runtime events,
  death times, hop timing) become f64. JSON readers see more digits only.
- `LifeAccumulator.elapsed_sec` becomes f64 and `lifetime_count` becomes `u64`.
- Log `[t=…]` values are printed from f64.

## 4. Stage 1: open end

### 4.1 Rhai API

`endless()` declares that the piece has no planned end.

- Valid once, at top level, before any playback event and before time advances.
  Otherwise it is a script error. Requiring it before any event keeps every `Spawn` of
  a piece in one ID mode (§5.7).
- Stage 1 already enforces the stage 2 rules for endless pieces, so a stage 1 endless
  script stays valid in stage 2:
  - the header rule (§5.6)
  - the late-LOO error (§5.8)
  - the `parallel` error (§5.1)
- When the script returns, `finish()` emits pending actions but appends no `Finish`.
  The ecology keeps sounding until the performer ends it (§4.4).
- Stage 1 still evaluates the script eagerly at load. An unbounded loop hangs at load
  exactly as it does today; stage 2 removes this limit.
- A piece that releases everything and returns stays silent until ended. There is no
  automatic end; a silent continuation may be intended.

The registry entry in `src/scripting/docs.rs` states this in the hover text. The
generated `rhai-defs/conchordal.d.rhai` and API reference are regenerated with
`gen_rhai_defs`.

### 4.2 IR

`Scenario.duration_sec` becomes `Scenario.end: ScenarioEnd`, with
`Finite { duration_sec: f64 }` and `Open`.

- `validate_scenario` keeps its finite rules.
- For `Open` (stage 1, eager), it requires no `Finish`. The "Scenario has no events"
  check stays.
- In stage 2, validation is split by when the information exists:
  - header values when the header is set
  - each item when it is emitted (well-formed actions; time not before the watermark,
    and `order` increasing, both by construction)
  - completion rules ("Scenario has no events") only when the script returns

  An empty prefix is allowed: `endless(); wait(1); place(…)` sets its header with no
  item yet.
- compile-only prints `end: open` instead of a duration.

### 4.3 End state

`Conductor::is_done()` is replaced by one runtime transition and one query.

- `WorkerState::begin_finish(now_tick)` is the only way a run starts finishing. It sets
  `finish_begun` and applies `Action::Finish` to the Community (`abort_requested`). Two
  callers use it:
  - the Conductor, when it dispatches a scripted `Finish` (finite pieces)
  - the end sequence (§4.4)
- Everything that used `is_done()` reads `finish_begun` instead: the end processing
  (`phases.rs:137-144`), stopping phonation batches, stopping respawn
  (`cleanup_dead(scenario_finished)`) and the monitor warning.
- After `begin_finish`, the run ends by the existing predicate (`worker.rs:700-707`):
  - at once without Flora (outside the render prototype)
  - otherwise once the renderer is idle and Flora are gone
- `pending_events()`, the queue length, is used only by the UI.

In a finite scenario `Finish` is the last event by construction (largest time, largest
order), so `finish_begun` becomes true exactly when `is_done()` did, and behaviour is
unchanged. In an open scenario an empty queue no longer clears the Voices or stops
respawn.

### 4.4 End request

An end request is a graceful end. Its sources are the GUI End button, the first Ctrl-C
(GUI and headless), and `--duration`. The Ctrl-C handler (`src/main.rs:35-39`) sets an
`end_request` flag on the first signal and the existing stop flag on the second.

On the first hop after the request, the worker:

1. In stage 2, closes the score (§5.10). In stage 1, drops the remaining queued
   events, including a pending `Finish`.
2. Releases each population that has live members and is not already released, with
   the fade a script `release` would use (the population template's
   `control.body.envelope.release_sec`). Releases each Flora not already released,
   with its `fade_sec`.
3. Records `EndSequence { finish_tick }` in `WorkerState`. `finish_tick` is `now`
   plus the largest remaining fade over every Voice and Flora that is fading after
   step 2, including those released earlier by the script or at the bound, rounded up
   to a hop. Voices and Flora already track their fade progress, so the runtime reads
   the remaining time. At `finish_tick` the worker calls `begin_finish`, outside the
   score. The run then ends by the same predicate as a finite piece's `Finish` (§4.3).
   Every fade has completed by then, as at a scripted `Finish` placed after the release
   tail.

Two kinds of request differ in their ordering:

- A human request (GUI End, the first Ctrl-C) takes effect at the next hop. It may end
  a watermark wait (§5.4) and discard undispatched score items. Its timing is the
  performer's.
- A `--duration` request is part of the deterministic run. At its hop, the runtime
  first satisfies the ordinary readiness predicate of §5.5: the watermark is past the
  hop, or the outbox is terminal. It then dispatches the due items, and only then runs
  the end sequence. Bound-time releases (§5.9) are therefore always applied before it,
  independent of thread scheduling.

If the audio stream is `Recovering` (§6.4) when an end request arrives, or becomes so
during the end sequence, nobody can hear the fade. The request then completes as an
immediate exit: `finish_run`, flushed reports, process exit. The worker checks
`end_request` and `exiting` while it waits for ring vacancy, so the request does not
depend on vacancy. A `--duration` end tick is not reached during an outage, because the
playhead does not advance. Such a run continues after recovery, or ends by Ctrl-C or
Exit.

A second Ctrl-C, the Exit button and window close keep today's immediate exit. A
`--duration` request is a tick (`round(duration · fs)`) and is therefore deterministic.
GUI and Ctrl-C requests take effect at the next hop. A finite piece that ends before
`--duration` ends normally. The produced length is therefore `duration` plus the end
fades (plus the renderer tail when Flora are present), which no startup check can know
in general.

### 4.5 CLI and output bounds

- `conchordal --duration <sec>`: end request at that time. It adds no disk write; the
  air-gap rule is unaffected.
- `conchordal-render --duration <sec>`: same meaning. An open scenario without
  `--duration` is rejected.
- WAV size is enforced during production, for every render, finite or open. Before a
  write would make the data chunk or the RIFF size exceed `u32::MAX`, the writer
  finalizes a valid file up to that point and fails the render with "WAV size limit
  reached at t=…". At startup, a `--duration` whose bytes alone exceed the limit is
  rejected early; that check is only a convenience.
- `--report` with an open scenario requires `--duration`. The whole-run summaries
  (1/f slopes, late-span mean) are whole-interval statistics, which DCC §1.1 places in
  an explicitly bounded interval. Their memory is then bounded by the declared run and
  the accumulators are not rewritten.
- `--profile` with an open scenario requires `--duration`. A duration whose hops alone
  exceed `PROFILE_HOP_CAPACITY` is rejected at start. The tail beyond it still triggers
  the existing truncation error; users leave room for the end fades.

### 4.6 GUI

See §8.

## 5. Stage 2: incremental evaluation

### 5.1 Scope

Incremental evaluation applies to pieces that declare `endless()`. Finite pieces keep
eager evaluation and the preloaded Conductor queue, unchanged, and compile-only still
validates them completely.

`parallel` is a script error in an endless piece. It rewinds the cursor, so a
block's items cannot be released until every branch ends; that requires buffering
without a bound, and pacing cannot apply inside a block. No bundled scenario uses
`parallel` (only the book's example does). Concurrent unbounded lines need a future
construct (§1 non-goals), which is the place to design concurrency for endless pieces.

### 5.2 Script thread

A dedicated script thread creates and owns the `Engine` and `ScriptContext`. The
engine is not `Send`, so it never leaves this thread. Only score items, the header,
warning updates and the terminal state cross, through the score outbox (§5.5).

### 5.3 Pacing

The runtime publishes `pace_limit_sec` (`AtomicU64` holding f64 bits). Its value is
`(playhead_tick + lead_ticks) / fs` and is updated once per hop, so the script never
needs `fs`. Until the runtime starts, the value is 0, so load-time evaluation stops at
the first wait that advances time.

At every cursor-advancing `wait()` in an endless piece, the script thread:

1. advances the watermark (§5.5);
2. while `cursor > pace_limit_sec`, records `parked_cursor_sec` and parks.

Before parking it re-checks the limit and `script_stop`. `std::thread::park` keeps a
wake token, so a wake between the check and the park is not lost.

The runtime unparks the script when `pace_limit_sec ≥ parked_cursor_sec + lead_sec/2`,
so a dense script is woken at most once per half lead. Since `parallel` is excluded
(§5.1), every advance of the cursor passes through this wait.

`lead_ticks` is 64 hops (0.68 s at 48 kHz / 512). It is an engineering constant, not a
tolerance.

- Lower bound: one wake latency of a normal-priority thread plus the compute of one
  half-lead segment.
- Upper bound: memory, through the items committed ahead (§5.5).
- Correctness does not depend on it (§5.4).
- V6b records the smallest observed margin between the watermark and the playhead,
  which decides whether it must grow.
- Changing it later changes one constant.

### 5.4 Runtime synchronization and determinism

Before dispatching at a hop starting at tick `a`, the runtime requires, under the
outbox lock, either a terminal outbox (§5.5) or `round(watermark_sec · fs) > a`. Every
item not yet committed will have a time at least the watermark, and rounding is
monotone, so no item due at or before `a` can still arrive.

While the condition fails, the worker waits like `wait_for_analysis`: it releases the
lock, sleeps 200 µs, re-locks and re-reads. On each iteration it also checks `exiting`
and a human `end_request`; either ends the wait (§4.4). A `--duration` request does not
end the wait (§4.4). This lockstep is
the same discipline the worker keeps with analysis (`MAX_LANDSCAPE_LAG_FRAMES = 1`,
`src/runtime/mod.rs:51`).

The runtime never dispatches past the watermark, so the dispatched sequence of
`(time, order, actions)` is a function of the script and seed alone. It is identical in
realtime, offline render and compile-only. Other realtime nondeterminism (analysis lag
0 or 1) is unchanged.

Overflow: if the worker is waiting for the watermark while the script is parked on a
full ring (§5.5), neither can proceed. The worker returns the credit of every item it
has consumed before it enters the wait, so at that point it holds no item. When the
script parks on credit, it records the pending item's requirement (actions and bytes)
in the outbox. Overflow is declared only when all of these hold under the lock:

- the worker is in the wait;
- it holds no unreturned credit;
- a pending requirement is recorded;
- that requirement exceeds the credit available now.

The last check makes the predicate independent of whether the producer has yet run
after a credit return. A pending item that fits now is admitted when the producer runs,
and no overflow is declared. This happens exactly when the items between two
dispatch hops exceed the ring's action or byte capacity. That is determined by the
score and the hop size. The runtime closes the score with the error "score items within
one hop exceed the ring capacity at t=…". The condition does not depend on timing, so
the error occurs at the same score point in realtime, render and compile-only, given
the same `fs` and hop.

Realtime watchdog: if an audio device is attached and the worker has waited for the
watermark longer than one lead (0.68 s of wall time) without that overflow, the script
cannot keep pace with real time. Examples are a loop that never advances the cursor and
a script slower than real time.

- The runtime closes the score with the error "script fell behind the performance at
  t=…" (§5.10).
- Underruns during the wait are counted as usual.
- The threshold reuses the lead: the script is granted one lead as its scheduling
  slack.

Offline render has no wall-clock watchdog, which keeps it deterministic. There, a loop
that never advances the cursor hangs, as such a loop hangs at load today.

### 5.5 Score outbox

```rust
struct ScoreOutbox {                 // in Mutex<_>, with a Condvar
    header: Option<SceneHeader>,
    ring: VecDeque<ScoreItem>,        // capacity C, reserved once; never grows
    watermark_sec: f64,               // every item not yet in `ring` has time >= this
    credit: Credit,                   // actions and bytes available (capacity C, B)
    unreturned: Credit,               // charged to items the worker popped, not yet consumed
    pending: Option<Credit>,          // requirement of the item waiting for admission
    script_parked_full: bool,
    worker_waiting: bool,             // the worker is in the watermark wait (§5.4)
    terminal: Option<Terminal>,       // Finished(Ok) | Failed(ScriptError) | Closed(reason)
    warnings: BTreeMap<WarningKey, WarningEntry>, // coalesced by key (§5.11)
}
enum ScoreItem {
    Event { time_sec: f64, order: u64, actions: Vec<Action> },
    Marker { time_sec: f64, order: u64, name: String },
}
```

Script side:

- Without `parallel`, items are emitted in `(time, order)` order at the current cursor.
  Each item is pushed when emitted, under the lock, in O(1).
- When the ring lacks credit (actions or bytes) for the next item, the thread records
  `pending`, sets `script_parked_full` under the lock and parks. Admission of that item
  and closure of the score both clear `pending` and `script_parked_full`. Credit returns when the worker has
  consumed an item (below), and the worker unparks the script at each credit return
  while `script_parked_full` is set. The script re-checks credit under the lock after
  every wake, so a wake between a pop and the return of its credit only makes it park
  again until that return.
- A cursor-advancing `wait` sets `watermark_sec` to the new cursor.
- Script return sets `terminal = Finished(Ok)` (watermark +∞).
- Every change notifies the condition variable.

Runtime side:

- Each hop, under the lock, the runtime pops due items from the front and reads
  `watermark_sec` and `terminal` together. This atomicity makes the watermark a
  statement about received items.
- Neither side allocates for the ring. Actions are allocated by the script thread when
  it builds them, and the worker frees them when dispatch consumes them, as today.
- Dispatching a marker sets `current_scene: Option<String>`. The UI reads that value
  instead of scanning a marker list. In open scenarios the report writes markers at
  dispatch; finite scenarios keep writing them at start.

Capacity. All actions due in one hop are applied within that hop, so a hop holding
more actions than the worker can apply within one hop budget cannot be performed in
real time anyway. The ring therefore counts actions, not items, and also counts bytes:

- `C` actions. Fixed at implementation from a measurement: the number of the cheapest
  action (an `UpdatePopulation` patch) that a release build applies within one hop
  budget on blanche, rounded down to a power of two. The measurement and the value are
  recorded in the ledger entry.
- `B = C · size_of::<Action>()` bytes. Each pushed item is charged its retained bytes:
  the inline size of its actions plus their heap payload (mode-ratio arrays, names,
  marker strings), computed by `ScoreItem::retained_bytes()`. `B` is the memory of `C`
  actions without payload, so payload-heavy items fit fewer per hop.
- An item counts its action count against `C` (a marker counts one) and its bytes
  against `B`. The ring's slot count `C` is reserved once.
- A single item whose actions exceed `C` or whose bytes exceed `B` can never fit. It is
  a deterministic script error at emission: "score item exceeds the ring capacity".
  This includes a scope cleanup that releases more than `C` entities at once, and it
  applies before the header as well (§5.6).
- Retained bytes are charged by capacity, not length (`Vec::capacity`,
  `String::capacity`), because capacity is what stays allocated.
- An item's charge is released when the worker has consumed it (applied its actions),
  not when it is popped. The worker returns the charge under the outbox lock, in O(1),
  and unparks a producer parked on credit. Items the worker holds therefore stay within
  the ring's `C` and `B`.
- Builders check the limits as they append. `emit_pending_actions`, `pop_scope` and
  every other builder of an item stop with the single-item error at the first action or
  byte beyond the limit. An item under construction therefore never exceeds `B`.
- Score storage outside the ring is bounded too:
  - At most one item exists outside the ring at a time: the one under construction,
    which waits for admission while the script is parked on a full ring.
  - Items emitted before the header are held in the ring as not yet visible, under the
    same accounting.
- Retained score memory is therefore at most `2B` plus `C` slot headers: the ring with
  the worker-held items, plus one item under construction. Values the
  composer holds in Rhai variables are outside this bound (§5.12). Outside overflow,
  items ahead of the playhead reach at most one wait step past `pace_limit_sec`.

Terminal states and waiters:

- `Finished(Ok)`: the script returned.
- `Failed(error)`: the script raised an error.
- `Closed(reason)`: the runtime closed the score (§5.10).

Each waiter has a predicate:

- Load waits until `header.is_some() || terminal.is_some()`. A failure before the
  header fails the load. A closure before the header cancels it.
- The runtime and the compile-only driver dispatch while
  `terminal.is_some() || round(watermark_sec · fs) > a`.
- After `Finished` or `Failed`, items still in the ring dispatch normally, as if the
  watermark were +∞. The compile-only driver stops only when the ring is empty. Only
  `Closed` discards items.
- The script, when parked, wakes on unpark and re-checks `script_stop`.

Setting the header, a push, a watermark advance, a terminal state and a closure all
call `notify_all`.

### 5.6 Header rule

In an endless piece, these declarations must come before time first advances:
`seed`, `output_gain`, `temporal_mode`, `scaffold_*`, `control_update_mode`,
`meter_stability`, `temporal_basin`, `harmonic_rhythm`, and
`observe_tonal_context()`. `endless` itself comes before any event
(§4.1). A later declaration is a script error: "… must be declared before time
advances in an endless scenario". Stage 1 already enforces this for endless pieces
(§4.1).

- The thread sets the header when the cursor first advances, or when the script
  returns. Wiring waits for it (§5.5) before building the runtime; `output_gain` is
  needed to create the audio output.
- Items emitted at cursor 0 are held in the ring as not yet visible until the header
  is set, so that `Spawn`s among them receive the final-basin rewrite. Later `Spawn`s read the
  header basin when emitted. The basin cannot change afterwards, so the outcome equals
  the finite semantics.
- No runtime exists yet to drain the ring before the header. Items before time first
  advances that exceed the ring's action or byte capacity are therefore a load error
  ("score items before time advances exceed the ring capacity"). The condition depends
  only on the score.
- The `temporal_mode` guard tests an `emitted_any_event` flag instead of
  `scenario.events.is_empty()`, in both modes, because streamed items are drained.

Author adoption, 2026-11-01 22:5x JST (IP-INT integration): add
`observe_tonal_context()` to this header list. An endless scene that declares it
reserves TC observation and its shared decoder table from run start. A declaration
after time advances is the existing positioned header-rule script error. Placing a
positive `follow_tonal_context` weight in an undeclared endless scene is a positioned
script error: "declare observe_tonal_context() before time advances to use
follow_tonal_context in an endless scenario". Finite scenes need no declaration;
if written, it warns and is ignored, and their existing placement pre-scan alone
determines whether observation and tables are constructed. This reservation word
is an Author rule, not an additional cognitive mechanism.

### 5.7 Voice IDs

`Action::Spawn.ids` becomes `SpawnIds`:

- `Assigned(Vec<u64>)`: finite pieces. Unchanged values and tests.
- `Allocate(count)`: endless pieces. `Community` allocates each ID at dispatch with
  `allocate_runtime_id`, the allocator respawn uses.

In offline render the ecology, and therefore the interleaving of respawns, is
deterministic, so these IDs are too. Two endless scripts that differ only after some
time get identical IDs before it, which is what `--reserve-runtime-ids-through`
provides for finite pieces; that flag still applies. In realtime the IDs share the
nondeterminism of respawn IDs.

On the script side, `PopulationState.live_ids: Vec<u64>` becomes `spawned: bool` in
both modes. The vector is never pruned and is only tested for emptiness
(`context.rs:264,335,360`), so the gating is identical.

### 5.8 Self-exclusion and Flora capabilities

- `loo_populations` moves from the immutable `WorkerConfig` into `WorkerState`.
  - Finite pieces initialise it from the pre-scan, as today.
  - In an endless piece, a population joins the set when its `Spawn` is dispatched, if
    its spec requests self-exclusion (`leave_self_out`, or Entrain Sustain
    `selection_environment`). Dispatch precedes render in the hop (`phases.rs:13`,
    then `phases.rs:420-459`), so capture starts at birth exactly as with the
    pre-scan. Flora capture already starts this way (`has_placed_flora`).
- In an endless piece, a patch that enables `leave_self_out` on a population that did
  not request it at placement is a script error: "enable leave_self_out at placement in
  an endless scenario". The observer has to be cloned at birth, and capturing every
  Voice from birth would add one analysis stream per Voice. Stage 1 already enforces
  this for endless pieces.
- In an endless piece, shared-stream resource power and the shared Flora meter are
  created at run start unconditionally. Their contract observes from run start, and a
  later placement cannot be foreseen. Cost: one `MeterNetwork` per hop plus resource
  power, measured in V6b. A consequence: non-metric Flora reports in open scenarios
  carry `acoustic_observation`. Finite pieces keep the pre-scan and its test.

Author adoption, 2026-11-01 22:5x JST (IP-INT integration): endless scenes also
reserve the shared onset-habituation observer unconditionally from run start.
Its small cost and the impossibility of foreseeing later placements give the same
reason as shared Flora resource power and the meter. Finite scenes keep their
existing positive-weight pre-scan. Measure the integrated cost in V6/V8, outside
IP-INT. TC follows the explicit `observe_tonal_context()` reservation in §5.6;
positive TC consumers register their self-exclusion history at birth. Undeclared
endless scenes construct neither TC observer nor decoder table.

### 5.9 Evaluation bound (compile-only and render)

Offline evaluation of an endless piece is bounded in script time, which keeps it
deterministic:

- compile-only of an endless piece requires `--duration`.
- render uses its `--duration` (§4.5).

A `wait` whose cursor would pass the bound behaves as follows:

1. It emits pending actions at the current cursor, as every `wait` does.
2. It sets the cursor to the bound and the watermark to the bound.
3. It returns `ErrorTerminated` with a bound-reached token, which script `try`/`catch`
   cannot intercept.

The token propagates through `play` and `section`. Their scope cleanup runs as it
already does, and emits its releases at the cursor, which is now the bound. The driver
then sets `Finished(Ok)`. Any other error propagates.

The bound is also latched, so that no script construct can resume past it:

- The first bound crossing sets `bound_reached`.
- Every later `wait` raises the token again.
- `on_progress` terminates evaluation whenever `bound_reached` is set, and that
  termination cannot be caught. Between the token's raise and the native scope
  cleanups of `play` and `section`, no script operation runs, so the cleanups always
  run first.
- `eval`, the one construct that wraps errors in a catchable form, is disabled (§5.13).

In render, those releases share their tick with the `--duration` end request. They are
dispatched first (§4.4), and the end sequence releases only what they left.

compile-only has no worker. Its driver:

- sets `pace_limit_sec` to +∞, so the script never parks for pacing;
- advances a virtual playhead hop by hop (config `fs` and hop) whenever the dispatch
  predicate of §5.5 holds;
- pops and prints due items;
- applies the same overflow rule;
- stops when the outbox is terminal and the ring is empty; `Closed` (Ctrl-C) discards
  the remainder.

It prints `end: open (evaluated through t=…)`.

### 5.10 Ending and stopping

`Engine::on_progress` checks `script_stop: AtomicBool` and terminates evaluation when it
is set. Every blocking state of the script thread is interruptible:

- Parked waits (pacing or a full ring) check the flag after each wake, and whoever sets
  the flag also unparks the thread.
- The outbox lock is held only for O(1) operations.

"Closing the score" means four steps, in this order:

1. Under the lock, set `terminal = Closed(reason)` and clear the ring.
2. Notify all.
3. Set `script_stop`.
4. Unpark the thread.

A push or a watermark advance after closure is discarded.

The end request, exit, the watchdog, overflow and load-time cancellation all close the
score. compile-only uses the same flag for Ctrl-C, and the script thread is joined on
shutdown.

The graceful `Finish` of an end request (§4.4) belongs to the runtime's own end
sequence (`EndSequence { finish_tick }` in `WorkerState`), never to the closed score.

### 5.11 Errors and warnings

- Before the header, a script error fails the load, as today.
- After the header, a script error sets `Failed(error)`:
  - Items already in the ring stay and dispatch. With `parallel` excluded, every pushed
    item is a well-formed action at its time.
  - The ecology continues (an open scenario has no `Finish`).
  - The GUI shows the message and position; headless runs print them.
- Warnings receive a static kind (for example `release_unknown`) at their call sites.
  The outbox keeps one entry per `(kind, position)`: the first full message and a count,
  updated in place. Their number is therefore bounded by the script's call sites, even
  when the messages contain changing IDs.

### 5.12 Script-side state

- Released `PopulationState` and `FloraState` entries are removed. IDs are monotonic,
  so an ID below the next ID and absent from the map is reported as inactive, as now.
- Scope frames no longer record created IDs. Each population and Flora state records
  the depth of the scope that created it. `pop_scope` scans the live maps for entries
  of the closing depth and releases them in ID order, which is creation order. This
  matches today's order and removes both the unbounded frame lists and the
  `expect("scoped Flora")` on pruned entries. Scope depth is unambiguous because
  scopes nest strictly; without `parallel` there are no sibling branches in an endless
  piece, and in finite pieces sibling branches run one after another.
- Values a script accumulates itself (for example an array grown in a loop) are the
  composer's responsibility.

### 5.13 Deterministic capability boundary (both modes)

The determinism argument (§5.4) needs the score to depend only on the script and the
seed. The script also must never block the thread outside the waits of §5.3. The
scenario engine therefore removes the standard capabilities that break either:

- `eval` is disabled (`Engine::disable_symbol("eval")`). It also wraps errors in a
  catchable form, which would let a script swallow the bound token.
- `import` has no module resolver (`DummyModuleResolver`). Reading files at run time
  would make the score depend on the file system.
- `sleep` and `timestamp` (Rhai standard packages, `lang_core.rs:138`,
  `time_basic.rs:41`) are overridden by natives that raise "not available in
  scenarios; use wait() for score time". `sleep` blocks uninterruptibly, and
  `timestamp`/`elapsed` read the wall clock.
- A test asserts that each of these is rejected. Precedence of the overrides over the
  package functions is checked by the same test.

This applies to finite pieces as well: no bundled scenario uses these capabilities, and
the same rules keep finite evaluation deterministic. `print` and `debug` remain; they
write to the log and do not affect the score.

## 6. Long-run robustness (both stages)

### 6.1 `beat_cycles`

Wrap it with `rem_euclid(12.0)` after each update. The lcm of `MEASURE_RATIOS =
[2, 3, 4]` (`meter.rs:80`) is 12, and 12/m is an integer, so both reads (`:753-759`,
`:850-853`) are mathematically unchanged. No read uses the integer part beyond 12.
`rem_euclid` is needed because corrections can make the value negative. Bits change
after the first wrap, with better precision.

### 6.2 `Recurrence` timestamps

`Recurrence` takes the hop start tick and `fs` and computes timestamps directly, with
no running sum:

- `time = (start_tick + hop) / fs`
- `onset_time = (start_tick + frac · hop) / fs`

Each timestamp then carries a fixed number of roundings, so the summation term
`4γ|time|/window` with `γ ≈ steps·ε` is replaced by a per-value bound
`c·ε·|time|/window`. The constant `c` is derived from the operations in `coverage` and
`accept`, and the existing `(8n+32)ε` term stays. `steps` leaves the bound.

The callers pass ticks: the production and Flora meters pass the hop start tick, and
ListenerTwin passes its generated frame id × hop. This aligns the Events variant with
the continuous variant, which already uses `u64` ticks. Near-tie decisions inside
existing piece lengths can change bits, because the old bound was looser.

### 6.3 Population pruning

`cleanup_dead` removes a `RuntimePopulationState` once it is released and has no live
member. The population also leaves the LOO set and the crowding visibility map.
Populations with no members that are not released are kept, because a later `Spawn` for
the same ID may refresh them.

Bounded state under repeated placement therefore depends on ownership. Every placement
inside an unbounded loop must end in an explicit `release` or in the cleanup of an
enclosing `section` or `play`. A loop that keeps placing populations it never releases
grows the population maps even when their Voices expire naturally. This is the
composer's ownership, like values held in Rhai variables (§5.12). No automatic
retirement is added. The book's endless chapter states the rule, and V2 uses scoped and
explicitly released placements.

### 6.4 Audio device recovery

Error classification (`cpal::ErrorKind`) in the error callback:

- Count only, stream continues: `Xrun`, `DeviceChanged`, `RealtimeDenied`.
- Lost stream: `DeviceNotAvailable`, `StreamInvalidated`, `HostUnavailable`,
  `DeviceBusy`.
- Any other kind: counted. A rebuild follows only if the liveness rule fires.

Liveness rule: the stream is also treated as lost when
`AudioCallbackCounters.callback_frames_total` (`src/audio/output.rs:143`) has not
advanced for longer than the ring buffer's duration. Once the buffered audio is
exhausted without a callback, the device has stopped pulling audio.

Stream states:

- `Running`.
- `Recovering { since, attempts }`. Entered on a lost stream. The supervisor retries
  opening the device: first at its own cadence, then with the interval doubling up to
  the 1 s stats cadence of `AudioMonitor`. This continues until a stream opens or the
  run ends. A PipeWire restart, a missing server or a missing device are all
  transient here.
- `Failed`. Terminal: a device opens but rejects the runtime sample rate or format
  (`UnsupportedConfig`). The runtime `fs` comes from the original device (§2.2), so a
  different rate cannot be accepted. The run ends with an error.

Supervisor:

- GUI: `App::logic`, once per frame.
- Headless: `run_headless` replaces the blocking `join_runtime` with a 50 ms poll
  loop, the cadence of the GUI's `watch_for_close` (`src/app.rs:224`). The loop checks
  the worker, the stop flag, the end request and stream health.

The callback state (ring consumer and limiter) lives in a wrapper whose `Drop` returns
it through a recycle channel, as `PcmPacket` already does (`src/core/stream/loo.rs:25-29`).
The state therefore comes back both when an old stream is dropped and when a failed
build drops its closure. The supervisor holds it between attempts.

The worker keeps its producer. During an outage it blocks on ring vacancy, so the
ecology pauses and then resumes. While blocked, it still observes `exiting` and
`end_request`; an end request during `Recovering` completes as an exit (§4.4).

Failure accounting:

- The exit on any error count (`worker.rs:262-268`) is removed.
- `finish_run` no longer records counted errors as failures (`worker.rs:349-355`).
  Only `Failed` is a run failure.
- The GUI shows the xrun and rebuild counts.

### 6.5 Logging

Hop-path lines whose rate the ecology sets move to `debug`: the per-hop "Cleaned up"
line and per-respawn body-build lines. Lines whose rate the score sets stay `info`
(Conductor actions). Monitor lines are already rate-limited. No non-blocking writer is
added unless V6 shows stdout stalls.

### 6.6 Report-only accumulators

`SlidingPlv` (report-only, death-record PLV) recomputes its sums from the buffer once
per window length. The refresh is exact, and its cost is O(1) amortized. Other report
accumulators are bounded by `--duration` (§4.5).

## 7. Cumulative estimators

`HistoryEnergyPrediction` (RLS-equivalent ridge regression, fixed ridge 0.01, 57
features) and the scalar sums of `EnergyForecastLearning`
(`src/core/temporal_expectation.rs:24-28`) and `comparison_square/cross` run on the
default path. They feed participation through `TemporalForecast`, whose energies are
f32 (`temporal_expectation.rs:486-501`).

The ledger (`technote-ledger.md:3310`) records the cumulative statistics as engineering
choices, not identified cognitive retention. DCC §6 requires retention to come from
elapsed-time and intervening-material comparisons, not from forecast count. This design
therefore keeps the estimators' definitions and changes only how they are computed.

### 7.1 Numerical implementation

Implementation:

- `G = 0.01I + Σ_issued xxᵀ` and each `b_h` (`response_cross`) are accumulated with
  compensated (Neumaier) summation in f64. The accumulation error then stays about
  `ε·|G|` instead of growing like `ε·N²/2` for N comparable terms.
  - At the 1/0.01 eigen-direction, the naive error overtakes the ridge near N ≈ 10⁷
    issues, about a day at one issue per hop.
  - The compensated error does not overtake it within about 4·10¹³ issues.
- The inverse is no longer propagated. A Cholesky factor `L` of `G` is kept:
  - Operation counts use multiply-adds. A rank-1 Cholesky update costs about
    n² = 3 249 for n = 57. A full factorization costs about n³/6 ≈ 30 900.
  - Each issue applies a rank-1 update.
  - Every R issues, `L` is recomputed exactly from the compensated `G`. R comes from
    cost balance: n³/6 spread over R issues equals one rank-1 update when R = n/6, so
    R = ⌈57/6⌉ = 10. The amortized cost is about two rank-1 updates per issue.
  - Each estimator offsets its refactorization phase by its observer index modulo R.
    This spreads load: with M observers, a hop carries about ⌈M/R⌉ refactorizations,
    not M.
  - A clone schedules its first refactorization at its own phase offset, not at the
    parent's deadline. Births therefore add no refactorization to their hop. A cloned
    factor carries at most 2R − 1 updates once, before its own phase applies; the V7
    criterion is output-based and covers that case.
  - Update error therefore accumulates over at most R updates and never across a run.
  - `projected` and the forecast correction use two triangular solves.
- Cadence: observations run at 100 Hz (`window_len = fs/100`,
  `temporal_expectation.rs:776-780`), independent of the 93.75 Hz output hop. A hop
  therefore carries one issue per observer, occasionally two.
  - The observers are the shared history and the `OwnSoundHistory` of each tracked
    Voice (`temporal_expectation.rs:170,282`, `schedule_renderer.rs:547`).
  - Amortized, with 64 Voices (M = 65 observers): about 65 × 2 × 3 249 ≈ 0.42 M
    multiply-adds per hop.
  - Steady-state peak hop: two issues per observer (130 rank-1 updates ≈ 0.42 M) plus
    about ⌈2·65/10⌉ = 13 refactorizations (≈ 0.40 M), about 0.8 M multiply-adds,
    before solves.
  - A birth hop additionally copies each clone (§7.1, about 330 KB more than today per
    clone). V8 measures both through p99 and over-budget hops, with birth hops
    included.
  - It is validated with the declared load matrix (V8).
- Transactional updates. An issue or refactorization writes into a scratch copy
  (`G`, its compensation terms and `L`), one scratch per thread that runs estimators,
  reused. It commits by swapping only after the whole operation succeeds. A failure
  discards the scratch, so the committed state remains the last successful issue.
- Factorization failure: `G` is positive definite by construction. If an update or a
  refactorization fails anyway (a non-positive pivot or a non-finite value), that
  estimator freezes:
  - The committed `G`, compensation and `L` are those of the last successful issue.
    `b_h` is kept as of the failure; later observations no longer update it, and the
    failed issue's later completion is never added.
  - The frozen pair is one instance of the estimator's definition, with the issued set
    up to the last success and the completed set up to the failure. Its forecasts stay
    coherent.
  - A status (`frozen at issue N`) and a counter appear in the report and the System
    panel.
  - A forced-failure test fails an update after partially writing the scratch and
    checks that the committed state is unchanged.
- Scalar ratios (`feature_square`, `residual_cross`, `comparison_square/cross`) also
  use compensated sums. The `EnergyForecastLearning` weight `1 − clamp(cross/square)`
  (`temporal_expectation.rs:90`) cancels when the ratio is near 1. It is computed as
  `clamp((square − cross)/square)`, the difference formed from the compensated `(hi,
  lo)` parts by error-free summation. This is a stable evaluation of the retained sums.
  It cannot recover information the accumulation itself lost, so it guarantees a
  relative error of order ε only with respect to those sums. If V7 finds a decidable
  failure in a scalar weight, these sums move to full double-double accumulation (an
  error-free sum at every step), the scalar counterpart of matrix refinement. With naive sums the relative error `N·ε/2` reaches half an f32
  ulp near N = 2²⁹ updates (about 66 days); compensated sums do not.
- Clone (`OwnSoundHistory::new` at each Voice birth) copies `G`, its compensation
  terms, `b_h` with compensation, and `L`.
  - `L` replaces the inverse at the same size (26 KB).
  - The additions are `G` (26 KB), its compensation (26 KB) and the compensation of
    `b_h` (275 KB), about 330 KB more per clone.
  - The clone already copies the inverse, `b_h` and the issued pending records. Its
    cost per birth is part of V8.
- Rewind (`clear_pending(true)`) resets `G` to `0.01I`, the compensation terms and
  `b_h` to zero, and `L` to `0.1I`.

Acceptance. The adopted object is the estimator's definition (ledger §9.3.37:
`G = 0.01I + Σxxᵀ`, `b_h`, `ê_h = max(0, e_t + x_tᵀG⁻¹b_h)`). The invariant this rework
protects is that production delivers that definition's forecasts at the precision the
consumer receives them.

- Measured outputs: the f32 outputs of each changed estimator.
  - The history-candidate energies of `HistoryEnergyPrediction::forecast`
    (`history_prediction.rs:242-258`).
  - The comparison mixture weights of `preview_comparison` (`:261-284`).
  - The weights `EnergyForecastLearning` issues (`temporal_expectation.rs:84-94`).

  The forecast that `TemporalForecast` delivers to participation is a fixed function of
  these and of the unchanged recurrence candidate, so its fidelity follows.
- Tolerance per value: `tol = ulp_f32(max(|v_ref|, f32::MIN_POSITIVE)) + 2ε₆₄·S`.
  - The first term asks the production value to equal the exact value rounded to f32,
    within one ulp.
  - The second term is the rounding that any f64 evaluation of the definition's final
    expression incurs. `S` is that expression's operand scale: `|e_t| + |x_tᵀw|` for a
    candidate energy before the clamp, and 0 for the weights. The weights' final
    operation is a ratio of accurately formed operands (including the complement
    above), already covered by the first term.

  This is a statement of numerical fidelity, not a claim that decisions cannot change:
  a decision whose margin is below the tolerance is decided by rounding in any
  implementation.
- Reference (test build): double-double arithmetic (about 106 bits).
  - It accumulates `G`, `b_h` and the scalar sums from the same raw inputs as
    production. Cancellation therefore affects only production, and the criterion sees
    its effect at the output.
  - It solves by a direct double-double Cholesky factorization.
  - It carries a running error bound `u_ref` for each output, summing:
    - the accumulation bound (2⁻¹⁰⁴ times the sum of absolute terms);
    - the solve bound (κ₂(G)·2⁻¹⁰⁴·|w|, κ₂ from the reference factor);
    - the bounds of the final dot product and sum, including cancellation in
      `e_t + x_tᵀw`.
- Decision per value, with measured difference `d = |v_prod − v_ref|`:
  - Pass when `d + u_ref ≤ tol_min`.
  - Fail when `d − u_ref > tol_max`.
  - Otherwise inconclusive. Such values are never counted as passed, and they are
    recomputed with a higher-precision reference (quad-double).

  `tol_min` and `tol_max` are the tolerance evaluated with the smallest and largest f32
  ulp over the interval `[v_ref − u_ref, v_ref + u_ref]`. This covers a tolerance that
  changes at an f32 binade boundary within the reference uncertainty.
- Short runs: the existing estimator fixtures, and the history observed during renders
  of the bundled samples (the current producer's actual output). All existing
  estimator and participation tests must pass.
- Long runs: V7.
- If a decidable comparison fails in the matrix path, production adds iterative
  refinement whose residual `b − G·w` is computed in double-double from the compensated
  `G`. A failure in a scalar weight moves its sums to double-double accumulation. V7
  decides both.
- The adoption's own evidence agreed with independent batch solves below 8.63·10⁻¹⁰
  over 1,008 components. It is retained as historical evidence, not used as a
  tolerance: it is an observed maximum on recordings that were wiped with `target/` on
  2026-10-07.

The change alters forecast bits from the first hop in every piece. This difference is
named in G4.

### 7.2 Behavioural

The adaptation gain falls as 1/N, so forecasts converge to run-long averages, and new
Voices clone the shared statistics (`OwnSoundHistory::new`). This already happens
within finite pieces at their length. Changing it is a temporal DCC design unit under
the "Memory and continuing operation" obligation, outside this design. It is recorded
as open in the ledger entry (§10).

## 8. GUI and CLI behaviour

- `SimulationMeta` carries:
  - `time_sec: f64`
  - `end: Option<f64>` (finite duration, or `None` for open)
  - in stage 2, a script status: evaluating through t, done, error with position, or
    stopped
  - a warning count
  - audio xrun, slow-hop, rebuild and recovering-state indicators
  - the estimator factorization-failure count
- Top bar:
  - Finite: today's progress bar.
  - Open: elapsed time as `H:MM:SS` (with days past 24 h), "open end", and the scene
    name.
  - `format_time` handles hours and days.
- Buttons: Start (NotStarted), End (Playing; graceful end, §4.4), Exit (immediate).
  Space still starts.
- `PlaybackState` gains `Ending`, between an end request and `Finish`.
- System panel: xrun, slow-hop and rebuild counts, the recovering state, population
  count, script status and deduplicated warnings.
- Headless: the first Ctrl-C prints "ending (fade)…" and requests an end; the second
  exits immediately.
- Not added: seek, timeline, scene list and long-term plots.

## 9. Validation

Each item states the decision it changes, why principle or saved data cannot answer
it, and whether it discriminates.

- V1. Aged-state regressions (U1, U2).
  - `beat_cycles`. The fix keeps it in [0, 12) after every write: the per-hop increment,
    the seed and both corrections. Tests check:
    - the invariant over long update sequences;
    - the per-hop measure-phase advance at the largest reachable value (just below
      12) against ω·dt within f32 tolerance;
    - the negative-correction case.

    The current code fails the invariant, because its value grows without bound.
  - `Recurrence`. A test-only constructor injects multi-day `time` and `steps` into the
    current type and shows the period tie for a fixed onset pattern; this reproduces
    the defect. With the fix, time derives from ticks. The same onset pattern at start
    ticks 0 and 2⁴⁰ (about 265 days) must give the same period choice.
  - Components that read absolute time are tested at tick offset 2⁴⁰: ListenerTwin `dt`,
    exact dispatch ticks and the scaffold phase.

  Decision: whether each fix removes the time dependence. It discriminates, because
  every case fails on the current code.
- V2. Bounded state (U3, U5). An endless scenario with a place/release loop inside a
  `section`, respawn churn and Flora place/release runs offline for three loop cycles.
  The sizes of `Community.populations`, the LOO set, the outbox ring occupancy at the
  hop following each cycle end, the script maps and the warning entries must be equal
  at the ends of cycles 2 and 3. Decision: whether retained state is independent of
  elapsed time. Growth of one entry per cycle is detected.
- V3. Stage 2 equivalence (U5). Every bundled scenario that satisfies the endless rules
  (no `parallel`; header declarations before time advances) is evaluated two ways:
  - eagerly, as a finite piece
  - through the script thread and outbox, with `endless()` added and the compile-only
    driver

  The two `(time, order, actions)` sequences must agree up to the script's return.
  Spawn IDs are compared as counts, because the endless piece allocates them at
  dispatch. Dedicated fixtures cover markers, sections, releases, header declarations
  and a full ring.

  Decision: whether the thread, commits, pacing-free driver and header preserve the
  score. The unchanged eager path is the reference.
- V4. Determinism, bounds and overflow (U5):
  - An endless piece rendered twice with `--duration` must give identical pre-synth
    state and report (the existing render determinism scope).
  - A loop whose actions within one hop exceed `C`, and another whose payload exceeds
    `B`, must fail with the overflow error at the same score time in render and
    compile-only. An item larger than the ring must fail at emission.
  - Interleaving tests drive two sequences:
    - pop, then a producer retry, then consumption;
    - a credit return, then the worker's next-hop readiness check before the producer
      runs, then the producer retry.

    In both, the producer must park on missing credit, wake at the credit return and
    proceed, with no overflow declared.
  - A bound reached inside `section` with a long pending `wait` must emit the cleanup
    releases at the bound and dispatch them before the end request.
  - A bound reached inside `play` and `section` must end evaluation normally, with
    their scope releases emitted.
- V5. Device recovery and ending (U3):
  - Unit tests: error classification, the state transitions, and an end request while
    `Recovering` (it completes as an exit).
  - Once on blanche:
    - Restart PipeWire during a GUI performance. Playback must resume, the rebuild count
      must increment, and the run must not fail.
    - Stop PipeWire, then press End in the GUI and press Ctrl-C in a headless run. Both
      must exit promptly.

  Decision: whether the recovery path works on the target host. cpal backends cannot be
  exercised in unit tests.
- V6. Soak. Telemetry is bounded: once per `AudioMonitor` stats interval (1 s), one
  summary line gives:
  - RSS from `/proc/self/statm`
  - hop-time mean and max for the interval
  - slow-hop, underrun and rebuild counts
  - voice, population and Flora counts, and births and deaths in the interval
  - in stage 2, the smallest watermark margin

  No per-hop record is kept, so the profile capacity does not apply.

  The scenario keeps live counts stationary: a fixed respawn capacity and a fixed set of
  Flora, with place/release cycles. Criterion: fit RSS against time over the second
  half. The 95 % interval of the slope must include zero or lie below it, and the slope
  is reported in bytes per day. In an indefinite run any positive slope exhausts
  memory, so a positive slope is a leak to investigate whatever its size.

  Decision: whether the technote may state continuing operation as supported, quoting
  the tested duration. Unknown leaks in dependencies and allocator behaviour cannot be
  derived from principle.
  - V6a, after U1–U4: stage 1 (ecology, runtime, recovery).
  - V6b, after U5: adds the script thread, `on_progress`, the always-on Flora
    capabilities and the watermark margin, whose minimum decides whether `lead_ticks`
    must grow.
- V7. Estimator precision on long runs (U2).
  - Path: `conchordal --nogui --play=false --duration D`. This is the instrument's
    existing offline worker path without audio output or WAV, used by
    `evaluate_rt.py --mode offline-check`. It runs faster than real time and has no
    WAV size limit.
  - Scenario: an endless piece with a sustained drone population and a respawning
    population, so that the shared observer and one long-lived Voice observer both
    accumulate throughout.
  - Checkpoints: 10⁶, 10⁷ and 3·10⁷ issues of the shared observer (100 issues per
    second of score), that is about 2.8 h, 28 h and 83 h of score;
    D = 3·10⁵ s. They bracket the naive-error crossover near 10⁷ issues (§7.1) and
    extend three times beyond it.
  - At each checkpoint a test build evaluates the §7.1 criterion against the
    double-double reference. The reference has accumulated from the same raw inputs
    since the start, and κ₂(G) is logged.
  - Decision: a decidable failure selects the repair for its path:
    - in the matrix path (history-candidate energies), iterative refinement with
      double-double residuals;
    - in a scalar weight, double-double accumulation of its sums (§7.1).

    After a repair, the same checkpoints must pass the criterion, and V8 is rerun. The
    κ₂(G) trend also gives the horizon at which factorization would fail.

  Principle fixes the criterion but cannot say whether the implementation meets it on
  the actual feature statistics.
- V8. Realtime load (U2, U5). `scripts/evaluate_rt.py` runs on the reference build and
  on the changed build, in the same session. The matrix is the existing one: 4, 16 and
  64 Voices × sine, harmonic and modal × report off/on × DCC 0/0.25.
  - Workloads for U2, all within supported load:
    - `entrained` and `flow` as they are: participation issues forecasts from every
      Voice's surroundings.
    - A churn variant of each, with an explicit respawn policy (`respawn_random()`),
      `respawn_capacity` equal to the Voice count and
      `respawn_background_death_rate(0.1)`. Presets default to no respawn
      (`spec.rs:94`), so the policy must be explicit. Each Voice is then reborn about
      once per 10 s measurement window (the window length is the origin of the rate),
      so births clone estimators inside a window whose load is otherwise the passing
      `entrained`/`flow` load.
    - A window is accepted only if births actually occurred in it: at least as many
      births as the configured Voice count, read from the runtime's birth counter in
      the per-second summary (V6 telemetry).
  - Workload for U5: an endless variant of the churn workload whose script loop, inside
    the window:
    - places and releases a one-Voice population every `stress_sec` (the existing 5 s);
    - patches the base population every settle period (the existing 2 s).

    The script thread therefore produces items continuously under a passing load.
  - The overloaded `recovery` workload is not used for acceptance, because its overload
    phase is allowed to fail.
  - The evaluator's existing activity checks require the original live count and that
    many distinct onset-producing IDs (`evaluate_rt.py:168,305`), which churn violates.
    The churn workloads replace them:
    - the live count stays at or above the configured count, except within one hop of a
      death;
    - every population produces at least two onsets per window;
    - the birth requirement above holds.
  - Criterion: every condition without underruns on the reference stays without
    underruns and without over-budget hops over the whole window, birth hops included,
    with p99 worker hop time below the hop duration. These are the acceptance terms used
    when the estimator was adopted (ledger §9.3.37: zero over-budget hops, p99 against
    10.667 ms). The audio deadline is their origin.
  - Decision: whether the estimator rework (§7.1) and the stage 2 machinery fit the
    existing load. If a condition regresses, the change is revised before
    integration.

Focused tests also cover:

- the `endless()` rules and validation of `Open`
- the `finish_dispatched` split (finite pieces unchanged)
- the end request sequence, `EndSequence` and `--duration`
- the CLI rejections and the WAV writer guard
- the header-rule, late-LOO and `parallel` errors in endless pieces
- warning coalescing and scope release order
- `format_time`

Existing tests for finite pieces (reserved IDs, late LOO patch, final basin, Flora meter
reservation) remain valid as written.

## 10. Documentation

- Rhai book `concepts/timeline.md` (and the Japanese adaptation) gains:
  - `endless()`, ending a performance and `--duration`
  - for stage 2: loops in endless pieces, the header rule, the `parallel` rule and the
    LOO rule
  - the ownership rule for placements inside unbounded loops (§6.3), and the scenario
    capability boundary (§5.13)
- `src/scripting/docs.rs` gains the registry entry; the generated files are regenerated.
- The technote changes only when an implementation lands: end semantics, continuing
  operation and the tested duration from V6.
- A new ledger entry records this design, the decisions A1–A4, and §7.2 as open.
- CLAUDE.md's allocation notes describe the score outbox: the worker never allocates
  for the score.
- A new endless sample is an author decision after implementation. It is not part of
  this design.

## 11. Implementation units

- U1. Time representation (§3).
- U2. Numerical long-run fixes (§6.1, §6.2, §6.6, §7.1), with V1, V7 and V8 for the
  estimator. Depends on U1 for ticks in `Recurrence`.
- U3. Operational fixes (§6.3–§6.5), the output bounds of §4.5, and the per-second
  summary telemetry of V6, which V8 also reads. Independent of U1.
- U4. Stage 1 (§4, §8). Depends on U1.
- U5. Stage 2 (§5). Depends on U4.
- U6. V6a after U1–U4; V6b after U5. V3, V4 and V8 for the stage 2 machinery run
  inside U5. The value of `C` (§5.5) is measured in U5 and recorded in the ledger.

U2, U3 and U4 touch mostly disjoint files (`core/meter.rs`/`core/history_prediction.rs`;
`community.rs`/`output.rs`/`main.rs`/`writer.rs`; `scenario.rs`/`scripting`/`ui`) and
can proceed in parallel after U1.

## 12. Risks and open items

- cpal error kinds differ by backend (ALSA, PipeWire through ALSA, CoreAudio). The
  liveness rule covers misclassification; V5 checks the target host.
- On blanche, external jobs can delay the script thread. The lead absorbs ordinary
  delays; a watchdog trip is reported, not hidden.
- `on_progress` runs per Rhai operation. Its cost is one relaxed atomic load; measured
  in V6b.
- Estimator precision (V7), its load (V8) and behavioural adaptation (§7.2).
- `parallel` is unavailable in endless pieces until a concurrency construct is designed
  (§5.1).
- stdout blocking (§6.5).

## Review record

Independent review rounds by Astra (`codex exec -m gpt-6-astra -c
model_reasoning_effort=high`). Raw outputs are kept under
`.orchestration/units/IP-1/` (machine-local).

Section numbers in each round's dispositions refer to the revision that answered it.

### Round 1 (revision 1): REVISE, 2 blockers, 9 major, 1 minor

1. The watermark did not establish receipt. Fixed by the mutex outbox: committed items
   and the watermark change together and are read together, and the worker re-reads
   while it waits (§5.4, §5.5).
2. Cancellation did not reach every blocking state. Fixed: no blocking send, unpark on
   stop, the end request inside the wait loop, a defined close order, and load-time and
   compile-only cancellation (§5.2, §5.4, §5.11).
3. The evaluation bound lost later `parallel` branches, and compile-only had no stop.
   Fixed by a branch-aware bound through an uncatchable token; compile-only of endless
   pieces requires `--duration` (§5.10).
4. The script lacked the sample rate. Fixed: the script works in seconds, the worker
   publishes the pace limit in seconds, and ticks are formed only on the runtime side
   (§3.2, §5.3, §5.4).
5. Event storage was unbounded and the worker allocated. Fixed: unfinished blocks stay
   on the script thread, and the worker only pops. The absence of a count cap is now
   justified (§5.5).
6. Warning keys, scope lists and the Flora `expect` would grow or panic under pruning.
   Fixed by `(kind, position)` keys and scope depth recorded on entities (§5.12, §5.13).
7. `--duration` did not bound output. Fixed: the WAV guard is enforced during
   production for all renders; startup checks are a convenience; profile tails are
   documented (§4.4, §4.5).
8. Recovery lacked a transient state, and counted errors stayed fatal at the end.
   Fixed: `Running`/`Recovering`/`Failed`, retry policy, callback-state return on
   failed builds, and failure accounting (§6.4).
9. The estimator repair lacked a consumer criterion. Fixed: f32-ulp requirement from the
   consumer, compensated accumulation, factorization per issue (no cadence constant),
   failure handling, clone/reset and cost, V7 decision (§7.1, V7).
10. Offsets did not reproduce aged state. Fixed by injected aged-state regressions (V1).
11. The soak could not validate stage-2 costs, and telemetry was undefined. Fixed: V6a
    and V6b, bounded per-second telemetry, and a slope criterion on stationary live
    counts (V6).
12. Minor: the `NeuralRhythms` path is corrected, and the lead is restated as an
    engineering constant with its bounds and measurement (§2.5, §5.3).

### Round 2 (revision 2): REVISE, 1 blocker, 8 major, 1 minor

Answered by revision 3.

1. Repeated `parallel` blocks bypassed pacing. Fixed: `parallel` is a script error in
   endless pieces (no bundled scenario uses it). Every cursor advance is then a paced
   wait (§5.1, §5.3).
2. `parallel` consuming the bound token could loop forever. Removed with `parallel`:
   the token propagates through `play` and `section` to the driver (§5.9).
3. Storage was unbounded, and appends held the lock through large copies. Fixed:
   - a fixed-capacity ring `C`, with O(1) pushes under the lock and backpressure;
   - deterministic overflow detection, the mutual wait that occurs when more than `C`
     items fall within one hop;
   - `C` derived from the per-hop application budget, measured at implementation;
   - warning entries coalesced in place by key (§5.4, §5.5, §5.11).
4. compile-only had no pacing consumer. Fixed: the driver sets the pace limit to +∞,
   advances a virtual playhead and applies the same predicates (§5.9).
5. Terminal transitions were incomplete. Fixed: terminal states, a predicate for each
   waiter, notifications, and the closing order; the graceful `Finish` is runtime-owned
   (`EndSequence`) (§4.4, §5.5, §5.10).
6. End could stall during an outage. Fixed: an end request during `Recovering` exits
   immediately, and vacancy waits observe the request (§4.4, §6.4, V5).
7. One f32 ulp did not establish unchanged decisions. Fixed: acceptance reuses the
   adopted numerical evidence, agreement with independent batch solves within
   8.63·10⁻¹⁰ (ledger §9.3.37), applied to inputs that exist now (§7.1, V7).
8. The fallback after a factorization failure was incoherent. Fixed: the estimator
   freezes G, b and L together, reports its status, and has a forced-failure test
   (§7.1).
9. The estimator cost was understated. Fixed: the cost is counted per observer; a
   rank-1 Cholesky update is used, with refactorization every ⌈n/3⌉ issues from cost
   balance; validation runs the existing `evaluate_rt.py` matrix with the adopted
   over-budget and p99 terms (§7.1, V8).
10. The aged `beat_cycles` test failed the fix as well. Fixed: an invariant test with
    states in [0, 12), and accuracy checked at the largest reachable value (V1).

### Round 3 (revision 3): REVISE, 0 blockers, 8 major, 1 minor

Answered by revision 4.

1. Ring slots did not bound payload. Fixed: the ring counts actions (`C`) and retained
   bytes (`B = C · size_of::<Action>()`, charged with heap payload). An item that can
   never fit is a deterministic error, before the header too (§5.5, §5.6).
2. compile-only could discard queued items on completion. Fixed: after `Finished` or
   `Failed` the ring drains normally; only `Closed` discards (§5.5, §5.9).
3. The bound had no timestamp for scope cleanup. Fixed: the bound-crossing `wait`
   flushes, clamps the cursor and watermark to the bound, and then raises. Cleanup
   releases land at the bound and are dispatched before the end request of the same
   tick (§4.4, §5.9).
4. Streaming conflicted with the empty-scenario check. Fixed: header, per-item and
   completion validation are separated, and an empty prefix is allowed (§4.2).
5. The runtime `Finish` was not connected to the end predicates. Fixed: one transition,
   `begin_finish`, sets `finish_begun` for both the Conductor and the end sequence. The
   run then ends by the existing predicate, unchanged for finite pieces (§4.3, §4.4).
6. The freeze lacked a commit rule. Fixed: transactional scratch with swap-on-success;
   the frozen state is the last success plus `b_h` at failure; partial-failure test
   (§7.1).
7. An observed maximum was used as a tolerance. Fixed: the tolerance is the
   backward-error bound of the method, `(R + 1)(n + 1)·ε·κ₂(G)`, protecting the adopted
   definition. 8.63·10⁻¹⁰ is kept as history (§7.1, V7).
8. V8 missed the changed paths. Fixed: the `entrained`/`flow` workloads for
   participation forecasts, `recovery` for in-window births, and a looping endless
   `recovery` for script activity (V8).
9. Minor: R = ⌈n/6⌉ = 10 with consistent multiply-add counts, phase offsets across
   observers, and the 100 Hz observation cadence with up to two issues per hop (§7.1).

### Round 4 (revision 4): REVISE, 0 blockers, 7 major, 3 minor

Answered by revision 5.

1. Rhai's native `sleep` defeated cancellation. Fixed: `sleep` is overridden by an error
   in every scenario (§5.13).
2. `timestamp`/`elapsed` broke determinism. Fixed: overridden; `import` has no
   resolver; this capability boundary applies to both modes (§5.13).
3. `eval` made the bound token catchable. Fixed: `eval` is disabled, and the bound is
   latched and enforced by uncatchable `on_progress` termination (§5.9, §5.13).
4. The end sequence could cut existing fades. Fixed: `finish_tick` uses the largest
   remaining fade over every fading Voice and Flora (§4.4).
5. Duration ordering depended on scheduling. Fixed: human requests may end a watermark
   wait; a `--duration` request first satisfies readiness and dispatches the due items
   (§4.4, §5.4).
6. The error contract was insufficient. Fixed:
   - metric: the delivered f32 energies;
   - criterion: one f32 ulp against a double-double reference that accumulates from the
     same raw inputs and solves directly;
   - an explicit validity condition κ₂·2⁻¹⁰⁴ ≤ 2⁻³⁰ (§7.1).
7. Birth acceptance could be vacuous. Fixed: churn variants of `entrained`/`flow` put
   births inside passing windows, all hops count, and the overloaded `recovery` is
   excluded (V8).
8. Minor: phase offsets are described as load spreading, amortized and peak costs are
   given, and clones keep factor age (§7.1).
9. Minor: retained bytes are charged by capacity; the pending item and the pre-header
   items are accounted, so storage is at most `2B` plus slot headers (§5.5, §5.6).
10. Minor: V7 uses the offline instrument path without WAV, with checkpoints at 10⁶,
    10⁷ and 3·10⁷ issues bracketing the crossover (V7).

### Round 5 (revision 5): REVISE, 0 blockers, 2 major, 4 minor

Answered by revision 6.

1. The churn workloads did not enable respawn. Fixed: an explicit `respawn_random()`
   policy, and windows accepted only with observed births (V8).
2. The reference validity did not certify the output comparison. Fixed:
   - the tolerance adds the f64 rounding of the final expression's operands;
   - the reference carries a running error bound covering accumulation, solve, dot
     product and final cancellation;
   - undecidable comparisons count as inconclusive and are rerun at higher precision
     (§7.1).
3. Minor: the measured outputs are the changed estimators' own f32 outputs (history
   candidate, comparison weights, learning weights), of which the delivered mixture is
   a fixed function (§7.1).
4. Minor: charges are held until the worker consumes the item, builders enforce limits
   while appending, and the `2B` bound is restated with its parts (§5.5).
5. Minor: clones refactor at their own phase, so births add no refactorizations; the
   peak is labelled steady-state, and birth hops are measured (§7.1, V8).
6. Minor: churn-aware evaluator checks replace the fixed-ID checks (V8).

### Round 6 (revision 6): REVISE, 0 blockers, 1 major, 2 minor

Answered by revision 7.

1. Consumption-based credit had a wake race. Fixed: the producer wakes on credit return
   and re-checks under the lock. Overflow requires the worker to be waiting with no
   unreturned credit. An interleaving test is added (§5.4, §5.5, V4).
2. Minor: the `EnergyForecastLearning` complement cancelled. Fixed: it is computed as
   `(square − cross)/square` from compensated parts (§7.1).
3. Minor: the decision rule straddled the boundary. Fixed: pass if `d + u_ref ≤
   tol_min`, fail if `d − u_ref > tol_max`, otherwise inconclusive and recomputed at
   higher precision (§7.1).

### Round 7 (revision 7): REVISE, 0 blockers, 1 major, 1 minor

Answered by revision 8.

1. Overflow could still depend on scheduling after a credit return. Fixed: the pending
   requirement is recorded, and overflow also requires that it exceed the credit
   available now, checked under the lock. The credit-return → readiness → retry
   interleaving is tested (§5.4, V4).
2. Minor: the complement's accuracy is stated relative to the retained sums. A scalar
   failure in V7 moves those sums to double-double accumulation (§7.1).

### Round 8 (revision 8): PASS, 0 blockers, 0 major, 3 minor

Answered by revision 9.

1. Minor: the outbox sketch now carries `credit`, `unreturned`, `pending` and
   `worker_waiting`, and states when `pending` is cleared (§5.5).
2. Minor: bounded repeated placement is qualified by ownership. Placements in unbounded
   loops must be released or scoped; the book states the rule (§6.3).
3. Minor: V7 names both repair branches and requires the checkpoints to pass after a
   repair (V7).

### Round 9 (revision 9): PASS, no findings

The confirmation round verified the three round-8 clarifications and their consistency
with §5.4, §5.12 and V2.

