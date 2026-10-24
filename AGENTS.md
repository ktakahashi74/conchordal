# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` with focused modules:
  - `core/` (Psychoacoustic DSP primitives):
    - `log2space.rs`: Log-frequency coordinate system (octave-based).
    - `auditory_frontend.rs` / `auditory_statistics.rs`: Causal complex filterbank,
      hop boxcars, reassigned power, critical-band compression and S6 density.
    - `nsgt_rt.rs` / `nsgt_kernel.rs`: Non-Stationary Gabor Transform (streaming RT
      variant and offline kernel variant). An older FFT variant is retired to `attic/`.
    - `harmonicity_kernel.rs`: Virtual pitch/root detection via harmonic templates.
    - `roughness_kernel.rs`: Sensory dissonance calculation via interference kernels.
    - `phase.rs`: Phase wrap/diff utilities (shared across rhythm + kernels).
    - `landscape.rs`: Real-time integration of Roughness and Harmonicity.
    - `fft.rs`: Symmetric Blackman-Harris window construction.
    - `stream/potentials.rs`: Shared harmonicity and roughness scratch for streaming,
      synchronous body analysis, and exact leave-self-out evaluation.
    - `stream/loo.rs`: actual habitat PCM self-removal and per-Voice analysis histories.
    - `erb.rs`: Equivalent Rectangular Bandwidth scales.
  - `audio/` (Real-time I/O):
    - `output.rs`: cpal stream management (ringbuffer producer).
    - `limiter.rs`: Output peak limiter, soft clipping, and interval meters.
    - `writer.rs`: Disk recording (WAV).
  - `synth/` (Synthesis engine):
    - Modal resonator bank and mode-table primitives (Hz/sec domain).
  - `ui/` (Egui views and plots):
    - `windows.rs` and `plots.rs` only. The frame types the worker publishes live in
      `src/viewdata.rs`, not here.
  - `life/` (Voices and behaviors):
    - Voice/Community models, temporal cores, fields, and phonation.
    - `flora.rs`: spectral biomass dynamics, ERB cells, lanes, and sound atoms.
    - `community.rs` holds the hop-path core; the `community/` submodules split out
      action dispatch (`actions.rs`), spawn-frequency choice (`frequency.rs`),
      respawn/inheritance (`respawn.rs`), and the Kuramoto/social paths (`social.rs`).
    - `voice.rs` owns the `#[path]` submodules `articulation_core.rs`,
      `pitch_controller.rs`, `pitch_core.rs`, `sound_body.rs` (files stay flat
      under `src/life/`).
    - `phonation_engine.rs` owns the flat `#[path]` modules `phonation_clock.rs`,
      `phonation_rules.rs`, and `phonation_timing.rs`; `temporal_participation.rs`,
      `sustained_participation.rs`, and `shape_participation.rs` implement participation
      decisions and their Voice-side application.
  - `scripting/` (Rhai engine, API registration, generated docs):
    - `spec.rs`: PopulationSpec builders and resolution.
    - `placement.rs`: Placement builders and routing values.
    - `context.rs`: Script context, handles, and positioned diagnostics.
    - `engine.rs`: Registration; `tests/`: spec, placement, and context suites.
  - `runtime/` (thread wiring, worker loop, scenario execution):
    - `wiring.rs`: `wire_runtime(WiringOptions)`, the single wiring path shared by
      `init_runtime` (GUI/headless) and `run_render` (offline), plus run/join entrypoints.
    - `worker.rs`: `WorkerConfig` / `WorkerChannels` / `WorkerState`, `worker_loop`,
      `process_hop`, and finalization. One hop delegates to the modules below.
    - `analysis_sync.rs`: `wait_for_analysis`, `apply_landscape_updates`,
      `wait_for_listener`, merge, and ecology/listener habituation.
    - `phases.rs`: `advance_population`, `emit_hop_reports`, `render_and_route_audio`,
      `drive_production_meter`, and body/temporal routing.
    - `ui_frame.rs`: UI frame construction and `send_runtime_ui_frame`.
    - `monitor.rs`: `AudioMonitor` and `log_guard_meter`.
    - `scenario_load.rs`: Script compilation, validation, and compile-only diagnostics.
  - `temporal_cognition/` (declared by `temporal_cognition.rs`): ordered auditory
    evidence, grouping, body observation, arrival, proposals, and opt-in T2 integration.
  - `listener_twin/` (listener-side model).
  - `scenario.rs` + `scenario/control.rs` / `scenario/lifecycle.rs`: scenario IR,
    voice control config, lifecycle config.
  - `viewdata.rs`: worker -> UI snapshot contract (`UiFrame` and friends). It is
    deliberately outside `ui/` so `runtime/` has no dependency on `ui/`.
  - `config.rs` / `dcc_coupler.rs`: TOML config, DCC coupling.
  - `cli.rs`: Instrument CLI arguments.
  - `runtime_profile.rs`: Hop timing, run profiles, and `profile-alloc` instrumentation.
  - `bin/`: `render.rs` (conchordal-render) and `gen_rhai_defs.rs`.
- Entrypoints: `src/main.rs` (instrument) and `src/app.rs` (GUI and shutdown
  ownership); engine/thread wiring lives in `wire_runtime` (`src/runtime/wiring.rs`).
- `web/` (Astro site sources; build with `npm ci && npm run build` in `web/`) lives at the repository root, not under `src/`.
- Samples: execution-test scripts live under `tests/scripts/`. Compile-only samples stay under `samples/`.

## Build, Test, and Development Commands
- Build (debug): `cargo build`
- Run (release): `cargo run --release` (Recommended for real-time DSP performance)
- Tests: `cargo test` (required after `src/` changes; run once on the final revision).
- Format: `cargo fmt --all`
- Lint: `cargo clippy --all-targets -- -D warnings`
- Verify all targets: `cargo check --all-targets`
- Test all targets: `cargo test --all-targets`
- Check examples explicitly: `cargo check --examples`
- Paper figures/manuscript flow is out of scope for this repository.

Choose commands for the affected targets; this list is not a per-task checklist.
`cargo test` also compiles the targets it covers. Do not precede it with redundant
build/check commands unless a distinct target or an actual failure needs them.
Clippy remains mandatory before a commit. Standalone prototypes outside the crate
use their own focused tests rather than a full crate build.


## End-of-Task Test Record

When a task modifies code under src/, finish by running the tests and recording the
status. `just test` runs the command below and writes both record files; `CARGO`
can select a wrapper such as cargo-slot. Other recipes include `just fmt`,
`just clippy`, `just gen_rhai_defs`, and `just mdbook`.

Run tests with full output and backtraces enabled, and record the exit code in the
same shell invocation. `$?` does not survive a separate shell — if you split these
into two commands, `test_status.txt` will report `exit=0` no matter what happened.

```bash
set -o pipefail
( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt
echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt
```

- test_report.txt must contain stdout + stderr of cargo test
- After a task that changes `src/`, test_status.txt must exist even if tests fail.

## Progress and Validation Scope

- For ordinary, reversible changes, proceed from accepted decisions directly to
  implementation and relevant checks. Separate design, Source review, independent
  review and evidence sealing are required only by an explicit plan or a concrete
  unresolved problem, not by default for every small repair.
- Fix acceptance criteria before implementation. Add a gate only for an uncovered
  requirement or an actual defect; do not recursively make historical recovery or
  supplementary documentation a prerequisite for an unrelated repair.
- Use focused tests while editing, then run the mandatory checks and full test
  record on the final revision. Repeat checks when code changes, a check fails or
  an unresolved concern requires it, not merely at each reporting boundary.
- Reuse inactive isolated worktrees and build caches when source identity and
  isolation remain intact. Preserve old acquired artifacts and test reports.
  Ordinary repairs need concise change and test records, not repeated sealing of
  every fixture and wrapper log. Scientific acquisitions retain their registered
  provenance, resource limits and acceptance requirements.
- Report implementation, scientific acceptance and author audition separately.
  Preserve the original plan's scope and real integration dependencies; do not
  turn every research or audition item into a gate for every ordinary repair.
- Escalate only unresolved author choices. An accepted decision does not need
  another approval before its prescribed implementation.
- When changing a long validation pipeline, exercise the affected path with a
  bounded case using the current producer's actual output shape before launching
  the full matrix. Reuse an unchanged successful case; synthetic mocks alone do
  not establish compatibility with the actual producer.
- Reuse deterministic work over immutable inputs within a run. Preserve the
  original checks, ordering and complete witness occurrences; do not repeatedly
  normalize or decode the same publication prefix merely for another decision.
  Keep large retained witnesses on disk rather than duplicating Python trees.
- Count progress by completed implementation behavior and closed registered
  criteria. Reviews, pins, plans and growing output are supporting work, not
  additional completion units. Treat a missing implementation algorithm as an
  unresolved design problem, not a queue of more verification artifacts.
- Estimate remaining time from completed comparable work and explicit remaining
  units. Do not extrapolate an unfinished case's elapsed time as its duration;
  report unresolved design and author waiting time separately.
- Numerical precision follows the consumer's adopted acceptance criteria. Do not
  promote an exact optimization's intermediate float bits, branch choices or
  evaluation order into an additional requirement for every predictive fast path.
  Reordered or approximate arithmetic may use the existing error budget, measured
  end to end against the unchanged reference; do not spend the whole budget anew
  at each stage. Preserve identity, clocks, routing, known/unknown status, declared
  support, finite output and the registered controls and resource requirements.
- Keep exact arithmetic where a concrete invariant depends on it. The adopted
  single-round closed-amplitude law retains its existing finite-zero certificate
  and bounded evaluator. Reuse that evidence; do not repeat its universal sweep
  for native integration or require analogous proofs of unrelated arithmetic.
  Existing exact scientific controls and explicitly scoped regression tests
  remain in force; they do not impose waveform compatibility on a new renderer.
- Before adding any evaluation, numerical or subjective, to a plan, state three
  things: which decision its result changes; why principle, the consumer's
  decision sensitivity and saved data cannot already answer it; and that it can
  discriminate the options. Without a decision that changes, do not add it.
- A principle-layer mechanism (a published model adopted as is) is accepted by
  numerical agreement with that model, the registered causal difference and
  real-time operation on the declared load. Subjective evaluation selects free
  parameters that neither principle nor a consumer fixes. It is not an acceptance
  condition for the principle layer, and it runs once, on the configuration it
  will apply to.
- Every strict precision, tolerance or load target names the consumer decision
  or invariant it protects. One that cannot is removed; it is not kept for lack
  of proof that it is unnecessary. Derive tolerances from the principle, from
  the error that changes the consumer's decision, or from the declared supported
  range.
- Where a proposal is enough, do not require the best: a Voice acts on local
  perception and reaction. Do not build exhaustive candidate tables or exact
  pre-hearing predictions unless a consumer needs them.


## Air-Gap Protocol
- The `conchordal` binary (instrument) MUST NOT write audio to disk
  in any build profile. Performances are ephemeral by design (manifesto).
  This is upheld by absence, not by a guard: `cli.rs` exposes no output-path
  flag, and `main.rs` never calls `run_render`. There is no regression test —
  if you add a disk-write path to the `conchordal` binary, nothing will stop you.
- Offline WAV rendering is provided by the separate `conchordal-render` binary
  (`src/bin/render.rs`), which shares the core engine but is not the instrument.
  The air-gap policy does not apply to `conchordal-render`.
- Do not add `--wav` or any disk-write capability to the `conchordal` binary.

## Coding Style & Naming Conventions
- **Comments**:  All comments must be in concise English.  Do not use Japanese in code comments.
- **DSP Efficiency**: Prefer `f32`. Minimize allocations in `worker_loop` and the
  cpal callback. Reuse scratch or pre-allocated ringbuffers. Audit the actual call path;
  the following is a conditional budget, not an exact allocation census.
  - `render_and_route_audio` (`src/runtime/phases.rs`) creates two `Arc<[f32]>`
    chunks for the presentation and habitat buses, shared with WAV/listener/analysis
    channels as needed. When a scenario has self-exclusion consumers, `process_hop`
    also transfers one recycled packet of per-Voice habitat PCM. Capture storage is
    allocated at birth and returned by the analysis worker; steady-state capture
    has no per-Voice allocation. Recycling backpressure counts toward worker time.
  - `apply_landscape_updates` (`src/runtime/analysis_sync.rs`) allocates the observed
    field `Arc` for `GeneratorModel::observe_consonance_field_level` only after analysis
    or valuation updates with an available analysis frame.
  - `drive_and_apply_habituation` (`src/runtime/analysis_sync.rs`) obtains an owned
    harmonicity projection from `potential_h_from_log2_spectrum`. Habituation is enabled
    by default. The ecology path runs per hop, and `observe_listener_analysis_results`
    runs the same projection per received listener frame; the default GUI has both
    paths. Explicitly disabling habituation skips these projections.
  - ListenerTwin groups bin masses into local-maximum components with cached mass,
    component-mass and ascent-owner buffers. Initial observation and frequency-space
    changes may allocate; ordinary observations reuse those buffers.
  - `build_runtime_ui_frame` (`src/runtime/ui_frame.rs`) clones the Landscape and
    builds spectrum, Voice, phase, and temporal/body snapshots. `send_runtime_ui_frame`
    rate-limits ordinary GUI frames with `UI_MIN_INTERVAL`; terminal delivery retains
    one pending snapshot for retry. Headless and offline render pass `ui_tx = None`
    and construct neither initial nor runtime UI frames.
  - Reporting allocates only when enabled, across `emit_hop_reports`, participation
    context/decision collection in `advance_population`, prediction/body/outcome paths
    in `render_and_route_audio` and `drive_production_meter` (`src/runtime/phases.rs`),
    listener contour/hop timing in `process_hop`, and `finish_run`
    (`src/runtime/worker.rs`). Do not treat `emit_hop_reports` as the whole report cost.
  - `merge_latest_analysis_results` (`src/runtime/analysis_sync.rs`) clones frequency
    spaces only on Log2Space reconfiguration. `idle_silence` (`src/runtime/worker.rs`)
    is allocated before the loop. The old analysis-lag warning allocation is gone;
    `AudioMonitor` (`src/runtime/monitor.rs`) aggregates slow-hop warnings at stats cadence.
  The `life` layer has its own conditional allocations. The prediction scan in
  `TerrainPredictor::predict_consonance_field_level_at` (`src/life/generator_model.rs`)
  builds a Vec/Arc when extrapolating two observations; `cache_pred_next_gate` reuses
  the gate result until invalidated. `Community::advance_substep_sequential_current`
  (`src/life/community.rs`) computes rotating indices directly, without an order Vec.
  `Community::respawn_on_new_deaths` (`src/life/community/respawn.rs`) returns before
  Vec/BTreeMap construction if no new death exists. `dying_ids` in
  `Community::apply_background_turnover` allocates only when a death is selected.
  Preserve reused `Community` buffers (`advance_scratch`, `gate_open_scratch`,
  `dead_id_scratch`), pitch proposal scratch (`src/life/pitch_core.rs`), and
  `PhonationEngine` buffers (`scratch_candidates`, `scratch_merged`, `scratch_grid`,
  `scratch_field`, `src/life/phonation_engine.rs`). Clear and refill them.
  Flora's own advance/render/capture has zero steady-state allocations, including lane births and retirements; placement, reconfiguration, and reporting may allocate.
  Flora capture allocates one hop buffer at placement and one recycled packet entry
  at first capture; existing storage is reused through fade and retirement. Flora
  power publication owns a scan and allocates. Reports include resource/consumption
  scans and lane state at seconds and lifecycle changes; their allocations are
  outside the renderer contract.
  Mixed/listener NSGT power and its spectral history are enabled once at wiring
  only for GUI, reports, temporal observation taps or mixed-bus Flora resources.
  With none of those consumers the NSGT state does not advance; gap publication
  still waits the same sample-counted window. Voice observers continue to skip it.
  The analysis worker (`src/core/analysis_worker.rs`) is a separate thread. Streaming
  and synchronous analysis share front-end/potential scratch; `AnalysisStream::process`
  (`src/core/stream/analysis.rs`) returns a borrowed Landscape. The worker clones only
  at publication (every Ordered frame or the last Latest frame), attaching a spectral
  history snapshot. This still allocates; the former per-hop count is obsolete.
  A scenario reserves observers for populations with pitch/environment self-exclusion,
  including a later enabling patch. The renderer captures those Voices by ID and
  generation from birth. Each analysis observer subtracts actual own habitat PCM
  before the auditory front end, using a clone of the mixed stream's pre-birth
  history. Both histories
  reset on gaps. Processing uses reused scratch; publication owns per-Voice H/R and
  score scans and therefore allocates. Flora has a separate power observer for each instance. It subtracts actual own
  habitat PCM, copies the mixed pre-birth raw/power history, and reuses the FFT
  and shared sparse power weights without evaluating H/R. SourceKind separates
  Voice and Flora identities. Voice observer clones skip NSGT and disable the mixed stream's
  resource tracker because they consume only H/R. The auditory filterbank, hop
  boxcars and ERB-context scratch allocate at construction and reuse storage.
  Body skip advances filters and hop statistics; selected updates apply S4–S6 once.
  Flora power publication requires full-window coverage; gaps hold timestamped
  consumption and geometry changes discard it. Allocation
  census with `profile-alloc` (NSGT/history enabled, 48 kHz, hop 512, window 16384, 690 bins, Ordered,
  one hop per batch) measured 19 allocations / 50192 bytes without LOO and
  23 / 58808 with one Voice LOO observer. Its first packet used 79 / 586496,
  including observer construction and test-only assay storage. A separate
  64-hop test measured zero allocations / zero bytes for the selected auditory
  front end, including body skip and intermittent updates.
  These counts exclude input PCM ownership and UI cloning. Births and
  gap/reconfiguration resets are outside the steady-state processing claim.
  Self-exclusion observers run sequentially on one analysis thread, so sustained
  analysis cost above the hop budget makes the audio worker wait and can interrupt
  output; cost estimates must include retiring observers.
  The shared density-normalization floor applies to general analysis as well as LOO,
  independently of Flora placement or self-exclusion consumers.
  Use `--features profile-alloc` when measuring or asserting allocations; without it,
  allocation counts are unavailable. Runtime measurements live in `runtime_profile.rs`.
- **Naming**: `snake_case` for modules/functions, `CamelCase` for structs/traits.

## Anti-Bloat Rules
- Enforce YAGNI strictly: do not add abstractions before a second concrete use appears.
- Handle only errors that can actually occur on the current path.
- Do not add comments that only restate code; keep comments for intent/constraints only.
- Inline helpers unless they are used in 3 or more places.
- Prefer plain structs and existing types over new nominal wrapper types.
- Keep the crate's external surface at what `tests/`, `src/bin/` and `examples/`
  actually consume. Most submodules under `core/` and `life/` are `pub(crate)`
  (see `src/core.rs`, `src/life/mod.rs`); `src/lib.rs`, `src/scenario.rs` and
  `src/scripting/mod.rs` restrict their own submodules the same way. Widen a module or
  item to `pub` only when an out-of-crate consumer needs it, and say which one.
- Shared scalar helpers live in `src/core/float.rs` (`sanitize01`, `finite_or`,
  `sanitize_nonnegative_finite`, `unit_gaussian`). Call them instead of respelling the
  formula; the documented exception is `core::roughness_kernel`, whose Gaussian
  spelling is pinned bit-exact by its own tests.

## Testing Policy

- Reusable auditory consumer acquisitions and their input/reference tools are
  described in [auditory validation](tests/research/auditory-validation.md).
- **Inline tests** (`#[cfg(test)] mod tests` in the same source file) are for module-internal logic and private APIs.
- **Integration tests** (`tests/` directory) are for public API and cross-module behavior; treat them as black-box specs.
- **Crate-internal test modules** hold larger suites that require private APIs:
  `src/life/tests.rs`, `src/scripting/tests/{mod,spec,placement,context}.rs`, runtime
  `tests.rs` / `shutdown_tests.rs`, and nested temporal-cognition `tests.rs` files.
  Flat `#[path = "*_tests.rs"]` modules (for example `phonation_engine_tests.rs`,
  `shape_policy_clock_tests.rs`, and `core/meter_hierarchy_tests.rs`) remain attached
  to the owning module for private access. Use these forms only when that access is
  needed; keep public/cross-module black-box specifications under `tests/`.

## Architecture Notes for Agents
- The core perception model is **Landscape**. It ingests PCM through a causal complex auditory filterbank (bandwidth factor
1/4), channel-dependent hop boxcars, frequency reassignment and critical-band
loudness compression. Its Log2Space-aligned density feeds two potentials:
  1. **Roughness (R)**: Amplitude fluctuations within critical bands (dissonance).
  2. **Harmonicity (H)**: Periodicity/Template matching (consonance/fusion).
- Sound bodies (synthesis methods) are replaceable modules; the boundary is in
  `docs/design-notes/timbre.md` ("Core and body modules"). Anything the ecology
  evaluates is computed by the core from radiated sound. A body's self-report (ratios,
  spectral or energy forecasts) is a fast path only and must match the body's own
  render in the conformance tests. New ecology code must not branch on body kind;
  existing branches are migrated in Phase 3 of
  `docs/superpowers/plans/2026-09-29-timbre-synthesis.md`.



## Terminology: predictive/perceptual vs potential/representation (R/H/C)

We use two orthogonal axes. Do not mix them.

### Axis A: WorldModel layer (origin)
- **predictive** (`pred_*`): hypothesis derived from NoteBoard / internal model (zero-latency).
- **perceptual** (`perc_*`): evidence derived from actual audio analysis (NSGT/filterbank; delayed).

`perceptual` is reserved for this axis only.

The `pred_` / `perc_` prefixes are used only where both origins coexist and must be
distinguished — chiefly the WorldModel layer (`src/life/generator_model.rs`,
`src/listener_twin/`). Kernel-layer scans that have only one possible origin carry no
prefix (`r_pot_scan`, `h_pot_scan`, `c_score_scan`). Do not add a prefix that
distinguishes nothing.

### Axis B: representation (kernel output vs transformed views)
- **potential** (`*_pot_*`): raw kernel output / physical-ish quantity (unnormalized).
- **score** (`*_score_*`): kernel score (unbounded real value).
- **level** (`*_level_*`): bounded level in `[0,1]`, `level = sigmoid(beta * (score - theta))`.
- **mass** (`*_mass_*`): non-negative, pre-normalization mass used before PMF normalization.
- **density** (`*_density_*`): normalized PMF/PDF.
- **energy** (`*_energy_*`): minimization form, `energy = -score`.
- **state** (`*_state_*`): a bounded `[0,1]` view of a potential, normalized against a
  reference (`src/core/psycho_state.rs`). Distinct from `level`, which is a sigmoid of a
  score.

Potential/representation is orthogonal to pred/perc. Real examples from the tree:
- unprefixed kernel scans: `r_pot_scan`, `h_pot_scan`, `c_score_scan`,
  `c_level_scan`, `c_density_scan`, `c_energy_scan`
- state views: `r_state01_scan`, `h_state01_scan`, produced by
  `r_pot_scan_to_r_state01_scan` / `h_pot_scan_to_h_state01_scan`
  (`src/core/psycho_state.rs`)
- prefixed, where both origins coexist: `pred_c_field_level_scan`,
  `perc_c_field_level_scan`, `perc_habituation_state_scan`

## Consonance: Field / Density
- Inputs are `H01` and `R01`, sanitized into `[0,1]`.
- Field is a bilinear evaluation terrain for behavior, hill-climb, prediction, and UI:
  `field_score = a*H01 + b*R01 + c*H01*R01 + d`.
- Density is a spawn distribution derived from non-negative mass then normalized to PMF.
- Density uses a minimal family to absorb roughness-scale arbitrariness:
  `K_density(H,R; rho) = max(0, H01 * (1 - rho*R01))`.
- Implementation is unified through `ConsonanceKernel::density_with_rho(rho)` (bilinear special case with coefficients `(1,0,-rho,0)`), while density freedom is limited to `rho` only.

## Consonance Variants (Current)
1. `consonance_field_score`
- Definition: `a*H01 + b*R01 + c*H01*R01 + d`.
- Implementation: `ConsonanceKernel` in core + `src/core/landscape.rs`.
- Usage: hill-climb evaluation in `src/life/pitch_core.rs`.
2. `consonance_field_level`
- Definition: `sigmoid(beta*(score-theta))`.
- Usage: base value. Consumed as the habituation drive in
  `drive_and_apply_habituation` (`src/runtime/analysis_sync.rs`) and as the source for the
  eroded view. Voice behavior and the listener read variant 6; Flora growth reads
  this base level because its consumer–resource dynamics already supply band fatigue.
3. `consonance_field_energy`
- Definition: `-score`.
- Usage: retained for minimization view and consistency checks.
4. `consonance_density_mass`
- Definition: `max(0, H01*(1-rho*R01))`.
- Implementation: `ConsonanceKernel::density_with_rho(rho)` + `src/core/landscape.rs`.
- Usage: Flora placement reads this base mass. It also supplies the eroded mass and
  `build_consonance_density` (`src/core/landscape.rs`). Range-local spawn reads
  variant 6, not this one.
5. `consonance_density` (base PMF)
- Definition: normalized base density mass; uniform fallback on all-zero totals.
- Implementation: `recompute_consonance` in `src/core/landscape.rs` normalizes it once.
  `apply_habituation` updates effective mass without renormalizing this base PMF.
  `build_consonance_density` separately builds an occupancy-masked base PMF on request.
- Usage: currently exercised only by inline tests. Spawn placement goes through
  `SpawnStrategy::Field` (see the Rhai Spawn API section), not through this PMF.
6. `consonance_field_score_eff` / `consonance_field_level_eff` / `consonance_density_mass_eff`
- Definition: variants 1, 2 and 4 after habituation erosion is applied.
- Implementation: fields at `src/core/landscape.rs`, written by
  `Landscape::apply_habituation` (`src/core/landscape.rs`); driven per hop from
  `apply_landscape_updates` (`src/runtime/analysis_sync.rs`) when
  `[psychoacoustics.habituation]` is enabled.
- Usage split:
  - Voice behavior, listener and Voice spawn read the eroded views:
    `src/life/community/frequency.rs`, `src/life/pitch_core.rs` (through
    `Landscape::evaluate_pitch_score_log2`, `src/core/landscape.rs`),
    `src/listener_twin/mod.rs`.
  - UI and diagnostics read the base views (`src/ui/windows.rs`), so the display
    shows the un-eroded terrain.
- With habituation explicitly disabled the `_eff` views equal their base variants
  bit-exact, so reading the eroded view is always correct for Voice behavior.
  Flora reads base level/mass and unchanged `harmonicity01`; its habitat audio still
  drives habituation. Every Flora uses beat opportunities with a shared
  habitat audio-only meter and a private CouplingClock. Wiring creates the meter
  only if the scenario contains a Flora placement; it observes from run
  start, including audio before that placement. Internal onset commands
  never drive that meter. Pending windows reserve current excess once; support
  loss and release retire immediately. Changes apply at sample crossings within
  the hop. Unknown, stale or unconfirmed observations use continuous allocation without resetting
  the clock; biomass/resources stay continuous. Flora has no `metric()` opt-in or opt-out.
  Flora residual rendering defaults to false; explicit
  `.residual(true)` remains available.

## Config Keys
- `[analysis].nfft`: retained NSGT window for spectrum/UI, reports, spectral history,
  research taps, gap publication warmup and Flora resource coverage. H/R use the
  auditory filterbank independently of this window. `kernel_align` applies to NSGT.
- `[analysis].tau_ms`: S6 density smoothing (default 10 ms).
- `[psychoacoustics].use_incoherent_power`: retained NSGT power only, not H/R density.
- The fixed auditory grid includes its 8175.326 Hz upper margin channel. Configured
  and actual device sample rates must put every channel below Nyquist (integer
  rates at least 16351 Hz for the runtime's 55..8000 Hz grid); unsupported rates
  return configuration errors before analysis construction.
- `[audio].output_gain_db` (default `4.5`): fixed dB gain before cpal/WAV guards, calibrated from bundled sample peaks and the 0.98 ceiling. Analysis, listener, report and UI inputs stay at their original levels.
- `[psychoacoustics.consonance.field.kernel]`
- `a, b, c, d` (defaults: `1.0, -1.35, 1.0, 0.0`)
- `[psychoacoustics.consonance.field.level]`
- `beta, theta` (defaults: `2.0, 0.0`)
- `[psychoacoustics.consonance.density]`
- `roughness_gain` (`rho`, default: `1.0`)
- `rho` is density roughness sensitivity. Config validation requires a finite value
  in `[0, 1000000]`; the kernel guards direct internal callers separately.
- `[psychoacoustics.habituation]`
- `enabled` (default: `true`), `satiation_sec` (`5.0`), `recovery_sec` (`8.0`),
  `ref_drive` (`0.25`)
- Defaults come from `HabituationParams::default()` in `src/core/habituation.rs`.
- When `enabled = false` the `_eff` consonance views equal their base variants.

Other top-level sections exist and are defined in `src/config.rs`:
`[analysis]`, `[dcc]`, `[playback]`. They are not enumerated here —
read `src/config.rs` for their keys and defaults.

Note: every config struct carries `#[serde(deny_unknown_fields)]`, so a misspelled TOML
key is a hard error rather than a silent fallback. `#[serde(default)]` still fills in
*missing* keys. `AppConfig::load_or_default` (`src/config.rs`) returns
`anyhow::Result`: only an omitted config path with no default `config.toml` writes and
returns defaults. An explicit missing `--config` path, unreadable/malformed file,
unknown key, or invalid numeric range fails the run.

## Rhai Spawn API
Placement builders produce a `Placement` (defined in `src/scripting/placement.rs`,
registered in `src/scripting/engine.rs`), which lowers to `SpawnStrategy`
in `src/scenario.rs`.

- Range form `(lo, hi)`: `consonance` / `dissonance` / `edge` / `gap` / `random`
  build `SpawnStrategy::Field` with `FieldTarget::Consonance` / `Dissonance` /
  `Edge` / `Gap` / `Uniform` respectively (`src/scenario.rs`).
- `line(lo, hi)` builds `SpawnStrategy::Linear`.
- `at(freq)` fixes pitch unless the final spec retains explicit `seek_consonance()`
  intent and Free pitch; for that mover it sets only the initial position.
- `consonance(root)` also has a one-argument root form; `.range(min_mul, max_mul)`
  is the multiplier band around that root.
- Modifiers: `.peak()` / `.density()` set `FieldSampling` (`src/scenario.rs`;
  `Density` is the default), `.tension(t)` sets the tension degree in `[0,1]`
  (Consonance target only), `.count(n)` sets the batch size, `.spacing(erb)` sets the
  minimum separation.
- Spawn sampling is range-local in `Community::decide_frequency`
  (`src/life/community/frequency.rs`): it builds local masses with occupancy masks
  and normalizes in-range.
- If range-local total mass is zero, fallback stays in-range and remains well-defined
  (unoccupied-uniform first, then full-range uniform if all occupied).

## Naming Note
- `field_level` is a 0..1 gate/strength used by behavior and prediction.
- `density_mass` is pre-normalization non-negative mass before PMF conversion.
- For prose, "level" and "mass" are preferred names.

### Suffix convention (avoid ambiguity)
Use explicit suffixes when needed:
- `_scan`: frequency-indexed arrays (Log2Space bins)
- `_scalar`: summary values (total/max/p95 etc.)

Example:
- `r_state01_scan` (Log2Space-aligned)
- `pred_c_field_level_scan` (prefixed because a `perc_` counterpart exists)
- `loudness_mass` (a scalar summary; `_scalar` is only needed when a `_scan` of the
  same name would otherwise be ambiguous)

## Frequency Space: Log2Space invariants

We represent frequency-direction terrains as **Log2Space-aligned scans**.

### Rules
- **F1**: Any vector suffixed with `_scan` MUST be aligned to Log2Space bins:
  `scan.len() == space.n_bins()`.
- **F2**: Any function that accepts/returns a `_scan` MUST assert the invariant at boundaries.
  The O(1) length comparison is a hard `assert`, live in release as well as debug:
  call `Log2Space::assert_scan_len_named` (`src/core/log2space.rs`) so the panic
  identifies the offending scan. (A bare `assert_scan_len` also exists,
  `src/core/log2space.rs`, but it has no production callers — do not reach for it.)
  Never return a sentinel for a length mismatch — it is a
  programming error, not a runtime condition (`sample_scan_linear_log2`,
  `src/core/log2space.rs`, panics on a length mismatch). Invalid or out-of-range
  frequency coordinates return `0.0` after that length check.
  Per-element and per-bin checks stay `debug_assert`. A hop-path function that can
  degrade instead of aborting the performance may keep a `debug_assert` plus a graceful
  fallback, but it must say so: see `exact_loo_consonance_score_scan`
  (`src/life/pitch_core.rs`), which falls back to the approximate score.
  Tests must cover every boundary (`tests/log2space_scan_invariants.rs`).
- **F3**: Hz / ERB (or other psychoacoustic coordinates) are allowed as internal representations
  (e.g. oscillators, note events, intermediate grids), but any exposed terrain field is converted to
  Log2Space bins.
- **F4**: Candidate evaluation against terrains MUST use log2->bin mapping (interpolation allowed).
  Never index `_scan` with linear-Hz indices.

### Naming
- `_scan`: Log2Space bins terrain vector
- `_hz`: linear frequency array in Hz
- `_erb`: ERB-domain array (psychoacoustic helper)
- `_log2`: log2-frequency coordinate
- `_idx` / `_bin`: bin index into Log2Space scans


## Scenario Script Authoring
**Keep simple things simple, and complex things possible.**

Scenarios should be approachable for newcomers while remaining expressive for advanced use cases:

- A minimal scenario should require only essential parameters
- Complex behaviors emerge from composition, not configuration bloat
- The full parameter space remains accessible for those who need it

The composer sets three things: the modeled listener's enculturation layer (valuation
such as objective sign and roughness aversion; priors such as meter stability and tempo
basin), the initial conditions of what the ecology evolves (founder bodies and
placements), and the macro form. Composer-facing controls never alter the sensation
model (front end, `roughness01`, `harmonicity01`), the listener's in-performance state,
or traits the ecology has taken over. See `docs/design-notes/technote-ledger.md`
§9.3.56.

Dedicated beat-carrier Voices or temporal scaffolds may be used in samples only
when explicit synchronization is essential to the demonstration or assay.
Do not add them as general musical support or merely to make synchronization
easier. Routing a beat carrier only to the habitat bus does not exempt it.
The current top-level exception is `07_heartbeat.rhai`, which demonstrates
explicit synchronization. Research scaffolds require synchronization to be an
explicit experimental variable, as in `temporal_scaffolding_*`.
This restriction concerns dedicated synchronization support, not pitch-only
`anchor()` or the collective's own `metric()` / `entrained()` behavior.


## Other
- Don't touch `web/` when editing sourcecode. Don't touch `src/` when editing `web/`, unless otherwise specified.
- The public technote (`web/content/technote.md` + `.ja.md`) describes only the implemented
  state. Its final chapter is a condensed ledger of Manifesto commitments (implemented /
  partial / open) plus a short open-problems list. Design decisions, first-principles
  findings, numerical registrations and upstream revisions go to
  `docs/design-notes/technote-ledger.md` (+ `.ja.md`), which preserves the former
  chapter-9 numbering (§9.2–§9.4, §9.3.x) and anchors. Update the technote only when an
  implementation lands; never add research state or pending decisions to it.
  `samples/README.md` is an index of the études, not a rules document.
- `web/content/technote.md` (+ `.ja.md`) is hand-curated; keep it in sync with the code when core algorithms change. There is no generation pipeline.
- `docs/rhai_book` is the English Rhai API reference; build with `mdbook build docs/rhai_book`.
- `rhai-defs/conchordal.d.rhai` and `docs/rhai_book*/src/reference/api.md` are generated
  by `cargo run --bin gen_rhai_defs`. Never hand-edit them; `tests/rhai_defs_in_sync.rs`
  fails if they drift from the registry.
- `samples/` top-level études must not pin a seed; `samples/research/` assays must pin one.
  This is enforced by `tests/sample_seed_policy.rs`.
- `docs/rhai_book_ja` is a Japanese adaptation, not a sentence-by-sentence translation. Follow
  `docs/rhai_book_ja/README.md`: write natural Japanese prose, preserve canonical ontology names
  such as Voice and Population in English, and translate only ordinary explanatory language. Build with
  `mdbook build docs/rhai_book_ja` and keep its chapter paths and executable Rhai examples aligned
  with the English book.

## Compatibility Policy
- During the alpha phase, do not preserve backward compatibility by default.
- Prefer clean architecture and correct behavior over compatibility shims, aliases, or migration layers.
- Require legacy behavior or waveform equality only for an explicit contract or
  scientific control. Keep those comparisons distinct from product compatibility;
  a synthesis change does not inherit an old-waveform identity requirement.

For orchestration, pass/fail timing, or long jobs on blanche, read [operations/orchestration.md](docs/operations/orchestration.md).

## Author Decision Requests
- For each recommended option state the implementation, validation and acquisition work it
  creates, the work that disappears if it is not taken, the origin of every number or
  condition (principle, consumer, existing practice or placeholder) and the cost of
  changing it later.
- Do not put a placeholder number into a bundle of recommendations offered for adoption.
- Present independent decisions as separate items in dependency order, and ask only for
  those needed now.

## Git Operation Policy
- Never create a commit unless the user explicitly asks for a commit in that turn.
- Before creating any commit, always run `cargo clippy -- -D warnings` and confirm it passes.
- If a commit is requested, commit only the files relevant to the requested task.
- While several sessions share the main checkout, commit from a worktree branch, never in
  the shared checkout, to isolate each session's index and changes. The tracked hook
  `.githooks/pre-commit` runs `cargo fmt --all -- --check` and never changes the index;
  enable it with `git config core.hooksPath .githooks`.
- Before every push, integrate the remote first with `git pull --rebase --autostash`.
  Two machines and several agents push to `main`.
- If two appends to a record file (e.g. `docs/roadmap/**`) conflict, keep both entries,
  upstream first. Edits to the same paragraph and conflicts in code need a real merge;
  stop and report when the intent of either side is unclear.
