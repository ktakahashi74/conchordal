# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` with focused modules:
  - `core/` (Psychoacoustic DSP primitives):
    - `log2space.rs`: Log-frequency coordinate system (octave-based).
    - `nsgt_rt.rs` / `nsgt_kernel.rs`: Non-Stationary Gabor Transform (streaming RT
      variant and offline kernel variant). An older FFT variant is retired to `attic/`.
    - `harmonicity_kernel.rs`: Virtual pitch/root detection via harmonic templates.
    - `roughness_kernel.rs`: Sensory dissonance calculation via interference kernels.
    - `phase.rs`: Phase wrap/diff utilities (shared across rhythm + kernels).
    - `landscape.rs`: Real-time integration of Roughness and Harmonicity.
    - `fft.rs`: FFT wrappers and Hilbert transform logic.
    - `erb.rs`: Equivalent Rectangular Bandwidth scales.
  - `audio/` (Real-time I/O):
    - `output.rs`: cpal stream management (ringbuffer producer).
    - `writer.rs`: Disk recording (WAV).
  - `synth/` (Synthesis engine):
    - Modal resonator bank and mode-table primitives (Hz/sec domain).
  - `ui/` (Egui views and plots):
    - `windows.rs` and `plots.rs` only. The frame types the worker publishes live in
      `src/viewdata.rs`, not here.
  - `life/` (Voices and behaviors):
    - Voice/Community models, temporal cores, fields, and phonation.
    - `community.rs` holds the hop-path core; the `community/` submodules split out
      action dispatch (`actions.rs`), spawn-frequency choice (`frequency.rs`),
      respawn/inheritance (`respawn.rs`), and the Kuramoto/social paths (`social.rs`).
    - `voice.rs` owns the `#[path]` submodules `articulation_core.rs`,
      `pitch_controller.rs`, `pitch_core.rs`, `sound_body.rs` (files stay flat
      under `src/life/`).
  - `scripting/` (Rhai engine, API registration, generated docs).
  - `runtime/` (thread wiring, `worker_loop`, scenario execution).
    - `wire_runtime(WiringOptions)` is the single wiring path shared by
      `init_runtime` (GUI/headless) and `run_render` (offline).
    - `worker_loop` owns `WorkerConfig` / `WorkerChannels` / `WorkerState` and
      delegates one hop to `process_hop`, which calls the phase functions
      (`wait_for_analysis`, `apply_landscape_updates`, `wait_for_listener`,
      `advance_population`, `emit_hop_reports`, `render_and_route_audio`,
      `drive_production_meter`, `log_guard_meter`, `send_runtime_ui_frame`).
  - `listener_twin/` (listener-side model).
  - `scenario.rs` + `scenario/control.rs` / `scenario/lifecycle.rs`: scenario IR,
    voice control config, lifecycle config.
  - `viewdata.rs`: worker -> UI snapshot contract (`UiFrame` and friends). It is
    deliberately outside `ui/` so `runtime/` has no dependency on `ui/`.
  - `config.rs` / `dcc_coupler.rs`: TOML config, DCC coupling.
  - `bin/`: `render.rs` (conchordal-render) and `gen_rhai_defs.rs`.
- Entrypoints: `src/main.rs` (binary) and `src/app.rs` (GUI/Thread wiring).
- `web/` (Zola site sources) lives at the repository root, not under `src/`.
- Samples: execution-test scripts live under `tests/scripts/`. Compile-only samples stay under `samples/`.

## Build, Test, and Development Commands
- Build (debug): `cargo build`
- Run (release): `cargo run --release` (Recommended for real-time DSP performance)
- Tests: `cargo test`.  Run tests always after modifying code.
- Format: `cargo fmt --all`
- Lint: `cargo clippy -- -D warnings`
- Verify all targets: `cargo check --all-targets`
- Test all targets: `cargo test --all-targets`
- Check examples explicitly: `cargo check --examples`
- Paper figures/manuscript flow is out of scope for this repository.


## Mandatory End-of-Task Procedure

At the end of EVERY task that modifies code under src/, the agent MUST:

### 1. Run cargo tests and record status

Run tests with full output and backtraces enabled, and record the exit code in the
same shell invocation. `$?` does not survive a separate shell — if you split these
into two commands, `test_status.txt` will report `exit=0` no matter what happened.

```bash
set -o pipefail
( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt
echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt
```

- test_report.txt must contain stdout + stderr of cargo test
- test_status.txt must always exist after a task
- Do NOT skip this step under any circumstances


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
- **DSP Efficiency**: Prefer `f32`. Minimize allocations in the audio thread (`worker_loop`)
  and the cpal callback. Use `Vec::with_capacity` or pre-allocated ringbuffers.
  This is a budget, not an absolute. After the decomposition into `process_hop` + phase
  functions, the runtime layer's own steady-state per-hop allocations are exactly these
  five:
  - the two audio `Arc<[f32]>` chunks in `render_and_route_audio`
    (`src/runtime/mod.rs:1934-1935`), handed to the WAV, listener and analysis channels;
  - the terrain `Arc` in `apply_landscape_updates` (`src/runtime/mod.rs:1669`), fed to
    `GeneratorModel::observe_consonance_field_level`, and only when analysis or
    landscape params actually changed;
  - the UI snapshot built by `build_runtime_ui_frame` (`src/runtime/mod.rs:221`), reached
    through `send_runtime_ui_frame` (`:2012`) and rate-limited by `UI_MIN_INTERVAL`;
  - the report path (`emit_hop_reports`, `src/runtime/mod.rs:1781`), which early-returns
    unless `--report` attached a reporter;
  - the harmonicity projection in `drive_and_apply_habituation`
    (`src/runtime/mod.rs:460`, `potential_h_from_log2_spectrum` returns an owned `Vec`),
    which only runs when `[psychoacoustics.habituation]` is enabled (default off).
  Two more sit off the steady-state path: the analysis-lag warning string in
  `AudioMonitor::update` (`src/runtime/mod.rs:125`, at most once a second) and the space
  clones in `merge_latest_analysis_results` (`src/runtime/mod.rs:382-384`, only on a
  Log2Space reconfiguration). `idle_silence` (`src/runtime/mod.rs:1383`) is allocated
  once before the loop, not per hop. The rule is "do not add allocations beyond what the
  handoff requires", not "zero allocations".
  Downstream of `runtime/`, the `life` layer allocates too, and that list is
  representative rather than exhaustive — audit the call path, do not trust this
  paragraph as a census. Known live examples: the prediction scan in
  `TerrainPredictor::predict_consonance_field_level_at`
  (`src/life/generator_model.rs:80`, once per theta gate because
  `cache_pred_next_gate` memoizes the gate tick); the visit-order permutation in
  `Community::advance_substep_sequential_current` (`src/life/community.rs:840`, once per
  control substep); and `dying_ids` in `Community::apply_background_turnover`
  (`src/life/community.rs:899`, which only reaches the heap when a death actually fires).
  Where a hop path has already been made allocation-free it stays that way: the
  `Community` / `Voice` / `PhonationEngine` paths reuse pre-allocated scratch buffers
  (`Community::advance_scratch` / `gate_open_scratch` / `dead_id_scratch`,
  `src/life/community.rs:103-107`; `scratch_candidates` / `scratch_merged` /
  `scratch_grid` / `scratch_field`, `src/life/phonation_engine.rs:968-971`). Clear and
  refill those instead of introducing a new `Vec` in a hop path.
  The analysis worker (`src/core/analysis_worker.rs`) runs on its own thread and is not
  bound by this rule; it allocates roughly 30 times per hop, dominated by the
  `Landscape` snapshot `AnalysisStream::process` clones
  (`src/core/stream/analysis.rs:63`) and sends over the channel.
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
- **Inline tests** (`#[cfg(test)] mod tests` in the same source file) are for module-internal logic and private APIs.
- **Integration tests** (`tests/` directory) are for public API and cross-module behavior; treat them as black-box specs.
- **Crate-internal test modules** (`src/<module>/tests.rs`, declared with
  `#[cfg(test)] mod tests;`) are used where a suite is too large to inline but needs
  access to private APIs across a module tree — currently `src/life/tests.rs` and
  `src/scripting/tests.rs`. Prefer one of the two forms above; reach for this only when
  the suite genuinely needs crate-internal visibility.

## Architecture Notes for Agents
- The core perception model is **Landscape**. It ingests audio, transforms it to Log2-frequency space via NSGT, and computes two potentials:
  1. **Roughness (R)**: Amplitude fluctuations within critical bands (dissonance).
  2. **Harmonicity (H)**: Periodicity/Template matching (consonance/fusion).



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
  (`src/core/psycho_state.rs:61,74`)
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
  `drive_and_apply_habituation` (`src/runtime/mod.rs:463`) and as the source for the
  eroded view. Behavior and the listener read variant 6, not this one.
3. `consonance_field_energy`
- Definition: `-score`.
- Usage: retained for minimization view and consistency checks.
4. `consonance_density_mass`
- Definition: `max(0, H01*(1-rho*R01))`.
- Implementation: `ConsonanceKernel::density_with_rho(rho)` + `src/core/landscape.rs`.
- Usage: base value. It is the source for the eroded mass and the input to
  `build_consonance_density` (`src/core/landscape.rs:351`). Range-local spawn reads
  variant 6, not this one.
5. `consonance_density_pmf`
- Definition: normalized PMF from density mass; uniform fallback on all-zero totals.
- Implementation: `build_consonance_density` in `src/core/landscape.rs`.
- Usage: currently exercised only by inline tests. Spawn placement goes through
  `SpawnStrategy::Field` (see the Rhai Spawn API section), not through this PMF.
6. `consonance_field_score_eff` / `consonance_field_level_eff` / `consonance_density_mass_eff`
- Definition: variants 1, 2 and 4 after habituation erosion is applied.
- Implementation: fields at `src/core/landscape.rs:100,102,104`, written by
  `Landscape::apply_habituation` (`src/core/landscape.rs:300`); driven per hop from
  `apply_landscape_updates` (`src/runtime/mod.rs:1656`) when
  `[psychoacoustics.habituation]` is enabled.
- Usage split:
  - Behavior, listener and spawn read the eroded views:
    `src/life/community/frequency.rs:16,21,52`, `src/life/pitch_core.rs` (through
    `Landscape::evaluate_pitch_score_log2`, `src/core/landscape.rs:240`),
    `src/listener_twin/mod.rs:181,187`.
  - UI and diagnostics read the base views (`src/ui/windows.rs:708,830`), so the display
    shows the un-eroded terrain.
- With habituation disabled (the default) the `_eff` views equal their base variants
  bit-exact, so reading the eroded view is always correct for behavior.

## Config Keys
- `[psychoacoustics.consonance.field.kernel]`
- `a, b, c, d` (defaults: `1.0, -1.35, 1.0, 0.0`)
- `[psychoacoustics.consonance.field.level]`
- `beta, theta` (defaults: `2.0, 0.0`)
- `[psychoacoustics.consonance.density]`
- `roughness_gain` (`rho`, default: `1.0`)
- `rho` is density roughness sensitivity; negative values clamp to `0`, non-finite values sanitize to `1`.
- `[psychoacoustics.habituation]`
- `enabled` (default: `false`), `satiation_sec` (`5.0`), `recovery_sec` (`8.0`),
  `ref_drive` (`0.25`)
- When `enabled = false` the `_eff` consonance views equal their base variants.

Other top-level sections exist and are defined in `src/config.rs`:
`[audio]`, `[analysis]`, `[dcc]`, `[playback]`. They are not enumerated here —
read `src/config.rs` for their keys and defaults.

Note: every config struct carries `#[serde(deny_unknown_fields)]`, so a misspelled TOML
key is a hard error rather than a silent fallback. `#[serde(default)]` still fills in
*missing* keys. `AppConfig::load_or_default` (`src/config.rs:403`) returns
`anyhow::Result`: an absent file still writes and returns defaults, but an existing file
that is unreadable, malformed, or carries an unknown key fails the run.

## Rhai Spawn API
Placement builders produce a `Placement` (defined in `src/scripting/mod.rs:945`,
registered in `src/scripting/engine.rs:749-817`), which lowers to `SpawnStrategy`
in `src/scenario.rs:545`.

- Range form `(lo, hi)`: `consonance` / `dissonance` / `edge` / `gap` / `random`
  build `SpawnStrategy::Field` with `FieldTarget::Consonance` / `Dissonance` /
  `Edge` / `Gap` / `Uniform` respectively (`src/scenario.rs:521`).
- `line(lo, hi)` builds `SpawnStrategy::Linear`.
- `at(freq)` places at a fixed frequency.
- `consonance(root)` also has a one-argument root form; `.range(min_mul, max_mul)`
  is the multiplier band around that root.
- Modifiers: `.peak()` / `.density()` set `FieldSampling` (`src/scenario.rs:536`;
  `Density` is the default), `.tension(t)` sets the tension degree in `[0,1]`
  (Consonance target only), `.count(n)` sets the batch size, `.spacing(erb)` sets the
  minimum separation.
- Spawn sampling is range-local in `Community::decide_frequency`
  (`src/life/community/frequency.rs:116`): it builds local masses with occupancy masks
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
  call `Log2Space::assert_scan_len_named` (`src/core/log2space.rs:58`) so the panic
  identifies the offending scan. (A bare `assert_scan_len` also exists,
  `src/core/log2space.rs:53`, but it has no production callers — do not reach for it.)
  Never return a sentinel for a length mismatch — it is a
  programming error, not a runtime condition (`sample_scan_linear_log2`,
  `src/core/log2space.rs:167`, panics rather than yielding `NEG_INFINITY`).
  Per-element and per-bin checks stay `debug_assert`. A hop-path function that can
  degrade instead of aborting the performance may keep a `debug_assert` plus a graceful
  fallback, but it must say so: see `exact_loo_consonance_score_scan`
  (`src/life/pitch_core.rs:1033`), which falls back to the approximate score.
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

## Git Operation Policy
- Never create a commit unless the user explicitly asks for a commit in that turn.
- Before creating any commit, always run `cargo clippy -- -D warnings` and confirm it passes.
- If a commit is requested, commit only the files relevant to the requested task.
- Before every push, integrate the remote first with `git pull --rebase --autostash`.
  Two machines and several agents push to `main`.
- If two appends to a record file (e.g. `docs/roadmap/**`) conflict, keep both entries,
  upstream first. Edits to the same paragraph and conflicts in code need a real merge;
  stop and report when the intent of either side is unclear.
