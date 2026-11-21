# Repository Guidelines

## Project Structure & Module Organization

Start with [architecture-current.md](docs/architecture-current.md) for the current
architecture and its invariants. See [engineering-reference.md](docs/operations/engineering-reference.md)
for the module map, subsystem contracts, and validation details.

- `src/` contains the shared native and web engine. Keep `runtime/` independent
  of `ui/` through `src/viewdata.rs`. Browser hosts live in `crates/conchordal-web/`
  and `gui/`. The Astro site lives in `web/`.
- The core computes everything the ecology evaluates from radiated sound. Body
  self-reports are fast paths and must match the render in conformance tests.
  New ecology code must not branch on body kind. See [the body boundary](docs/design-notes/timbre.md).

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

## Mechanism Evidence and Composer Controls

Follow [ledger §9.3.104](docs/design-notes/technote-ledger.md#mechanism-evidence-and-composer-controls)
(adopted 2026-10-27). Classify each mechanism as principle, candidate or author rule;
separate mixed-tier parts of a term. A candidate has a published formal model but an
unidentified brain mechanism or task-fitted parameters: preserve its source equations
and values, and identify author-rule additions. This does not change §9.3.57's
source-matched versus original computation rule or §9.3.56's five listener layers.

- Do not introduce human-authored harmony without warning: written progressions,
  scores, or stylistic priors such as key and functional-harmony profiles.
  Under the author's 2026-10-27 criterion, generatively computed functional harmony
  may remain an optional composition choice outside the main line; fixed harmonic
  profiles are presets and must be removed. Research assays may use authored
  harmony only when the brief, listening guide and report state at the top what
  was written, by whom, and that the ecology did not generate it.
- Keep defaults > presets > individual axes. Principles operate by default;
  candidate and author-rule axes are neutral by default. Qualifying candidates may
  enter one default listener preset after source agreement and author adoption by
  listening; allow removal or replacement of that preset with one term. Check the
  preset and existing vocabulary before adding a term, and prune displaced author
  rules after candidate layers arrive.
- Existing author-rule defaults that run the instrument (movement cost, tessitura
  gravity, consonance evaluation coefficients, habituation constants, participation,
  metabolism and lifecycle rules) remain the default base behavior with author-rule
  labels. "Neutral by default" applies to mechanism axes added under §9.3.104;
  changing the existing base is a separate change with its own audition.
- The composer chooses hypothesis use and weight, and style-dependent priors, in
  layer-3 cognitive coordinates. Do not expose layer-4 memory/learning dynamics or
  change layer-1 sensation. Initial conditions and macro form remain author choices.
- Accept principles and candidates by source agreement, registered causal difference
  and real-time operation on the declared load. Candidate acceptance verifies the
  implementation of a hypothesis, not its truth. Author rules require implementation
  checks (default agreement, causal difference, real time) and author listening.
  Audition a bundled default listener preset once; do not rank parameter values by
  listening. Composer-control upper bounds come only from failure or saturation,
  not cost. Principle-layer acceptance still has no subjective gate.
- §9.3.55's three conditions now qualify default-active principles; they no longer
  prohibit candidate or author-rule connections to generation under this policy.
  Default-off passive diagnostics and research code remain allowed.
- Add evidence labels to the registry, LSP hover and generated API pages as those
  surfaces are implemented: Principle / Candidate (source) / Author rule. These are
  distinct from Core / Experimental / Tuning / Research disclosure tiers. Keep the
  bilingual composer guide aligned; describe labels and the default listener preset
  in the public technote only after implementation. Inventory existing terms and
  return divided classifications, including default-active author-rule parts, to
  the author; do not silently change their defaults during documentation work.

## Listener Profile

- `listener_profile("sensory" | "tonal")` is an initial Director setting before
  scene actions/scopes. `DEFAULT_LISTENER_PROFILE` in `scripting/context.rs` is
  `"sensory"`; changing it needs separate author adoption.
- Tonal inherits SQ 0.05, ME 0.00076066292, PX 1 and final-section arch/ritard 1/1.
  These are audition weights, not adopted defaults. Explicit Population weights,
  including zero, win; unsupported final forms are silently skipped.
- TC is excluded from tonal: its table costs about 1 GB/80 seconds, and the
  declared approximation acceptance range is [0.25,1] (measured at 0.25/0.5/1).
  This is not an upper bound or an extra warning on the individual word.
- Implicit expression uses the last completed section and existing whole-scene
  PE-1 support. No suitable section means no effect; explicit expressive_section
  (including zero) wins. Runtime rejection is silent only for profile expression.
- Profile selection has no mechanism evidence claim. Keep the constituent
  candidate and author-rule labels; do not extend their supported consumers.

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
- The browser version follows the same rule: no path saves or sends audio or
  audio-derived values. Its measurement page posts performance counters only (an
  explicit allowlist in `crates/conchordal-web/www/page.js`). The public demo page
  (`web/src/scripts/demo.js`) sends one counter record per performance, with a
  visible notice and an off switch, to `web/metrics-worker/`, which rebuilds it
  from its own allowlist; hardware details are added only on a post-dropout opt-in.

## Coding Style & Naming Conventions
- **Comments**:  All comments must be in concise English.  Do not use Japanese in code comments.
- **Incremental score**: Endless scripts own their Engine and action builders on a
  dedicated thread. The preallocated score outbox accounts for action count and heap
  capacity; popped items stay charged until dispatch consumes them. The worker does
  not allocate score queues or copy action payloads; it returns credit and frees
  consumed items. Runtime action effects, LOO registration and UI/report snapshots
  retain their separate allocation costs.
- **DSP Efficiency**: Prefer `f32`. Minimize allocations in `worker_loop` and the
  cpal callback. Avoid new steady-state heap allocations. Reuse scratch or
  pre-allocated ringbuffers and audit the actual call path.
  See the [conditional allocation budget](docs/operations/engineering-reference.md#dsp-allocation-budget)
  for subsystem rules, allocation measurements, and test protocols.
  Use `--features profile-alloc` when measuring or asserting allocations; without it,
  allocation counts are unavailable. Runtime measurements live in `runtime_profile.rs`.
- **SIMD**: Use `fearless_simd`. Select the level once per call and use generic
  `#[inline(always)]` kernels. Values reaching the ecology must be bit-identical
  across native SIMD levels. Do not use `mul_add`. Reduce across lanes at a fixed
  logical width and order. FFT dispatch can vary low bits in NSGT and Flora resource
  power. Wasm may differ from native. Test each supported level against scalar.
  Use a faster kernel only on levels where it helps and preserve a bit-identical
  path elsewhere. Use `Level::baseline()` for per-sample synthesis. See the
  [SIMD implementation and baseline details](docs/operations/engineering-reference.md#simd-implementation-and-native-build-baseline).
- **Naming**: `snake_case` for modules/functions, `CamelCase` for structs/traits.

## Anti-Bloat Rules
- Enforce YAGNI strictly: do not add abstractions before a second concrete use appears.
- Handle only errors that can actually occur on the current path.
- Do not add comments that only restate code; keep comments for intent/constraints only.
- Inline helpers unless they are used in 3 or more places.
- Prefer plain structs and existing types over new nominal wrapper types.
- Keep submodules and items `pub(crate)` or private unless `tests/`, `src/bin/`
  or `examples/` need them. Name the out-of-crate consumer when widening visibility.
- Reuse shared scalar helpers from `src/core/float.rs`. Preserve the bit-exact
  Gaussian spelling in `core::roughness_kernel`. See the
  [crate visibility and helper details](docs/operations/engineering-reference.md#anti-bloat-rules).

## Testing Policy

Use inline or crate-internal tests for private APIs and module logic. Keep public
and cross-module black-box specs in `tests/`. See the
[testing reference](docs/operations/engineering-reference.md#testing-policy) for
test locations and auditory validation acquisitions.

## Terminology and Naming

Reserve `perceptual` for audio evidence and use `pred_` / `perc_` only where
both origins coexist. Keep origin independent of representation and use explicit
suffixes where needed. See [R/H/C terminology](docs/operations/engineering-reference.md#frequency-and-consonance).

## Frequency Space: Log2Space invariants

We represent frequency-direction terrains as **Log2Space-aligned scans**.

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

## Scenario Script Authoring

Keep minimal scenarios approachable and the full parameter space accessible.
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

See [configuration and Rhai details](docs/operations/engineering-reference.md#scene-configuration-and-rhai)
and [mechanism-specific contracts](docs/operations/engineering-reference.md#optional-mechanisms-and-observer-reservations)
when working on those systems.

## Documentation and Samples
- Don't touch `web/` when editing sourcecode. Don't touch `src/` when editing `web/`, unless otherwise specified.
- Keep the hand-curated public technote aligned with implemented core changes.
  Record research and decisions in the bilingual ledger. See the
  [documentation reference](docs/operations/engineering-reference.md#documentation-and-sample-maintenance).
- `rhai-defs/conchordal.d.rhai` and `docs/rhai_book*/src/reference/api.md` are generated
  by `cargo run --bin gen_rhai_defs`. Never hand-edit them; `tests/rhai_defs_in_sync.rs`
  fails if they drift from the registry.
- `samples/` top-level études must not pin a seed; `samples/research/` assays must pin one.
  This is enforced by `tests/sample_seed_policy.rs`.
- Keep `src/scripting/docs.rs`, `src/scripting/engine.rs`, and the generated Rhai
  definitions and API references in sync. Follow the Japanese book's
  [adaptation rules](docs/rhai_book_ja/README.md) and keep its chapter paths and
  executable examples aligned with the English book.

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
