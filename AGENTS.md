# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` with focused modules:
  - `core/` (Psychoacoustic DSP primitives):
    - `log2space.rs`: Log-frequency coordinate system (octave-based).
    - `nsgt*.rs`: Non-Stationary Gabor Transform implementations (RT, FFT, and Kernel variants).
    - `harmonicity_kernel.rs`: Virtual pitch/root detection via harmonic templates.
    - `roughness_kernel.rs`: Sensory dissonance calculation via interference kernels.
    - `phase.rs`: Phase wrap/diff utilities (shared across rhythm + kernels).
    - `landscape.rs`: Real-time integration of Roughness and Harmonicity.
    - `fft.rs`: FFT wrappers and Hilbert transform logic.
    - `erb.rs`: Equivalent Rectangular Bandwidth scales.
  - `audio/` (Real-time I/O):
    - `output.rs`: cpal stream management (ringbuffer producer).
    - `buffer.rs`: Interleaved audio buffer types.
    - `writer.rs`: Disk recording (WAV).
  - `synth/` (Synthesis engine):
    - Phase-vocoder/additive synthesis based on NSGT bins.
  - `ui/` (UI and visualization):
    - Egui views, plots, and visualization logic.
  - `life/` (Agents and behaviors):
    - Individual/Population models, temporal cores, fields, and scenario scripting.
  - `web/` (Project website):
    - Zola site sources.
- Entrypoints: `src/main.rs` (binary) and `src/app.rs` (GUI/Thread wiring).
- Samples: execution-test scripts live under `tests/scripts/`. Compile-only samples stay under `samples/`.

## Build, Test, and Development Commands
- Build (debug): `cargo build`
- Run (release): `cargo run --release` (Recommended for real-time DSP performance)
- Tests: `cargo test`.  Run tests always after modifying code.
- Format: `cargo fmt --all`
- Lint: `cargo clippy -- -D warnings`


## Mandatory End-of-Task Procedure

At the end of EVERY task that modifies code under src/, the agent MUST:

### 1. Run cargo tests
Run tests with full output and backtraces enabled.

```bash
set -o pipefail
( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt

```

### 2. Record test status

Always write the exit code and timestamp, even if tests fail.

```bash
echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt
```

- test_report.txt must contain stdout + stderr of cargo test
- test_status.txt must always exist after a task
- Do NOT skip this step under any circumstances


## Air-Gap Protocol
- In release builds, audio file export is forbidden. Do not add or restore any functionality that writes audio to disk (e.g., WAV export).

## Coding Style & Naming Conventions
- **Comments**:  All comments must be in concise English.  Do not use Japanese in code comments.
- **DSP Efficiency**: Prefer `f32`. Avoid allocations in the audio thread (`worker_loop`). Use `Vec::with_capacity` or pre-allocated ringbuffers.
- **Naming**: `snake_case` for modules/functions, `CamelCase` for structs/traits.

## Testing Policy
- **Inline tests** (`#[cfg(test)] mod tests` in the same source file) are for module-internal logic and private APIs.
- **Integration tests** (`tests/` directory) are for public API and cross-module behavior; treat them as black-box specs.

## Architecture Notes for Agents
- The core perception model is **Landscape**. It ingests audio, transforms it to Log2-frequency space via NSGT, and computes two potentials:
  1. **Roughness (R)**: Amplitude fluctuations within critical bands (dissonance).
  2. **Harmonicity (H)**: Periodicity/Template matching (consonance/fusion).



## Terminology: predictive/perceptual vs potential/state (R/H)

We use two orthogonal axes. Do not mix them.

### Axis A: WorldModel layer (origin)
- **predictive** (`pred_*`): hypothesis derived from NoteBoard / internal model (zero-latency).
- **perceptual** (`perc_*`): evidence derived from actual audio analysis (NSGT/filterbank; delayed).
- **error** (`err_*`): `err_* = perc_* - pred_*`.

`perceptual` is reserved for this axis only.

### Axis B: representation (kernel output vs normalized state)
- **potential** (`*_pot_*`): raw kernel output / physical-ish quantity (unnormalized; references not applied yet).
- **state** (`*_state_*`): normalized / referenced / composed quantities used for decision making or logging
  (e.g. 0..1, and `C_state01 = sigmoid(beta * (C_score - theta))`).

`C_score = alpha * H01 - w(H) * R01` and `w(H) = w0 + w1 * (1 - H01)`.

Potential/state is orthogonal to pred/perc:
- `pred_h_pot_scan`, `pred_h_state01_scan`
- `perc_r_pot_scan`, `perc_r_state01_scan`
- `perc_c_score_scan`, `perc_c_state01_scan`
- `err_c_state01_scan = perc_c_state01_scan - pred_c_state01_scan`

### Suffix convention (avoid ambiguity)
Use explicit suffixes when needed:
- `_scan`: frequency-indexed arrays (Log2Space bins)
- `_scalar`: summary values (total/max/p95 etc.)

Example:
- `perc_r_state01_scalar`
- `pred_c_state01_scan`
- `perc_c_score_scan`

## Frequency Space: Log2Space invariants

We represent frequency-direction terrains as **Log2Space-aligned scans**.

### Rules
- **F1**: Any vector suffixed with `_scan` MUST be aligned to Log2Space bins:
  `scan.len() == space.n_bins()`.
- **F2**: Any function that accepts/returns a `_scan` MUST assert the invariant at boundaries
  (debug_assert is acceptable; tests must cover it).
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


## Other
- Don't touch `web/` when editing sourcecode. Don't touch `src/` when editing `web/`, unless otherwise specified.
- `docs/generated`, `docs/schemas`, `docs/rhai-book` are auto-generated by scripts. Don't touch them directly.

## Git Operation Policy
- Never create a commit unless the user explicitly asks for a commit in that turn.
- Before creating any commit, always run `cargo clippy -- -D warnings` and confirm it passes.
- If a commit is requested, commit only the files relevant to the requested task.
