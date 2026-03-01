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
- Verify all targets: `cargo check --all-targets`
- Test all targets: `cargo test --all-targets`
- Check examples explicitly: `cargo check --examples`
- Paper figures/manuscript flow is out of scope for this repository.


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
- The `conchordal` binary (instrument) MUST NOT write audio to disk
  in any build profile. Performances are ephemeral by design (manifesto).
  The `#[cfg(debug_assertions)]` guards in `cli.rs` and `app.rs` enforce this.
- Offline WAV rendering is provided by the separate `conchordal-render` binary
  (`src/bin/render.rs`), which shares the core engine but is not the instrument.
  The air-gap policy does not apply to `conchordal-render`.
- Do not add `--wav` or any disk-write capability to the `conchordal` binary.

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



## Terminology: predictive/perceptual vs potential/representation (R/H/C)

We use two orthogonal axes. Do not mix them.

### Axis A: WorldModel layer (origin)
- **predictive** (`pred_*`): hypothesis derived from NoteBoard / internal model (zero-latency).
- **perceptual** (`perc_*`): evidence derived from actual audio analysis (NSGT/filterbank; delayed).
- **error** (`err_*`): `err_* = perc_* - pred_*`.

`perceptual` is reserved for this axis only.

### Axis B: representation (kernel output vs transformed views)
- **potential** (`*_pot_*`): raw kernel output / physical-ish quantity (unnormalized).
- **score** (`*_score_*`): kernel score (unbounded real value).
- **level** (`*_level_*`): bounded level in `[0,1]`, `level = sigmoid(beta * (score - theta))`.
- **mass** (`*_mass_*`): non-negative, pre-normalization mass used before PMF normalization.
- **density** (`*_density_*`): normalized PMF/PDF.
- **energy** (`*_energy_*`): minimization form, `energy = -score`.

Potential/representation is orthogonal to pred/perc:
- `pred_h_pot_scan`, `pred_h_state_scan`
- `perc_r_pot_scan`, `perc_r_state_scan`
- `perc_c_field_score_scan`, `perc_c_field_level_scan`
- `perc_c_density_scan`, `perc_c_field_energy_scan`
- `err_c_field_level_scan = perc_c_field_level_scan - pred_c_field_level_scan`

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
- Usage: individual behavior, world model, UI in `src/life/population.rs`, `src/life/world_model.rs`, `src/ui/windows.rs`.
3. `consonance_field_energy`
- Definition: `-score`.
- Usage: retained for minimization view and consistency checks.
4. `consonance_density_mass`
- Definition: `max(0, H01*(1-rho*R01))`.
- Implementation: `ConsonanceKernel::density_with_rho(rho)` + `src/core/landscape.rs`.
- Usage: range-local spawn mass in `src/life/population.rs`.
5. `consonance_density_pmf`
- Definition: normalized PMF from density mass; uniform fallback on all-zero totals.
- Implementation: global PMF cache in `src/core/landscape.rs`.
- Usage: `SpawnStrategy::ConsonanceDensity` and Rhai spawn API.

## Config Keys
- `[psychoacoustics.consonance.field.kernel]`
- `a, b, c, d` (defaults: `1.0, -1.35, 1.0, 0.0`)
- `[psychoacoustics.consonance.field.level]`
- `beta, theta` (defaults: `2.0, 0.0`)
- `[psychoacoustics.consonance.density]`
- `roughness_gain` (`rho`, default: `1.0`)
- `rho` is density roughness sensitivity; negative values clamp to `0`, non-finite values sanitize to `1`.

## Rhai Spawn API
- `consonance_density_pmf(min_freq, max_freq)` builds `SpawnStrategy::ConsonanceDensity`.
- Spawn sampling is range-local in `Population`: it builds local masses with occupancy masks and normalizes in-range.
- If range-local total mass is zero, fallback stays in-range and remains well-defined (unoccupied-uniform first, then full-range uniform if all occupied).

## Naming Note
- `field_level` is a 0..1 gate/strength used by behavior and prediction.
- `density_mass` is pre-normalization non-negative mass before PMF conversion.
- For prose, "level" and "mass" are preferred names.

### Suffix convention (avoid ambiguity)
Use explicit suffixes when needed:
- `_scan`: frequency-indexed arrays (Log2Space bins)
- `_scalar`: summary values (total/max/p95 etc.)

Example:
- `perc_r_state_scalar`
- `pred_c_field_level_scan`
- `perc_c_field_score_scan`

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

## Compatibility Policy
- During the alpha phase, do not preserve backward compatibility by default.
- Prefer clean architecture and correct behavior over compatibility shims, aliases, or migration layers.

## Git Operation Policy
- Never create a commit unless the user explicitly asks for a commit in that turn.
- Before creating any commit, always run `cargo clippy -- -D warnings` and confirm it passes.
- If a commit is requested, commit only the files relevant to the requested task.
