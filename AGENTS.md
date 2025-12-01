# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` with focused modules:
  - `core/` (Psychoacoustic DSP primitives):
    - `log2space.rs`: Log-frequency coordinate system (octave-based).
    - `nsgt*.rs`: Non-Stationary Gabor Transform implementations (RT, FFT, and Kernel variants).
    - `harmonicity_kernel.rs`: Virtual pitch/root detection via harmonic templates.
    - `roughness_kernel.rs`: Sensory dissonance calculation via interference kernels.
    - `landscape.rs`: Real-time integration of Roughness and Harmonicity.
    - `fft.rs`: FFT wrappers and Hilbert transform logic.
    - `erb.rs`: Equivalent Rectangular Bandwidth scales.
  - `audio/` (Real-time I/O):
    - `output.rs`: cpal stream management (ringbuffer producer).
    - `buffer.rs`: Interleaved audio buffer types.
    - `writer.rs`: Disk recording (WAV).
  - `synth/`: Synthesis engine (phase-vocoder/additive based on NSGT bins).
  - `ui/`: Egui views, plots, and visualization logic.
  - `life/`: Evolutionary components (Genotype/Phenotype).
- Entrypoints: `src/main.rs` (binary) and `src/app.rs` (GUI/Thread wiring).

## Build, Test, and Development Commands
- Build (debug): `cargo build`
- Run (release): `cargo run --release` (Recommended for real-time DSP performance)
- Tests: `cargo test`.  Run tests always after modifying code.
- Format: `cargo fmt --all`
- Lint: `cargo clippy -- -D warnings`

## Coding Style & Naming Conventions
- **Comments**:  All comments must be in concise English.  Do not use Japanese in code comments.
- **DSP Efficiency**: Prefer `f32`. Avoid allocations in the audio thread (`worker_loop`). Use `Vec::with_capacity` or pre-allocated ringbuffers.
- **Naming**: `snake_case` for modules/functions, `CamelCase` for structs/traits.

## Architecture Notes for Agents
- The core perception model is **Landscape**. It ingests audio, transforms it to Log2-frequency space via NSGT, and computes two potentials:
  1. **Roughness (R)**: Amplitude fluctuations within critical bands (dissonance).
  2. **Harmonicity (H)**: Periodicity/Template matching (consonance/fusion).
