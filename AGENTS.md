# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` with focused modules:
  - `core/` (DSP primitives: `fft.rs`, `gammatone.rs`, `erb.rs`, `hilbert.rs`, `roughness.rs`, `landscape.rs`, `util.rs`)
  - `audio/` (I/O: `buffer.rs`, `output.rs`, `writer.rs`)
  - `synth/` (synthesis engine)
  - `ui/` (views, plots, windows)
  - `life/` (evolutionary components: `individual.rs`, `population.rs`, `meta.rs`)
- Entrypoints: `src/main.rs` (binary) and `src/app.rs` (app wiring). Config lives in `src/config.rs`.
- Sample assets like `a.wav` are acceptable for dev; prefer an `assets/` folder for larger or generated files.

## Build, Test, and Development Commands
- Build (debug): `cargo build` — compiles the crate and dependencies.
- Run (debug): `cargo run` — runs the main binary.
- Run (optimized): `cargo run --release` — useful for audio/DSP perf.
- Tests: `cargo test` — runs unit/integration tests.
- Format: `cargo fmt --all` — applies `rustfmt`.
- Lint: `cargo clippy -- -D warnings` — lints and fails on warnings.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; rely on `rustfmt` with default settings.
- Naming: `snake_case` for functions/modules, `CamelCase` for types/traits, `SCREAMING_SNAKE_CASE` for consts.
- Files: one module per file where practical; keep public APIs minimal (`pub(crate)` when possible).
- Audio/DSP code: prefer `f32`, avoid allocations in real‑time paths; document units and ranges.

## Testing Guidelines
- Unit tests colocated via `#[cfg(test)] mod tests { ... }` in each module.
- Integration tests (if added) live under `tests/` and use `*_tests.rs` naming.
- Favor deterministic fixtures for DSP (seeded data) and assert spectral/temporal properties.
- Run `cargo test` before PRs; keep tests fast (<1s per module when possible).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), optional body. Prefix with scope when helpful (e.g., `core:`, `audio:`, `ui:`).
- PRs: clear description, rationale, before/after notes; link issues; include screenshots for UI/plot changes and short audio notes for synthesis changes.
- CI hygiene: ensure `cargo fmt`, `clippy`, and `test` pass locally.

## Security & Configuration Tips
- Centralize tunables in `src/config.rs`. Do not commit secrets or large generated audio; ignore via `.gitignore`.
- Prefer environment variables (`std::env`) or config files for local overrides.
