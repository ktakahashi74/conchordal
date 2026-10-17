# Changelog

## Unreleased

### Scenario authoring and runtime

- Rhai enum values now require exact, lowercase canonical names. Unknown names,
  aliases, surrounding whitespace, and case variants produce positioned errors;
  an unknown `modal_table` no longer silently selects the harmonic table.
- Explicit `glide`, `pitch_core`, and `viability_scope` values survive preset
  resolution regardless of builder order. Ineffective settings now produce
  positioned warnings; `--compile-only` evaluates and validates scripts and lists
  their warnings. Generated API definitions include argument names, ranges,
  defaults, and hover documentation.
- `--wait-user-exit` enables waiting without a value; use
  `--wait-user-exit=false` to disable it. An explicit missing `--config` path is an
  error. Only an omitted path may generate a default `config.toml`. Invalid numeric
  config values fail validation rather than being silently adjusted.
- Shutdown finishes and flushes reports/profiles once, joins observation workers,
  and reports worker, callback, capture, and output failures through the exit status.
  A stalled GUI closes only after runtime cleanup. Headless and offline runs skip
  UI snapshots.
- SoftClip preserves signals below its knee instead of amplifying them, and remains
  strictly below its ceiling. Limiter meters retain interval maxima. Limiter mode
  comes from `[audio].limiter`; the `CONCHORDAL_LIMITER` environment override was
  removed. The default remains the peak limiter.
- Terrain sampling at invalid or out-of-range frequencies returns zero;
  scan-length mismatches remain hard errors. Latest analysis delivery retains the
  newest result when the queue is full. Habituation defaults have one definition,
  and the base consonance PMF is normalized once, independently of erosion.

### API and report changes

| Previous API | Current API |
| --- | --- |
| `pitch_smooth(tau_sec)` | Removed; it had no runtime reader |
| `variant(spec)` | Assign the value and apply builders to the copy |
| `selection_approx_loo(true/false)` | `viability_scope("environment"/"total")` |
| `respawn_min_c_level(level)` | `respawn_min_consonance(level)` |
| `set_control_update_mode(name)` | `control_update_mode(name)` |
| `set_scaffold_off()` | `scaffold_off()` |
| `set_scaffold_shared(freq_hz)` | `scaffold_shared(freq_hz)` |
| `set_scaffold_scrambled(freq_hz, seed)` | `scaffold_scrambled(freq_hz, seed)` |

- `unison`, `top_k`, `random_candidates`, and `respawn_capacity` take integers only.
  Floating-point parameters accept integer arguments where applicable.
- Canonical enum strings are `gate_snap`/`glide`/`rearticulate` for
  `pitch_apply_mode`, `hill_climb`/`peak_sampler` for `pitch_core`,
  `legacy`/`proposal_interval` for `move_cost_time_scale`, `approx`/`exact` for
  `leave_self_out_mode`, `attack_phase_match`/`none` for `rhythm_reward`,
  `consonance`/`dissonance` for `set_pitch_objective`, and
  `snapshot_phased`/`sequential_rotating` for `control_update_mode`.
- Participation JSONL fields `observed_external_band_energy` and
  `memory_external_band_energy` are now `perc_external_band_energy` and
  `perc_memory_external_band_energy`. Body-observation records expose capture state.
- Rust callers use `Landscape`; the `LandscapeFrame` alias was removed.
  Removed internal fields and unused update APIs have no compatibility aliases.

### Development

- Streaming, body analysis, and exact leave-self-out evaluation share perception
  scratch and numerical kernels. Analysis snapshots clone at publication, and
  participation, pitch proposals, social updates, and no-death respawn reuse storage.
  Allocation assertions require the `profile-alloc` feature.
- Runtime, phonation, and scripting modules are split by responsibility. The
  [documentation guide](docs/README.md) links current plans, preserved histories,
  design records, and operating instructions.
- CI checks formatting, Clippy across all targets, Python tests, English/Japanese
  books, allocation profiles, and release recovery. Linux also checks `plotcheck`
  and `--no-default-features`. Rust and scientific Python dependencies are pinned;
  documentation-only changes skip Rust CI and superseded runs are cancelled.
- The tracked `.githooks/pre-commit` checks formatting without staging files.
  Enable it with `git config core.hooksPath .githooks`. `just test` records the full
  Rust test output and exit status. Python tests live under `scripts/tests`.
- Book HTML is generated during site deployment rather than tracked in Git.
  Concurrent listener fixtures use unique file names.
