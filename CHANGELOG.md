# Changelog

## Unreleased

### Composer-facing

- `steady()` offers private, regular opportunities; acoustic context selects sound or skip without shifting the grid.
  It joins `entrained()`, `flow()`, and `metric()` as a timing preset.
- `intrinsic_rate(hz)` sets the founder bodily rate for entrained, steady, and flow timing; omission retains 2 Hz.
  Metric Voices instead start at the final scene tempo-basin center, with independent phases before beat confirmation.
- Explicit `sustain()` combines with acoustic participation or metric synchronization in either builder order.
  A living Tone can continue across withheld opportunities; the initial Hold onset remains immediate.
- `pitch_apply_mode("rearticulate")` supports attacks at adopted pitch targets and live pitch-mode changes.
  `rearticulate_above(cents)` measures net distance from the last sounded onset; its default is zero.
- Sustained participation defers GateSnap targets until a sounded opportunity; bounded Rearticulate slides small moves and defers large ones.
  Finite note lengths and `rest()`/`rest(periods)` support gaps measured from completion of all Tone tails.
- `articulation_persistence(p)` retains a private Base/Connect policy for eligible free-pitch sustained Voices.
  It requires Rearticulate and a positive movement boundary; the initial, template-inherited value defaults to zero.
- `tie(p)` probabilistically extends pending-free opportunities in free-pitch sustained GateSnap/Rearticulate.
  Omission is zero for participation and one for metric synchronization; explicit `tie(0)` requests repeated beat attacks there.
- `at(freq)` sets only the initial position when a Free-pitch spec explicitly retains `seek_consonance()` intent.
  Other `at()` placements keep fixed pitch.
- Samples 13–16 add Three Windows, Stone and Wire, Folded Air, and Open Court as composed scenes.
  Samples 1–12 remain capability demonstrations; sample 12 no longer supplies a dedicated beat carrier.
- Samples 08, 10, and 12 use rounded lifecycle constants; Generations uses `endurance(9.0)`.
- Rhai enums require exact lowercase canonical names; unknown names, aliases, whitespace, and case variants give positioned errors.
  Unknown `modal_table` names no longer silently select the harmonic table.
- Explicit `glide`, `pitch_core`, and `viability_scope` survive preset resolution regardless of builder order.
  Ineffective settings produce positioned warnings; generated API definitions include names, ranges, defaults, and hover documentation.

### Behavior

- Landscape habituation is enabled by default for ecology and listener analysis, eroding behavioral and spawn views under sustained activity.
  `enabled = false` under `[psychoacoustics.habituation]` restores the base views; the UI shows the un-eroded terrain.
- `set_roughness_aversion(w)` changes roughness valuation in consonance fields and spawn mass while leaving sensed roughness unchanged.
- Entrained and flow participation use causal external-audio context to choose near-due onsets, waits, or skips.
  Only actual onsets update onset-context memory; hold lengths follow the intrinsic period independently of cadence adaptation.
- Phonation invariant failures are reported without terminating the performance.

### Runtime and CLI

- `--compile-only` evaluates and validates scripts without starting audio or the GUI, and lists positioned warnings.
- `--wait-user-exit` enables waiting without a value; `--wait-user-exit=false` disables it.
- An explicit missing `--config` path is an error; only omission may generate a default `config.toml`.
  Unknown keys and invalid numeric values fail validation instead of falling back or being silently adjusted.
- Headless `--profile` writes hop timings and device counters; `conchordal-render --report` writes JSONL telemetry.
  The renderer's `--reserve-runtime-ids-through` keeps generated Voice IDs above a comparison's reserved range.
- Shutdown flushes reports/profiles once, joins observation workers, and propagates worker, callback, capture, and output failures to exit status.
  A stalled GUI closes after runtime cleanup; headless and offline runs skip UI snapshots.
- Output-underrun accounting starts after the first audio chunk, avoiding startup counter races.
- SoftClip preserves signals below its knee and stays below its ceiling; limiter meters retain interval maxima.
  `limiter` under `[audio]` selects the mode, defaulting to peak limiting; the `CONCHORDAL_LIMITER` override was removed.

### Analysis and DCC

- Acoustic onset recurrence corroborates an adaptive beat with integer-ratio subdivision and measure layers.
  Beat confidence stays zero until corroboration; `meter_stability` also weights the cultural tempo prior within `temporal_basin`.
- Production timing follows habitat radiation, while ListenerTwin timing follows presentation audio.
- The top-level `temporal_meter_t2 = true` opts the production meter into continuous raw habitat-T2 recurrence and later forecast confirmation.
  Its default is false; it shares the existing meter and is independent of the `[temporal_t2]` group-accent path.
- Optional whole-bus T2 preprocessing supplies direct group accents and Periodic arrival forecasts.
  Arrival costs feed the configured participation comparison; representative body-onset footprints can replace its energy proxy.
- Body-footprint delivery uses deterministic observation cutoffs in offline renders and preserves source-generation identity.
  Participation reports expose decision terms, observation coverage, and actual-onset joins.
- Terrain sampling outside the valid frequency range returns zero; scan-length mismatches remain hard errors.
  Latest analysis delivery retains the newest result when its queue is full.
- Habituation defaults have one definition; the base consonance PMF is normalized once, independently of erosion.

### Sound bodies

- `render_prototype = true` opts both the instrument and offline renderer into excitation-based synthesis; legacy rendering remains the default.
  ADSR shapes excitation, body radiation has separate state, and closed tails retain their source-generation identity.
- Excitation Sine/Harmonic follows the requested attack without added free ringing; fluctuation is disabled by default.
- Optional body-aware Field birth placement scores supported Sine/Harmonic recipes using integrated body spectra.
  Legacy final respawn selection also uses body scores for supported policy/family combinations; native final selection retains point scoring.
- `[body_metabolism]` with `enabled = true` uses current-position body scoring, source-PCM self-subtraction, and fair round-robin updates.
  Sine/Harmonic support both renderers; unavailable or unsupported evaluations use point scores, and unvisited Voices retain their last score.
- Block rendering shares oscillator kernels and reuses unchanged oscillator/modal coefficients and roughness geometry.
  Tone retirement is checked once per source generation; energy spans split at bank refreshes and modal forecasts match rendering magnitude.
- Tone updates from other source generations are ignored, and source energy remains attributed across slot reuse.

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
- `pitch_apply_mode`: `gate_snap`/`glide`/`rearticulate`; `pitch_core`: `hill_climb`/`peak_sampler`.
- `move_cost_time_scale`: `legacy`/`proposal_interval`; `leave_self_out_mode`: `approx`/`exact`.
- `rhythm_reward`: `attack_phase_match`/`none`; `set_pitch_objective`: `consonance`/`dissonance`.
- `control_update_mode`: `snapshot_phased`/`sequential_rotating`.
- Participation JSONL renames `observed_external_band_energy` to `perc_external_band_energy`
  and `memory_external_band_energy` to `perc_memory_external_band_energy`; body observations expose capture state.
- Rust callers use `Landscape`; the `LandscapeFrame` alias was removed.
  Removed internal fields and unused update APIs have no compatibility aliases.

### Developer

- Streaming, body analysis, and exact leave-self-out evaluation share perception scratch and numerical kernels.
  Analysis clones at publication; participation, pitch proposals, social updates, and no-death respawn reuse storage.
- Allocation profiling and assertions require `profile-alloc`; runtime profiles record the work performed along enabled paths.
- Runtime, phonation, and scripting modules are split by responsibility.
  The [documentation guide](docs/README.md) links current plans, preserved histories, design records, and operating instructions.
- CI checks formatting, all-target Clippy, Python tests, English/Japanese books, allocation profiles, and release recovery.
  Linux also checks `plotcheck` and `--no-default-features`; Rust and scientific Python dependencies are pinned.
- Documentation-only changes skip Rust CI, and superseded runs are cancelled.
  Concurrent profile/listener fixtures use unique names; render-binary identification and failure tests respect platform differences.
- The tracked `.githooks/pre-commit` checks formatting without staging files; enable it with `git config core.hooksPath .githooks`.
  `just test` records full Rust output and exit status; Python tests live under `scripts/tests`.
- Successful render tests remove temporary output; failed-output retention is bounded.
- The public site uses Astro; English/Japanese book HTML is built during site deployment rather than tracked in Git.
