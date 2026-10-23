# Changelog

## Unreleased

- Flora lowers its fixed Allee half-saturation density from 0.576 to 0.144 mass
  per octave, reducing the growth barrier for small biomass. Seed rain, rate,
  renewal and metric lane scheduling retain their existing rules.

- Flora now supports opt-in `metric()` lane opportunities (T1), using a shared
  habitat audio-only meter and private CouplingClocks. Births and replacements
  apply at sample crossings; pending windows reserve current excess once, while
  unsupported tones, separation conflicts and release retire immediately.
  Unknown beat observations use continuous allocation. Default T0, biomass,
  resources, seed and Allee dynamics remain unchanged.

- Flora resources now consume other habitat sound as well as their own latent
  biomass. Each instance subtracts its actual habitat PCM before power analysis;
  base C/H still include self-sound. Power uses a fixed nominal sine reference
  of 0.03, with Parseval-normalized NSGT support weights and existing band
  integration. Unknown windows retain resource state or timestamped consumption.
  Resource and lane telemetry expose depletion and lifecycle changes. Temporal
  participation defaults to uncoupled T0; no seed or Allee constants change.

- Bundled samples set per-scene playback gain with `output_gain(...)`, measured
  across seeds 21–23 at the default +4.5 dB config gain. Values move average
  integrated loudness toward -23 LUFS within the existing 0.98 sample-peak
  ceiling, rounded toward zero in 0.5 dB steps; the samples retain fresh seeds.

- Rhai `output_gain(db)` sets scene-wide playback gain (default 0 dB; last
  declaration wins). It adds to `[audio].output_gain_db` immediately before the
  instrument/WAV output limiter. Analysis, listener, ecology, reports and UI bus
  meters keep their original levels. Invalid or unrepresentable gains report the
  declaration position.

- `[audio].output_gain_db` applies fixed playback gain immediately before the
  output limiter in both the instrument and offline renderer. Ecology, listener,
  reports and UI meters retain the original bus levels. The +4.5 dB default is calibrated
  from the peaks of all 17 bundled studies across three seeds, below the 0.98
  peak-limiter ceiling; set it to `0.0` for the original output level.
  With +4.5 dB and seed 21, research runs `hereditary_adaptation_heredity_selection` / `temporal_scaffolding_shared` recorded maximum limiter reductions of 1.832700 / 3.741124 dB, respectively.

- Voice self-exclusion now subtracts each Voice's actual habitat PCM before the
  shared NSGT, smoothing and H/R/C pipeline. Pitch movement and environment
  viability reuse the same frame and the usual habituation. Observers retain the
  complete mixed history before birth, then follow Voice identity and generation.
  All registered Flora cases, including nearby residual sound, are assertions.
  Production spectral ownership, residual redistribution and the Flora-only
  analysis stream are removed; the existing shared silence floor is unchanged.
  Source capture and per-Voice analysis reuse scratch; publication still allocates.
  The A-to-B comparison (spectral ownership at `af8b6b0` to PCM self-removal at
  `b59a830`, seed 21 for top-level samples) confirms changed WAV and telemetry for
  `08_murmuration`, `13_three_windows` and `16_open_court`; their scripts are unchanged.
  Earlier redistribution and shared-floor comparisons, recorded at `05d9d37`,
  also changed `10_generations` and `12_emergence_and_resolution` against their
  `a30b44c` captures, and `14_stone_and_wire` and `15_folded_air` against their
  `a02e529` captures. Those comparisons used matching scripts, configuration and
  seed 21; the four are historical differences, not newly acquired A-to-B differences.

- Flora adds an Experimental composition mode: spectral biomass grows over the shared
  consonance landscape, with sine or resonant tones and optional residual sound.
  It coexists with Populations through sound alone. Its consumer–resource dynamics
  are authored rules. Sample 17 demonstrates the mode; the Rhai Book documents it
  in English and Japanese.

- Flora grows and seeds from the base consonance field and density mass before
  habituation. Its consumer–resource depletion supplies its band fatigue; Voices
  and the listener retain habituation, including the drive from Flora's sound.
  Residual band noise is now off by default; `.residual(true)` enables it.
  Research assay templates explicitly retain their residual-on configurations.

## 0.5.0 — 2026-10-18

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
- `leave_self_out(true)` and `viability_scope("environment")` now remove only the Voice's own spectral share,
  estimated from each Voice's body projection, instead of the whole field at its partials; samples 08, 10, 12 and 13–16 sound different.

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
| `leave_self_out_mode(name)` | Removed; one self-exclusion method remains |
| `leave_self_out_harmonics(count)` | Removed; the body projection supplies the partials |
| `set_control_update_mode(name)` | `control_update_mode(name)` |
| `set_scaffold_off()` | `scaffold_off()` |
| `set_scaffold_shared(freq_hz)` | `scaffold_shared(freq_hz)` |
| `set_scaffold_scrambled(freq_hz, seed)` | `scaffold_scrambled(freq_hz, seed)` |

- `unison`, `top_k`, `random_candidates`, and `respawn_capacity` take integers only.
  Floating-point parameters accept integer arguments where applicable.
- `pitch_apply_mode`: `gate_snap`/`glide`/`rearticulate`; `pitch_core`: `hill_climb`/`peak_sampler`.
- `move_cost_time_scale`: `legacy`/`proposal_interval`.
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
