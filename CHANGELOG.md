# Changelog

## Unreleased

- Remove `follow_own_tonality(w)` and its tonal listener-profile weight. Its fixed
  Krumhansl–Kessler major/minor profiles are style presets, not generatively
  computed harmony, and fail the author's criterion for retaining such priors.
  Sequence learning, melodic continuity and heard tonal context remain available;
  sensory stays the default. The removed word now uses Rhai's ordinary unknown-function error.

- Extend optional `listener_profile("tonal")` with the melodic continuity audition
  weight 0.00076066292. Explicit values, including zero, override inheritance;
  unsupported final Voices remain silent. Sensory stays the default. Tonal context
  remains an individual axis because of table cost and approximation scope.
- Add initial `listener_profile("sensory" | "tonal")`. Sensory remains the default;
  tonal bundles provisional sequence, phrase and final-section expression
  audition weights. Explicit axes (including zero) win; unsupported consumers are
  silently skipped without widening their support. Author adoption remains pending.
- Add initial-only `melodic_continuity(w)`, neutral by default. A von Hippel (2000)
  candidate predicts pitch from committed self-onset range and mobility; explicit
  author rules estimate prefix statistics and score existing proposals. Known
  predictions replace movement cost and fixed-center gravity; unknown or failed
  predictions restore them. SQ-1 remains independent. Legal-pitch scoring, bounded
  execution histories and numerical support checks preserve the selection contract.
- Add opt-in `follow_tonal_context(w)` for stepwise sustained Voices. The IPEM
  sensory-memory candidate observes continuous PCM; an authored tabulated sine
  probe rescores existing pitches and couples to dwell. Weight 0 preserves the
  existing path. Birth, ownership changes and gaps retain explicit context
  availability; presentation and habitat observations stay separate.

- Label all scripting terms and AppConfig keys by mechanism evidence in the
  documentation registry, generated bilingual API references and function hover.
  Mixed terms distinguish principle inputs, candidate models (including pending
  source matches) and author rules. Add English and Japanese composer guides;
  disclosure tiers and existing defaults are unchanged. This metadata change itself introduces no listener profile.

- Add initial-only `phrase_boundary(w)`, neutral by default, for discrete
  sustained Sine Voices. The LBDM candidate analyzes six executed On events;
  author rules publish local boundaries and soften one later legal On without
  changing strength, clocks or release. Renderer receipts remain distinct from
  SQ-1 issuance records. Veto, expiry, cancellation and loss of support discard
  the reaction; unsupported final forms warn and ignore the word.

- Add `expressive_section(name, arch, ritard, callback)` for a known final interval.
  Candidate Todd intensity and Friberg–Sundberg timing curves are separated from
  authored gain and clock rules. Zero keeps the original path; unsupported scenes
  warn and ignore the effect. Initial support covers Once without free rearticulation
  and non-sustained synchronized metric Voices, excluding Flora and participation.
  Gate durations follow the common score map; auditory and body time stay physical.

- Add initial-only `follow_own_sequence(w)` for stepwise sustained Voices, default
  zero. A fixed PPM-Decay candidate model learns executed self pitch intervals;
  an author rule adds positive follow after candidate generation. Sample-ordered
  execution records and paired counterfactual decisions distinguish learned sound
  from rejected targets. Glide and unsupported configurations warn and disable it.
  This does not learn rhythm or harmonic context, form phrases, or change defaults.

- Extend the shared production analysis grid down to 55·2^(−7/8) Hz
  (approximately 29.989 Hz), adding 84 low-frequency bins and four auditory
  channels. Runtime analysis, sample-rate validation, and body fallbacks use
  the same definition. H/R/C formulas and the 8 kHz upper bound remain unchanged;
  normalization and field-driven behavior can change across the whole grid.

- Flora now holds tonal and residual amplitude targets and their joint normalization
  between confirmed action opportunities, retaining slew and immediate support-loss
  exits. Unknown, stale or unconfirmed timing continues to update each hop.
  Faster-than-beat ecological renewal uses confirmed subdivisions (ratios 2–4)
  without resetting beat phase. Ecology and retirement saturate at one renewal
  per nominal opportunity, replacing the previous per-hop cap. Span/gap transitions
  retain beat crossings. Flora sound changes under these adopted rules.

- Flora adds initial-only `rest_on_habituation(w)` (default 0). Positive weights
  reuse the shared acoustic attack history to alternate spans and gaps. Exit uses
  tonal target-power-weighted habituation; all lanes retire at a private beat
  crossing while ecology continues. Return waits for every tail and private
  recovery, then permits births at a crossing. Reports identify Flora span/gap
  transitions; zero weight preserves the previous audio and report.

- Flora adds initial/live `pitch_apply_mode("glide" | "rearticulate")`.
  Glide remains the default. Rearticulate holds each tonal lane's pitch through
  retirement and replaces it when its peak moves beyond the 50-cent claim radius,
  using existing beat opportunities and tails. Live changes preserve sounding lanes.
  Unsupported values, including Voice's `gate_snap`, warn and retain the mode.

- Flora adds initial and live `pace(k)`, default 1, to scale ecological time relative
  to its private beat clock. Growth, spread, seeding, recovery, consumption and
  retirement share the multiplier. Beat opportunities, birth slew and macro fade
  retain their clocks. Pace retains the 1/64 finite-retirement minimum with no cost-based upper
  bound; ecological time now saturates at 10 seconds per nominal opportunity. The composer owns
  whole-scene real-time cost. `rest_on_habituation(w)` and `dwell(w)` accept
  finite nonnegative weights above 1 without changing their laws.

- Flora now removes its actual own habitat PCM by default before computing growth
  consonance and seed harmonicity, sharing the subtraction with external resource power.
  Other Flora remain audible. Base views retain the mixed landscape's current
  kernel and sigmoid parameters; reports expose the consumed scans. Placement
  still uses mixed density mass before own sound exists. Presentation-only Flora
  reads mixed C/H without an H/R observer. Analysis gaps retain the existing
  runtime stop rule, with no mixed fallback for own-excluded observations.
  Initial-only `flora().leave_self_out(false)` selects mixed growth/seed C/H
  without an H/R observer, retaining resource-power self-exclusion. Flora defaults
  to true and Voice to false: Flora's many partials can sustain their own lattice.

- Initial-only `dwell(w)` slows free `seek_consonance()` pitch proposals while the
  current habituated environment level is high. A separate weighted phase leaves
  adaptation in real seconds and preserves existing movement costs and legal
  onset decisions. Default zero retains the previous behavior. Reports include
  proposal clocks, environment source and terminal state. This is an author rule,
  not a model of tonal stability or motive formation.

- Initial-only `rest_on_habituation(w)` adds probabilistic spans and tail-aware
  pauses to explicit sustained metric/entrained/steady/flow Voices. Habitat PCM
  supplies shared attack habituation; executed private attacks govern return.
  Existing opportunities and vetoes remain in force. Default zero allocates no
  observer or private random stream. Reports include state transitions and the
  shared scan. Recovery excludes front-end restart transients using the existing
  analysis window, band boxcars and filter-derived settling times.

- Every Flora now uses the existing T1 beat-opportunity rule. The shared habitat
  audio-only meter runs from scene start whenever Flora is placed; each Flora has
  its own confidence-weighted clock. Unknown, stale or unconfirmed observations
  retain continuous allocation and basin pace. `FloraSpec.metric` and
  `flora().metric()` are removed. Birth slew, finite retirement, resource laws and
  routing are unchanged. The old T0/T1 research pair is archived in history;
  the remaining basin pair compares tempo priors under the new default.

- Auditory validation retains the V2 consumer tools and ordinary conformance
  tests while removing completed one-shot acquisitions. V3 onset and V4 history
  checks now run as ordinary tests without writing acquisition artifacts.

- Runtime wiring now skips NSGT power and spectral history when no GUI, report,
  temporal observation tap or Flora resource consumer needs them. Listener streams
  also skip this work without a GUI, report or listener-side temporal tap.
  Auditory H/R/C, behavior and gap recovery retain their existing rules.

- Listener stability and resolvability now aggregate components defined by local
  mass maxima. Adjacent bins representing one component no longer offer an
  internal resolution move. Evidence thresholds, movement costs and tension
  coefficients retain their values. The validation fixture compares stable,
  resolved and settled tension with the tension cluster, accounting for audible
  upper partials that the former input pruned.

- H/R perception now uses a causal complex auditory filterbank (bandwidth factor
  1/4), channel-dependent hop boxcars, reassigned power and critical-band
  loudness compression. Voice self-exclusion skips NSGT; mixed/listener streams
  retain spectral power, and Flora resource analysis keeps its existing law.
  Full-hop boundaries and sample-rate validation apply before analysis. Reported
  loudness and roughness values follow the new density; configuration adds no keys.

- All Flora now derives ecological pace from temporal DCC: T0 keeps the
  tempo-basin center (2 Hz by default); metric T1 follows confirmed heard beats
  and falls back to the center. All retiring lanes use a finite metabolic tail,
  bounded by macro fade; explicit fade(0) remains immediate. Independent `rate`
  and `renewal` methods are removed. Research assays use tempo priors, changing
  former growth-only comparisons into whole-ecology pace comparisons. Sample 17
  uses T0 with the 2–4 Hz basin for its sustained scene.

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
