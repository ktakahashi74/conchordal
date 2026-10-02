# Body fitness: moving non-Sine runtime registration v2

Date: 2026-09-27. Status: fixed before v2 acquisition. This is a new functional acquisition following the failed `at(freq)` movement condition recorded in [the original registration](body-fitness-runtime-changes-registration-20260927.md). The original result remains a failed movement result; v2 does not retroactively pass it.

## Single scenario correction

Keep seed 17, 48,000 Hz/512-sample hops, `body_fitness_action = true`, the phase-zero frequency representative recipe, Harmonic and Modal Voices, drone articulation, brightness values, buses, ADSR, pitch parameters, update times, scene lengths, and two-run deterministic comparisons exactly as in the original registration. Replace only the initial placement expressions `at(220.0)` and `at(330.0)` with `line(220.0, 220.0)` and `line(330.0, 330.0)`. A zero-width line selects the same spawn frequency but carries `Placement.freq_hz = None`. `place_population_spec` therefore does not call `PopulationSpec::set_freq` after `seek_consonance()`. The existing `spawn_strategy_respects_free_pitch_mode` test establishes that `SpawnStrategy::Linear` retains `PitchMode::Free` while setting the Voice frequency. No scenario duration, seed, temperature, or source count changes.

## Required observations

Scenario A must meet the original non-Sine conditions: each source consumes body-prepared scores, at least one reported consumed target differs before versus after, physical frequency differs from its initial value, job age is at most 4,800 samples, and bounded caches remain below 256 entries and 1 MiB. The two runs must match in WAV bytes, action reports, and population-frequency reports. The target and physical movement are observations of the whole pitch policy at temperature 1.0, not proof that body fitness alone caused the move.

Scenario B must meet the original update conditions: both sources consume before change; Harmonic brightness raises body generation and is followed by new-generation consumption; the old control decision is rejected as `Consume(ControlChanged)` at a due gate after the control update and later consumption recovers; Modal generation remains fixed and its consumption continues. Both runs must match in WAV bytes, action reports, and population-frequency reports. No route-mutation conclusion follows from these normal-render scenes.

Acquire into a fresh evidence directory. If v2 fails, retain its raw reports and identify the failing condition. Do not tune the registered seed or temperature after seeing output.

## Acquisition result

The focused `body_fitness_action_changes` test passed with the v2 scene. Full raw reports, WAVs, scenario scripts, config files, `summary.json`, and binary/source SHA manifest are in `target/body-fitness-ecology-build-20260927/runtime-action-ecology-evidence/run-142-1790482642154955205/`. The command log is `target/runtime-action-ecology-validation/non-sine-changes-v2.log`. The render binary SHA-256 is `079c51c9d2ba3f073a1648187abb3d999f8d9e23adac1471a9c06e8fd62535dc`. The two runs of each scene matched in WAV bytes, action reports, and population-frequency reports.

In Scenario A, both sources consumed 30 prepared decisions by report frame 144, with body generation 1 throughout and positive score-use counts. The last reported Harmonic target moved from 7.54178 to 7.51053 log2 Hz and its physical frequency was 189.67 Hz; the Modal target moved from 6.86469 to 6.71886 log2 Hz and its physical frequency was 116.18 Hz. Their initial frequencies were 220 and 330 Hz. The last reported cache contained 256 entries/727,040 bytes per source, within the registered bounds. All observed consumed decision timestamps met the 4,800-sample age bound.

In Scenario B, both sources had consumed 10 decisions by frame 48. By frame 96, Harmonic body generation rose from 1 to 2 and its latest consumed decision used generation 2. By frame 144, a stale decision had been rejected once as `Consume(ControlChanged)`; by frame 192, Harmonic consumption had risen to 40 with generation 2. Modal body generation stayed 1 and consumption rose to 41. The registered change and recovery conditions passed.

The target and physical frequency changes establish that the normal runtime consumed prepared decisions while the actual Voice moved under the registered complete pitch policy. They do not isolate the causal contribution of the body-fitness term from temperature, gravity, or other pitch terms. The original `at(freq)` scene and its failed movement outcome remain documented separately.
