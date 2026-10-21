# Research assays

Comparison fixtures for studying the instrument, not for playing it:

- `hereditary_adaptation_*.rhai` — heredity/selection ablations (E6, the
  paper's Hereditary Adaptation experiment; see
  `docs/hereditary_adaptation_samples.md`), including the approximate
  leave-one-out variants.
- `temporal_scaffolding_*.rhai` — external-scaffold rhythm controls
  (off / shared / scrambled) against the emergent meter.
- `consonance_field_control.rhai` — field-state navigation mechanism study.
- `hillclimb_stable_convergence.rhai` — movement-mechanism convergence study.
- `drift_flow.rhai` — early drift/flow sketch (cited by the technote case
  studies).
- `flora_lone.rhai`, `flora_low.rhai`, `flora_overlap.rhai` — closed-loop
  spectral-growth assays in the middle register, low register, and overlapping extents.
- `flora_mixing_tuning.rhai` — 110 Hz drone, 110–1760 Hz Flora, seed 41101.
- `flora_mixing_evaluation_{1,2,3}.rhai` — 146.83 Hz drone, 146.83–2349.3 Hz
  Flora, held-out seeds 42101–42103.
- `flora_succession_drone.rhai` — 110 Hz drone, retuned to 146.83 Hz at 42 s; seed 43001.
- `flora_succession_mute.rhai` — Flora muted at 42 s and restored at 62 s; seed 43002.
- `flora_succession_population.rhai` — habitat drone and five moving seekers; seed 43003.
- `flora_succession_habitat_only.rhai` — Flora routed only to habitat; seed 43004.
- `flora_succession_presentation_only.rhai` — Flora routed only to presentation; seed 43005.
- `flora_succession_rate8.rhai` — Flora at rate 8/s against a 110 Hz drone; seed 43006.
- `flora_succession_low.rhai` — 55 Hz drone and 55–220 Hz Flora; seed 43007.
- `flora_succession_overlap.rhai` — 110–880 Hz and 220–1760 Hz Flora; seed 43008.

Headless report runs are recorded in
`docs/roadmap/v0.4.0-rhythm-report-runs.md`.

The assay scripts intentionally duplicate their baseline setup so each file
can run alone and the test suite can compile-check every comparison point.

## Hereditary adaptation axis

| File suffix | Spawn rule | Selection | Selection score |
|---|---|---|---|
| `random_only` | random respawn | off | none |
| `heredity_only` | consonance respawn | off | none |
| `random_selection` | random respawn | on | total |
| `heredity_selection` | consonance respawn | on | total |
| `random_selection_approx_loo` | random respawn | on | approximate leave-one-out |
| `heredity_selection_approx_loo` | consonance respawn | on | approximate leave-one-out |

## Temporal scaffold axis

| File suffix | Global scaffold | Expected behavior |
|---|---|---|
| `off` | none | attacks drift freely |
| `shared` | one shared phase | attacks align to a common grid |
| `scrambled` | per-voice randomized phase | local reward remains, global lock weakens |

## Flora campaigns

These offline integration tests are ignored by the default test run:

```bash
cargo test --test flora_assays closed_loop -- --ignored --exact --nocapture
cargo test --test flora_assays mixing_tuning -- --ignored --exact --nocapture
cargo test --test flora_assays mixing_evaluation -- --ignored --exact --nocapture
cargo test --release --test flora_assays succession -- --ignored --nocapture
```

Each runs sine with residual, sine without residual, and resonant with residual.
The closed-loop campaign runs all three extents in all three configurations.
Generated scenarios, explicit 48 kHz / 512-sample-hop configurations, WAVs, logs,
raw reports, and `per_second.jsonl` go under `target/flora-assays/<campaign>/`.
Only `conchordal-render` writes WAVs; the instrument is not used for recording.

The renderer's opt-in `--report` includes one Flora snapshot at each elapsed
second's first completed hop. Each has the latent PMF and peak frequencies,
rendered lane tuples `(frequency_hz, nominal_power, is_tonal)`, cumulative births,
retirements, flapping, and tonal/residual energy. Births and retirements count tonal
lane selection events. Flapping means a newly born tonal lane retires within 80 ms.
`per_second.jsonl` adds counter differences and interval-mean tonal/residual power.
These powers use the renderer's deterministic lane-power proxy, including ramps
and retiring lanes, without random envelope fluctuations or cross terms. They are
not estimates of the complete waveform's acoustic power. Latent peaks and the
rendered partial set are snapshots at the second boundary, not interval averages.

Mixing sends the drone to the habitat bus and Flora to both buses. Thus the drone
participates in the closed loop while the WAV contains only Flora. The drone has
16 explicit harmonic modes. Measurements use seconds [18, 34), after Flora has
been sounding for 16 seconds. Latent fractions average the PMFs at the 16 ending
boundaries. Rendered fractions use summed one-second Hann-windowed power spectra
of the PCM WAV, zero-padded to 65536 FFT points. Both measurements condition on
the Flora extent and compare against equal mass per in-extent Log2Space bin
(96 bins/octave). Masks count mass within 0.5 ERB-rate of any drone partial and
within ±25 cents of any multiple of f0/2. They count each bin at most once.
`spectrum_per_second.jsonl` records the two waveform fractions for each measured
second; `mixing.json` records aggregate latent, rendered, and uniform fractions.

`tests/fixtures/flora_mixing_protocol.json` freezes the seeds, window, reference,
and pass rule before evaluation. Passing requires both representations, in every
configuration and every evaluation seed, to avoid drone partials below the uniform
fraction and exceed the uniform common-root fraction by the respective fixed
margins. Margins are currently unregistered (`null`), so the evaluation test remains
ignored and reports measurements with `accepted: null`. Successful execution is
not mixing acceptance. Never choose margins from these evaluation results; a future
registered acceptance campaign needs fresh evaluation seeds. No growth parameter
is adjusted by either campaign, and the known settling dynamics remain unchanged.

The succession campaign renders all eight `flora_succession_*` samples in all three
configurations and writes `succession/summary.json` and `succession/summary.md`.
The two 106 s samples release at 102 s; the six 92 s samples release at 88 s.
All Flora are placed at 2 s. Summary measurements use [22 s, release), with
additional mute level windows [22, 42), [42, 62), and [67, 102) s. Snapshot windows
use the report's integer second; counter windows use the intervals ending within
(begin, end], with each rate divided by its actual interval duration. The JSON
includes every per-second normalized spectrum and metric, including release tails,
and marks undefined measurements as `null`.

Rendered spectra use all reported partials, including residual and retiring lanes,
on a 0.05 ERB-rate grid over each extent, smoothed with sigma 0.25 ERB. Each
Gaussian is normalized on the extent before depositing its partial's power, so edge
partials retain their weight. Silent snapshots are excluded from spectrum distances,
lattice fractions, and participation. Recurrence compares each non-silent second
with all non-silent seconds at least 5 s earlier after warm-up; its denominator
includes only seconds with an eligible predecessor. Overlapping Flora are also
measured together by pooling their partials and summing counters and biomass,
without waveform interference terms. No acceptance threshold is applied.
