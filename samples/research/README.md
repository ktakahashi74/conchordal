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

## Flora tempo priors

`flora_metric_basin_slow.rhai` and `flora_metric_basin_fast.rhai` use the same
90-second sustained field and seed 21. Their only parameter difference is the
tempo basin, 1–2 versus 2–4 Hz. Metric Flora follows the confidence-weighted heard
beat and uses the basin center when evidence is unavailable; the chosen prior
therefore does not prescribe a fixed realized tempo. The fast basin is one octave
above the slow basin, a controlled compositional contrast, not a fitted optimum.

The earlier `flora_succession_*`, `flora_mixing_*`, and ownership assays retain
continuous T0 with their explicit rate/renewal and residual settings. Their
historical measurements still describe T0. They are not silently migrated to the
new metric pace or retirement law, and do not establish its closed-loop behavior.
