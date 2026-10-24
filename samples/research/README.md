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

`flora_basin_slow.rhai` and `flora_basin_fast.rhai` retain the former T1 scenes:
90 seconds of sustained sound with seed 21, differing only in the tempo basin,
1–2 versus 2–4 Hz. Every Flora now uses the heard beat with confidence-weighted
coupling and returns to the basin center when evidence is unknown, stale or
unconfirmed. The prior does not prescribe a fixed realized tempo. The fast basin
is one octave above the slow basin, a compositional contrast, not a fitted optimum.

The T0 pair that cut beat feedback has been removed; the T1 pair no longer calls
`metric()` and uses the shorter filenames above. The last commit containing both
T0 and T1 controls is `16ccf89682deaeb89cc8e5af5b108bef43ba878d`.
Historical T0 acquisitions include the 2026-10-23 pace/retirement comparison
recorded in the Flora design note's Implementation Findings. Their saved input,
source and binary hashes identify the old implementation. Those recordings remain
historical evidence; running the current pair cannot reproduce a T0/T1 causal
ablation. It compares priors with beat feedback enabled in both scenes.

The succession, mixing, lone, low-register, overlap, routing and ownership assays
retain their placement, bus, muting and resource contrasts. They now exercise the
same default beat feedback; their old waveform and succession measurements are
not numerical baselines for the new version. `flora_succession_rate8` retains its
historical filename but compares whole ecological pace, not growth alone or a
fixed realized rate. Former rate 4/s assays use a 1 Hz prior and former rate 8/s
assays a 2 Hz prior. Recovery, consumption, diffusion and seed rain share that pace.
Producer-compatibility assays still exercise the two bodies and residual choices.
Archived reduced-model experiments remain historical analyses, not executable
specifications of the current instrument.
