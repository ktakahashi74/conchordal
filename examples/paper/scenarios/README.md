# Paper Scenarios

These scripts are lightweight, headless-friendly demos that mirror the paper figure concepts.
They are intended for quick visual and behavioral checks, not for generating plots.
For E4, prefer fixed-population scripts to avoid density/history confounds.

## Run

```bash
cargo run -- --nogui examples/paper/scenarios/e1_landscape_scan_demo.rhai
cargo run -- --nogui examples/paper/scenarios/e1_landscape_scan_demo.rhai
```

## Scenarios

- `e1_landscape_scan_demo.rhai`: Anchor + ratio sweep to illustrate the E1 landscape scan idea.
- `e2_emergent_harmony_demo.rhai`: Conceptual sketch of emergent harmony around a shifting anchor.
- `e3_metabolic_selection_demo.rhai`: Conceptual sketch of selection pressure via repeated spawns.
- `e4_mirror_sweep_demo.rhai`: Fixed-population step response (`mirror_weight` sweep, no respawn inside loop).
- `e4_mirror_sweep_between_runs.rhai`: Between-run sweep (per-weight spawn/reset) for cleaner statistics.
- `e5_rhythmic_entrainment_demo.rhai`: Conceptual sketch using two pulse trains (entrainment flavor).

## E4 Notes

- Main analysis metrics should be `Root Affinity (R_A)`, `Overtone Affinity (O_A)`,
  `Binding Strength (S)`, and `Harmonic Tilt (τ)`.
- Keep interval heatmaps as secondary figures with guide lines (112/316/386/498/702 cents).
