# Paper experiments

Example runner for generating plots used in the paper.

Paper scenarios used for headless behavioral checks are under `examples/paper/scenarios/`.

## Run

```bash
cargo run --example paper
```

## Just recipes

```bash
just --justfile examples/paper/justfile paper --exp e2
```

```bash
just --justfile examples/paper/justfile paper-pdf --exp e2
```

`paper-pdf` runs the plot generator and converts all emitted SVG files in
`examples/paper/plots/` to PDF. Conversion uses `rsvg-convert` or `inkscape`.
`paper` rejects concurrent runs with a lock at `examples/paper/.paper_plots.lock`.
If a previous run crashed, remove that lock directory and retry.

## Build check

```bash
cargo check --example paper
```

```bash
cargo check --examples
```

```bash
cargo check --all-targets
```

```bash
cargo test --examples
```

## Options

```bash
cargo run --example paper -- --exp e3
```

```bash
cargo run --example paper -- --exp e4 --e4-hist on
```

```bash
cargo run --example paper -- --exp e2 --e2-phase normal
```

Default E2 phase is `dissonance_then_consonance` when `--e2-phase` is omitted.

Outputs are written to `examples/paper/plots/<exp>/` (for example, `examples/paper/plots/e2/`).
Plot images are emitted as `.svg` files (vector output).
`examples/paper/plots` is cleared on each run.

## Manual verification

```bash
cargo check --example paper
cargo check --examples
cargo check --all-targets
cargo test --examples
cargo run --example paper -- --exp e2
cargo run --example paper -- --exp e2 --e2-phase normal
cargo run --example paper -- --exp e4
cargo run --example paper -- --exp e4 --e4-hist on
```

Verify `e4_seed_slopes.csv`, `e4_run_level_regression.csv`, `e4_seed_slope_meta.csv`, and
`e4_total_third_mass.csv` are non-empty, and that eps labels show `12.5c` when enabled.

Notes:
- `e4_seed_slope_meta.csv` is a seed-level slope sign test with mean CI (not a mixed-effects model).
- `e4_run_level_regression.csv` is descriptive only; use the seed-level summary for primary claims.

Run twice to confirm `examples/paper/plots` is cleared each time (no stale outputs remain).

## Paper scenarios (headless demos)

```bash
cargo run -- --nogui examples/paper/scenarios/e1_landscape_scan_demo.rhai
```

Files:
- `e1_landscape_scan_demo.rhai`
- `e2_emergent_harmony_demo.rhai`
- `e3_metabolic_selection_demo.rhai`
- `e4_mirror_sweep_demo.rhai`
- `e4_mirror_sweep_between_runs.rhai`
- `e5_rhythmic_entrainment_demo.rhai`
