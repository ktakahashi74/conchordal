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
cargo run --example paper -- --exp e4 --e4-legacy on
```

```bash
cargo run --example paper -- --exp e4 --e4-debug-fit-metrics on
```

```bash
cargo run --example paper -- --exp e4 --e4-env-partials 9 --e4-env-decay 0.8
```

```bash
cargo run --example paper -- --exp e4 --e4-dyn-exploration 0.9 --e4-dyn-persistence 0.1 --e4-dyn-step-cents 75
```

```bash
cargo run --example paper -- --exp e2 --e2-phase normal
```

Default E2 phase is `dissonance_then_consonance` when `--e2-phase` is omitted.
E4 legacy outputs are disabled by default and are emitted only with `--e4-legacy on`.
E4 fit debug CSV outputs are disabled by default and emitted only with `--e4-debug-fit-metrics on`.

Outputs are written to `examples/paper/plots/<exp>/` (for example, `examples/paper/plots/e2/`).
Plot images are emitted as `.svg` files (vector output).
Default behavior: only selected experiment directories are cleared and regenerated.
Use `--clean` to clear `examples/paper/plots` entirely before generation.

## Manual verification

```bash
cargo check --example paper
cargo check --examples
cargo check --all-targets
cargo test --examples
cargo run --example paper -- --clean --exp e4
find examples/paper/plots/e4 -maxdepth 1 -type f | rg 'delta_bind|root_fit|ceiling_fit|e4_fit_metrics|paper_e4_fit_metrics|e4_bind_metrics|e4_bind_summary' || true
```

Primary E4 outputs to verify:
- `paper_e4_binding_metrics_raw.csv`
- `paper_e4_binding_metrics_summary.csv`
- `paper_e4_harmonic_tilt.png`
- `paper_e4_binding_phase_diagram.png`

Secondary output:
- `paper_e4_fingerprint_heatmap.png` (fingerprint view)

Use `--clean` when you need strict reproducibility from a fully fresh `plots` tree.

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
