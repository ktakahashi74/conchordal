# Paper experiments

Example runner for generating plots used in the paper.

## Run

```bash
cargo run --example paper
```

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

Outputs are written to `target/plots/paper/<exp>/` (for example, `target/plots/paper/e2/`).
`target/plots/paper` is cleared on each run.

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

Run twice to confirm `target/plots/paper` is cleared each time (no stale outputs remain).
