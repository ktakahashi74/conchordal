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
cargo run --example paper -- --exp e2
cargo run --example paper -- --exp e2 --e2-phase normal
cargo run --example paper -- --exp e4
cargo run --example paper -- --exp e4 --e4-hist on
```

Run twice to confirm `target/plots/paper` is cleared each time (no stale outputs remain).
