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

Outputs are written to `target/plots/paper/<exp>/` (for example, `target/plots/paper/e2/`).
