set shell := ["bash", "-euo", "pipefail", "-c"]

cargo := env_var_or_default("CARGO", "cargo")

default:
    @just --list

fmt:
    {{cargo}} fmt --all

clippy:
    {{cargo}} clippy -- -D warnings

test:
    #!/usr/bin/env bash
    set -o pipefail
    ( RUST_BACKTRACE=1 {{cargo}} test -- --nocapture ) 2>&1 | tee test_report.txt
    rc=$?
    echo "cargo test exit=$rc @ $(date -Iseconds)" > test_status.txt
    exit "$rc"

gen_rhai_defs:
    {{cargo}} run --bin gen_rhai_defs

mdbook:
    mdbook build docs/rhai_book
    mdbook build docs/rhai_book_ja
