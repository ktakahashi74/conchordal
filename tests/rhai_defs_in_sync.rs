//! Drift check: `rhai-defs/conchordal.d.rhai` must match the output of
//! `cargo run --bin gen_rhai_defs`. If a new `register_fn` was added without
//! regenerating the file, editor completion / diagnostics fall out of sync
//! with the engine. This test catches that on CI.
//!
//! Fix: `cargo run --bin gen_rhai_defs > rhai-defs/conchordal.d.rhai`

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn rhai_defs_match_generator_output() {
    let bin = env!("CARGO_BIN_EXE_gen_rhai_defs");
    let output = Command::new(bin)
        .output()
        .expect("failed to run gen_rhai_defs");
    assert!(
        output.status.success(),
        "gen_rhai_defs exited non-zero: {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let generated = String::from_utf8(output.stdout).expect("non-UTF8 stdout");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rhai-defs/conchordal.d.rhai");
    let committed =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));

    if generated != committed {
        let first_diff_line = generated
            .lines()
            .zip(committed.lines())
            .position(|(a, b)| a != b);
        let hint = match first_diff_line {
            Some(i) => format!(
                "first diff at line {}:\n  generated: {}\n  committed: {}",
                i + 1,
                generated.lines().nth(i).unwrap_or(""),
                committed.lines().nth(i).unwrap_or(""),
            ),
            None => format!(
                "line counts differ: generated={} committed={}",
                generated.lines().count(),
                committed.lines().count(),
            ),
        };
        panic!(
            "rhai-defs/conchordal.d.rhai is stale.\n\
             Regenerate with: cargo run --bin gen_rhai_defs > rhai-defs/conchordal.d.rhai\n\
             {hint}"
        );
    }
}
