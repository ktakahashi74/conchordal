//! Drift checks for the artifacts generated from the scripting surface.
//!
//! `rhai-defs/conchordal.d.rhai` and `docs/rhai_book/src/reference/api.md`
//! must match the output of `cargo run --bin gen_rhai_defs`, and every
//! registered function must have a doc entry in `src/scripting/docs.rs`.
//! If any of these fail, editor completion/diagnostics or the book reference
//! fall out of sync with the engine.
//!
//! Fix: update `src/scripting/docs.rs` if a function was added/removed, then
//! regenerate with `cargo run --bin gen_rhai_defs`.

use std::fs;
use std::path::PathBuf;

use conchordal::scripting::defs_gen;

#[test]
fn docs_registry_matches_engine_surface() {
    if let Err(err) = defs_gen::check() {
        panic!("src/scripting/docs.rs is out of sync with the engine surface:\n{err}");
    }
}

#[test]
fn rhai_defs_match_generator_output() {
    assert_generated_matches("rhai-defs/conchordal.d.rhai", &defs_gen::render_d_rhai());
}

#[test]
fn book_api_reference_matches_generator_output() {
    assert_generated_matches(
        "docs/rhai_book/src/reference/api.md",
        &defs_gen::render_reference_md(),
    );
}

fn assert_generated_matches(rel_path: &str, generated: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
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
            "{rel_path} is stale.\n\
             Regenerate with: cargo run --bin gen_rhai_defs\n\
             {hint}"
        );
    }
}

#[test]
fn rhai_lsp_config_includes_generated_definitions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("Rhai.toml");
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let parsed: toml::Table =
        toml::from_str(&contents).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let include = parsed
        .get("source")
        .and_then(|source| source.get("include"))
        .and_then(|include| include.as_array())
        .unwrap_or_else(|| panic!("Rhai.toml must define source.include"));
    let include: Vec<&str> = include.iter().filter_map(|value| value.as_str()).collect();
    for required in [
        "rhai-defs/**/*.d.rhai",
        "samples/**/*.rhai",
        "tests/scripts/**/*.rhai",
    ] {
        assert!(
            include.contains(&required),
            "Rhai.toml source.include must contain {required:?}"
        );
    }
}

#[test]
fn rhai_defs_use_lsp_parseable_function_names() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rhai-defs/conchordal.d.rhai");
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    for (idx, line) in contents.lines().enumerate() {
        let Some(rest) = line.strip_prefix("fn ") else {
            continue;
        };
        let Some((name, _)) = rest.split_once('(') else {
            continue;
        };
        assert!(
            is_rhai_identifier(name),
            "line {} uses a non-identifier function name not parseable by Rhai LSP: {name:?}",
            idx + 1
        );
    }
}

fn is_rhai_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
