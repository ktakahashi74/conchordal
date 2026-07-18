//! Every `rhai` code fence in the book must execute against the real script
//! engine. This is the drift guard that keeps the book honest: an example
//! calling a removed or renamed function fails here, not in a reader's hands.
//!
//! Conventions:
//! - ```` ```rhai ```` fences are extracted and run (scenario compilation,
//!   no audio); examples must be self-contained.
//! - ```` ```rhai,ignore ```` fences are skipped (signature fragments in the
//!   generated API reference, deliberately non-runnable snippets).

use std::fs;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use conchordal::app::compile_scenario_from_script;
use conchordal::cli::Args;
use conchordal::config::AppConfig;

struct Fence {
    source: PathBuf,
    start_line: usize,
    code: String,
}

fn extract_rhai_fences(path: &Path) -> Vec<Fence> {
    let contents =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut fences = Vec::new();
    let mut current: Option<(usize, Vec<&str>)> = None;
    for (idx, line) in contents.lines().enumerate() {
        let trimmed = line.trim_start();
        match &mut current {
            Some((start, code)) => {
                if trimmed.starts_with("```") {
                    fences.push(Fence {
                        source: path.to_path_buf(),
                        start_line: *start + 1,
                        code: code.join("\n"),
                    });
                    current = None;
                } else {
                    code.push(line);
                }
            }
            None => {
                if let Some(info) = trimmed.strip_prefix("```")
                    && info.trim() == "rhai"
                {
                    current = Some((idx, Vec::new()));
                }
            }
        }
    }
    assert!(
        current.is_none(),
        "unterminated code fence in {}",
        path.display()
    );
    fences
}

#[test]
fn book_rhai_examples_execute() {
    let mut fences = Vec::new();
    for book_src in [
        Path::new("docs/rhai_book/src"),
        Path::new("docs/rhai_book_ja/src"),
    ] {
        for entry in WalkDir::new(book_src).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            fences.extend(extract_rhai_fences(path));
        }
    }
    assert!(
        !fences.is_empty(),
        "no `rhai` code fences found in either Rhai book"
    );

    let tmp_dir = std::env::temp_dir().join(format!("rhai_book_examples_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    for (idx, fence) in fences.iter().enumerate() {
        let script_path = tmp_dir.join(format!("fence_{idx}.rhai"));
        fs::write(&script_path, &fence.code)
            .unwrap_or_else(|e| panic!("write {}: {e}", script_path.display()));
        let args = Args {
            play: false,
            scenario_path: script_path.to_string_lossy().to_string(),
            config: "config.toml".to_string(),
            wait_user_exit: None,
            wait_user_start: None,
            nogui: false,
            compile_only: false,
            report: None,
            seed: None,
        };
        compile_scenario_from_script(&script_path, &args, &AppConfig::default()).unwrap_or_else(
            |e| {
                panic!(
                    "book example failed ({}:{}):\n{}\n\nerror: {e}",
                    fence.source.display(),
                    fence.start_line,
                    fence.code
                )
            },
        );
    }

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn translated_book_has_the_same_chapter_paths() {
    fn chapter_paths(root: &Path) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .map(|entry| entry.into_path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("md"))
            .map(|path| {
                path.strip_prefix(root)
                    .expect("book path prefix")
                    .to_path_buf()
            })
            .collect();
        paths.sort();
        paths
    }

    let english = chapter_paths(Path::new("docs/rhai_book/src"));
    let japanese = chapter_paths(Path::new("docs/rhai_book_ja/src"));
    assert_eq!(
        english, japanese,
        "English and Japanese chapter paths drifted"
    );
}

#[test]
fn translated_book_keeps_rhai_examples_identical() {
    let english_root = Path::new("docs/rhai_book/src");
    let japanese_root = Path::new("docs/rhai_book_ja/src");

    for entry in WalkDir::new(english_root)
        .into_iter()
        .filter_map(Result::ok)
    {
        let english_path = entry.path();
        if english_path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let relative = english_path
            .strip_prefix(english_root)
            .expect("English book path prefix");
        let japanese_path = japanese_root.join(relative);
        let english: Vec<String> = extract_rhai_fences(english_path)
            .into_iter()
            .map(|fence| fence.code)
            .collect();
        let japanese: Vec<String> = extract_rhai_fences(&japanese_path)
            .into_iter()
            .map(|fence| fence.code)
            .collect();
        assert_eq!(
            english,
            japanese,
            "Rhai examples differ between {} and {}",
            english_path.display(),
            japanese_path.display()
        );
    }
}
