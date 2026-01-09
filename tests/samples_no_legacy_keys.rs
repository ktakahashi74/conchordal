use std::fs;
use std::path::{Path, PathBuf};

const PLAN_A: &str = "plan";
const PLAN_B: &str = "ning";
const BIRTH_A: &str = "on";
const BIRTH_B: &str = "_birth";
const SUSTAIN_A: &str = "sustain";
const SUSTAIN_B: &str = "_update";
const TIMING_A: &str = "ti";
const TIMING_B: &str = "ming";

fn collect_sample_files(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(err) => panic!("failed to read {:?}: {err}", root),
    };
    for entry in entries {
        let entry = entry.expect("sample entry");
        let path = entry.path();
        if path.is_dir() {
            collect_sample_files(&path, out);
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext, "rhai" | "json" | "yaml" | "yml") {
            out.push(path);
        }
    }
}

#[test]
fn samples_have_no_legacy_keys() {
    let root = Path::new("samples");
    let mut files = Vec::new();
    collect_sample_files(root, &mut files);
    assert!(!files.is_empty(), "no sample files found under {root:?}");

    let banned_raw = [
        format!("{PLAN_A}{PLAN_B}"),
        format!("{BIRTH_A}{BIRTH_B}"),
        format!("{SUSTAIN_A}{SUSTAIN_B}"),
    ];
    let phonation_key = ["pho", "nation"].concat();
    let quote = "\"";
    let mut banned_compact = Vec::new();
    let timing_key = format!("{TIMING_A}{TIMING_B}");
    banned_compact.push(format!("{timing_key}:{quote}"));
    banned_compact.push(format!("{quote}{timing_key}{quote}:"));
    for value in ["once", "off", "immediate"] {
        banned_compact.push(format!("{phonation_key}:{quote}{value}{quote}"));
        banned_compact.push(format!(
            "{quote}{phonation_key}{quote}:{quote}{value}{quote}"
        ));
    }

    for path in files {
        let contents = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {:?}: {err}", path));
        for token in &banned_raw {
            assert!(
                !contents.contains(token),
                "legacy token \"{token}\" found in {path:?}"
            );
        }
        let compact: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
        for token in &banned_compact {
            assert!(
                !compact.contains(token),
                "legacy token \"{token}\" found in {path:?}"
            );
        }
    }
}
