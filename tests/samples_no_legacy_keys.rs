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

fn contains_identifier(contents: &str, needle: &str) -> bool {
    contents.match_indices(needle).any(|(idx, _)| {
        let before = contents[..idx].chars().next_back();
        let after = contents[idx + needle.len()..].chars().next();
        !before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char)
    })
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

#[test]
fn samples_have_no_legacy_keys() {
    let mut files = Vec::new();
    collect_sample_files(Path::new("samples"), &mut files);
    collect_sample_files(Path::new("tests/scripts"), &mut files);
    assert!(!files.is_empty(), "no sample or script files found");

    let banned_raw = [
        format!("{PLAN_A}{PLAN_B}"),
        format!("{BIRTH_A}{BIRTH_B}"),
        format!("{SUSTAIN_A}{SUSTAIN_B}"),
        ".energy(".to_string(),
        ".mode(".to_string(),
        ".pitch_apply(".to_string(),
        ".voices(".to_string(),
        "create(".to_string(),
        "sine(\"".to_string(),
        "harmonic(\"".to_string(),
        "modal(\"".to_string(),
        "saw(\"".to_string(),
        "square(\"".to_string(),
        "noise(\"".to_string(),
        "variant(\"".to_string(),
        "derive(".to_string(),
        ".accent(".to_string(),
        ".sync(".to_string(),
        ".gates(".to_string(),
        ".field()".to_string(),
        ".field_window(".to_string(),
        ".field_curve(".to_string(),
        ".field_drop(".to_string(),
        ".send(field)".to_string(),
        ".send(presentation)".to_string(),
        "field | presentation".to_string(),
        ".crowding(".to_string(),
        "set_harmonic_mirror(".to_string(),
        ".movement_glide(".to_string(),
        ".pitch_glide(".to_string(),
        "consonance_density(".to_string(),
        "random_log(".to_string(),
        "linear(".to_string(),
        ".field_only(".to_string(),
        ".presentation_only(".to_string(),
        // Pre-continuum rhythm verbs (replaced by metric()/entrained()/flow()
        // + entrainment()/rhythm_role()/microtiming() and director
        // meter_stability()/temporal_basin()).
        "metric_beat(".to_string(),
        "entrained_beat(".to_string(),
        "flow_timing(".to_string(),
        "beat_strength(".to_string(),
    ];
    let phonation_key = ["pho", "nation"].concat();
    let banned_identifiers = ["field_bus", "generator_field_bus"];
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
        for token in &banned_identifiers {
            assert!(
                !contains_identifier(&contents, token),
                "legacy identifier \"{token}\" found in {path:?}"
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
