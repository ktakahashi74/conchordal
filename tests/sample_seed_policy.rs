use std::fs;
use std::path::Path;

fn has_seed_statement(contents: &str) -> bool {
    contents.lines().any(|line| {
        let code = line.split("//").next().unwrap_or("").trim_start();
        code.starts_with("seed(")
    })
}

#[test]
fn top_level_samples_are_unseeded_and_research_assays_are_seeded() {
    let mut sample_count = 0;
    for entry in fs::read_dir(Path::new("samples")).expect("read samples dir") {
        let path = entry.expect("sample entry").path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        sample_count += 1;
        let contents = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        assert!(
            !has_seed_statement(&contents),
            "top-level sample should use a fresh scenario seed: {}",
            path.display()
        );
    }
    assert_eq!(sample_count, 18, "expected the eighteen top-level samples");

    let mut research_count = 0;
    for entry in fs::read_dir(Path::new("samples/research")).expect("read research dir") {
        let path = entry.expect("research entry").path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        research_count += 1;
        let contents = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        assert!(
            has_seed_statement(&contents),
            "research assay should keep a fixed seed: {}",
            path.display()
        );
    }
    assert!(research_count > 0, "expected research assay scripts");
}

/// Demo pieces are performed live in the browser, so each start must be a new performance.
#[test]
fn demo_pieces_are_listed_and_unseeded() {
    let dir = Path::new("samples/demo");
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(dir.join("pieces.json")).expect("read samples/demo/pieces.json"),
    )
    .expect("parse samples/demo/pieces.json");
    let mut listed = Vec::new();
    for piece in manifest.as_array().expect("pieces.json is an array") {
        let id = piece["id"].as_str().expect("piece id");
        assert!(
            !id.is_empty()
                && id
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "piece id must be lowercase ASCII, digits or '-': {id}"
        );
        for key in ["score", "config"] {
            let Some(name) = piece.get(key).and_then(|v| v.as_str()) else {
                assert_ne!(key, "score", "{id} has no score");
                continue;
            };
            assert!(
                !name.contains('/') && dir.join(name).is_file(),
                "{id}: {key} must name a file in samples/demo: {name}"
            );
            listed.push(name.to_owned());
        }
        let score = fs::read_to_string(dir.join(piece["score"].as_str().unwrap())).unwrap();
        assert!(
            !has_seed_statement(&score),
            "demo piece should use a fresh scenario seed: {id}"
        );
        for key in ["title", "authored", "generated"] {
            for locale in ["en", "ja"] {
                assert!(
                    piece[key][locale].as_str().is_some_and(|s| !s.is_empty()),
                    "{id}: missing {key}.{locale}"
                );
            }
        }
        assert!(
            piece.get("authored_harmony").is_some(),
            "{id}: authored_harmony must be stated, null when absent"
        );
    }
    for entry in fs::read_dir(dir).expect("read samples/demo") {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".rhai") || name.ends_with(".toml") {
            assert!(listed.contains(&name), "unlisted demo file: {name}");
        }
    }
}

#[test]
fn top_level_sample_material_uses_sample_framing() {
    let mut paths = vec![
        Path::new("README.md").to_path_buf(),
        Path::new("samples/README.md").to_path_buf(),
    ];
    paths.extend(
        fs::read_dir(Path::new("samples"))
            .expect("read samples dir")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("rhai")),
    );

    for path in paths {
        let contents = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        for stale_label in ["Étude", "étude"] {
            assert!(
                !contents.contains(stale_label),
                "{} frames a sample as a musical work with {stale_label:?}",
                path.display()
            );
        }
    }
}
