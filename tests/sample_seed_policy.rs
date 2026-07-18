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
    assert_eq!(sample_count, 12, "expected the twelve top-level samples");

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
