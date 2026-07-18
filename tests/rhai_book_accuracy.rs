//! Regression checks for factual errors previously found in the Rhai books.

use std::fs;
use std::path::Path;

use clap::Parser;
use conchordal::cli::Args;
use walkdir::WalkDir;

#[test]
fn books_do_not_restore_known_stale_claims() {
    let forbidden = [
        ("split into three tiers", "the scripting API has four tiers"),
        ("スクリプティング面は三層", "スクリプティングAPIは四層"),
        (
            "`conchordal.toml`",
            "the default config file is config.toml",
        ),
        (
            "max_exploration_bonus",
            "the DCC key is max_temperature_bonus",
        ),
        (
            "Respawn at random locations",
            "respawn_random is scene-weighted",
        ),
        (
            "Modes sampled from the live landscape",
            "landscape mode selection is deterministic",
        ),
        (
            "0 targets the strongest consonance",
            "tension(0) leaves ordinary placement unchanged",
        ),
        ("Étude", "the collection is a set of samples, not works"),
        ("étude", "the collection is a set of samples, not works"),
        ("作品例", "サンプルは音楽作品として位置づけない"),
        (
            "The Life of a Voice",
            "the chapter explains the Voice and Population object model",
        ),
        ("Voiceの一生", "章はVoiceとPopulationの仕組みを説明する"),
        (
            "The Ecological Loop",
            "the chapter explains concrete Voice-Landscape feedback",
        ),
        ("生態学的ループ", "章はVoiceとLandscapeのfeedbackを説明する"),
        (
            "# Voice and Population",
            "the heading must state the Population identity concept",
        ),
        (
            "# VoiceとPopulation",
            "見出しはPopulationの同一性というconceptを示す",
        ),
        (
            "# Voice–Landscape Feedback",
            "the heading must state both directions of the feedback concept",
        ),
        (
            "# VoiceとLandscapeのフィードバック",
            "見出しはfeedbackの両方向を示す",
        ),
        (
            "Observing a Performance",
            "the tutorial chapter covers the complete performance workflow",
        ),
        (
            "演奏を観察する",
            "章は観察だけでなく演奏の実行、記録、再現を扱う",
        ),
        (
            "open-loop",
            "state directly what the scenario does and does not determine",
        ),
        (
            "開いた制御系",
            "scenarioが決めるものと実行時に決まるものを直接説明する",
        ),
        (
            "composing loop",
            "describe the concrete run, inspect, and revise procedure",
        ),
    ];

    let mut violations = Vec::new();
    for root in ["docs/rhai_book/src", "docs/rhai_book_ja/src"] {
        for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("md") {
                continue;
            }
            let source = fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            for (stale, correction) in forbidden {
                if source.contains(stale) {
                    violations.push(format!(
                        "{} contains {stale:?}; {correction}",
                        path.display()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "stale factual claims returned:\n{}",
        violations.join("\n")
    );
}

#[test]
fn routing_chapters_name_the_runtime_dcc_keys() {
    for path in [
        Path::new("docs/rhai_book/src/concepts/routing.md"),
        Path::new("docs/rhai_book_ja/src/concepts/routing.md"),
    ] {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(source.contains("[dcc]"), "{} omits [dcc]", path.display());
        assert!(
            source.contains("coupling_strength") && source.contains("max_temperature_bonus"),
            "{} does not document both DCC runtime keys",
            path.display()
        );
    }
}

#[test]
fn concept_headings_name_the_term_and_explain_it() {
    for (path, expected_heading) in [
        (
            Path::new("docs/rhai_book/src/concepts/voice_life.md"),
            "# Population — A Persistent Unit of Voices",
        ),
        (
            Path::new("docs/rhai_book/src/concepts/ecological_loop.md"),
            "# Voice and Landscape — Sound–Environment Feedback",
        ),
        (
            Path::new("docs/rhai_book/src/concepts/consonance.md"),
            "# Consonance Field — A Terrain for Evaluating Pitch",
        ),
        (
            Path::new("docs/rhai_book_ja/src/concepts/voice_life.md"),
            "# Population — Voiceをまとめる持続単位",
        ),
        (
            Path::new("docs/rhai_book_ja/src/concepts/ecological_loop.md"),
            "# VoiceとLandscape — 音と環境のフィードバック",
        ),
        (
            Path::new("docs/rhai_book_ja/src/concepts/consonance.md"),
            "# Consonance Field — 音高を評価する地形",
        ),
    ] {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(
            source.lines().next(),
            Some(expected_heading),
            "{} has an unclear concept heading",
            path.display()
        );
    }
}

#[test]
fn performance_chapter_teaches_execution_and_replay_before_reports() {
    for (path, performance_heading, replay_heading, report_heading) in [
        (
            Path::new("docs/rhai_book/src/tutorial/observing.md"),
            "## Performing with the GUI",
            "## Replaying a performance",
            "## Reports",
        ),
        (
            Path::new("docs/rhai_book_ja/src/tutorial/observing.md"),
            "## GUIで演奏する",
            "## 演奏を再現する",
            "## レポート",
        ),
    ] {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let performance = source
            .find(performance_heading)
            .unwrap_or_else(|| panic!("{} omits {performance_heading:?}", path.display()));
        let replay = source
            .find(replay_heading)
            .unwrap_or_else(|| panic!("{} omits {replay_heading:?}", path.display()));
        let report = source
            .find(report_heading)
            .unwrap_or_else(|| panic!("{} omits {report_heading:?}", path.display()));
        assert!(
            performance < replay && replay < report,
            "{} must explain performance and replay before reports",
            path.display()
        );
    }

    let wait = Args::try_parse_from([
        "conchordal",
        "samples/10_generations.rhai",
        "--wait-user-start",
        "--wait-user-exit=false",
    ])
    .expect("parse documented cued GUI command");
    assert_eq!(wait.wait_user_start, Some(true));
    assert_eq!(wait.wait_user_exit, Some(false));

    let silent = Args::try_parse_from([
        "conchordal",
        "samples/10_generations.rhai",
        "--nogui",
        "--play=false",
        "--report",
        "run.jsonl",
    ])
    .expect("parse documented silent report command");
    assert!(silent.nogui);
    assert!(!silent.play);
    assert_eq!(silent.report.as_deref(), Some("run.jsonl"));
}
