//! Guardrails for the authored Japanese Rhai book prose.

use std::fs;
use std::path::Path;

use walkdir::WalkDir;

const FORBIDDEN_TRANSLATIONESE: &[&str] = &[
    "VoiceのPopulation",
    "founder Voice",
    "founder voice",
    "live patch",
    "script cursor",
    "runtime",
    "default",
    "handle",
    "target",
    "cloud",
    "builder method",
    "constructor",
    "modifier",
    "behavior",
    "scope",
    "Report",
    "mover",
    "配置前の集団設計",
    "配置済みの集団",
    "実行中の共同体",
    "個々の声",
    "声の一生",
    "配置方法",
    "Listener Twin",
    "複数の声",
    "すべての声",
    "一つの声",
    "集団",
    "共同体",
    "協和の地形",
    "知覚地形",
    "作品例",
    "étude",
    "エチュード",
];

#[test]
fn authored_japanese_prose_avoids_known_translationese() {
    let root = Path::new("docs/rhai_book_ja/src");
    let mut violations = Vec::new();

    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md")
            || path.ends_with("reference/api.md")
        {
            continue;
        }
        let source =
            fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let mut in_fence = false;
        for (index, line) in source.lines().enumerate() {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence {
                continue;
            }
            let prose = without_inline_code(line);
            for phrase in FORBIDDEN_TRANSLATIONESE {
                if prose.contains(phrase) {
                    violations.push(format!(
                        "{}:{} contains {phrase:?}: {}",
                        path.display(),
                        index + 1,
                        line.trim()
                    ));
                }
            }
            if contains_voice_alias(&prose) {
                violations.push(format!(
                    "{}:{} replaces Voice with the generic noun 声: {}",
                    path.display(),
                    index + 1,
                    line.trim()
                ));
            }
            if prose.contains("ephemeral") && !prose.contains("一過的な出来事（ephemeral）")
            {
                violations.push(format!(
                    "{}:{} leaves ephemeral unexplained: {}",
                    path.display(),
                    index + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Japanese prose violates docs/rhai_book_ja/README.md:\n{}",
        violations.join("\n")
    );
}

fn contains_voice_alias(prose: &str) -> bool {
    prose.match_indices('声').any(|(index, _)| {
        let previous = prose[..index].chars().next_back();
        let next = prose[index + '声'.len_utf8()..].chars().next();
        !matches!(previous, Some('和' | '音' | '歌'))
            && matches!(
                next,
                None | Some('が' | 'は' | 'を' | 'の' | 'へ' | 'に' | 'と' | 'で' | '、' | '。')
            )
    })
}

fn without_inline_code(line: &str) -> String {
    let mut prose = String::with_capacity(line.len());
    let mut in_code = false;
    for part in line.split('`') {
        if !in_code {
            prose.push_str(part);
        }
        in_code = !in_code;
    }
    prose
}
