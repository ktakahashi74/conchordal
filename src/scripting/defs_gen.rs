//! Generators for the artifacts derived from the scripting surface:
//! the Rhai LSP definition file (`rhai-defs/conchordal.d.rhai`) and the English
//! and Japanese book API references.
//!
//! Signatures come from the live engine (`ScriptHost::create_engine`), prose
//! comes from the documentation registry (`docs`). `check()` enforces that
//! the two match exactly; both renderers refuse to run on a mismatch.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use super::docs::{self, FnDoc, Owner, Patch, Style, Tier};
use super::{ScriptContext, ScriptHost};

/// Rhai reserved keywords that cannot appear as fn names in a `.d.rhai`.
/// See https://rhai.rs/book/language/keywords.html.
const RHAI_RESERVED: &[&str] = &[
    "var",
    "static",
    "is",
    "goto",
    "exit",
    "match",
    "case",
    "public",
    "protected",
    "new",
    "use",
    "with",
    "module",
    "package",
    "super",
    "spawn",
    "thread",
    "go",
    "sync",
    "async",
    "await",
    "yield",
    "default",
    "void",
    "null",
    "nil",
];

#[derive(Clone, Debug)]
struct Sig {
    name: String,
    /// Mapped parameter type names (e.g. `PopulationSpec`, `f64`, `String`, `[?]`).
    params: Vec<String>,
    /// Mapped return type; `None` for unit.
    ret: Option<String>,
}

impl Sig {
    fn owner(&self) -> Owner {
        match self.params.first().map(String::as_str) {
            Some("PopulationSpec") | Some("Population") => Owner::Population,
            Some("Placement") => Owner::Placement,
            Some("ModePattern") => Owner::ModePattern,
            _ => Owner::Global,
        }
    }

    fn decl(&self) -> String {
        let params = self
            .params
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("arg{i}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        match &self.ret {
            Some(ret) => format!("fn {}({params}) -> {ret};", self.name),
            None => format!("fn {}({params});", self.name),
        }
    }
}

fn collect() -> (Vec<Sig>, Vec<String>) {
    let ctx = Arc::new(Mutex::new(ScriptContext::default()));
    let engine = ScriptHost::create_engine(ctx);

    let mut sigs = Vec::new();
    let mut skipped = Vec::new();
    for raw in engine.gen_fn_signatures(false) {
        let name_end = raw.find('(').expect("signature has '('");
        let name = &raw[..name_end];
        if RHAI_RESERVED.contains(&name) || !is_rhai_identifier(name) {
            skipped.push(raw);
            continue;
        }
        sigs.push(parse_signature(&raw));
    }
    skipped.sort();
    (sigs, skipped)
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

/// Parse one `gen_fn_signatures` line: `name(_: T1, _: T2) -> R` (R optional).
fn parse_signature(raw: &str) -> Sig {
    let (head, ret) = match raw.split_once(" -> ") {
        Some((h, r)) => (h, Some(r)),
        None => (raw, None),
    };
    let open = head.find('(').expect("signature has '('");
    let close = head.rfind(')').expect("signature has ')'");
    let name = head[..open].to_string();
    let inside = &head[open + 1..close];
    let params = if inside.is_empty() {
        Vec::new()
    } else {
        inside
            .split(", ")
            .map(|p| map_type(p.trim_start_matches("_: ")))
            .collect()
    };
    let ret = ret.map(map_type).filter(|r| r != "()");
    Sig { name, params, ret }
}

/// Map a Rust type string from `gen_fn_signatures` to a `.d.rhai` type name.
fn map_type(t: &str) -> String {
    let t = t.trim();
    if let Some(inner) = unwrap_result(t) {
        return map_type(&inner);
    }
    let last = t.rsplit("::").next().unwrap_or(t).trim();
    match last {
        "string" => "String".to_string(),
        "Dynamic" => "?".to_string(),
        "array" => "[?]".to_string(),
        "PopulationSpecHandle" => "PopulationSpec".to_string(),
        "PopulationHandle" => "Population".to_string(),
        other => other.to_string(),
    }
}

/// Strip `core::result::Result<T, ...>` and return T.
fn unwrap_result(t: &str) -> Option<String> {
    let rest = t.strip_prefix("core::result::Result<")?;
    let mut depth: i32 = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                if depth == 0 {
                    return Some(rest[..i].to_string());
                }
                depth -= 1;
            }
            ',' if depth == 0 => return Some(rest[..i].to_string()),
            _ => {}
        }
    }
    None
}

type Populations = BTreeMap<(String, Owner), Vec<Sig>>;

fn population(sigs: Vec<Sig>) -> Populations {
    let mut populations: Populations = BTreeMap::new();
    for sig in sigs {
        populations
            .entry((sig.name.clone(), sig.owner()))
            .or_default()
            .push(sig);
    }
    for sigs in populations.values_mut() {
        sigs.sort_by_key(Sig::decl);
    }
    populations
}

fn doc_index() -> Result<BTreeMap<(&'static str, Owner), &'static FnDoc>, Vec<String>> {
    let mut index = BTreeMap::new();
    let mut errors = Vec::new();
    for doc in docs::FN_DOCS {
        if index.insert((doc.name, doc.owner), doc).is_some() {
            errors.push(format!(
                "duplicate doc entry: {} ({:?})",
                doc.name, doc.owner
            ));
        }
        if !docs::CATEGORY_DOCS.iter().any(|c| c.id == doc.category) {
            errors.push(format!(
                "doc entry {} references unknown category {:?}",
                doc.name, doc.category
            ));
        }
    }
    if errors.is_empty() {
        Ok(index)
    } else {
        Err(errors)
    }
}

fn usage_arity(usage: &str, name: &str) -> Result<usize, String> {
    let Some(rest) = usage.strip_prefix(name) else {
        return Err(format!("usage {usage:?} does not start with {name:?}"));
    };
    let Some(inside) = rest.strip_prefix('(').and_then(|r| r.strip_suffix(')')) else {
        return Err(format!("usage {usage:?} is not of the form name(args)"));
    };
    if inside.trim().is_empty() {
        Ok(0)
    } else {
        Ok(inside.split(',').count())
    }
}

/// Verify that the doc registry and the registered engine surface match.
pub fn check() -> Result<(), String> {
    let (sigs, _) = collect();
    let populations = population(sigs);
    let index = match doc_index() {
        Ok(index) => index,
        Err(errors) => return Err(errors.join("\n")),
    };
    let mut errors = Vec::new();

    for ((name, owner), sigs) in &populations {
        let Some(doc) = index.get(&(name.as_str(), *owner)) else {
            errors.push(format!(
                "registered fn {name:?} ({owner:?}) has no entry in src/scripting/docs.rs"
            ));
            continue;
        };
        let receiver = match doc.style {
            Style::Method => 1,
            Style::Free => 0,
        };
        let actual: BTreeSet<usize> = sigs.iter().map(|s| s.params.len() - receiver).collect();
        if *owner == Owner::Population {
            let receivers: BTreeSet<&str> = sigs
                .iter()
                .filter_map(|sig| sig.params.first().map(String::as_str))
                .collect();
            let expected = match doc.patch {
                Patch::Initial => Some(BTreeSet::from(["PopulationSpec"])),
                Patch::Live => Some(BTreeSet::from(["Population", "PopulationSpec"])),
                Patch::Na => None,
            };
            if let Some(expected) = expected
                && receivers != expected
            {
                errors.push(format!(
                    "doc patch contract for {name:?} is {:?}, but engine receivers are {receivers:?}",
                    doc.patch
                ));
            }
        }
        let mut documented = BTreeSet::new();
        for usage in doc.usage {
            match usage_arity(usage, doc.name) {
                Ok(arity) => {
                    documented.insert(arity);
                }
                Err(err) => errors.push(err),
            }
        }
        if documented != actual {
            errors.push(format!(
                "doc usage arities for {name:?} ({owner:?}) are {documented:?}, \
                 engine has {actual:?}"
            ));
        }
    }

    for (name, owner) in index.keys() {
        if !populations.contains_key(&(name.to_string(), *owner)) {
            errors.push(format!(
                "doc entry {name:?} ({owner:?}) has no registered engine function (stale?)"
            ));
        }
    }

    let bus_names: Vec<&str> = docs::CONST_DOCS.iter().map(|c| c.name).collect();
    if bus_names != ["habitat_bus", "presentation_bus"] {
        errors.push(format!(
            "CONST_DOCS out of sync with built-in buses: {bus_names:?}"
        ));
    }

    let doc_names: BTreeSet<&str> = docs::FN_DOCS.iter().map(|d| d.name).collect();
    let mut tiered = BTreeSet::new();
    for (name, tier) in docs::tier_lists() {
        if !doc_names.contains(name) {
            errors.push(format!(
                "tier list entry {name:?} ({tier:?}) has no doc entry (stale?)"
            ));
        }
        if !tiered.insert(name) {
            errors.push(format!("tier list entry {name:?} appears more than once"));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn checked_populations() -> (Populations, Vec<String>) {
    if let Err(err) = check() {
        panic!("scripting docs registry out of sync with engine surface:\n{err}");
    }
    let (sigs, skipped) = collect();
    (population(sigs), skipped)
}

/// Render `rhai-defs/conchordal.d.rhai`.
pub fn render_d_rhai() -> String {
    let (populations, skipped) = checked_populations();
    let index = doc_index().expect("doc index validated by check()");

    let mut out = String::new();
    out.push_str("// Auto-generated by `cargo run --bin gen_rhai_defs`. DO NOT EDIT.\n");
    out.push_str(
        "// Source of truth: src/scripting/engine.rs (signatures) + src/scripting/docs.rs (docs).\n\n",
    );
    out.push_str("module static;\n\n");
    out.push_str("// Built-in buses (registered as module constants).\n");
    for c in docs::CONST_DOCS {
        let _ = writeln!(out, "/// {}", c.summary);
        let _ = writeln!(out, "const {}: {};", c.name, c.ty);
    }

    for ((name, owner), sigs) in &populations {
        let doc = index
            .get(&(name.as_str(), *owner))
            .expect("doc presence validated by check()");
        let tag = match docs::tier_of(doc.name) {
            Tier::Core => "",
            Tier::Experimental => "[experimental] ",
            Tier::Tuning => "[tuning] ",
            Tier::Research => "[research] ",
        };
        out.push('\n');
        for sig in sigs {
            let _ = writeln!(out, "/// {tag}{}", doc.summary);
            let _ = writeln!(out, "{}", sig.decl());
        }
    }

    if !skipped.is_empty() {
        out.push_str("\n// Skipped (reserved keyword or non-identifier function name):\n");
        for s in &skipped {
            let _ = writeln!(out, "// - {s}");
        }
    }
    out
}

/// Render `docs/rhai_book/src/reference/api.md`.
pub fn render_reference_md() -> String {
    render_reference_md_for(ReferenceLanguage::English)
}

/// Render `docs/rhai_book_ja/src/reference/api.md`.
pub fn render_reference_md_ja() -> String {
    render_reference_md_for(ReferenceLanguage::Japanese)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReferenceLanguage {
    English,
    Japanese,
}

fn render_reference_md_for(language: ReferenceLanguage) -> String {
    let (populations, _) = checked_populations();

    let mut out = String::new();
    match language {
        ReferenceLanguage::English => out.push_str("# API Reference\n\n"),
        ReferenceLanguage::Japanese => out.push_str("# APIリファレンス\n\n"),
    }
    out.push_str("<!-- Auto-generated by `cargo run --bin gen_rhai_defs`. DO NOT EDIT. -->\n\n");
    match language {
        ReferenceLanguage::English => out.push_str(
            "This page is generated from the engine's registered scripting surface joined with \
the documentation registry (`src/scripting/docs.rs`). A CI test fails whenever this page \
drifts from the engine. Regenerate with:\n\n```bash\ncargo run --bin gen_rhai_defs\n```\n\n",
        ),
        ReferenceLanguage::Japanese => out.push_str(
            "このページは、engineに登録されたスクリプティング面とドキュメントregistry \
（`src/scripting/docs.rs`）から生成されます。engineとの差異はCIテストで検出されます。\
再生成するには次を実行します：\n\n```bash\ncargo run --bin gen_rhai_defs\n```\n\n",
        ),
    }
    match language {
        ReferenceLanguage::English => out.push_str(
            "Builder methods return their receiver and are chainable. Integer and float literals \
are interchangeable wherever both overloads are registered; the generated LSP definitions \
(`rhai-defs/conchordal.d.rhai`) list the exact overloads.\n\n",
        ),
        ReferenceLanguage::Japanese => out.push_str(
            "構築用のメソッドは操作対象を返すため、続けて呼び出せます。整数と小数は、\
両方のoverloadが登録されている箇所では交換可能です。正確なoverloadは生成されたLSP定義\
（`rhai-defs/conchordal.d.rhai`）に記載されています。\n\n",
        ),
    }
    match language {
        ReferenceLanguage::English => out.push_str(
            "The surface is split into four tiers. **Core API** is the curated composing \
surface — it is enough for every curated sample. **Experimental** contains candidate Core \
verbs under audition. **Mechanism Tuning** adjusts the mechanisms behind the core verbs \
when a piece needs a behavior the core surface does not express. **Research Controls** \
exist for studying the instrument, not composing with it.\n\n",
        ),
        ReferenceLanguage::Japanese => out.push_str(
            "スクリプティング面は四層に分かれます。**Core API**は選別された作曲用の面で、\
すべてのサンプルを記述できます。**Experimental**は試聴中のCore候補です。\
**Mechanism Tuning**は、Coreでは表せない挙動を作品が必要とするとき、動詞の背後にある\
機構を調整します。**Research Controls**は作曲ではなく、instrumentを研究するためのものです。\
関数ごとの技術説明は、LSP hoverと同じ英語の単一ソースを掲載しています。\n\n",
        ),
    }

    match language {
        ReferenceLanguage::English => {
            out.push_str("## Types\n\n| Type | Description |\n|------|-------------|\n")
        }
        ReferenceLanguage::Japanese => {
            out.push_str("## 型\n\n| 型 | 説明 |\n|------|-------------|\n")
        }
    }
    for t in docs::TYPE_DOCS {
        let _ = writeln!(
            out,
            "| `{}` | {} |",
            t.name,
            localized_type_summary(t, language)
        );
    }
    match language {
        ReferenceLanguage::English => out.push_str(
            "\n## Built-in Constants\n\n| Constant | Type | Description |\n|----------|------|-------------|\n",
        ),
        ReferenceLanguage::Japanese => out.push_str(
            "\n## 組み込み定数\n\n| 定数 | 型 | 説明 |\n|----------|------|-------------|\n",
        ),
    }
    for c in docs::CONST_DOCS {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {} |",
            c.name,
            c.ty,
            localized_const_summary(c, language)
        );
    }

    const ENGLISH_PARTS: &[(Tier, &str, &str)] = &[
        (
            Tier::Core,
            "Core API",
            "The curated composing surface. These verbs are enough for every curated sample.",
        ),
        (
            Tier::Experimental,
            "Experimental",
            "Candidate core verbs under audition: composing surface by intent, but with \
research-grade stability until validated.",
        ),
        (
            Tier::Tuning,
            "Mechanism Tuning",
            "Fine-grained control over the mechanisms behind the core verbs. Defaults are \
calibrated; reach for these when a piece needs a specific behavior the core surface does \
not express.",
        ),
        (
            Tier::Research,
            "Research Controls",
            "For studying the instrument, not composing with it. Normal composing never \
touches this tier, and it has the weakest stability guarantee: entries may change or \
disappear as their research questions settle.",
        ),
    ];
    const JAPANESE_PARTS: &[(Tier, &str, &str)] = &[
        (
            Tier::Core,
            "Core API",
            "選別された作曲用の面です。すべてのサンプルは、ここにある動詞だけで記述できます。",
        ),
        (
            Tier::Experimental,
            "Experimental",
            "Core候補として試聴中の動詞です。作曲用ですが、検証が済むまでは研究段階の安定性です。",
        ),
        (
            Tier::Tuning,
            "Mechanism Tuning",
            "Core動詞の背後にある機構を細かく制御します。既定値は調整済みです。Coreでは表せない特定の挙動を作品が必要とするときに使います。",
        ),
        (
            Tier::Research,
            "Research Controls",
            "instrumentを研究するための面であり、作曲用ではありません。通常の作曲では触れません。研究上の問いが定まるにつれて変更または削除される可能性があります。",
        ),
    ];
    let parts = match language {
        ReferenceLanguage::English => ENGLISH_PARTS,
        ReferenceLanguage::Japanese => JAPANESE_PARTS,
    };

    for (tier, part_title, part_intro) in parts {
        let _ = write!(out, "\n## {part_title}\n\n{part_intro}\n");
        if !docs::FN_DOCS.iter().any(|d| docs::tier_of(d.name) == *tier) {
            out.push_str(match language {
                ReferenceLanguage::English => "\nEmpty right now.\n",
                ReferenceLanguage::Japanese => "\n現在は空です。\n",
            });
            continue;
        }
        for category in docs::CATEGORY_DOCS {
            let entries: Vec<&docs::FnDoc> = docs::FN_DOCS
                .iter()
                .filter(|d| d.category == category.id && docs::tier_of(d.name) == *tier)
                .collect();
            if entries.is_empty() {
                continue;
            }
            let _ = write!(
                out,
                "\n### {}\n",
                localized_category_title(category, language)
            );
            if *tier == Tier::Core {
                let _ = write!(out, "\n{}\n", localized_category_intro(category, language));
            }
            for doc in entries {
                let sigs = populations
                    .get(&(doc.name.to_string(), doc.owner))
                    .expect("population presence validated by check()");
                let _ = write!(out, "\n#### `{}`\n\n```rhai,ignore\n", doc.name);
                let ret = free_return(doc, sigs);
                for usage in doc.usage {
                    match &ret {
                        Some(ret) => {
                            let _ = writeln!(out, "{usage} -> {ret}");
                        }
                        None => {
                            let _ = writeln!(out, "{usage}");
                        }
                    }
                }
                out.push_str("```\n\n");
                if let Some(applies) = applies_line(doc, sigs, language) {
                    let _ = writeln!(out, "{applies}\n");
                }
                out.push_str(doc.summary);
                if !doc.details.is_empty() {
                    out.push(' ');
                    out.push_str(doc.details);
                }
                out.push('\n');
            }
        }
    }
    out
}

fn localized_type_summary(doc: &docs::TypeDoc, language: ReferenceLanguage) -> &'static str {
    if language == ReferenceLanguage::English {
        return doc.summary;
    }
    match doc.name {
        "PopulationSpec" => {
            "再利用可能なPopulationSpec。初代のVoiceの特性に、生存過程、respawn、個体数の規則を組み合わせる。"
        }
        "Population" => {
            "`place()`が返すPopulation。Voiceが死亡し世代交代しても、同一性を保つ安定した参照。"
        }
        "Placement" => {
            "初代のVoiceをどこへ配置するかを表す。`at()`、`consonance()`、`dissonance()`、`edge()`、`gap()`、`random()`、`line()`で作る。"
        }
        "ModePattern" => "モーダル合成の発音体に使う周波数構成。`*_modes()`関数で作る。",
        "Bus" => "二つのmono出力busの一つ。`|`で結合すると`BusSet`になる。",
        "BusSet" => "`bus | bus`で作るbusの組み合わせ。`send()`へ渡す。",
        _ => doc.summary,
    }
}

fn localized_const_summary(doc: &docs::ConstDoc, language: ReferenceLanguage) -> &'static str {
    if language == ReferenceLanguage::English {
        return doc.summary;
    }
    match doc.name {
        "habitat_bus" => "解析bus：NSGTからLandscapeへ送られ、生態系が知覚する。",
        "presentation_bus" => "提示bus：cpal出力、録音、UI meterへ送られ、聴き手が耳にする。",
        _ => doc.summary,
    }
}

fn localized_category_title(
    category: &docs::CategoryDoc,
    language: ReferenceLanguage,
) -> &'static str {
    if language == ReferenceLanguage::English {
        return category.title;
    }
    match category.id {
        "population_specs" => "PopulationSpec",
        "placement" => "配置",
        "timeline" => "TimelineとPopulation",
        "body" => "Bodyと音色",
        "phonation" => "Phonationとリズム",
        "pitch" => "Pitch Movement",
        "neighbors" => "近傍の知覚",
        "lifecycle" => "LifecycleとViability",
        "respawn" => "Respawn",
        "modes" => "Mode Pattern",
        "routing" => "Routing",
        "director" => "Directorとglobal parameter",
        _ => category.title,
    }
}

fn localized_category_intro(
    category: &docs::CategoryDoc,
    language: ReferenceLanguage,
) -> &'static str {
    if language == ReferenceLanguage::English {
        return category.intro;
    }
    match category.id {
        "population_specs" => {
            "PopulationSpecは、初代のVoiceの特性とPopulation全体の生存規則を定める。生成関数で作り、構築用メソッドで調整し、`place()`でPopulationにする。"
        }
        "placement" => {
            "Placementは、初代のVoiceが周波数空間へ入る場所を決める。生成関数で作り、`place()`へ渡す前に調整する。"
        }
        "timeline" => {
            "`place()`は現在のスクリプト時刻にPopulationを配置する。`wait()`は時刻を進め、`flush()`は時刻を進めずに保留中の更新を発行する。有効範囲が終わると、内部のPopulationは自動的に解放される。"
        }
        "body" => "Voiceの音響bodyを構成するlevel、spectrum、detuning、envelope。",
        "phonation" => {
            "Voiceがいつ、どれだけの間鳴るかを決める。Tier 1はリズム結合連続体の領域を選び、Tier 2はwhen/durationを明示し、Tier 3は専門的な調整を行う。同じ軸では最後の指定が優先される。"
        }
        "pitch" => {
            "VoiceがConsonance Field内をどう移動するかを決める。`seek_consonance()`、`glide()`、`temperature()`が基本の作曲用APIで、残りは山登り探索とpeak samplerを調整する。"
        }
        "neighbors" => {
            "Fieldを評価するとき、ほかのVoiceと自分自身のspectrum footprintをどう知覚するかを決める。"
        }
        "lifecycle" => {
            "時間に沿った生存と回復。`endurance()`は適合度がゼロのときの寿命、`recovery()`は最大の連続回復に要する時間を表す。`consonance_viability()`は回復範囲を定め、既定では環境相対評価を使う。"
        }
        "respawn" => {
            "Populationの世代交代。respawn policyはVoiceが死亡したときのreplacementの出現位置を決め、capacityとacceptance thresholdが生態系規模の挙動を形づくる。"
        }
        "modes" => {
            "`modal()`の発音体に使う周波数関係。生成関数は`ModePattern`を返し、構築用メソッドで調整できる。Landscape-awareな構成は実行中のLandscapeから値を取る。"
        }
        "routing" => {
            "各Voiceは、独立した二つのモノラルバスへ寄与する。presentation busは作品として聴かれる音、habitat busはNSGT解析を通じて生態系が知覚する音である。初期状態では両方へ送る。"
        }
        "director" => {
            "scene全体の地形形成と研究用control。director verbはsoft priorであり、地形を形づくるが、beatをscheduleしたりmeasureを強制したりはしない。"
        }
        _ => category.intro,
    }
}

fn free_return(doc: &FnDoc, sigs: &[Sig]) -> Option<String> {
    if doc.style != Style::Free {
        return None;
    }
    let rets: BTreeSet<&String> = sigs.iter().filter_map(|s| s.ret.as_ref()).collect();
    if rets.len() == 1 && sigs.iter().all(|s| s.ret.is_some()) {
        rets.first().map(|r| (*r).clone())
    } else {
        None
    }
}

fn applies_line(doc: &FnDoc, sigs: &[Sig], language: ReferenceLanguage) -> Option<String> {
    if doc.style != Style::Method {
        return None;
    }
    match doc.owner {
        Owner::Population => {
            let has_spec = sigs
                .iter()
                .any(|s| s.params.first().map(String::as_str) == Some("PopulationSpec"));
            let has_population = sigs
                .iter()
                .any(|s| s.params.first().map(String::as_str) == Some("Population"));
            let receivers = match (language, has_spec, has_population) {
                (ReferenceLanguage::English, true, true) => "`PopulationSpec` and `Population`",
                (ReferenceLanguage::English, true, false) => "`PopulationSpec` only",
                (ReferenceLanguage::English, false, true) => "`Population` only",
                (ReferenceLanguage::Japanese, true, true) => "`PopulationSpec`と`Population`",
                (ReferenceLanguage::Japanese, true, false) => "`PopulationSpec`のみ",
                (ReferenceLanguage::Japanese, false, true) => "`Population`のみ",
                (_, false, false) => unreachable!("Voice population without voice receiver"),
            };
            let patch = match (language, doc.patch) {
                (ReferenceLanguage::English, Patch::Live) => {
                    " Live-patchable: updates running voices in a `Population`."
                }
                (ReferenceLanguage::English, Patch::Initial) => {
                    " Initial-only: configure the `PopulationSpec` before `place()`."
                }
                (ReferenceLanguage::Japanese, Patch::Live) => {
                    " 実行中に更新可能：`Population`内で鳴っているVoiceを更新する。"
                }
                (ReferenceLanguage::Japanese, Patch::Initial) => {
                    " 初期設定専用：`place()`より前に`PopulationSpec`へ設定する。"
                }
                (_, Patch::Na) => "",
            };
            Some(match language {
                ReferenceLanguage::English => format!("Applies to: {receivers}.{patch}"),
                ReferenceLanguage::Japanese => format!("適用対象：{receivers}。{patch}"),
            })
        }
        Owner::Placement => Some(match language {
            ReferenceLanguage::English => "Applies to: `Placement`.".to_string(),
            ReferenceLanguage::Japanese => "適用対象：`Placement`。".to_string(),
        }),
        Owner::ModePattern => Some(match language {
            ReferenceLanguage::English => "Applies to: `ModePattern`.".to_string(),
            ReferenceLanguage::Japanese => "適用対象：`ModePattern`。".to_string(),
        }),
        Owner::Global => None,
    }
}
