//! Generators for the artifacts derived from the scripting surface:
//! the Rhai LSP definition file (`rhai-defs/conchordal.d.rhai`) and the book
//! API reference (`docs/rhai_book/src/reference/api.md`).
//!
//! Signatures come from the live engine (`ScriptHost::create_engine`), prose
//! comes from the documentation registry (`docs`). `check()` enforces that
//! the two match exactly; both renderers refuse to run on a mismatch.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use super::docs::{self, FnDoc, Owner, Patch, Style};
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
    /// Mapped parameter type names (e.g. `Material`, `f64`, `String`, `[?]`).
    params: Vec<String>,
    /// Mapped return type; `None` for unit.
    ret: Option<String>,
}

impl Sig {
    fn owner(&self) -> Owner {
        match self.params.first().map(String::as_str) {
            Some("Material") | Some("Participant") => Owner::Voice,
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
        "SpeciesHandle" => "Material".to_string(),
        "GroupHandle" => "Participant".to_string(),
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

type Groups = BTreeMap<(String, Owner), Vec<Sig>>;

fn group(sigs: Vec<Sig>) -> Groups {
    let mut groups: Groups = BTreeMap::new();
    for sig in sigs {
        groups
            .entry((sig.name.clone(), sig.owner()))
            .or_default()
            .push(sig);
    }
    for sigs in groups.values_mut() {
        sigs.sort_by_key(Sig::decl);
    }
    groups
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
    let groups = group(sigs);
    let index = match doc_index() {
        Ok(index) => index,
        Err(errors) => return Err(errors.join("\n")),
    };
    let mut errors = Vec::new();

    for ((name, owner), sigs) in &groups {
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
        if !groups.contains_key(&(name.to_string(), *owner)) {
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

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn checked_groups() -> (Groups, Vec<String>) {
    if let Err(err) = check() {
        panic!("scripting docs registry out of sync with engine surface:\n{err}");
    }
    let (sigs, skipped) = collect();
    (group(sigs), skipped)
}

/// Render `rhai-defs/conchordal.d.rhai`.
pub fn render_d_rhai() -> String {
    let (groups, skipped) = checked_groups();
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

    for ((name, owner), sigs) in &groups {
        let doc = index
            .get(&(name.as_str(), *owner))
            .expect("doc presence validated by check()");
        out.push('\n');
        for sig in sigs {
            let _ = writeln!(out, "/// {}", doc.summary);
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
    let (groups, _) = checked_groups();

    let mut out = String::new();
    out.push_str("# API Reference\n\n");
    out.push_str("<!-- Auto-generated by `cargo run --bin gen_rhai_defs`. DO NOT EDIT. -->\n\n");
    out.push_str(
        "This page is generated from the engine's registered scripting surface joined with \
the documentation registry (`src/scripting/docs.rs`). A CI test fails whenever this page \
drifts from the engine. Regenerate with:\n\n```bash\ncargo run --bin gen_rhai_defs\n```\n\n",
    );
    out.push_str(
        "Builder methods return their receiver and are chainable. Integer and float literals \
are interchangeable wherever both overloads are registered; the generated LSP definitions \
(`rhai-defs/conchordal.d.rhai`) list the exact overloads.\n\n",
    );

    out.push_str("## Types\n\n| Type | Description |\n|------|-------------|\n");
    for t in docs::TYPE_DOCS {
        let _ = writeln!(out, "| `{}` | {} |", t.name, t.summary);
    }
    out.push_str("\n## Built-in Constants\n\n| Constant | Type | Description |\n|----------|------|-------------|\n");
    for c in docs::CONST_DOCS {
        let _ = writeln!(out, "| `{}` | `{}` | {} |", c.name, c.ty, c.summary);
    }

    for category in docs::CATEGORY_DOCS {
        let _ = write!(out, "\n## {}\n\n{}\n", category.title, category.intro);
        for doc in docs::FN_DOCS.iter().filter(|d| d.category == category.id) {
            let sigs = groups
                .get(&(doc.name.to_string(), doc.owner))
                .expect("group presence validated by check()");
            let _ = write!(out, "\n### `{}`\n\n```rhai,ignore\n", doc.name);
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
            if let Some(applies) = applies_line(doc, sigs) {
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
    out
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

fn applies_line(doc: &FnDoc, sigs: &[Sig]) -> Option<String> {
    if doc.style != Style::Method {
        return None;
    }
    match doc.owner {
        Owner::Voice => {
            let has_material = sigs
                .iter()
                .any(|s| s.params.first().map(String::as_str) == Some("Material"));
            let has_participant = sigs
                .iter()
                .any(|s| s.params.first().map(String::as_str) == Some("Participant"));
            let receivers = match (has_material, has_participant) {
                (true, true) => "`Material` and `Participant`",
                (true, false) => "`Material` only",
                (false, true) => "`Participant` only",
                (false, false) => unreachable!("Voice group without voice receiver"),
            };
            let patch = match doc.patch {
                Patch::Live => " Live-patchable: updates running voices on a live `Participant`.",
                Patch::Draft => " Draft-only: ignored with a warning on a live `Participant`.",
                Patch::Na => "",
            };
            Some(format!("Applies to: {receivers}.{patch}"))
        }
        Owner::Placement => Some("Applies to: `Placement`.".to_string()),
        Owner::ModePattern => Some("Applies to: `ModePattern`.".to_string()),
        Owner::Global => None,
    }
}
