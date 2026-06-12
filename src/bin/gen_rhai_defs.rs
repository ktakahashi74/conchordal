//! gen_rhai_defs: regenerate the artifacts derived from the scripting surface:
//!
//!   - `rhai-defs/conchordal.d.rhai` (Rhai LSP definitions with hover docs)
//!   - `docs/rhai_book/src/reference/api.md` (book API reference)
//!
//! Run with:
//!
//!     cargo run --bin gen_rhai_defs
//!
//! Sources of truth are `ScriptHost::create_engine` (signatures) and
//! `src/scripting/docs.rs` (prose). Regenerate whenever either changes so
//! editor support and the book stay in sync; CI tests catch stale output.

use std::fs;
use std::path::Path;

use conchordal::scripting::defs_gen;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (rel, contents) in [
        ("rhai-defs/conchordal.d.rhai", defs_gen::render_d_rhai()),
        (
            "docs/rhai_book/src/reference/api.md",
            defs_gen::render_reference_md(),
        ),
    ] {
        let path = root.join(rel);
        fs::write(&path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("wrote {rel}");
    }
}
