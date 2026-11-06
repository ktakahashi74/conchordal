#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CARGO:=cargo}"
: "${CARGO_TARGET_DIR:=$here/target}"
export CARGO_TARGET_DIR
# Separate targets prevent the GUI cdylib/rlib from overwriting the engine's data-only rlib.
# Both artifacts use the same SIMD and validated stack settings.
export RUSTFLAGS='-C target-feature=+simd128 -C link-arg=-zstack-size=8388608'
"$CARGO" build --target-dir "$CARGO_TARGET_DIR/engine-wasm" --manifest-path "$here/Cargo.toml" --locked --target wasm32-unknown-unknown --release --jobs "${CARGO_BUILD_JOBS:-4}"
wasm-bindgen --target web --out-dir "$here/www/pkg" \
  "$CARGO_TARGET_DIR/engine-wasm/wasm32-unknown-unknown/release/conchordal_web.wasm"

"$CARGO" build --target-dir "$CARGO_TARGET_DIR/gui-wasm" --manifest-path "$here/../../gui/Cargo.toml" --locked --target wasm32-unknown-unknown --release --features browser --lib --jobs "${CARGO_BUILD_JOBS:-4}"
wasm-bindgen --target web --out-dir "$here/www/gui-pkg" \
  "$CARGO_TARGET_DIR/gui-wasm/wasm32-unknown-unknown/release/conchordal_web_ui.wasm"
