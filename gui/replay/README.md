# Display replay and host boundary

This standalone crate builds the existing Conchordal egui windows and plots for
WebGL. The root crate depends only on its display data types; the `browser`
feature adds eframe and wasm-bindgen to the GUI artifact. The engine and device
I/O are not linked into this artifact.

Build from the repository root with the project's assigned cargo wrapper and
build directory:

```sh
RUSTFLAGS='-C target-feature=+simd128' "$CARGO" build --manifest-path gui/Cargo.toml --release --target wasm32-unknown-unknown --features browser --lib
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/conchordal_web_ui.wasm" --target web --out-dir "$REPLAY_DIR/pkg"
```

Copy `index.html` and the acquired `frames.jsonl` to `$REPLAY_DIR`. Serve that
directory over localhost HTTP. The page reads local display snapshots and has no
audio playback, recording, upload, or download control. It is an acquisition
viewer, not the final Astro product page.

The native acquisition is the explicitly ignored
`runtime::gui_support_tests::capture_web_gui_frames` test. Set `WP3_SCRIPT` to the
DEMO-B1 script and `WP3_FRAMES` to the output JSONL path. Acquisition requests
stop after 512 ordinary GUI deliveries, then drains terminal delivery. Set
`WP3_CAPTURE_STATS` only under the timing lock to record projection and encoding
costs separately. It runs the native runtime
with GUI snapshots and without device output or a WAV writer. Each JSONL line is
a version 1 envelope. This test does not add an output option to the instrument.

Integration with WP-1/WP-2:

1. Convert an engine `UiFrame` using `web_snapshot().encode()` in the computation
   Worker. Send the resulting UTF-8 bytes as an `ArrayBuffer` with a transfer list.
2. Pass the received bytes to `WebGui.push_snapshot()` on the main thread. Keep
   the bytes intact: parsing and stringifying them through JS `Number` loses u64
   identity precision. Consume every delivered snapshot so its batched rhythm
   observations are retained. Backpressure/coalescing belongs before snapshot
   publication in the engine.
3. Poll `take_commands()` after browser animation frames. It returns JSON strings
   `start`, `end`, and `exit`. Forward these to the host's lifecycle controller;
   the GUI never closes the page or terminates a Worker itself. Call `destroy()`
   when the host disposes the canvas.
4. A new performance session should construct a new `WebGui`; stream time moving
   backward also resets plot histories. The host remains responsible for UI
   playback state and any error presentation outside the canvas.

The wire schema keeps the displayed fields, waveform display samples, listener
forecasts, identities, and all batched rhythm hops. Harmonic support transfers its
root distribution and the three displayed per-Voice scans, not FFT plans or
scratch. Optional native body/temporal diagnostics become ordered display labels;
they do not become active browser engine consumers. Native and web renderers use
the same windows and plots. Rhai positions retain their producer formatting.

`set_measurement(true)` enables bounded local measurement buffers.
`benchmark_codec(bytes)` measures decode and encode without returning snapshot
bytes. `metrics()` returns timings only. `layout_check()` applies the existing
1200×900 R/H-bottom and Exit-right criteria to the current snapshot. The replay
page's `feedAll()` displays one acquired snapshot per animation frame for a
bounded rendering-cost experiment; ordinary replay follows source timestamps.
Frame CPU time comes from eframe (including its CPU paint path); it is not a GPU
completion measurement. RAF intervals also include browser scheduling.
