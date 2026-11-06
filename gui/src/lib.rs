pub mod frame;
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
pub use frame as viewdata;
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
mod browser;
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[allow(dead_code)] // Reuse native dB semantics; only amp_to_db is used by the GUI.
#[path = "../../src/core/db.rs"]
mod db;
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
mod ui;
