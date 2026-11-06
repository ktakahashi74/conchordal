#[path = "../../../src/ui/plots.rs"]
pub mod plots;
#[path = "../../../src/ui/windows.rs"]
pub mod windows;
pub mod model {
    pub use crate::db::amp_to_db;
    pub use crate::frame::{
        ConsonanceKernel, Diagnostic, Landscape, MeterBand, MeterState, Snapshot,
    };
    pub fn diagnostics(frame: &crate::frame::UiFrame) -> Vec<Diagnostic> {
        frame.diagnostics.clone()
    }
    pub fn support_scans(voice: &crate::frame::VoiceStateInfo, _: usize) -> Option<[Vec<f64>; 3]> {
        voice.support_scans.clone()
    }
}
