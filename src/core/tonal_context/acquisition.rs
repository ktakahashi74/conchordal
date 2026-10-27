//! Test-only capture of actual runtime decisions, before any TC feedback.
use crate::core::consonance_kernel::ConsonanceRepresentationParams;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::OnceLock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Decision {
    pub id: u64,
    pub generation: u32,
    pub tick: u64,
    pub n0: u64,
    pub hz: f32,
    pub dwell: f32,
    pub excluded: bool,
    pub field: f32,
    pub beta: f32,
    pub theta: f32,
    pub baseline: (f32, f32),
    pub candidates: Vec<(f32, f32)>,
}
#[derive(Default)]
struct Capture {
    owner: (u64, u32, u64, f32, f32),
    rows: Vec<Decision>,
}
thread_local! { static CAPTURE: RefCell<Capture> = RefCell::default(); }
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TC1B_CAPTURE").is_some())
}
pub(crate) fn owner(id: u64, generation: u32, tick: u64, hz: f32, dwell: f32) {
    if enabled() {
        CAPTURE.with_borrow_mut(|c| c.owner = (id, generation, tick, hz, dwell));
    }
}
pub(crate) fn decision(
    candidates: &[(f32, f32)],
    baseline: f32,
    baseline_score: f32,
    field: f32,
    params: ConsonanceRepresentationParams,
    excluded: bool,
) {
    if enabled() {
        CAPTURE.with_borrow_mut(|c| {
            let (id, generation, tick, hz, dwell) = c.owner;
            c.rows.push(Decision {
                id,
                generation,
                tick,
                n0: 0,
                hz,
                dwell,
                excluded,
                field,
                beta: params.beta,
                theta: params.theta,
                baseline: (baseline, baseline_score),
                candidates: candidates.to_vec(),
            });
        });
    }
}
pub(crate) fn take(n0: u64) -> Vec<Decision> {
    CAPTURE.with_borrow_mut(|c| {
        let mut rows = std::mem::take(&mut c.rows);
        for row in &mut rows {
            row.n0 = n0;
        }
        rows
    })
}
