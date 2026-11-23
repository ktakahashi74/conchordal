use super::{JsonlReporter, WorkerConfig, WorkerState};

#[inline(always)]
pub(super) fn write_rhythm(_: &mut Option<JsonlReporter>, _: &mut crate::life::voice::Voice) {}

#[inline(always)]
pub(super) fn write_beat(_: &mut WorkerState, _: &WorkerConfig) {}
