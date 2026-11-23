use super::*;

pub(super) fn write_rhythm(
    reporter: &mut Option<JsonlReporter>,
    voice: &mut crate::life::voice::Voice,
) {
    let id = voice.id();
    if let Some(records) = voice.phonation_engine.felt.drain_records() {
        for record in records {
            report_try(reporter, "felt rhythm", |writer| {
                writer.write_felt_rhythm(id, voice.metadata.generation, &record)
            });
        }
    }
}

pub(super) fn write_beat(state: &mut WorkerState, cfg: &WorkerConfig) {
    if state.reporter.is_some()
        && let Some(felt) = state
            .temporal
            .flora_meter
            .as_ref()
            .and_then(|m| m.felt_beat())
    {
        let end = state.timebase.frame_end_tick(state.frame_idx);
        report_try(&mut state.reporter, "felt beat", |writer| {
            writer.write_felt_beat(end, cfg.fs as u32, felt)
        });
    }
}
