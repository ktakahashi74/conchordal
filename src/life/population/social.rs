use super::PhonationBatch;
use crate::core::timebase::Tick;
use crate::life::social_density::SocialDensityTrace;

pub(super) fn kuramoto_order_from_phases(phases: &[f32]) -> Option<f32> {
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    let mut count = 0u32;
    for &phase in phases {
        if !phase.is_finite() {
            continue;
        }
        sum_cos += phase.cos();
        sum_sin += phase.sin();
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let n = count as f32;
    let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;
    Some(r.clamp(0.0, 1.0))
}

pub(super) fn mix_pred_gate_gain(sync: f32, gain_raw: f32) -> f32 {
    let sync = sync.clamp(0.0, 1.0);
    let gain01 = 0.2 + 0.8 * gain_raw.powf(2.0);
    let gain = 1.0 + (gain01 - 1.0) * sync;
    if gain.is_finite() { gain.max(0.0) } else { 1.0 }
}

pub(super) fn build_social_trace_from_batches(
    phonation_batches: &[PhonationBatch],
    frame_end: Tick,
    hop_tick: Tick,
    bin_ticks: u32,
    smooth: f32,
    population_size: usize,
) -> SocialDensityTrace {
    let mut onset_ticks = Vec::new();
    for batch in phonation_batches {
        for onset in &batch.onsets {
            onset_ticks.push((onset.onset_tick.saturating_add(hop_tick), onset.strength));
        }
    }
    SocialDensityTrace::from_onsets(
        frame_end,
        frame_end.saturating_add(hop_tick),
        bin_ticks,
        smooth,
        population_size,
        &onset_ticks,
    )
}

pub(super) fn social_trace_params(hop_tick: Tick) -> (u32, f32) {
    let auto_bin = (hop_tick / 64).max(1);
    let bin_ticks = auto_bin.min(u32::MAX as Tick) as u32;
    (bin_ticks, 0.0)
}

pub(super) fn social_trace_enabled_from_couplings<I>(couplings: I) -> bool
where
    I: IntoIterator<Item = f32>,
{
    couplings.into_iter().any(|coupling| coupling != 0.0)
}
