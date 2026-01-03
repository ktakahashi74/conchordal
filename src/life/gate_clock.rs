use std::f32::consts::TAU;

use crate::core::modulation::RhythmBand;
use crate::core::phase::wrap_0_tau;
use crate::core::timebase::Tick;

const PHI_EPSILON: f32 = 1e-6;

pub fn next_gate_tick(
    now_tick: Tick,
    fs: f32,
    theta: RhythmBand,
    target_phase: f32,
) -> Option<Tick> {
    if !fs.is_finite() || fs <= 0.0 {
        return None;
    }
    if !theta.freq_hz.is_finite() || theta.freq_hz <= 0.0 {
        return None;
    }
    if !theta.phase.is_finite() || !target_phase.is_finite() {
        return None;
    }

    let phi0 = wrap_0_tau(theta.phase);
    let phi_t = wrap_0_tau(target_phase);
    let mut dphi = (phi_t - phi0).rem_euclid(TAU);
    if dphi < PHI_EPSILON {
        dphi += TAU;
    }

    let dt_sec = dphi / (TAU * theta.freq_hz);
    let dt_tick_f = dt_sec * fs;
    if !dt_tick_f.is_finite() {
        return None;
    }
    if dt_tick_f > Tick::MAX as f32 {
        return None;
    }

    let dt_tick = dt_tick_f.round().max(1.0) as Tick;

    let next_tick = now_tick.saturating_add(dt_tick);
    if next_tick <= now_tick {
        return None;
    }
    Some(next_tick)
}
