use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;
use crate::core::timebase::Tick;
use crate::life::intent::Intent;

#[derive(Clone, Debug)]
pub struct PredKernelInputs {
    pub eval_tick: Tick,
    pub pred_env_scan: Vec<f32>,
    pub pred_den_scan: Vec<f32>,
}

pub fn build_pred_kernel_inputs_from_intents(
    space: &Log2Space,
    intents: &[Intent],
    eval_tick: Tick,
) -> PredKernelInputs {
    let n = space.n_bins();
    let mut pred_env_scan = vec![0.0f32; n];
    let mut pred_den_scan = vec![0.0f32; n];
    let (_erb, du) = erb_grid(space);
    debug_assert_eq!(du.len(), n);

    for intent in intents {
        if !intent_active_at(intent, eval_tick) {
            continue;
        }
        add_intent_energy(intent, space, &du, &mut pred_env_scan, &mut pred_den_scan);
    }

    space.assert_scan_len_named(&pred_env_scan, "pred_env_scan");
    space.assert_scan_len_named(&pred_den_scan, "pred_den_scan");

    PredKernelInputs {
        eval_tick,
        pred_env_scan,
        pred_den_scan,
    }
}

fn intent_active_at(intent: &Intent, eval_tick: Tick) -> bool {
    let end = intent.onset.saturating_add(intent.duration);
    intent.onset <= eval_tick && eval_tick < end
}

fn add_intent_energy(
    intent: &Intent,
    space: &Log2Space,
    du: &[f32],
    pred_env_scan: &mut [f32],
    pred_den_scan: &mut [f32],
) {
    if intent.freq_hz <= 0.0 || !intent.freq_hz.is_finite() {
        return;
    }
    if !intent.amp.is_finite() {
        return;
    }
    if intent.freq_hz < space.fmin || intent.freq_hz > space.fmax {
        return;
    }
    let mut amp = intent.amp.max(0.0);
    if let Some(body) = &intent.body {
        amp *= body.amp_scale.clamp(0.0, 1.0);
    }
    if amp <= 0.0 {
        return;
    }

    // HarmonicityStream consumes the log2-space amplitude spectrum (linear amplitude on log2 bins).
    add_log2_energy(pred_env_scan, space, intent.freq_hz, amp);

    let power = amp * amp;
    add_log2_density(pred_den_scan, space, du, intent.freq_hz, power);
}

fn add_log2_energy(amps: &mut [f32], space: &Log2Space, freq_hz: f32, energy: f32) {
    if !freq_hz.is_finite() || energy == 0.0 {
        return;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return;
    }
    let pos = match space.bin_pos_of_freq(freq_hz) {
        Some(pos) => pos,
        None => return,
    };
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < amps.len() {
        amps[idx] += energy * (1.0 - frac);
        amps[idx + 1] += energy * frac;
    } else if idx < amps.len() {
        amps[idx] += energy * (1.0 - frac);
    }
}

fn add_log2_density(density: &mut [f32], space: &Log2Space, du: &[f32], freq_hz: f32, mass: f32) {
    if !freq_hz.is_finite() || mass == 0.0 {
        return;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return;
    }
    let pos = match space.bin_pos_of_freq(freq_hz) {
        Some(pos) => pos,
        None => return,
    };
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < density.len() {
        let d0 = du[idx].max(1e-12);
        let d1 = du[idx + 1].max(1e-12);
        density[idx] += (mass * (1.0 - frac)) / d0;
        density[idx + 1] += (mass * frac) / d1;
    } else if idx < density.len() {
        let d0 = du[idx].max(1e-12);
        density[idx] += (mass * (1.0 - frac)) / d0;
    }
}
