use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::Intent;
use tracing::debug;

#[derive(Clone, Copy, Debug)]
pub struct RhythmBandSpec {
    pub freq_hz: f32,
    pub tau_sec: f32,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RhythmBandState {
    pub freq_hz: f32,
    pub z_re: f32,
    pub z_im: f32,
    pub strength01: f32,
    pub phase_rad: f32,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct PredictiveRhythmBank {
    pub now: Tick,
    pub bands: Vec<RhythmBandState>,
}

impl PredictiveRhythmBank {
    pub fn is_informative(&self, threshold: f32) -> bool {
        let thresh = threshold.max(0.0);
        self.bands
            .iter()
            .any(|band| band.strength01.is_finite() && band.strength01 > thresh)
    }

    pub fn prior01_at_tick(&self, tb: &Timebase, tick: Tick) -> f32 {
        if tick < self.now {
            return 0.0;
        }
        let dt = tick.saturating_sub(self.now) as f64;
        let dt_sec = dt / tb.fs.max(1.0) as f64;
        let mut weighted_sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for band in &self.bands {
            let freq_hz = band.freq_hz;
            let strength01 = band.strength01.clamp(0.0, 1.0);
            let weight = band.weight.max(0.0);
            if !freq_hz.is_finite() || freq_hz <= 0.0 || strength01 == 0.0 || weight == 0.0 {
                continue;
            }
            let phi = std::f64::consts::TAU * (freq_hz as f64) * dt_sec;
            let phase_rad = band.phase_rad as f64;
            let align = (phi - phase_rad).cos();
            let band_score01 = 0.5 + 0.5 * align;
            let w = (weight as f64) * (strength01 as f64);
            weighted_sum += w * band_score01;
            weight_sum += w;
        }
        if weight_sum <= 1e-12 {
            return 0.0;
        }
        let prior = (weighted_sum / weight_sum).clamp(0.0, 1.0);
        prior as f32
    }
}

pub fn default_pred_rhythm_specs() -> Vec<RhythmBandSpec> {
    vec![
        RhythmBandSpec {
            freq_hz: 1.0,
            tau_sec: 2.0,
            weight: 1.0,
        },
        RhythmBandSpec {
            freq_hz: 2.0,
            tau_sec: 2.0,
            weight: 1.0,
        },
        RhythmBandSpec {
            freq_hz: 4.0,
            tau_sec: 1.5,
            weight: 1.0,
        },
        RhythmBandSpec {
            freq_hz: 8.0,
            tau_sec: 1.0,
            weight: 0.8,
        },
    ]
}

pub fn build_pred_rhythm_bank_from_intents(
    tb: &Timebase,
    now: Tick,
    intents: &[Intent],
    specs: &[RhythmBandSpec],
    horizon_future: Tick,
) -> PredictiveRhythmBank {
    let mut bands = Vec::with_capacity(specs.len());
    let end = now.saturating_add(horizon_future);
    let fs = tb.fs.max(1.0) as f64;
    for spec in specs {
        let freq_hz = spec.freq_hz;
        let tau_sec = spec.tau_sec;
        let weight = spec.weight.max(0.0);
        if !freq_hz.is_finite() || freq_hz <= 0.0 || weight == 0.0 {
            bands.push(RhythmBandState {
                freq_hz,
                z_re: 0.0,
                z_im: 0.0,
                strength01: 0.0,
                phase_rad: 0.0,
                weight,
            });
            continue;
        }
        let mut z_re = 0.0f64;
        let mut z_im = 0.0f64;
        for intent in intents {
            if intent.onset < now || intent.onset >= end {
                continue;
            }
            if !intent.freq_hz.is_finite() || intent.freq_hz <= 0.0 {
                if cfg!(debug_assertions) {
                    debug!(
                        target: "rhythm::pred",
                        "skip intent freq_hz={}",
                        intent.freq_hz
                    );
                }
                continue;
            }
            let amp = intent.amp.max(0.0);
            if amp == 0.0 {
                continue;
            }
            let dt = intent.onset.saturating_sub(now) as f64;
            let dt_sec = dt / fs;
            let phase = std::f64::consts::TAU * (freq_hz as f64) * dt_sec;
            let w_time = if tau_sec > 0.0 {
                (-dt_sec / tau_sec as f64).exp()
            } else {
                1.0
            };
            let w = (amp as f64) * w_time;
            z_re += w * phase.cos();
            z_im += w * phase.sin();
        }
        let mag = (z_re * z_re + z_im * z_im).sqrt();
        let strength01 = if mag > 0.0 {
            let k = 1.0f64;
            (1.0 - (-mag * k).exp()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let phase_rad = if mag > 0.0 { z_im.atan2(z_re) } else { 0.0 };
        bands.push(RhythmBandState {
            freq_hz,
            z_re: z_re as f32,
            z_im: z_im as f32,
            strength01: strength01 as f32,
            phase_rad: phase_rad as f32,
            weight,
        });
    }

    PredictiveRhythmBank { now, bands }
}
