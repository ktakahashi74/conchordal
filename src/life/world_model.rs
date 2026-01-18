use std::sync::Arc;

use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::core::modulation::NeuralRhythms;
use crate::core::stream::dorsal::DorsalMetrics;
use crate::core::timebase::{Tick, Timebase};
use crate::life::gate_clock::next_gate_tick;
use tracing::debug;

#[derive(Clone, Debug, Default)]
struct TerrainPredictor {
    prev_obs: Option<(Tick, Arc<[f32]>)>,
    last_obs: Option<(Tick, Arc<[f32]>)>,
    cache_pred_next_gate: Option<(Tick, Arc<[f32]>)>,
}

impl TerrainPredictor {
    fn reset(&mut self) {
        self.prev_obs = None;
        self.last_obs = None;
        self.cache_pred_next_gate = None;
    }

    fn n_theta_per_delta(rhythm: &NeuralRhythms) -> u32 {
        let theta_hz = rhythm.theta.freq_hz;
        let delta_hz = rhythm.delta.freq_hz;
        if theta_hz.is_finite() && delta_hz.is_finite() && theta_hz > 0.0 && delta_hz > 0.0 {
            (theta_hz / delta_hz).round().clamp(2.0, 8.0) as u32
        } else {
            2
        }
    }

    fn theta_period_tick(time: &Timebase, rhythm: &NeuralRhythms) -> Tick {
        let theta_hz = rhythm.theta.freq_hz;
        if theta_hz.is_finite() && theta_hz > 0.0 {
            time.sec_to_tick(1.0 / theta_hz).max(1)
        } else {
            (time.hop as Tick).max(1)
        }
    }

    fn tau_horizon_ticks(time: &Timebase, rhythm: &NeuralRhythms) -> (Tick, Tick) {
        // If theta_hz is invalid, theta_period_tick falls back to hop-sized ticks.
        let n_theta = Self::n_theta_per_delta(rhythm).max(1) as Tick;
        let theta_period_tick = Self::theta_period_tick(time, rhythm);
        let tau_tick = theta_period_tick.saturating_mul(n_theta).max(1);
        let horizon_tick = theta_period_tick
            .saturating_mul(n_theta.saturating_mul(2))
            .max(1);
        (tau_tick, horizon_tick)
    }

    fn observe_consonance01(&mut self, tick: Tick, scan: Arc<[f32]>, space: &Log2Space) {
        space.assert_scan_len_named(&scan, "perc_c_state01_scan");
        self.prev_obs = self.last_obs.take();
        self.last_obs = Some((tick, scan));
        self.cache_pred_next_gate = None;
    }

    fn predict_consonance01_at(
        &self,
        tick: Tick,
        time: &Timebase,
        rhythm: &NeuralRhythms,
        space: &Log2Space,
    ) -> Option<Arc<[f32]>> {
        let (last_tick, last_scan) = self.last_obs.as_ref()?;
        space.assert_scan_len_named(last_scan, "pred_c_state01_scan");
        if let Some((prev_tick, prev_scan)) = self.prev_obs.as_ref() {
            space.assert_scan_len_named(prev_scan, "pred_c_state01_scan_prev");
            if prev_scan.len() != last_scan.len() {
                return Some(Arc::clone(last_scan));
            }
            let dt = last_tick.saturating_sub(*prev_tick).max(1) as f32;
            let a = tick.saturating_sub(*last_tick) as f32;
            let scale = a / dt;
            let (tau_tick, horizon_tick) = Self::tau_horizon_ticks(time, rhythm);
            let dist_tick = tick.saturating_sub(*last_tick).min(horizon_tick);
            let decay = (-(dist_tick as f32) / (tau_tick as f32)).exp();
            let mut out = Vec::with_capacity(last_scan.len());
            for (&last, &prev) in last_scan.iter().zip(prev_scan.iter()) {
                let pred = last + (last - prev) * scale * decay;
                let val = if pred.is_finite() {
                    pred.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                out.push(val);
            }
            Some(Arc::from(out))
        } else {
            Some(Arc::clone(last_scan))
        }
    }
}

pub struct WorldModel {
    pub time: Timebase,
    pub space: Log2Space,
    pub now: Tick,
    pub percept_landscape: Option<LandscapeFrame>,
    pub dorsal_metrics: Option<DorsalMetrics>,
    pub next_gate_tick_est: Option<Tick>,
    last_pred_next_gate: Option<(Tick, Arc<[f32]>)>,
    last_rhythm: NeuralRhythms,
    terrain_predictor: TerrainPredictor,
}

impl WorldModel {
    pub fn new(time: Timebase, space: Log2Space) -> Self {
        Self {
            time,
            space,
            now: 0,
            percept_landscape: None,
            dorsal_metrics: None,
            next_gate_tick_est: None,
            last_pred_next_gate: None,
            last_rhythm: NeuralRhythms::default(),
            terrain_predictor: TerrainPredictor::default(),
        }
    }

    pub fn advance_to(&mut self, now_tick: Tick) {
        self.now = now_tick;
    }

    pub fn set_space(&mut self, space: Log2Space) {
        self.space = space;
        self.terrain_predictor.reset();
        self.last_pred_next_gate = None;
    }

    pub fn update_gate_from_rhythm(&mut self, now_tick: Tick, rhythm: &NeuralRhythms) {
        self.next_gate_tick_est = next_gate_tick(now_tick, self.time.fs, rhythm.theta, 0.0);
        self.last_rhythm = *rhythm;
        self.last_pred_next_gate = None;
        if self.next_gate_tick_est.is_none() && cfg!(debug_assertions) {
            debug!(
                target: "gate",
                "next_gate_tick_est None: now_tick={} fs={:.3} theta_hz={:.3} theta_phase={:.3}",
                now_tick,
                self.time.fs,
                rhythm.theta.freq_hz,
                rhythm.theta.phase
            );
        }
    }

    pub fn observe_consonance01(&mut self, tick: Tick, scan: Arc<[f32]>) {
        // NSGT is right-aligned, so observations are stamped at frame_end_tick.
        self.terrain_predictor
            .observe_consonance01(tick, scan, &self.space);
        self.last_pred_next_gate = None;
    }

    pub fn predict_consonance01_at(&self, tick: Tick) -> Option<Arc<[f32]>> {
        self.terrain_predictor.predict_consonance01_at(
            tick,
            &self.time,
            &self.last_rhythm,
            &self.space,
        )
    }

    pub fn predictor_tau_horizon_ticks(&self, rhythm: &NeuralRhythms) -> (Tick, Tick) {
        TerrainPredictor::tau_horizon_ticks(&self.time, rhythm)
    }

    pub fn predictor_n_theta_per_delta(&self, rhythm: &NeuralRhythms) -> u32 {
        TerrainPredictor::n_theta_per_delta(rhythm)
    }

    pub fn predict_consonance01_next_gate(&mut self) -> Option<(Tick, Arc<[f32]>)> {
        let eval_tick = self.next_gate_tick_est?;
        if let Some((_, cached_scan)) = self
            .terrain_predictor
            .cache_pred_next_gate
            .as_ref()
            .filter(|(cached_tick, _)| *cached_tick == eval_tick)
        {
            let out = (eval_tick, Arc::clone(cached_scan));
            self.last_pred_next_gate = Some((eval_tick, Arc::clone(cached_scan)));
            return Some(out);
        }
        let scan = self.predict_consonance01_at(eval_tick)?;
        self.terrain_predictor.cache_pred_next_gate = Some((eval_tick, Arc::clone(&scan)));
        self.last_pred_next_gate = Some((eval_tick, Arc::clone(&scan)));
        Some((eval_tick, scan))
    }

    pub fn last_pred_next_gate(&self) -> Option<(Tick, Arc<[f32]>)> {
        self.last_pred_next_gate
            .as_ref()
            .map(|(tick, scan)| (*tick, Arc::clone(scan)))
    }

    pub fn sample_scan01(&self, scan: &[f32], freq_hz: f32) -> f32 {
        self.space
            .assert_scan_len_named(scan, "pred_c_state01_scan");
        let raw = sample_scan_linear_log2(&self.space, scan, freq_hz);
        if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
