use std::f32::consts::TAU;
use tracing::debug;

use crate::core::phase::wrap_pm_pi;

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RhythmBand {
    pub phase: f32,   // wrapped [-pi, pi]
    pub freq_hz: f32, // tracked tempo
    pub mag: f32,     // presence
    pub alpha: f32,   // precision
    pub beta: f32,    // prediction error
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeuralRhythms {
    pub theta: RhythmBand,
    pub delta: RhythmBand,
    pub env_open: f32,
    pub env_level: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveRhythmParams {
    pub f_min_hz: f32,
    pub f_max_hz: f32,
    pub init_hz: f32,
    pub eta_mag: f32,
    pub eta_omega: f32,
    pub eta_phi: f32,
    pub tau_beta: f32,
    pub alpha_k: f32,
    pub noise_floor: f32,
    pub shape_p: f32,
}

#[derive(Clone, Debug)]
pub struct AdaptiveRhythm {
    pub phi: f32,   // unwrapped phase
    pub omega: f32, // rad/s
    pub mag: f32,
    pub beta: f32,
    pub alpha: f32,
    // Parameters
    f_min_hz: f32,
    f_max_hz: f32,
    eta_mag: f32,
    eta_omega: f32,
    eta_phi: f32,
    tau_beta: f32,
    alpha_k: f32,
    noise_floor: f32,
    shape_p: f32,
}

impl AdaptiveRhythm {
    pub fn new(params: AdaptiveRhythmParams) -> Self {
        let init_hz = params.init_hz.clamp(params.f_min_hz, params.f_max_hz);
        Self {
            phi: 0.0,
            omega: TAU * init_hz,
            mag: 0.1,
            beta: 0.0,
            alpha: 0.5,
            f_min_hz: params.f_min_hz,
            f_max_hz: params.f_max_hz,
            eta_mag: params.eta_mag,
            eta_omega: params.eta_omega,
            eta_phi: params.eta_phi,
            tau_beta: params.tau_beta,
            alpha_k: params.alpha_k,
            noise_floor: params.noise_floor,
            shape_p: params.shape_p,
        }
    }

    fn template(&self, phi: f32) -> f32 {
        let c = phi.cos();
        if c <= 0.0 { 0.0 } else { c.powf(self.shape_p) }
    }

    fn template_prime(&self, phi: f32) -> f32 {
        let c = phi.cos();
        if c <= 0.0 {
            0.0
        } else {
            let base = c.powf(self.shape_p - 1.0);
            -self.shape_p * base * phi.sin()
        }
    }

    pub fn update(&mut self, dt: f32, u: f32, vitality: f32) -> RhythmBand {
        let dt = dt.max(1e-4);
        let v = vitality.clamp(0.0, 1.0);
        let u = u.clamp(0.0, 1.0);

        let phi_wrapped = wrap_pm_pi(self.phi);
        let g = self.template(phi_wrapped);
        let y = self.mag * g;
        let r = u - y;

        let a_beta = (-dt / self.tau_beta.max(1e-4)).exp();
        let target_beta = (r * r).clamp(0.0, 1.0);
        self.beta = a_beta * self.beta + (1.0 - a_beta) * target_beta;
        self.beta = self.beta.clamp(0.0, 1.0);

        let alpha_raw = 1.0 / (1.0 + self.alpha_k * self.beta + self.noise_floor);
        self.alpha = alpha_raw.clamp(0.0, 1.0);

        // Vitality scales learning rates with a non-zero floor to avoid stalling.
        let v_scale = 0.5 + 0.5 * v;
        let eta_mag = self.eta_mag * v_scale;
        let eta_omega = self.eta_omega * v_scale;
        let eta_phi = self.eta_phi * v_scale;

        self.mag += eta_mag * self.alpha * r * g;
        self.mag = self.mag.clamp(0.0, 1.0);

        let grad = self.alpha * r * self.mag * self.template_prime(phi_wrapped);
        self.omega += eta_omega * grad;
        let min_omega = TAU * self.f_min_hz;
        let max_omega = TAU * self.f_max_hz;
        self.omega = self.omega.clamp(min_omega, max_omega);

        self.phi += (self.omega + eta_phi * grad) * dt;
        let phi_out = wrap_pm_pi(self.phi);

        RhythmBand {
            phase: phi_out,
            freq_hz: self.omega / TAU,
            mag: self.mag,
            alpha: self.alpha,
            beta: self.beta,
        }
    }

    fn clamp_omega(&self, omega: f32) -> f32 {
        let min_omega = TAU * self.f_min_hz;
        let max_omega = TAU * self.f_max_hz;
        omega.clamp(min_omega, max_omega)
    }
}

#[derive(Clone, Debug)]
pub struct RhythmEngine {
    pub theta: AdaptiveRhythm,
    pub delta: AdaptiveRhythm,
    last: NeuralRhythms,
    debug_timer: f32,
    metrics_timer: f32,
    prev_u_theta: f32,
    onset_baseline: f32,
    onset_var: f32,
    onset_timer: f32,
    onset_sum_cos: f32,
    onset_sum_sin: f32,
    onset_count: u32,
    time_since_last_onset: f32,
    ioi_ema: f32,
    ioi_valid: bool,
    last_ioi: f32,
    last_phase_err: f32,
}

impl Default for RhythmEngine {
    fn default() -> Self {
        // Defaults: theta 3-12 Hz, delta 0.2-3 Hz; tau_beta=0.12/0.3s; eta_mag=0.45/0.35,
        // eta_omega=0.08/0.05, eta_phi=0.04/0.03 to keep adaptation stable.
        let theta = AdaptiveRhythm::new(AdaptiveRhythmParams {
            f_min_hz: 3.0,
            f_max_hz: 12.0,
            init_hz: 6.0,
            eta_mag: 0.45,
            eta_omega: 0.08,
            eta_phi: 0.04,
            tau_beta: 0.12,
            alpha_k: 4.0,
            noise_floor: 0.02,
            shape_p: 4.0,
        });
        let delta = AdaptiveRhythm::new(AdaptiveRhythmParams {
            f_min_hz: 0.2,
            f_max_hz: 3.0,
            init_hz: 1.0,
            eta_mag: 0.35,
            eta_omega: 0.05,
            eta_phi: 0.03,
            tau_beta: 0.3,
            alpha_k: 3.0,
            noise_floor: 0.03,
            shape_p: 4.0,
        });
        Self {
            theta,
            delta,
            last: NeuralRhythms::default(),
            debug_timer: 0.0,
            metrics_timer: 0.0,
            prev_u_theta: 0.0,
            onset_baseline: 0.0,
            onset_var: 0.0,
            onset_timer: 0.0,
            onset_sum_cos: 0.0,
            onset_sum_sin: 0.0,
            onset_count: 0,
            time_since_last_onset: 0.0,
            ioi_ema: 0.0,
            ioi_valid: false,
            last_ioi: 0.0,
            last_phase_err: 0.0,
        }
    }
}

impl RhythmEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, dt: f32, u_theta: f32, u_delta: f32, vitality: f32) -> NeuralRhythms {
        let phi_before = self.theta.phi;
        let phase_before = wrap_pm_pi(phi_before);
        let omega_before = self.theta.omega;
        let tau = 1.0;
        let a = (-dt / tau).exp();
        self.onset_baseline = a * self.onset_baseline + (1.0 - a) * u_theta;
        let dev = u_theta - self.onset_baseline;
        self.onset_var = a * self.onset_var + (1.0 - a) * dev * dev;
        let std = self.onset_var.max(1e-6).sqrt();
        let onset_th_raw = self.onset_baseline + 2.0 * std;
        let onset_th = onset_th_raw.clamp(0.01, 0.95);
        let refractory = 0.08;
        self.onset_timer = (self.onset_timer - dt).max(0.0);
        self.time_since_last_onset += dt;
        let mut onset_fired = false;
        let mut onset_phase = phase_before;
        if self.onset_timer <= 0.0 && self.prev_u_theta < onset_th && u_theta >= onset_th {
            let denom = (u_theta - self.prev_u_theta).max(1e-6);
            let frac = ((onset_th - self.prev_u_theta) / denom).clamp(0.0, 1.0);
            onset_phase = wrap_pm_pi(phi_before + omega_before * dt * frac);
            self.onset_sum_cos += onset_phase.cos();
            self.onset_sum_sin += onset_phase.sin();
            self.onset_count += 1;
            self.onset_timer = refractory;
            onset_fired = true;
        }
        self.prev_u_theta = u_theta;

        let mut theta = self.theta.update(dt, u_theta, vitality);
        if onset_fired {
            let ioi = self.time_since_last_onset;
            let ioi_ok = (0.05..=0.5).contains(&ioi);
            if ioi_ok {
                if !self.ioi_valid {
                    self.ioi_ema = ioi;
                    self.ioi_valid = true;
                } else {
                    self.ioi_ema = self.ioi_ema + (ioi - self.ioi_ema) * 0.25;
                }
                self.last_ioi = ioi;
                let omega_meas = TAU / self.ioi_ema.max(1e-4);
                self.theta.omega = self.theta.omega + (omega_meas - self.theta.omega) * 0.35;
                self.theta.omega = self.theta.clamp_omega(self.theta.omega);
                let phase_err = onset_phase;
                self.last_phase_err = phase_err;
                self.theta.phi -= 0.5 * phase_err;
            }
            self.time_since_last_onset = 0.0;
            theta = RhythmBand {
                phase: wrap_pm_pi(self.theta.phi),
                freq_hz: self.theta.omega / TAU,
                mag: self.theta.mag,
                alpha: self.theta.alpha,
                beta: self.theta.beta,
            };
        }
        let delta = self.delta.update(dt, u_delta, vitality);
        let env_wave = 0.5 + 0.5 * delta.phase.cos();
        let delta_conf = (delta.mag * delta.alpha).clamp(0.0, 1.0);
        let env_open = compute_env_open(delta.phase, delta.mag, delta.alpha);
        let env_level = u_delta.clamp(0.0, 1.0);

        self.debug_timer += dt;
        if self.debug_timer >= 1.0 {
            debug!(
                target: "rhythm::engine",
                u_theta,
                u_delta,
                theta_freq_hz = theta.freq_hz,
                theta_mag = theta.mag,
                theta_alpha = theta.alpha,
                theta_beta = theta.beta,
                theta_phase = theta.phase,
                delta_freq_hz = delta.freq_hz,
                delta_mag = delta.mag,
                delta_alpha = delta.alpha,
                delta_beta = delta.beta,
                delta_phase = delta.phase,
                delta_conf,
                env_wave,
                env_open,
                env_level
            );
            self.debug_timer = 0.0;
        }

        self.metrics_timer += dt;
        if self.metrics_timer >= 1.0 {
            let (plv, mean_phase) = if self.onset_count > 0 {
                let sum_cos = self.onset_sum_cos;
                let sum_sin = self.onset_sum_sin;
                let denom = self.onset_count as f32;
                let plv = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / denom;
                (plv, sum_sin.atan2(sum_cos))
            } else {
                (0.0, 0.0)
            };
            debug!(
                target: "rhythm::metrics",
                onset_count = self.onset_count,
                onset_plv_theta = plv,
                onset_mean_phase_theta = mean_phase,
                onset_th,
                u_theta,
                onset_baseline = self.onset_baseline,
                onset_std = std,
                theta_ioi_last = self.last_ioi,
                theta_ioi_ema = self.ioi_ema,
                theta_freq_from_ioi = if self.ioi_valid {
                    1.0 / self.ioi_ema.max(1e-4)
                } else {
                    0.0
                },
                theta_phase_err_at_onset = self.last_phase_err,
                theta_freq_hz = theta.freq_hz,
                theta_alpha = theta.alpha,
                theta_beta = theta.beta
            );
            self.metrics_timer = 0.0;
            self.onset_sum_cos = 0.0;
            self.onset_sum_sin = 0.0;
            self.onset_count = 0;
        }

        self.last = NeuralRhythms {
            theta,
            delta,
            env_open,
            env_level,
        };
        self.last
    }

    pub fn last(&self) -> NeuralRhythms {
        self.last
    }
}

fn compute_env_open(delta_phase: f32, delta_mag: f32, delta_alpha: f32) -> f32 {
    let env_wave = 0.5 + 0.5 * delta_phase.cos();
    let delta_conf = (delta_mag * delta_alpha).clamp(0.0, 1.0);
    let min_depth = 0.15;
    let mag_scale = smoothstep(0.02, 0.08, delta_mag);
    let depth = (min_depth + (1.0 - min_depth) * delta_conf) * (0.2 + 0.8 * mag_scale);
    (1.0 - depth * (1.0 - env_wave)).clamp(0.0, 1.0)
}

impl NeuralRhythms {
    pub fn advance_in_place(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.theta.phase = wrap_pm_pi(self.theta.phase + TAU * self.theta.freq_hz * dt);
        self.delta.phase = wrap_pm_pi(self.delta.phase + TAU * self.delta.freq_hz * dt);
        self.env_open = compute_env_open(self.delta.phase, self.delta.mag, self.delta.alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_locks_to_pulse_rate() {
        let mut engine = RhythmEngine::default();
        let dt = 0.01;
        let mut t = 0.0;
        for _ in 0..3000 {
            let phase = (t * 5.0) % 1.0;
            let u_theta = if phase < 0.1 { 1.0 } else { 0.0 };
            let u_delta = 0.2;
            engine.update(dt, u_theta, u_delta, 1.0);
            t += dt;
        }
        let freq = engine.last().theta.freq_hz;
        assert!(
            (freq - 5.0).abs() < 1.0,
            "theta should converge near 5Hz, got {freq}"
        );
    }

    #[test]
    fn delta_tracks_slow_envelope() {
        let mut engine = RhythmEngine::default();
        let dt = 0.02;
        let mut t = 0.0;
        for _ in 0..2000 {
            let u_theta = 0.0;
            let u_delta = 0.5 + 0.5 * (TAU * 0.5 * t).sin();
            engine.update(dt, u_theta, u_delta, 1.0);
            t += dt;
        }
        let freq = engine.last().delta.freq_hz;
        assert!(
            (freq - 0.5).abs() < 0.4,
            "delta should sit near 0.5Hz, got {freq}"
        );
    }

    #[test]
    fn env_level_tracks_u_delta_when_confidence_low() {
        let mut engine = RhythmEngine::default();
        engine.delta.mag = 0.0;
        engine.delta.alpha = 0.0;
        engine.delta.beta = 1.0;
        engine.delta.phi = 0.0;
        engine.delta.omega = TAU;
        let u_delta = 0.7;
        let rhythms = engine.update(0.01, 0.0, u_delta, 1.0);
        assert!(
            (rhythms.env_level - u_delta).abs() < 0.1,
            "env_level should stay near u_delta when delta confidence is low"
        );
        assert!(
            (rhythms.env_open - 1.0).abs() < 0.2,
            "env_open should stay open when delta confidence is low"
        );
    }

    #[test]
    fn env_open_modulates_when_confidence_high() {
        let mut engine = RhythmEngine::default();
        engine.delta.mag = 1.0;
        engine.delta.alpha = 1.0;
        engine.delta.beta = 0.0;
        engine.delta.phi = 0.0;
        engine.delta.omega = TAU;
        let mut min_env: f32 = 1.0;
        let mut max_env: f32 = 0.0;
        for _ in 0..8 {
            let r = engine.update(0.25, 0.0, 1.0, 1.0);
            min_env = min_env.min(r.env_open);
            max_env = max_env.max(r.env_open);
        }
        assert!(
            (max_env - min_env) > 0.2,
            "env_open should oscillate when delta confidence is high"
        );
    }

    #[test]
    fn theta_phase_locks_to_pulse_train_plv() {
        let mut engine = RhythmEngine::default();
        let dt = 0.01;
        let mut t = 0.0;
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        let mut count: f32 = 0.0;
        for _ in 0..4000 {
            let phase = (t * 6.0) % 1.0;
            let u_theta = if phase < 0.05 { 1.0 } else { 0.0 };
            let rhythms = engine.update(dt, u_theta, 0.4, 1.0);
            if t > 2.0 && phase < 0.01 {
                sum_cos += rhythms.theta.phase.cos();
                sum_sin += rhythms.theta.phase.sin();
                count += 1.0;
            }
            t += dt;
        }
        let plv = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / count.max(1.0);
        let mean_phase = sum_sin.atan2(sum_cos);
        assert!(plv > 0.9, "expected PLV > 0.9, got {plv}");
        assert!(
            mean_phase.abs() < 0.3,
            "expected mean phase near 0, got {mean_phase}"
        );
    }

    #[test]
    fn theta_tracks_constant_8hz_pulse_train() {
        let mut engine = RhythmEngine::default();
        let dt = 0.01;
        let mut t = 0.0;
        for _ in 0..3000 {
            let phase = (t * 8.0) % 1.0;
            let u_theta = if phase < 0.05 { 1.0 } else { 0.0 };
            engine.update(dt, u_theta, 0.4, 1.0);
            t += dt;
        }
        let freq = engine.last().theta.freq_hz;
        assert!(
            (freq - 8.0).abs() < 0.5,
            "expected theta near 8Hz, got {freq}"
        );
    }

    #[test]
    fn theta_tracks_step_change_6_to_8() {
        let mut engine = RhythmEngine::default();
        let dt = 0.01;
        let mut t = 0.0;
        for _ in 0..500 {
            let phase = (t * 6.0) % 1.0;
            let u_theta = if phase < 0.05 { 1.0 } else { 0.0 };
            engine.update(dt, u_theta, 0.4, 1.0);
            t += dt;
        }
        for _ in 0..500 {
            let phase = (t * 8.0) % 1.0;
            let u_theta = if phase < 0.05 { 1.0 } else { 0.0 };
            engine.update(dt, u_theta, 0.4, 1.0);
            t += dt;
        }
        let freq = engine.last().theta.freq_hz;
        assert!(
            (freq - 8.0).abs() < 0.5,
            "expected theta near 8Hz after step, got {freq}"
        );
    }

    #[test]
    fn neuralrhythms_advance_wraps_phase() {
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            delta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            env_level: 0.5,
        };
        for _ in 0..4 {
            rhythms.advance_in_place(0.25);
        }
        assert!(
            rhythms.theta.phase.abs() < 1e-4,
            "theta phase should wrap to 0"
        );
        assert!(
            rhythms.delta.phase.abs() < 1e-4,
            "delta phase should wrap to 0"
        );
    }

    #[test]
    fn neuralrhythms_env_open_updates_with_delta_phase() {
        let mut rhythms = NeuralRhythms {
            theta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            delta: RhythmBand {
                phase: 0.0,
                freq_hz: 1.0,
                mag: 1.0,
                alpha: 1.0,
                beta: 0.0,
            },
            env_open: 1.0,
            env_level: 0.5,
        };
        let env_before = rhythms.env_open;
        rhythms.advance_in_place(0.1);
        let env_after = rhythms.env_open;
        assert!(env_after.is_finite());
        assert!(env_after >= 0.0 && env_after <= 1.0);
        assert!(
            (env_before - env_after).abs() > 1e-4,
            "env_open should respond to delta phase advance"
        );
    }
}
