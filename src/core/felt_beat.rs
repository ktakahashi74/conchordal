//! Joint beat/Poisson posterior on a co-moving circular phase grid.
use serde::{Deserialize, Serialize};

// Accepted only after comparison with the fine grid and consumer sampling noise.
const GRID: usize = 2048;
// Author rule: support the same half-cycle position used by generation.
const WEAK_PHASE: f32 = 0.5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeltBeatParams {
    /// Phase-noise variance rate in the source's seconds coordinate.
    pub sigma2: f32,
    pub v: f32,
    pub lambda: f32,
    pub lambda_0: f32,
    pub weak_v: f32,
    pub weak_lambda: f32,
}

impl Default for FeltBeatParams {
    fn default() -> Self {
        Self {
            sigma2: 0.05 * 0.05,
            v: 0.0001,
            lambda: 0.05,
            lambda_0: 0.01,
            weak_v: 0.0005,
            weak_lambda: 0.01,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct FeltBeatState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<crate::core::felt_input::InputState>,
    pub phase: f32,
    pub phase_offset: f32,
    pub c: f32,
    pub r: f32,
    pub pi: f32,
    pub margin: f32,
    pub m_plus: f32,
    pub p_beat: f32,
    pub onset_surprisal: Option<f32>,
    pub onset_bin_cycles: f32,
    pub onset_frac: Option<f32>,
    pub meter_frequency_hz: f32,
    pub meter_confidence: f32,
    /// Descriptive crossings between the seed coordinate's two half-circles.
    pub flips: u32,
    pub unknown: bool,
    pub seeded: bool,
    pub hopf_phase_difference: f32,
}

#[derive(Debug)]
pub(crate) struct FeltBeat {
    params: FeltBeatParams,
    q: [f32; GRID],
    p0: f32,
    tau: f64,
    cp: [f32; GRID],
    inverse: [f32; GRID],
    z: [f32; GRID],
    cos: [f32; GRID],
    sin: [f32; GRID],
    seeded: bool,
    half: Option<bool>,
    state: FeltBeatState,
}

// Wrapped normal mass over [a,b]. Complete turns are handled by the caller.
fn peak_mass(a: f32, b: f32, sd: f32) -> f32 {
    let tail = (-2.0 * f32::EPSILON.ln()).sqrt() * sd;
    let first = (a - tail).ceil() as i32;
    let last = (b + tail).floor() as i32;
    let mut mass = 0.0;
    for image in first..=last {
        let l = (a - image as f32) / (sd * std::f32::consts::SQRT_2);
        let r = (b - image as f32) / (sd * std::f32::consts::SQRT_2);
        mass += if l >= 0.0 {
            0.5 * (libm::erfcf(l) - libm::erfcf(r))
        } else if r <= 0.0 {
            0.5 * (libm::erfcf(-r) - libm::erfcf(-l))
        } else {
            1.0 - 0.5 * libm::erfcf(-l) - 0.5 * libm::erfcf(r)
        };
    }
    mass.max(0.0)
}

impl FeltBeat {
    pub(crate) fn new(params: FeltBeatParams) -> Self {
        Self {
            params,
            q: [0.5 / GRID as f32; GRID],
            p0: 0.5,
            tau: 0.0,
            cp: [0.0; GRID],
            inverse: [0.0; GRID],
            z: [0.0; GRID],
            cos: std::array::from_fn(|i| (std::f32::consts::TAU * i as f32 / GRID as f32).cos()),
            sin: std::array::from_fn(|i| (std::f32::consts::TAU * i as f32 / GRID as f32).sin()),
            seeded: false,
            half: None,
            state: FeltBeatState {
                pi: 0.5,
                p_beat: 0.5,
                unknown: true,
                ..Default::default()
            },
        }
    }

    fn seed(&mut self) {
        self.q.fill(0.5 / GRID as f32);
        self.p0 = 0.5;
        self.tau = 0.0;
        self.seeded = true;
        self.half = None;
    }

    pub(crate) fn invalidate(&mut self) {
        self.seeded = false;
        self.half = None;
        self.state.unknown = true;
        self.state.seeded = false;
        self.state.onset_surprisal = None;
        self.state.onset_frac = None;
        self.state.flips = 0;
    }

    fn normalize(&mut self) {
        let total = self.q.iter().sum::<f32>() + self.p0;
        if !(total > 0.0 && total.is_finite()) || self.q.iter().any(|q| !q.is_finite()) {
            self.invalidate();
            return;
        }
        for q in &mut self.q {
            *q = (*q / total).max(0.0);
        }
        self.p0 /= total;
    }

    fn rate(&self, coordinate: f32, period: f32, gain: f32) -> f32 {
        let half_cell = 0.5 / GRID as f32;
        self.params.lambda_0 * period
            + gain
                * GRID as f32
                * (self.params.lambda
                    * peak_mass(
                        coordinate - half_cell,
                        coordinate + half_cell,
                        self.params.v.sqrt() / period,
                    )
                    + self.params.weak_lambda
                        * peak_mass(
                            coordinate - WEAK_PHASE - half_cell,
                            coordinate - WEAK_PHASE + half_cell,
                            self.params.weak_v.sqrt() / period,
                        ))
    }

    // Cyclic tridiagonal backward Euler, using a Sherman-Morrison correction.
    fn diffuse(&mut self, k: f32) {
        let b = 1.0 + 2.0 * k;
        for i in 0..GRID {
            let diagonal = if i == 0 {
                2.0 * b
            } else if i == GRID - 1 {
                b + k * k / b
            } else {
                b
            };
            self.inverse[i] = 1.0 / (diagonal + if i > 0 { k * self.cp[i - 1] } else { 0.0 });
            self.cp[i] = -k * self.inverse[i];
            self.q[i] = (self.q[i] + if i > 0 { k * self.q[i - 1] } else { 0.0 }) * self.inverse[i];
            let u = if i == 0 {
                -b
            } else if i == GRID - 1 {
                -k
            } else {
                0.0
            };
            self.z[i] = (u + if i > 0 { k * self.z[i - 1] } else { 0.0 }) * self.inverse[i];
        }
        for i in (0..GRID - 1).rev() {
            self.q[i] -= self.cp[i] * self.q[i + 1];
            self.z[i] -= self.cp[i] * self.z[i + 1];
        }
        let factor =
            (self.q[0] + k / b * self.q[GRID - 1]) / (1.0 + self.z[0] + k / b * self.z[GRID - 1]);
        for i in 0..GRID {
            self.q[i] -= factor * self.z[i];
        }
    }

    fn interval(&mut self, duration: f32, period: f32, gain: f32) {
        if duration <= 0.0 || !self.seeded {
            return;
        }
        // Refinement is fixed by the consumer error test, not a model parameter.
        let duration = duration / 8.0;
        for _ in 0..8 {
            let whole = duration.floor();
            let rest = duration - whole;
            let base = (-self.params.lambda_0 * period * duration
                - gain * (self.params.lambda + self.params.weak_lambda) * whole)
                .exp();
            for i in 0..GRID {
                let a = (i as f32 / GRID as f32 + self.tau as f32 + 0.5).rem_euclid(1.0) - 0.5;
                let strong = peak_mass(a, a + rest, self.params.v.sqrt() / period);
                let weak = peak_mass(
                    a - WEAK_PHASE,
                    a - WEAK_PHASE + rest,
                    self.params.weak_v.sqrt() / period,
                );
                self.q[i] *= base
                    * (-gain * (self.params.lambda * strong + self.params.weak_lambda * weak))
                        .exp();
            }
            let flat = self.params.lambda_0 * period
                + gain * (self.params.lambda + self.params.weak_lambda);
            self.p0 *= (-flat * duration).exp();
            let variance = self.params.sigma2 / period * duration;
            self.diffuse(0.5 * variance * (GRID * GRID) as f32);
            let switch = -(-2.0 * std::f32::consts::PI.powi(2) * 2.0 * self.params.sigma2 / period
                * duration)
                .exp_m1()
                * 0.5;
            let pb = self.q.iter().sum::<f32>();
            for q in &mut self.q {
                *q = (1.0 - switch) * *q + switch * self.p0 / GRID as f32;
            }
            self.p0 = (1.0 - switch) * self.p0 + switch * pb;
            self.tau = (self.tau + duration as f64).rem_euclid(1.0);
            self.normalize();
            if !self.seeded {
                return;
            }
        }
    }

    fn event(&mut self, period: f32, gain: f32, bin: f32) {
        if !self.seeded {
            return;
        }
        let flat =
            self.params.lambda_0 * period + gain * (self.params.lambda + self.params.weak_lambda);
        self.p0 *= flat;
        for i in 0..GRID {
            let a = i as f32 / GRID as f32 + self.tau as f32;
            self.q[i] *= self.rate(a, period, gain);
        }
        let prediction = self.q.iter().sum::<f32>() + self.p0;
        self.state.onset_surprisal = Some(-(prediction * bin).ln());
        self.normalize();
    }

    fn readout(&mut self, period: f32, gain: f32, hopf: f32) {
        if !self.seeded {
            return;
        }
        let mut re = 0.0;
        let mut im = 0.0;
        let mut pb = 0.0;
        for i in 0..GRID {
            pb += self.q[i];
            re += self.q[i] * self.cos[i];
            im += self.q[i] * self.sin[i];
        }
        let offset = im.atan2(re);
        let xhat = offset / std::f32::consts::TAU;
        let flat =
            self.params.lambda_0 * period + gain * (self.params.lambda + self.params.weak_lambda);
        let mut mass = 0.0;
        let mut br = 0.0;
        let mut bi = 0.0;
        let mut hyp_basin = 0.0;
        let mut hyp_total = self.p0 * flat;
        let mut hyp_beat = 0.0;
        for i in 0..GRID {
            let x = i as f32 / GRID as f32;
            let basin = ((x - xhat + 0.5).rem_euclid(1.0) - 0.5).abs() < 0.25;
            let hypothetical = self.q[i] * self.rate(x + 0.5 - xhat, period, gain);
            hyp_total += hypothetical;
            hyp_beat += hypothetical;
            if basin {
                mass += self.q[i];
                br += self.q[i] * self.cos[i];
                bi += self.q[i] * self.sin[i];
                hyp_basin += hypothetical;
            }
        }
        self.state.phase_offset = offset;
        self.state.phase =
            crate::core::phase::wrap_pm_pi(offset + std::f32::consts::TAU * self.tau as f32);
        self.state.p_beat = pb.clamp(0.0, 1.0);
        self.state.c = re.hypot(im).clamp(0.0, self.state.p_beat);
        self.state.r = if mass > 0.0 {
            (br.hypot(bi) / mass).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.state.pi = if pb > 0.0 {
            (mass / pb).clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.state.margin = 2.0 * self.state.pi - 1.0;
        self.state.m_plus = if hyp_total > 0.0 {
            ((2.0 * hyp_basin - hyp_beat) / hyp_total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let half = re >= 0.0;
        self.state.flips = u32::from(self.half.is_some_and(|old| old != half));
        self.half = Some(half);
        self.state.unknown = false;
        self.state.seeded = true;
        self.state.hopf_phase_difference = crate::core::phase::wrap_pm_pi(self.state.phase - hopf);
    }

    pub(crate) fn process(
        &mut self,
        dt: f32,
        frequency: f32,
        onset: crate::core::onset::Onset,
        seeded_now: bool,
        stability: f32,
        hopf_phase: f32,
    ) {
        self.state.onset_surprisal = None;
        self.state.onset_frac = None;
        self.state.flips = 0;
        self.state.onset_bin_cycles = frequency * dt;
        let period = 1.0 / frequency;
        let gain = 1.0 + stability;
        let duration = frequency * dt;
        if seeded_now {
            self.seed();
            self.interval(duration * (1.0 - onset.frac), period, gain);
        } else if self.seeded {
            if onset.fired {
                self.interval(duration * onset.frac, period, gain);
                self.event(period, gain, duration);
                self.interval(duration * (1.0 - onset.frac), period, gain);
            } else {
                self.interval(duration, period, gain);
            }
        }
        self.readout(period, gain, hopf_phase);
        self.state.onset_frac = onset.fired.then_some(onset.frac);
    }

    pub(crate) fn state(&self) -> FeltBeatState {
        self.state
    }
}

#[cfg(test)]
include!("felt_beat_source.rs");
#[cfg(test)]
#[path = "felt_beat_tests.rs"]
mod tests;
