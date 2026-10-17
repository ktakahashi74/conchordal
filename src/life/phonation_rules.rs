use super::*;

pub enum OnsetRule {
    None,
    /// Fire on every live candidate, each onset carrying `strength` (1.0 = plain
    /// beat, > 1.0 = accent). The coupling clock emits candidates exactly at
    /// onset times, so no further timing decision is needed here.
    Always {
        strength: f32,
    },
    Accumulator {
        rate_hz: f32,
        refractory_gates: u32,
        acc: f32,
        next_allowed_gate: u64,
    },
    #[cfg(test)]
    Custom(Box<dyn FnMut(&IntervalInput, &CoreState) -> Option<OnsetKick> + Send>),
}

impl OnsetRule {
    pub fn accumulator(rate_hz: f32, refractory_gates: u32, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let acc = rng.random_range(0.7..1.0);
        OnsetRule::Accumulator {
            rate_hz,
            refractory_gates,
            acc,
            next_allowed_gate: 0,
        }
    }

    /// Renewal inter-onset interval used by the coupling clock to shape the
    /// free-running (low-coupling) limit: a regular period when `depth == 0` and
    /// a clustered, gappy non-metric texture as `depth` rises.
    pub(crate) fn flow_next_ioi(
        mean_rate_hz: f32,
        depth: f32,
        cluster_remaining: &mut u8,
        rng: &mut SmallRng,
    ) -> f32 {
        let mean_rate_hz = if mean_rate_hz.is_finite() {
            mean_rate_hz.max(0.01)
        } else {
            1.0
        };
        let base_ioi = 1.0 / mean_rate_hz;
        let depth = depth.clamp(0.0, 1.0);

        if depth <= f32::EPSILON {
            *cluster_remaining = 0;
            return base_ioi.clamp(0.035, 4.0);
        }

        let mult = if *cluster_remaining > 0 {
            *cluster_remaining = cluster_remaining.saturating_sub(1);
            rng.random_range(0.30..0.62)
        } else {
            let p_cluster = 0.05 + 0.17 * depth;
            let p_gap = 0.04 + 0.10 * depth;
            let draw = rng.random_range(0.0..1.0);
            if draw < p_cluster {
                *cluster_remaining = rng.random_range(1..=2);
                rng.random_range(0.34..0.72)
            } else if draw < p_cluster + p_gap {
                rng.random_range(1.18..1.72)
            } else {
                let local = rng.random_range(-1.0..1.0) + rng.random_range(-1.0..1.0);
                (0.22 * depth * local).exp().clamp(0.58, 1.58)
            }
        };
        (base_ioi * mult).clamp(0.035, 4.0)
    }

    pub fn on_candidate(&mut self, c: &IntervalInput, state: &CoreState) -> Option<OnsetKick> {
        match self {
            OnsetRule::None => None,
            OnsetRule::Always { strength } => {
                if !state.is_alive || c.weight <= 0.0 {
                    return None;
                }
                Some(OnsetKick {
                    strength: *strength,
                })
            }
            OnsetRule::Accumulator {
                rate_hz,
                refractory_gates,
                acc,
                next_allowed_gate,
            } => {
                const ACC_MAX: f32 = 32.0;
                if !state.is_alive {
                    return None;
                }
                if !rate_hz.is_finite() || *rate_hz <= 0.0 {
                    return None;
                }
                let weight = c.weight.max(0.0);
                *acc += *rate_hz * weight * c.dt_sec;
                if !acc.is_finite() || *acc > ACC_MAX {
                    *acc = ACC_MAX;
                }
                let refractory_ok = c.gate >= *next_allowed_gate;
                if *acc >= 1.0 && refractory_ok {
                    *acc -= 1.0;
                    *next_allowed_gate = c.gate + *refractory_gates as u64 + 1;
                    return Some(OnsetKick { strength: 1.0 });
                }
                None
            }
            #[cfg(test)]
            OnsetRule::Custom(f) => f(c, state),
        }
    }

    pub(super) fn from_config(config: &OnsetConfig, seed: u64) -> Self {
        match config {
            OnsetConfig::None => OnsetRule::None,
            OnsetConfig::Always { strength } => OnsetRule::Always {
                strength: *strength,
            },
            OnsetConfig::Accumulator { rate, refractory } => {
                OnsetRule::accumulator(*rate, *refractory, seed)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OnsetContext {
    pub tone_id: ToneId,
    pub tick: Tick,
    pub gate: u64,
    pub exc_gate: f32,
    pub exc_slope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DurationPlan {
    None,
    HoldTheta(f32),
}

pub enum DurationRule {
    FixedGate {
        length_gates: u32,
    },
    Field {
        hold_min_theta: f32,
        hold_max_theta: f32,
        curve_k: f32,
        curve_x0: f32,
        drop_gain: f32,
    },
    #[cfg(test)]
    Custom(Box<dyn FnMut(OnsetContext) -> DurationPlan + Send>),
}

impl DurationRule {
    pub fn fixed_gate(length_gates: u32) -> Self {
        DurationRule::FixedGate { length_gates }
    }

    pub fn field(
        hold_min_theta: f32,
        hold_max_theta: f32,
        curve_k: f32,
        curve_x0: f32,
        drop_gain: f32,
    ) -> Self {
        let hold_min_theta = if hold_min_theta.is_finite() {
            hold_min_theta.max(0.0)
        } else {
            0.25
        };
        let mut hold_max_theta = if hold_max_theta.is_finite() {
            hold_max_theta.max(0.0)
        } else {
            1.0
        };
        if hold_max_theta < hold_min_theta {
            hold_max_theta = hold_min_theta;
        }
        let curve_k = if curve_k.is_finite() { curve_k } else { 4.0 };
        let curve_x0 = if curve_x0.is_finite() { curve_x0 } else { 0.5 };
        let drop_gain = if drop_gain.is_finite() {
            drop_gain
        } else {
            0.0
        };
        DurationRule::Field {
            hold_min_theta,
            hold_max_theta,
            curve_k,
            curve_x0,
            drop_gain,
        }
    }

    pub fn on_note_on(&mut self, onset: OnsetContext) -> DurationPlan {
        #[cfg(test)]
        if let DurationRule::Custom(f) = self {
            return f(onset);
        }
        DurationPlan::HoldTheta(self.estimate_hold(onset.exc_gate, onset.exc_slope).unwrap())
    }

    pub(super) fn estimate_hold(&self, exc_gate: f32, exc_slope: f32) -> Option<f32> {
        match self {
            DurationRule::FixedGate { length_gates } => Some(*length_gates as f32),
            DurationRule::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => {
                let exc_gate = exc_gate.clamp(0.0, 1.0);
                let min = hold_min_theta.max(0.0);
                let max = hold_max_theta.max(min);
                let p = sigmoid01_stable(*curve_k * (exc_gate - *curve_x0));
                let mut hold = min + (max - min) * p;
                if *drop_gain > 0.0 && exc_slope.is_finite() {
                    let drop_val = (-exc_slope).max(0.0).clamp(0.0, 1.0);
                    hold *= 1.0 - drop_gain.clamp(0.0, 1.0) * drop_val;
                }
                Some(hold.max(0.0))
            }
            #[cfg(test)]
            DurationRule::Custom(_) => None,
        }
    }

    pub(super) fn from_config(config: &DurationConfig) -> Self {
        match config {
            DurationConfig::FixedGate { length_gates } => DurationRule::fixed_gate(*length_gates),
            DurationConfig::Field {
                hold_min_theta,
                hold_max_theta,
                curve_k,
                curve_x0,
                drop_gain,
            } => DurationRule::field(
                *hold_min_theta,
                *hold_max_theta,
                *curve_k,
                *curve_x0,
                *drop_gain,
            ),
        }
    }
}
