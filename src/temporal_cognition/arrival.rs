//! Exchangeable next-accent forecasts on the same observed-accent clock and support.

use super::{
    accents::periods::Peak,
    features::{Accent, RawDescriptor},
    ridge::Handle,
};
use crate::config::{ArrivalModel, TemporalPeriodConfig};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(in crate::temporal_cognition) struct Context {
    pub peak: Option<Peak>,
    pub word: Option<bool>,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Forecast {
    pub model: ArrivalModel,
    pub version: u64,
    pub group: Handle,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
    pub issued_at: u64,
    pub horizon_end: u64,
    pub last_accent: u64,
    pub elapsed_seconds: [f64; 2],
    pub reset_unknown: bool,
    pub probability: Option<[f64; 2]>,
    pub observed_survival: Option<f64>,
    pub evaluations: usize,
}

impl Forecast {
    pub(crate) fn valid_for(&self, group: Handle, model: ArrivalModel, cut: u64) -> bool {
        self.group == group
            && self.model == model
            && self.version == 1
            && self.source_start <= self.source_end
            && self.source_end <= self.available
            && self.available <= self.issued_at
            && self.issued_at <= cut
            && cut <= self.horizon_end
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Engine {
    config: TemporalPeriodConfig,
    last: Option<Accent>,
    intervals: [Option<f64>; 4],
    interval_starts: [u64; 4],
    cut: Option<u64>,
    uncertain_since: Option<u64>,
    survival: Option<f64>,
    context: Context,
}

mod projection;
pub(in crate::temporal_cognition) use projection::Scratch;
pub(crate) use projection::{Frozen, Projection};

impl Engine {
    pub(crate) fn new(config: TemporalPeriodConfig) -> Result<Self, &'static str> {
        if config
            .coefficients
            .iter()
            .chain(&config.means)
            .any(|v| !v.is_finite())
            || config.deviations.iter().any(|v| !v.is_finite() || *v < 0.)
            || !config.horizon_sec.is_finite()
            || config.horizon_sec <= 0.
            || config.horizon_sec > 32.
        {
            return Err("invalid frozen arrival coefficients, scales or horizon");
        }
        Ok(Self {
            config,
            last: None,
            intervals: [None; 4],
            interval_starts: [u64::MAX; 4],
            cut: None,
            uncertain_since: None,
            survival: None,
            context: Context::default(),
        })
    }

    fn features(&self, elapsed: f64, context: Context) -> [f64; 18] {
        let mut values = [None; 8];
        for (j, value) in values[..3].iter_mut().enumerate() {
            *value = self.intervals[j + 1]
                .zip(self.intervals[0])
                .map(|(a, b)| (a / b).ln());
        }
        if let Some(peak) = context.peak {
            let phase = std::f64::consts::TAU * (elapsed / peak.period_seconds).fract();
            values[3] = Some(peak.period_seconds.log2());
            values[4] = Some(peak.support);
            values[5] = Some(phase.cos());
            values[6] = Some(phase.sin());
            values[7] = context.word.map(f64::from);
        }
        let mut x = [0.; 18];
        x[0] = 1.;
        x[1] = elapsed.ln_1p();
        for (j, value) in values.into_iter().enumerate() {
            let index = if j < 3 { 2 + j } else { 5 + j };
            let missing = if j < 3 { 5 + j } else { 10 + j };
            x[index] = value.map_or(0., |v| {
                (v - self.config.means[j]) / self.config.deviations[j].max(1e-6)
            });
            x[missing] = f64::from(value.is_none());
        }
        x
    }

    fn hazard(&self, elapsed: f64, context: Context) -> Option<f64> {
        let z: f64 = self
            .features(elapsed, context)
            .iter()
            .zip(self.config.coefficients)
            .map(|(x, c)| x * c)
            .sum();
        z.is_finite().then(|| z.max(0.) + (-z.abs()).exp().ln_1p())
    }

    // Two-point Gauss-Legendre with bounded step doubling: at most 62 evaluations.
    fn integral(&self, lo: f64, hi: f64, context: Context) -> (Option<f64>, usize) {
        if hi == lo {
            return (Some(0.), 0);
        }
        // A phase-independent hazard needs no samples per period constraint.
        let phase_period = context
            .peak
            .filter(|_| self.config.coefficients[10] != 0. || self.config.coefficients[11] != 0.)
            .map(|p| p.period_seconds);
        let mut previous = None;
        let mut evaluations = 0;
        for n in [1, 2, 4, 8, 16] {
            let width = (hi - lo) / n as f64;
            let mut sum = 0.;
            for i in 0..n {
                let mid = lo + (i as f64 + 0.5) * width;
                let offset = width / (2. * 3_f64.sqrt());
                let (a, b) = (
                    self.hazard(mid - offset, context),
                    self.hazard(mid + offset, context),
                );
                evaluations += 2;
                let (Some(a), Some(b)) = (a, b) else {
                    return (None, evaluations);
                };
                sum += 0.5 * width * (a + b);
            }
            if !sum.is_finite() {
                return (None, evaluations);
            }
            if phase_period.is_none_or(|period| width <= period / 4.)
                && previous.is_some_and(|v: f64| (v - sum).abs() <= 1e-8 + 1e-6 * sum.abs())
            {
                return (Some(sum), evaluations);
            }
            previous = Some(sum);
        }
        (None, evaluations)
    }

    // Interval arithmetic encloses all elapsed/reset alternatives, including internal phase extrema.
    fn uncertain_probability(&self, elapsed: [f64; 2], context: Context) -> Option<[f64; 2]> {
        let x = self.features(elapsed[0], context);
        let mut z = [0.; 2];
        for (i, &coefficient) in self.config.coefficients.iter().enumerate() {
            let interval = if i == 1 {
                [
                    elapsed[0].ln_1p(),
                    (elapsed[1] + self.config.horizon_sec).ln_1p(),
                ]
            } else if context.peak.is_some() && (i == 10 || i == 11) {
                let j = if i == 10 { 5 } else { 6 };
                let scale = self.config.deviations[j].max(1e-6);
                [
                    (-1. - self.config.means[j]) / scale,
                    (1. - self.config.means[j]) / scale,
                ]
            } else {
                [x[i]; 2]
            };
            let a = coefficient * interval[0];
            let b = coefficient * interval[1];
            z[0] += a.min(b);
            z[1] += a.max(b);
        }
        if z.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(
            z.map(|v| {
                -(-(v.max(0.) + (-v.abs()).exp().ln_1p()) * self.config.horizon_sec).exp_m1()
            }),
        )
    }

    fn probability(
        &self,
        elapsed: [f64; 2],
        context: Context,
        uncertain: bool,
    ) -> (Option<[f64; 2]>, usize) {
        match self.config.model {
            ArrivalModel::Hazard if uncertain => (self.uncertain_probability(elapsed, context), 0),
            ArrivalModel::Hazard => {
                let (integral, n) =
                    self.integral(elapsed[1], elapsed[1] + self.config.horizon_sec, context);
                (integral.map(|v| [-(-v).exp_m1(); 2]), n)
            }
            ArrivalModel::Periodic => (
                context.peak.map(|p| {
                    if self.config.horizon_sec >= p.period_seconds {
                        [1.; 2]
                    } else if uncertain {
                        [0., 1.]
                    } else {
                        let wait = p.period_seconds - (elapsed[1] % p.period_seconds);
                        [f64::from(wait <= self.config.horizon_sec); 2]
                    }
                }),
                0,
            ),
        }
    }

    pub(in crate::temporal_cognition) fn advance(
        &mut self,
        raw: Option<&RawDescriptor>,
        accent: Option<Accent>,
        context: Context,
        cut: u64,
        rate: u32,
    ) -> Result<Option<Forecast>, &'static str> {
        if rate == 0
            || self.cut.is_some_and(|old| cut < old)
            || context.available > cut
            || context.source_end > context.available
        {
            return Err("noncausal arrival context");
        }
        if let Some(raw) = raw {
            if raw.start >= raw.end
                || raw.end > raw.source_end
                || raw.known_samples > raw.end - raw.start
                || self.last.is_some_and(|a| a.group != raw.group)
                || raw.end > cut
                || raw.source_end > raw.available_end
                || raw.available_end > cut
            {
                return Err("unavailable arrival acquisition");
            }
            if self.cut.is_some_and(|old| raw.start > old) {
                self.uncertain_since = Some(raw.start);
                self.intervals = [None; 4];
                self.interval_starts = [u64::MAX; 4];
                self.survival = None;
            }
        }
        let observed = raw.is_some_and(|r| r.known_samples == r.end - r.start);
        if !observed || raw.is_some_and(|r| r.end < cut) {
            self.uncertain_since = Some(cut);
            self.intervals = [None; 4];
            self.interval_starts = [u64::MAX; 4];
            self.survival = None;
        }
        let mut reset = false;
        if let Some(a) = accent {
            if !a.at_cut(cut)
                || a.event_start >= a.event_end
                || a.event_end > a.source_end
                || a.source_end > a.available_end
                || a.weight > 1.
                || a.weight <= 0.
                || !a.weight.is_finite()
            {
                return Err("unsupported arrival anchor");
            }
            if let Some(old) = self.last {
                if a.group != old.group
                    || a.event_end < old.event_end
                    || (a.event_end == old.event_end && a != old)
                {
                    return Err("foreign or stale arrival anchor");
                }
                if a.event_end > old.event_end {
                    if self.uncertain_since.is_none()
                        && a.observed_prefix.checked_sub(old.observed_prefix)
                            == Some(a.event_end - old.event_end)
                    {
                        self.intervals.rotate_right(1);
                        self.interval_starts.rotate_right(1);
                        self.interval_starts[0] = old.source_start;
                        self.intervals[0] =
                            Some((a.event_end - old.event_end) as f64 / f64::from(rate));
                    } else {
                        self.intervals = [None; 4];
                        self.interval_starts = [u64::MAX; 4];
                    }
                    reset = true;
                }
            } else {
                reset = true;
            }
            if reset {
                self.last = Some(a);
                // A delayed accent resolves only gaps preceding its observed four-hop support.
                if self.uncertain_since.is_none_or(|t| t <= a.source_end) {
                    self.uncertain_since = None;
                    self.survival = Some(1.);
                }
            }
        }
        let Some(last) = self.last else {
            self.cut = Some(cut);
            self.context = context;
            return Ok(None);
        };
        let upper = (cut - last.event_end) as f64 / f64::from(rate);
        let lower = self
            .uncertain_since
            .map_or(upper, |t| (cut - t) as f64 / f64::from(rate));
        let begin = if reset {
            last.event_end
        } else {
            self.cut.unwrap_or(cut)
        };
        if let Some(survival) = self.survival {
            let c = if reset { context } else { self.context };
            self.survival = match self.config.model {
                ArrivalModel::Hazard => {
                    let (integral, _) =
                        self.integral((begin - last.event_end) as f64 / f64::from(rate), upper, c);
                    integral.map(|v| survival * (-v).exp())
                }
                ArrivalModel::Periodic => None,
            };
        }
        let (probability, evaluations) =
            self.probability([lower, upper], context, self.uncertain_since.is_some());
        let horizon = (self.config.horizon_sec * f64::from(rate)).ceil() as u64;
        let forecast = Forecast {
            model: self.config.model,
            version: 1,
            group: last.group,
            source_start: (if context.source_end == 0 {
                last.source_start
            } else {
                context.source_start.min(last.source_start)
            })
            .min(*self.interval_starts.iter().min().unwrap()),
            source_end: context.source_end.max(last.source_end),
            available: context.available.max(last.available_end),
            issued_at: cut,
            horizon_end: cut.checked_add(horizon).ok_or("arrival horizon overflow")?,
            last_accent: last.event_end,
            elapsed_seconds: [lower, upper],
            reset_unknown: self.uncertain_since.is_some(),
            probability,
            observed_survival: self.survival,
            evaluations,
        };
        self.cut = Some(cut);
        self.context = context;
        Ok(Some(forecast))
    }
}

#[cfg(test)]
mod tests;
