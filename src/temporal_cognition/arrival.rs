//! Periodic next-accent forecasts on the observed-accent clock and support.

use super::{
    accents::periods::Peak,
    features::{Accent, RawDescriptor},
    ridge::Handle,
};
use crate::config::{ArrivalModel, TemporalPeriodConfig};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(in crate::temporal_cognition) struct Context {
    pub peak: Option<Peak>,
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
    interval_starts: [u64; 4],
    cut: Option<u64>,
    uncertain_since: Option<u64>,
    context: Context,
}

mod frozen;
mod payload;
pub(crate) use frozen::Frozen;
pub(crate) use payload::Payload;

impl Engine {
    #[cfg(test)]
    pub(in crate::temporal_cognition) fn latest_accent(&self) -> Option<Accent> {
        self.last
    }

    pub(crate) fn new(config: TemporalPeriodConfig) -> Result<Self, &'static str> {
        if !config.horizon_sec.is_finite() || config.horizon_sec <= 0. || config.horizon_sec > 32. {
            return Err("invalid periodic arrival horizon");
        }
        Ok(Self {
            config,
            last: None,
            interval_starts: [u64::MAX; 4],
            cut: None,
            uncertain_since: None,
            context: Context::default(),
        })
    }

    fn probability(
        &self,
        elapsed: [f64; 2],
        context: Context,
        uncertain: bool,
    ) -> (Option<[f64; 2]>, usize) {
        (
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
        )
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
                self.interval_starts = [u64::MAX; 4];
            }
        }
        let observed = raw.is_some_and(|r| r.known_samples == r.end - r.start);
        if !observed || raw.is_some_and(|r| r.end < cut) {
            self.uncertain_since = Some(cut);
            self.interval_starts = [u64::MAX; 4];
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
                        self.interval_starts.rotate_right(1);
                        self.interval_starts[0] = old.source_start;
                    } else {
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
            evaluations,
        };
        self.cut = Some(cut);
        self.context = context;
        Ok(Some(forecast))
    }
}

#[cfg(test)]
mod tests;
