//! Bounded next-accent CDF publication; the I6 one-horizon forecast is unchanged.

use super::*;
use crate::temporal_cognition::action_profiles::consumer;
use serde::Serialize;

pub(crate) const CDF_LEN: usize = 201;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CdfUnknown {
    NoForecast,
    UnsupportedHorizon,
    UncertainElapsed,
    ResetUnknown,
    Nonperiodic,
    InvalidClock,
    IntegralFailed,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupArrival {
    pub group: Handle,
    pub periodic: bool,
    pub forecast: Option<Forecast>,
    #[serde(serialize_with = "serialize_cdf")]
    pub cdf: Option<[f64; CDF_LEN]>,
    pub unknown: Option<CdfUnknown>,
    pub evaluations: usize,
}

fn serialize_cdf<S>(cdf: &Option<[f64; CDF_LEN]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    cdf.as_ref()
        .map(|values| values.as_slice())
        .serialize(serializer)
}

impl GroupArrival {
    fn unknown(
        group: Handle,
        periodic: bool,
        forecast: Option<Forecast>,
        unknown: CdfUnknown,
        evaluations: usize,
    ) -> Self {
        Self {
            group,
            periodic,
            forecast,
            cdf: None,
            unknown: Some(unknown),
            evaluations,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ArrivalIssue {
    pub model: ArrivalModel,
    pub version: u64,
    pub bus: u8,
    pub epoch: u64,
    pub sample_rate: u32,
    pub issued_at: u64,
    pub horizon_end: u64,
    pub groups: [Option<GroupArrival>; 7],
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct ArrivalPublication {
    pub issue: ArrivalIssue,
    pub profile_key: Option<consumer::Key>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct ArrivalResources {
    pub publications: u64,
    pub cdf_generated: u64,
    pub cdf_failed: u64,
    pub horizon_overflow: u64,
    pub evaluations: u64,
    pub generation_us_last: u64,
    pub generation_us_total: u64,
    pub generation_us_max: u64,
}

impl Engine {
    pub(in crate::temporal_cognition) fn cdf(
        &self,
        group: Handle,
        cut: u64,
        sample_rate: u32,
        forecast: Option<Forecast>,
    ) -> GroupArrival {
        let periodic = self.context.peak.is_some();
        let failed = |reason, evaluations| {
            GroupArrival::unknown(group, periodic, forecast, reason, evaluations)
        };
        if self.config.horizon_sec != 4.0 {
            return failed(CdfUnknown::UnsupportedHorizon, 0);
        }
        let Some(forecast) = forecast else {
            return failed(CdfUnknown::NoForecast, 0);
        };
        let Some(horizon_end) = u64::from(sample_rate)
            .checked_mul(4)
            .and_then(|span| cut.checked_add(span))
        else {
            return failed(CdfUnknown::InvalidClock, 0);
        };
        if sample_rate == 0
            || self.cut != Some(cut)
            || self
                .last
                .is_none_or(|a| a.group != group || a.event_end != forecast.last_accent)
            || !forecast.valid_for(group, self.config.model, cut)
            || forecast.issued_at != cut
            || forecast.horizon_end != horizon_end
            || forecast.available > cut
            || self.context.available > cut
            || self.context.source_end > self.context.available
        {
            return failed(CdfUnknown::InvalidClock, 0);
        }
        if forecast.reset_unknown || self.uncertain_since.is_some() {
            return failed(CdfUnknown::ResetUnknown, 0);
        }
        let [lower, elapsed] = forecast.elapsed_seconds;
        if lower != elapsed || !elapsed.is_finite() || elapsed < 0.0 {
            return failed(CdfUnknown::UncertainElapsed, 0);
        }
        let Some(expected_elapsed) = self
            .last
            .and_then(|last| cut.checked_sub(last.event_end))
            .map(|samples| samples as f64 / f64::from(sample_rate))
        else {
            return failed(CdfUnknown::InvalidClock, 0);
        };
        if elapsed != expected_elapsed {
            return failed(CdfUnknown::InvalidClock, 0);
        }
        let mut values = [0.0; CDF_LEN];
        let mut evaluations = 0;
        match self.config.model {
            ArrivalModel::Hazard => {
                let mut integral_sum = 0.0;
                for (k, value) in values.iter_mut().enumerate().skip(1) {
                    let lo = elapsed + (k - 1) as f64 / 50.0;
                    let hi = elapsed + k as f64 / 50.0;
                    let (integral, count) = self.integral(lo, hi, self.context);
                    evaluations += count;
                    let Some(integral) = integral else {
                        return failed(CdfUnknown::IntegralFailed, evaluations);
                    };
                    if !integral.is_finite() || integral < 0.0 {
                        return failed(CdfUnknown::IntegralFailed, evaluations);
                    }
                    integral_sum += integral;
                    *value = -(-integral_sum).exp_m1();
                    if !integral_sum.is_finite()
                        || !value.is_finite()
                        || !(0.0..=1.0).contains(value)
                    {
                        return failed(CdfUnknown::IntegralFailed, evaluations);
                    }
                }
            }
            ArrivalModel::Periodic => {
                let Some(peak) = self.context.peak else {
                    return failed(CdfUnknown::Nonperiodic, 0);
                };
                if !peak.period_seconds.is_finite() || peak.period_seconds <= 0.0 {
                    return failed(CdfUnknown::Nonperiodic, 0);
                }
                let wait = peak.period_seconds - (elapsed % peak.period_seconds);
                if !wait.is_finite() {
                    return failed(CdfUnknown::InvalidClock, 0);
                }
                for (k, value) in values.iter_mut().enumerate().skip(1) {
                    *value = f64::from((k as f64 / 50.0) >= wait);
                }
            }
        }
        GroupArrival {
            group,
            periodic,
            forecast: Some(forecast),
            cdf: Some(values),
            unknown: None,
            evaluations,
        }
    }
}

#[cfg(test)]
mod tests;
