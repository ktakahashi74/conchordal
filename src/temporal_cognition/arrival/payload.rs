//! A frozen, right-continuous step at the next Periodic arrival.

use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Payload {
    pub group: Handle,
    pub sample_rate: u32,
    pub model: ArrivalModel,
    pub version: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
    pub issued_at: u64,
    pub horizon_end: u64,
    pub last_accent: u64,
    pub period_seconds: f64,
    pub next_at: f64,
}

impl Frozen {
    pub(crate) fn payload(&self, forecast: Forecast) -> Result<Payload, &'static str> {
        if self.group != forecast.group
            || self.issued_at != forecast.issued_at
            || !forecast.valid_for(self.group, ArrivalModel::Periodic, self.issued_at)
        {
            return Err("arrival_identity_mismatch");
        }
        if forecast.reset_unknown || forecast.elapsed_seconds[0] != forecast.elapsed_seconds[1] {
            return Err("arrival_elapsed_uncertain");
        }
        let period = self
            .engine
            .context
            .peak
            .ok_or("arrival_period_absent")?
            .period_seconds;
        let elapsed = forecast.elapsed_seconds[1];
        if !period.is_finite() || period <= 0. || !elapsed.is_finite() || elapsed < 0. {
            return Err("arrival_invalid_period");
        }
        let next_at =
            self.issued_at as f64 + (period - elapsed % period) * f64::from(self.sample_rate);
        if !next_at.is_finite() || next_at <= self.issued_at as f64 {
            return Err("arrival_invalid_clock");
        }
        Ok(Payload {
            group: self.group,
            sample_rate: self.sample_rate,
            model: forecast.model,
            version: forecast.version,
            source_start: forecast.source_start,
            source_end: forecast.source_end,
            available: forecast.available,
            issued_at: self.issued_at,
            horizon_end: forecast.horizon_end,
            last_accent: forecast.last_accent,
            period_seconds: period,
            next_at,
        })
    }
}

impl Payload {
    pub(crate) fn matches(&self, forecast: Forecast, sample_rate: u32) -> bool {
        self.sample_rate == sample_rate
            && self.group == forecast.group
            && self.model == forecast.model
            && self.version == forecast.version
            && self.source_start == forecast.source_start
            && self.source_end == forecast.source_end
            && self.available == forecast.available
            && self.issued_at == forecast.issued_at
            && self.horizon_end == forecast.horizon_end
            && self.last_accent == forecast.last_accent
            && !forecast.reset_unknown
            && forecast.elapsed_seconds[0] == forecast.elapsed_seconds[1]
    }

    pub(crate) fn window_probability(&self, at: f64, width: f64) -> Option<f64> {
        if !at.is_finite()
            || !width.is_finite()
            || width < 0.
            || at < self.issued_at as f64
            || at + width > self.horizon_end as f64
            || !self.next_at.is_finite()
            || self.next_at <= self.issued_at as f64
        {
            return None;
        }
        let lo = (at - width).max(self.issued_at as f64);
        Some(f64::from(self.next_at <= at + width) - f64::from(self.next_at <= lo))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{accent, context, raw};
    use super::*;

    #[test]
    fn periodic_step_uses_actual_rate_and_window_endpoints() {
        for rate in [44_100, 48_000, 96_000] {
            let mut engine = Engine::new(TemporalPeriodConfig {
                model: ArrivalModel::Periodic,
                horizon_sec: 1.25,
            })
            .unwrap();
            let forecast = engine
                .advance(
                    Some(&raw(0, 110)),
                    Some(accent(100)),
                    context(110),
                    110,
                    rate,
                )
                .unwrap()
                .unwrap();
            let payload = engine
                .freeze(forecast.group, 110, rate)
                .unwrap()
                .payload(forecast)
                .unwrap();
            assert_eq!(payload.next_at, 100. + 0.5 * f64::from(rate));
            let next = payload.next_at;
            assert_eq!(payload.window_probability(next - 10., 10.), Some(1.));
            assert_eq!(payload.window_probability(next + 10., 10.), Some(0.));
            assert_eq!(payload.window_probability(next - 11., 10.), Some(0.));
            assert_eq!(payload.window_probability(109., 10.), None);
            assert_eq!(
                payload.window_probability(forecast.horizon_end as f64 - 10., 10.),
                Some(0.)
            );
            assert_eq!(
                payload.window_probability(forecast.horizon_end as f64, 10.),
                None
            );
            engine.advance(None, None, context(120), 120, rate).unwrap();
            let unknown = engine
                .advance(None, None, context(130), 130, rate)
                .unwrap()
                .unwrap();
            assert!(
                engine
                    .freeze(unknown.group, 130, rate)
                    .unwrap()
                    .payload(unknown)
                    .is_err()
            );
            // Later accents do not rewrite an already issued step.
            assert_eq!(payload.next_at, next);
        }
    }
}
