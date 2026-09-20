//! Frozen conditional forecasts; candidate receipts cannot update an observed engine.

use super::*;
use crate::temporal_cognition::feature_projection::Feature;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Frozen {
    pub group: Handle,
    pub issued_at: u64,
    pub sample_rate: u32,
    engine: Engine,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Projection {
    pub group: Handle,
    pub issued_at: u64,
    pub evaluated_at: u64,
    pub horizon_end: u64,
    pub original_last_accent: u64,
    pub candidate_last_accent: Option<u64>,
    pub original_reset_unknown: bool,
    pub reset_unknown: bool,
    pub candidate_accents: usize,
    pub unsupported_receipts: usize,
    pub elapsed_seconds: f64,
    pub processed_through: u64,
    pub incomplete_tail: bool,
    pub(in crate::temporal_cognition) probability: Feature,
    pub evaluations: usize,
}

pub(in crate::temporal_cognition) struct Scratch {
    frozen: Frozen,
    engine: Engine,
    end: u64,
    last: u64,
    candidate_last: Option<u64>,
    unknown: bool,
    receipts_end: u64,
    center_end: Option<u64>,
    accents: usize,
    unsupported: usize,
}

impl Engine {
    pub(in crate::temporal_cognition) fn freeze(
        &self,
        group: Handle,
        issue: u64,
        rate: u32,
    ) -> Option<Frozen> {
        let last = self.last?;
        (rate > 0
            && self.cut == Some(issue)
            && last.group == group
            && last.event_end <= issue
            && last.at_cut(issue)
            && self.context.available <= issue
            && self.context.source_end <= self.context.available)
            .then_some(Frozen {
                group,
                issued_at: issue,
                sample_rate: rate,
                engine: *self,
            })
    }
}

impl Frozen {
    pub(in crate::temporal_cognition) fn project(&self, end: u64) -> Option<Scratch> {
        if end < self.issued_at || end - self.issued_at > 4 * u64::from(self.sample_rate) {
            return None;
        }
        Some(Scratch {
            frozen: *self,
            engine: self.engine,
            end,
            last: self.engine.last?.event_end,
            candidate_last: None,
            unknown: self.engine.uncertain_since.is_some(),
            receipts_end: self.issued_at,
            center_end: None,
            accents: 0,
            unsupported: 0,
        })
    }
}

impl Scratch {
    pub(in crate::temporal_cognition) fn receipt(
        &mut self,
        interval: [u64; 2],
        available: u64,
        supported: bool,
        weight: Option<f64>,
    ) -> Result<(), &'static str> {
        let [start, end] = interval;
        if start >= end
            || end < self.frozen.issued_at
            || end >= available
            || available > self.end
            || available <= self.receipts_end
            || self.center_end.is_some_and(|prior| start < prior)
            || (weight.is_some() && end <= self.last)
            || weight.is_some_and(|v| !supported || !v.is_finite() || v <= 0. || v > 1.)
        {
            return Err("invalid hypothetical arrival receipt");
        }
        let gap = self
            .center_end
            .map_or(end > self.frozen.issued_at, |prior| start > prior);
        self.receipts_end = available;
        self.center_end = Some(end);
        if !supported || gap {
            self.unknown = true;
            self.engine.intervals = [None; 4];
            self.unsupported += 1;
        }
        if weight.is_some() {
            if self.unknown {
                self.engine.intervals = [None; 4];
            } else {
                self.engine.intervals.rotate_right(1);
                self.engine.intervals[0] =
                    Some((end - self.last) as f64 / f64::from(self.frozen.sample_rate));
            }
            self.last = end;
            self.candidate_last = Some(end);
            self.unknown = false;
            self.accents += 1;
        }
        Ok(())
    }

    pub(in crate::temporal_cognition) fn finish(&self) -> Option<Projection> {
        let original_unknown = self.frozen.engine.uncertain_since.is_some();
        let elapsed = (self.end - self.last) as f64 / f64::from(self.frozen.sample_rate);
        let incomplete_tail = self.receipts_end != self.end;
        let (probability, evaluations) = if original_unknown || self.unknown || incomplete_tail {
            (None, 0)
        } else {
            self.engine
                .probability([elapsed; 2], self.frozen.engine.context, false)
        };
        let probability = probability.filter(|p| p[0] == p[1]).map(|p| p[0]);
        Some(Projection {
            group: self.frozen.group,
            issued_at: self.frozen.issued_at,
            evaluated_at: self.end,
            horizon_end: self.end.checked_add(
                (self.engine.config.horizon_sec * f64::from(self.frozen.sample_rate)).ceil() as u64,
            )?,
            original_last_accent: self.frozen.engine.last?.event_end,
            candidate_last_accent: self.candidate_last,
            original_reset_unknown: original_unknown,
            reset_unknown: original_unknown || self.unknown || incomplete_tail,
            candidate_accents: self.accents,
            unsupported_receipts: self.unsupported,
            elapsed_seconds: elapsed,
            processed_through: self.receipts_end,
            incomplete_tail,
            probability: probability.map_or(Feature::Unsupported, |p| {
                if self.end == self.frozen.issued_at {
                    Feature::Observed(p)
                } else {
                    Feature::Projected(p)
                }
            }),
            evaluations,
        })
    }
}

#[cfg(test)]
mod tests;
