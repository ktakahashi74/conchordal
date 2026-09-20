//! Candidate accent support stays separate from the observed event ledger.

use super::*;
use crate::temporal_cognition::features::{Status, accent_salience, accent_status};

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct AccentDensity {
    pub(in crate::temporal_cognition) value: Feature,
    pub observed_samples: u64,
    pub projected_samples: u64,
    pub observed_weight: f64,
    pub projected_weight: f64,
    pub projected_accents: usize,
    pub latest_projected_interval: Option<[u64; 2]>,
}

impl Group {
    pub(super) fn projected_accent_density(
        &self,
        [start, end]: [u64; 2],
        issue: u64,
        rate: u32,
        (means, deviations): ([f64; 2], [f64; 2]),
        mut arrival: Option<&mut crate::temporal_cognition::arrival::Scratch>,
        future: impl Iterator<Item = window::Frame>,
    ) -> Result<AccentDensity, &'static str> {
        if start > end
            || issue > end
            || rate == 0
            || means.iter().chain(&deviations).any(|v| !v.is_finite())
            || deviations.iter().any(|v| *v < 0.)
        {
            return Err("invalid candidate accent window or scales");
        }
        let mut result = AccentDensity {
            value: Feature::Unsupported,
            observed_samples: self
                .history
                .iter()
                .filter(|s| s.raw.available_end <= issue)
                .map(|s| {
                    s.raw
                        .end
                        .min(end)
                        .min(issue)
                        .saturating_sub(s.raw.start.max(start))
                })
                .sum(),
            projected_samples: 0,
            observed_weight: self
                .accents
                .iter()
                .filter(|a| a.event_end >= start && a.event_end <= end && a.at_cut(issue))
                .map(|a| a.weight)
                .sum(),
            projected_weight: 0.,
            projected_accents: 0,
            latest_projected_interval: None,
        };
        #[derive(Clone, Copy)]
        struct Hop {
            start: u64,
            end: u64,
            known: bool,
            salience: Option<f64>,
        }
        let prefix = self.frames().skip(self.history.len().saturating_sub(3));
        let mut saved: [Option<Hop>; 4] = [None; 4];
        for frame in prefix.chain(future) {
            if frame.start >= frame.end {
                return Err("invalid candidate accent hop");
            }
            // No partial-hop lookahead or reanalysis is available in these profiles.
            if frame.end > end {
                break;
            }
            let projected = frame.end > issue;
            for feature in [frame.energy, frame.raw[3], frame.raw[5]] {
                if let Some(value) = feature.value() {
                    if !value.is_finite() || value < 0. {
                        return Err("invalid candidate accent component");
                    }
                    let valid = match feature {
                        Feature::Observed(_) => {
                            !projected
                                && frame.source_end >= frame.end
                                && frame.source_end <= frame.available
                                && frame.available <= issue
                        }
                        Feature::Projected(_) => {
                            projected
                                && frame.start >= issue
                                && frame.source_end >= frame.end
                                && frame.source_end <= frame.available
                                && frame.available <= end
                        }
                        Feature::Unsupported => unreachable!(),
                    };
                    if !valid {
                        return Err("candidate accent crosses issue provenance");
                    }
                }
            }
            if let Some(last) = saved[3] {
                if frame.start < last.end {
                    return Err("unordered candidate accent hops");
                }
                if frame.start > last.end {
                    saved.fill(None);
                }
            }
            let salience = frame.raw[3]
                .value()
                .zip(frame.raw[5].value())
                .map(|(rise, flux)| accent_salience([rise, flux], means, deviations));
            if salience.is_some_and(|v| !v.is_finite()) {
                return Err("candidate accent standardization overflow");
            }
            saved.rotate_left(1);
            saved[3] = Some(Hop {
                start: frame.start,
                end: frame.end,
                known: frame.energy.value().is_some(),
                salience,
            });
            if !projected {
                continue;
            }
            let Some(center) = saved[2] else { continue };
            let saliences = saved[1]
                .and_then(|s| s.salience)
                .zip(center.salience)
                .zip(saved[3].and_then(|s| s.salience));
            let supported = saved.iter().all(|s| s.is_some_and(|s| s.known)) && saliences.is_some();
            let weight = saliences
                .filter(|_| supported)
                .and_then(|((left, middle), right)| {
                    (accent_status([left, middle, right], 1.) == Status::Admitted)
                        .then_some((middle - 1.).clamp(0., 1.))
                });
            if let Some(scratch) = arrival.as_deref_mut() {
                scratch.receipt([center.start, center.end], frame.end, supported, weight)?;
            }
            if !supported {
                continue;
            }
            result.projected_samples += center
                .end
                .min(end)
                .saturating_sub(center.start.max(start).max(issue));
            if center.end >= start
                && center.end <= end
                && let Some(weight) = weight
            {
                result.projected_weight += weight;
                result.projected_accents += 1;
                result.latest_projected_interval = Some([center.start, center.end]);
            }
        }
        let samples = result.observed_samples + result.projected_samples;
        if samples > 0
            && samples as f64 >= 0.9 * (end - start) as f64
            && self.evicted_accent.is_none_or(|t| t < start)
        {
            let value = (result.observed_weight + result.projected_weight) * f64::from(rate)
                / samples as f64;
            result.value = if result.projected_samples > 0 || result.projected_accents > 0 {
                Feature::Projected(value)
            } else {
                Feature::Observed(value)
            };
        }
        Ok(result)
    }
}
