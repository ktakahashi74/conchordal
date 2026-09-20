//! Earlier-stage phrase context, frozen at completed acoustic cuts before joint reweighting.

use super::{acoustics, correspondence, grouping};
use crate::temporal_cognition::{ridge::Handle, section::input::Observation};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Snapshot {
    pub group: Handle,
    pub cut: u64,
    pub values: [Option<f64>; 12],
    pub query: Option<correspondence::Support>,
    pub truncated: usize,
}

#[derive(Clone, Copy)]
struct Sample {
    start: u64,
    end: u64,
    alpha: f64,
    grouping: f64,
}

pub(super) struct Cache {
    group: Handle,
    rate: u32,
    hop: u64,
    born: u64,
    previous: Snapshot,
    current: Snapshot,
    history: VecDeque<Sample>,
}

impl Cache {
    pub fn new(group: Handle, start: u64, rate: u32, hop: u64) -> Result<Self, &'static str> {
        if group.bus > 1 || group.generation <= 1 || rate == 0 || hop == 0 {
            return Err("invalid phrase-input owner or clock");
        }
        let empty = Snapshot {
            group,
            cut: start,
            values: [None; 12],
            query: None,
            truncated: 0,
        };
        Ok(Self {
            group,
            rate,
            hop,
            born: start,
            previous: empty,
            current: empty,
            history: VecDeque::with_capacity((2 * u64::from(rate)).div_ceil(hop) as usize + 1),
        })
    }

    pub fn at(&self, group: Handle, cut: u64) -> Result<&Snapshot, &'static str> {
        if self.group != group {
            return Err("foreign phrase-input owner");
        }
        if self.current.cut == cut {
            Ok(&self.current)
        } else if self.previous.cut == cut {
            Ok(&self.previous)
        } else {
            Err("phrase inputs belong to another completed cut")
        }
    }

    pub fn advance(
        &mut self,
        observation: &Observation,
        acoustics: &acoustics::Cache,
        grouping: &grouping::Cache,
        correspondence: &correspondence::Cache,
        retained: &[(u64, u64)],
        rms_reference: f64,
    ) -> Result<&Snapshot, &'static str> {
        let [start, end] = observation.interval;
        if observation.group != self.group
            || observation.rate != self.rate
            || start != self.current.cut
            || end <= start
            || end - start < self.hop
            || (observation.observed && end - start != self.hop)
            || grouping.group != self.group
            || grouping.cut != end
            || correspondence.group != self.group
            || correspondence.cut != end
            || correspondence.rate != self.rate
        {
            return Err("invalid phrase-input source owner or clock");
        }
        let alpha = acoustics.assignment(observation)?;
        let mut next = Snapshot {
            cut: end,
            ..self.current
        };
        let window = end.saturating_sub(2 * u64::from(self.rate)).max(self.born);
        let sample = observation
            .raw
            .filter(|_| observation.observed)
            .and_then(|u| {
                Some(Sample {
                    start: u.raw.start,
                    end: u.raw.end,
                    alpha: alpha?,
                    grouping: grouping.admission?,
                })
            });
        if observation.observed {
            next.values = [None; 12];
            let short = acoustics.short(end, rms_reference)?;
            for (v, f) in next.values[4..8].iter_mut().zip(short.values) {
                *v = f.value();
            }
            // No selected joint parent participates in this stage-1 bootstrap distribution.
            let bootstrap = correspondence.proposals(
                None,
                true,
                (end - start) as f64 / f64::from(self.rate),
                retained,
            )?;
            next.query = correspondence.support;
            next.truncated = bootstrap.fresh_truncated;
            if next.query.is_some() {
                let mut episodes = [None; 16];
                let mut count = 0;
                let mut no_memory = false;
                for entry in bootstrap.list.entries[..bootstrap.list.len]
                    .iter()
                    .flatten()
                {
                    let Some(id) = entry.id else {
                        continue;
                    };
                    let state = bootstrap
                        .states
                        .iter()
                        .flatten()
                        .find(|s| s.id == id)
                        .unwrap();
                    let Some(m) = state.matched else {
                        no_memory = true;
                        continue;
                    };
                    let target = (m.episode_id, m.episode_generation);
                    let i = episodes[..count]
                        .iter()
                        .position(|e: &Option<((u64, u64), f64)>| e.unwrap().0 == target)
                        .unwrap_or_else(|| {
                            let i = count;
                            episodes[i] = Some((target, 0.));
                            count += 1;
                            i
                        });
                    episodes[i].as_mut().unwrap().1 += entry.log_weight.exp();
                }
                let total = episodes[..count].iter().flatten().map(|e| e.1).sum::<f64>();
                next.values[10] = Some(total);
                next.values[11] = if count > 1 {
                    Some(
                        -episodes[..count]
                            .iter()
                            .flatten()
                            .map(|e| {
                                let q = e.1 / total;
                                q * q.ln()
                            })
                            .sum::<f64>()
                            / (count as f64).ln(),
                    )
                } else if count == 1 || no_memory {
                    Some(0.)
                } else {
                    None
                };
            }
        }
        // All fallible source checks precede history mutation.
        while self.history.front().is_some_and(|s| s.end <= window) {
            self.history.pop_front();
        }
        if let Some(sample) = sample {
            assert!(self.history.len() < self.history.capacity());
            self.history.push_back(sample);
        }
        if observation.observed {
            let (mut numerator, mut denominator, mut physical) = (0., 0., 0_u64);
            for s in &self.history {
                let n = s.end.min(end).saturating_sub(s.start.max(window));
                numerator += n as f64 * s.alpha * s.grouping;
                denominator += n as f64 * s.alpha;
                physical += n;
            }
            next.values[8] = (denominator > 0. && physical as f64 >= 0.9 * (end - window) as f64)
                .then(|| numerator / denominator);
        }
        self.previous = self.current;
        self.current = next;
        Ok(&self.current)
    }
}

#[cfg(test)]
mod tests;
