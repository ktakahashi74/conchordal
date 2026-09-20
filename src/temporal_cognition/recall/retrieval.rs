//! Causal retained-episode score bounds from an explicit residual-distance rule.

use super::*;
use crate::config::TemporalMemoryConfig;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Entry {
    pub episode: u64,
    pub generation: u64,
    pub acoustic_score: f64,
    pub availability: retention::Availability,
    pub score: [f64; 2],
    pub weight: [f64; 2],
    #[serde(skip)]
    prefix: [f64; 2],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Group {
    pub group: Handle,
    pub query_id: u64,
    pub support_start_sample: u64,
    pub support_end_sample: u64,
    pub available_at_sample: u64,
    pub evaluated_at_sample: u64,
    pub no_memory_bias: f64,
    pub recognition: retention::Recognition,
    pub discarded_weight: [f64; 2],
    pub scored_episodes: usize,
    pub search_pruned: usize,
    pub cutoff_tie: bool,
    pub entries: [Option<Entry>; 16],
}

#[cfg(test)]
pub(in crate::temporal_cognition) fn fixture(
    group: Handle,
    cut: u64,
    weights: &[(u64, [f64; 2])],
) -> Group {
    Group {
        group,
        query_id: cut,
        support_start_sample: cut.saturating_sub(100),
        support_end_sample: cut,
        available_at_sample: cut,
        evaluated_at_sample: cut,
        no_memory_bias: 0.,
        recognition: retention::Recognition {
            lower: 0.,
            upper: 1.,
            supported_matches: weights.len(),
        },
        discarded_weight: [0.; 2],
        scored_episodes: weights.len(),
        search_pruned: 0,
        cutoff_tie: false,
        entries: std::array::from_fn(|i| {
            weights.get(i).map(|&(episode, weight)| Entry {
                episode,
                generation: episode,
                acoustic_score: 0.,
                score: [0.; 2],
                weight,
                prefix: [0.; 2],
                availability: retention::Availability {
                    handle: episode,
                    log_lower: 0.,
                    log_upper: 0.,
                    elapsed_sec: 0.,
                    strength: 1.,
                    observed_interference: 0.,
                    coarse_unknown_interference: 0.,
                    missing_seconds_lower: 0.,
                    missing_seconds_upper: 0.,
                    gap_interference_upper: 0.,
                    envelope_invalid: false,
                    envelope_unverified: false,
                    rate_history_unknown: false,
                    clock_history_lost: false,
                },
            })
        }),
    }
}

fn logadd(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    if maximum == f64::NEG_INFINITY {
        maximum
    } else {
        maximum + (a.min(b) - maximum).exp().ln_1p()
    }
}

impl MatchSnapshot {
    /// Explicit rule with declared scales; no fitted coefficients. Coordinates without
    /// residual support contribute nothing rather than an imputed mean.
    pub(in crate::temporal_cognition) fn acoustic_score(
        &self,
        config: &TemporalMemoryConfig,
    ) -> Option<f64> {
        if self.ambiguous {
            return None;
        }
        let p = config.retention?;
        let r = self.residuals.filter(|r| r.observed > 0)?;
        let observed = f64::from(r.observed);
        let mut distance = 0.;
        let mut coordinates = 0.;
        for j in 0..10 {
            if r.coordinate_count[j] > 0 {
                distance += (r.coordinate_squared_error[j] / observed).sqrt() / config.scales[j];
                coordinates += 1.;
            }
        }
        if coordinates > 0. {
            distance /= coordinates;
        }
        if r.motion_count > 0 {
            distance +=
                (r.motion_squared_error / f64::from(r.motion_count)).sqrt() / p.motion_scale;
        }
        if r.interval_count > 0 {
            distance +=
                (r.interval_squared_error / f64::from(r.interval_count)).sqrt() / p.interval_scale;
        }
        let edits = f64::from(r.inserted + r.deleted) / observed;
        let score = -(distance / p.match_temperature) - p.edit_penalty * edits;
        score.is_finite().then_some(score)
    }
}

impl Recall {
    pub(super) fn refresh_retrieval(&mut self, cut: u64) -> Result<(), &'static str> {
        let (Some(retention), Some(clock), Some(parameters)) = (
            self.retention.as_mut(),
            self.acquisition.as_ref(),
            self.config.retention,
        ) else {
            return Ok(());
        };
        let time = cut as f64 / f64::from(self.rate);
        self.snapshot.retention = Some(retention.snapshot(clock, time)?);
        for (index, group) in self.groups.iter().enumerate() {
            self.snapshot.retrieval[index] = None;
            let Some((group, q)) = group.as_ref().and_then(|g| g.latest.map(|q| (g, q))) else {
                continue;
            };
            if q.received_at > cut
                || q.deadline < cut
                || q.supporting_audio_end.is_none()
                || cut.saturating_sub(q.support_end_sample) >= u64::from(self.rate) / 2
            {
                continue;
            }
            self.retrieval_scratch.clear();
            self.recognition_scratch.clear();
            for row in group.matches.iter() {
                let best = row
                    .iter()
                    .flatten()
                    .filter_map(|m| m.acoustic_score(&self.config).map(|s| (*m, s)))
                    .max_by(|(a, x), (b, y)| x.total_cmp(y).then(b.episode_id.cmp(&a.episode_id)));
                let Some((m, acoustic_score)) = best else {
                    continue;
                };
                let Some(slot) = retention
                    .records
                    .iter()
                    .position(|r| r.handle == m.episode_id)
                else {
                    continue;
                };
                if !self.episodes.iter().any(|e| {
                    e.identity.id == m.episode_id && e.identity.generation == m.episode_generation
                }) {
                    continue;
                }
                let availability = retention.availability(clock, slot, time)?.unwrap();
                let score = [availability.log_lower, availability.log_upper]
                    .map(|v| logadd(v, 1e-300_f64.ln()) + acoustic_score);
                self.retrieval_scratch.push(Entry {
                    episode: m.episode_id,
                    generation: m.episode_generation,
                    acoustic_score,
                    availability,
                    score,
                    weight: [0.; 2],
                    prefix: [0.; 2],
                });
                self.recognition_scratch
                    .push((m.episode_id, acoustic_score));
            }
            self.recognition_scratch.sort_unstable_by_key(|(id, _)| *id);
            let recognition = retention.recognition(
                clock,
                &self.recognition_scratch,
                parameters.no_memory_bias,
                time,
            )?;
            let mut prefix = [parameters.no_memory_bias; 2];
            for entry in &mut self.retrieval_scratch {
                entry.prefix = prefix;
                for (p, score) in prefix.iter_mut().zip(entry.score) {
                    *p = logadd(*p, score);
                }
            }
            let mut suffix = [f64::NEG_INFINITY; 2];
            for entry in self.retrieval_scratch.iter_mut().rev() {
                for i in 0..2 {
                    let others = logadd(entry.prefix[1 - i], suffix[1 - i]);
                    entry.weight[i] = (entry.score[i] - logadd(entry.score[i], others)).exp();
                }
                for (s, score) in suffix.iter_mut().zip(entry.score) {
                    *s = logadd(*s, score);
                }
            }
            self.retrieval_scratch.sort_unstable_by(|a, b| {
                b.weight[0]
                    .total_cmp(&a.weight[0])
                    .then(a.episode.cmp(&b.episode))
            });
            let discarded_weight = std::array::from_fn(|i| {
                self.retrieval_scratch
                    .iter()
                    .skip(16)
                    .map(|e| e.weight[i])
                    .sum()
            });
            self.snapshot.retrieval[index] = Some(Group {
                group: group.handle,
                query_id: q.query_id,
                support_start_sample: q.support_start_sample,
                support_end_sample: q.support_end_sample,
                available_at_sample: q.received_at,
                evaluated_at_sample: cut,
                no_memory_bias: parameters.no_memory_bias,
                recognition,
                discarded_weight,
                scored_episodes: self.retrieval_scratch.len(),
                search_pruned: q.pruned_candidates,
                cutoff_tie: q.cutoff_tie,
                entries: std::array::from_fn(|i| self.retrieval_scratch.get(i).copied()),
            });
        }
        Ok(())
    }
}
