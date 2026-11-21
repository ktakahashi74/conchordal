use super::{PitchHillClimbPitchCore, TargetProposal};
use rand::{Rng, RngExt};

/// Minimum field improvement required to accept a greedy hill-climb move.
const IMPROVEMENT_THRESHOLD: f32 = 0.1;

impl PitchHillClimbPitchCore {
    pub(in crate::life::voice) fn select_with_sequence<R: Rng + Clone>(
        &self,
        current_target_log2: f32,
        current_adjusted: f32,
        scored: &mut Vec<(f32, f32)>,
        counterfactual: &mut Vec<(f32, f32)>,
        rng: &mut R,
    ) -> TargetProposal {
        if !self.sequence_enabled {
            return self.select_with_pitch_set(
                current_target_log2,
                current_adjusted,
                scored,
                rng,
                None,
            );
        }
        counterfactual.clear();
        counterfactual.extend_from_slice(scored);
        let mut reason = None;
        let mut terms = Vec::new();
        if let Some(snapshot) = &self.sequence {
            if scored.len() > if self.pitch_set.is_some() { 66 } else { 64 } {
                reason = Some("candidate_resource_limit");
            } else {
                for (pitch, score) in scored.iter_mut() {
                    match snapshot.contribution(*pitch, current_target_log2) {
                        Some(term) if ((*score as f64 + term) as f32).is_finite() => {
                            let exact = *score as f64 + term;
                            let combined = exact as f32;
                            if self.sequence_trace {
                                terms.push(crate::life::own_sequence::CandidateScore {
                                    pitch_log2: *pitch,
                                    existing: *score,
                                    sequence: term,
                                    combined,
                                    rounding_error: (combined as f64 - exact).abs(),
                                });
                            }
                            *score = combined;
                        }
                        _ => {
                            reason = Some("nonfinite_sequence_score");
                            break;
                        }
                    }
                }
            }
        }
        if reason.is_some() {
            scored.copy_from_slice(counterfactual);
        }
        let mut counter_rng = rng.clone();
        let without = self.select_with_pitch_set(
            current_target_log2,
            current_adjusted,
            counterfactual,
            &mut counter_rng,
            None,
        );
        // The SQ-only f32 audit does not bound the composed PS candidate/affinity error.
        let mut precision = (self.sequence_trace
            && self.sequence.is_some()
            && self.pitch_set.is_none())
        .then(|| crate::life::own_sequence::SelectionPrecision {
            score_error_bound: terms.iter().map(|t| t.rounding_error).fold(0.0, f64::max)
                + 4e-12 * self.sequence.as_ref().unwrap().weight,
            ..Default::default()
        });
        let selected = self.select_with_pitch_set(
            current_target_log2,
            current_adjusted,
            scored,
            rng,
            precision.as_mut(),
        );
        *self.sequence_decision.borrow_mut() = Some(crate::life::own_sequence::SequenceDecision {
            at: self.sequence.as_ref().map_or(0, |s| s.at),
            epoch: self.sequence.as_ref().map_or(0, |s| s.epoch),
            version: self.sequence.as_ref().map_or(0, |s| s.version),
            active: self.sequence.is_some() && reason.is_none(),
            reason,
            baseline_log2: current_target_log2,
            selected_log2: selected.target_pitch_log2,
            without_sequence_log2: without.target_pitch_log2,
            changed: selected.target_pitch_log2.to_bits() != without.target_pitch_log2.to_bits(),
            previous_hz: self.sequence.as_ref().map(|s| s.previous_hz),
            distribution: self
                .sequence
                .as_ref()
                .filter(|_| self.sequence_trace)
                .map_or_else(Vec::new, |s| s.probabilities[..s.alphabet].to_vec()),
            candidates: terms,
            precision,
        });
        selected
    }

    pub(in crate::life::voice::pitch_core) fn select_with_pitch_set<R: Rng + Clone>(
        &self,
        current: f32,
        baseline: f32,
        scored: &mut Vec<(f32, f32)>,
        rng: &mut R,
        precision: Option<&mut crate::life::own_sequence::SelectionPrecision>,
    ) -> TargetProposal {
        let Some(ps) = &self.pitch_set else {
            return self.select_scored(current, baseline, scored, rng, precision);
        };
        let affinity = |p: f32| {
            ps.snapshot
                .as_ref()
                .map_or(0., |s| s.affinity(f64::from(p)))
        };
        let scale = ps.weight.max(1.);
        let mut shadow = rng.clone();
        let choose = |items: &[(f32, f32)], weight: f64, rng: &mut R| {
            let scale = weight.max(1.);
            let score = |p: f32, b: f32| f64::from(b) / scale + (weight / scale) * affinity(p);
            let mut best_pitch = current;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_distance = f32::INFINITY;
            for &(pitch, base) in items.iter() {
                let value = score(pitch, base);
                let distance = (pitch - current).abs();
                let better = value > best_score + 1e-6 / scale;
                let same = (value - best_score).abs() <= 1e-6 / scale;
                let closer = distance < best_distance - 1e-6;
                let same_distance = (distance - best_distance).abs() <= 1e-6;
                let random = same && same_distance && rng.random_range(0.0_f32..1.0) < 0.5;
                if better || (same && closer) || random {
                    best_pitch = pitch;
                    best_score = value;
                    best_distance = distance;
                }
            }
            let current_score = score(current, baseline);
            let improvement = best_score - current_score;
            let mut target = current;
            if improvement > f64::from(IMPROVEMENT_THRESHOLD) / scale {
                target = best_pitch;
            } else if self.temperature > 0. {
                let count = items
                    .iter()
                    .filter(|(p, _)| (*p - current).abs() > 1e-9)
                    .count();
                if count > 0 {
                    let pick = rng.random_range(0..count);
                    let &(p, b) = items
                        .iter()
                        .filter(|(p, _)| (*p - current).abs() > 1e-9)
                        .nth(pick)
                        .unwrap();
                    let delta = score(p, b) - current_score;
                    let draw = if delta >= 0. {
                        None
                    } else {
                        Some(f64::from(rng.random_range(0.0_f32..1.0)))
                    };
                    let ratio = (delta / (f64::from(self.temperature) / scale)).exp();
                    let accepted = delta >= 0. || draw.is_some_and(|u| u < ratio);
                    if accepted {
                        target = p;
                    }
                }
            }
            if let Some(temporal) = self.loo_input.relations.as_ref().filter(|u| u.active()) {
                let baseline_index = items
                    .iter()
                    .position(|(p, _)| *p == target)
                    .unwrap_or(items.len());
                let hold_missing = !items.iter().any(|(p, _)| *p == current);
                let temperature = if improvement > f64::from(IMPROVEMENT_THRESHOLD) / scale {
                    0.
                } else {
                    f64::from(self.temperature) / scale
                };
                let selected = temporal.choose_conditioned(
                    items.len() + usize::from(hold_missing),
                    baseline_index,
                    temperature,
                    1. / scale,
                    |i| {
                        let (p, b) = items.get(i).copied().unwrap_or((current, baseline));
                        (Some(p), temporal.source(), score(p, b), p == current)
                    },
                );
                target = items.get(selected).map_or(current, |(p, _)| *p);
            }
            TargetProposal {
                target_pitch_log2: target,
                salience: (improvement / (0.2 / scale)).clamp(0., 1.) as f32,
            }
        };
        let selected = choose(scored, ps.weight, rng);
        let target = selected.target_pitch_log2;
        if ps.trace {
            let added = ps.added.borrow();
            let subset: Vec<_> = scored
                .iter()
                .copied()
                .filter(|(p, _)| {
                    !added
                        .iter()
                        .any(|a| a.exp2().to_bits() == p.exp2().to_bits())
                })
                .collect();
            let without_candidates = choose(&subset, ps.weight, &mut shadow.clone());
            let outer_probes = [1. / 64., 16.].map(|weight| {
                (
                    weight,
                    choose(scored, weight, &mut shadow.clone()).target_pitch_log2,
                )
            });
            let mut same_candidates = scored.clone();
            let without =
                self.select_scored(current, baseline, &mut same_candidates, &mut shadow, None);
            *ps.decision.borrow_mut() = Some(crate::core::tonal_context::pitch_set::Decision {
                at: None,
                epoch: ps.snapshot.as_ref().map(|s| s.epoch),
                version: ps.snapshot.as_ref().map(|s| s.version),
                support_end: ps.snapshot.as_ref().map(|s| s.support_end),
                available_at: ps.snapshot.as_ref().map(|s| s.available_at),
                exposure: ps.snapshot.as_ref().map_or(0., |s| s.exposure),
                weight: ps.weight,
                baseline: (current, baseline, affinity(current)),
                candidates: scored.iter().map(|&(p, b)| (p, b, affinity(p))).collect(),
                selected: target,
                without_term: without.target_pitch_log2,
                outer_probes,
                added_candidates: added.clone(),
                without_candidates: without_candidates.target_pitch_log2,
                tc_failed_added: ps.tc_failed_added.get(),
                saturated_base: scored.iter().any(|&(p, b)| {
                    let a = (ps.weight / scale) * affinity(p);
                    b != 0. && a + f64::from(b) / scale == a
                }),
                changed: target.exp2().to_bits() != without.target_pitch_log2.exp2().to_bits(),
                salience: selected.salience,
            });
        }
        selected
    }

    pub(in crate::life::voice::pitch_core) fn select_scored<R: Rng + Clone>(
        &self,
        current_target_log2: f32,
        current_adjusted: f32,
        scored: &mut Vec<(f32, f32)>,
        rng: &mut R,
        mut precision: Option<&mut crate::life::own_sequence::SelectionPrecision>,
    ) -> TargetProposal {
        let mut best_pitch = current_target_log2;
        let mut best_score = f32::MIN;
        let mut best_distance = f32::INFINITY;
        for &(pitch, score) in scored.iter() {
            let distance = (pitch - current_target_log2).abs();
            if let Some(audit) = precision.as_deref_mut() {
                let error = 2.0 * audit.score_error_bound
                    + f32::EPSILON as f64 * (score.abs() as f64 + best_score.abs() as f64);
                if ((score as f64 - best_score as f64).abs() - 1e-6).abs() <= error {
                    audit.uncertain_rank_comparisons += 1;
                }
            }
            let better_score = score > best_score + 1e-6;
            let same_score = (score - best_score).abs() <= 1e-6;
            let closer = distance < best_distance - 1e-6;
            let same_distance = (distance - best_distance).abs() <= 1e-6;
            let random_tie_break = same_score && same_distance && rng.random_range(0.0..1.0) < 0.5;
            if better_score || (same_score && closer) || random_tie_break {
                best_score = score;
                best_pitch = pitch;
                best_distance = distance;
            }
        }

        let improvement = best_score - current_adjusted;
        let mut target_pitch_log2 = current_target_log2;
        if let Some(audit) = precision.as_deref_mut() {
            let error = audit.score_error_bound
                + f32::EPSILON as f64 * (best_score.abs() as f64 + current_adjusted.abs() as f64);
            audit.greedy_margin = improvement as f64 - IMPROVEMENT_THRESHOLD as f64;
            audit.greedy_uncertain = audit.greedy_margin.abs() <= error;
        }

        if improvement > IMPROVEMENT_THRESHOLD {
            target_pitch_log2 = best_pitch;
        } else if self.temperature > 0.0 {
            // Metropolis step: pick a random movable candidate and accept it if it
            // improves, or with Boltzmann probability if it is downhill.
            scored.retain(|(pitch, _)| (*pitch - current_target_log2).abs() > 1e-9);
            if !scored.is_empty() {
                let pick = rng.random_range(0..scored.len());
                let (candidate_pitch, candidate_score) = scored[pick];
                let delta = candidate_score - current_adjusted;
                let draw = if delta >= 0.0 {
                    None
                } else {
                    Some(rng.random_range(0.0..1.0))
                };
                let probability = if delta >= 0.0 {
                    1.0
                } else {
                    (delta / self.temperature).exp()
                };
                if let Some(audit) = precision {
                    let error = audit.score_error_bound
                        + f32::EPSILON as f64
                            * (candidate_score.abs() as f64 + current_adjusted.abs() as f64);
                    let relative = (error / self.temperature as f64).exp_m1();
                    audit.acceptance_probability = Some(probability);
                    audit.acceptance_draw = draw;
                    audit.acceptance_relative_error = Some(relative);
                    audit.acceptance_uncertain = (delta as f64).abs() <= error
                        || draw.is_some_and(|u| {
                            (u as f64 - probability as f64).abs()
                                <= probability as f64 * relative + 2.0 * f32::EPSILON as f64
                        });
                }
                let accepted = delta >= 0.0 || draw.is_some_and(|u| u < probability);
                if accepted {
                    target_pitch_log2 = candidate_pitch;
                }
            }
        }
        // else: temperature == 0, settle (greedy hill-climb stays put).

        if let Some(temporal) = self.loo_input.relations.as_ref().filter(|u| u.active()) {
            let baseline = scored
                .iter()
                .position(|(p, _)| *p == target_pitch_log2)
                .unwrap_or(scored.len());
            let hold_missing = !scored.iter().any(|(p, _)| *p == current_target_log2);
            let temperature = if improvement > IMPROVEMENT_THRESHOLD {
                0.
            } else {
                f64::from(self.temperature)
            };
            let selected = temporal.choose_conditioned(
                scored.len() + usize::from(hold_missing),
                baseline,
                temperature,
                1.,
                |i| {
                    let (p, s) = scored
                        .get(i)
                        .copied()
                        .unwrap_or((current_target_log2, current_adjusted));
                    (
                        Some(p),
                        temporal.source(),
                        f64::from(s),
                        p == current_target_log2,
                    )
                },
            );
            target_pitch_log2 = scored
                .get(selected)
                .map_or(current_target_log2, |(p, _)| *p);
        }

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}
