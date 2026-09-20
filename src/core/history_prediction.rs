//! Matching-horizon energy regression from observed auditory history.

use crate::core::temporal_expectation::{ENERGY_PENDING_LEN, FORECAST_LEN, FORECAST_STRIDE};
use crate::core::temporal_history::AuditoryHistorySnapshot;

const FEATURES: usize = 57;
const RIDGE: f64 = 0.01;
pub(crate) const SCORE_LEADS: [usize; 7] = [0, 5, 10, 25, 50, 100, 200];

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct PredictionMatch<'a> {
    pub(crate) issued_step: u64,
    pub(crate) target_step: u64,
    pub(crate) requested_frame: Option<u64>,
    pub(crate) completed_before_issue: u64,
    pub(crate) issued_features: Option<&'a [f64]>,
    pub(crate) history_weight: [f32; 3],
    pub(crate) recurrence: [f32; 3],
    pub(crate) history: [f32; 3],
    pub(crate) mixed: [f32; 3],
    pub(crate) observed: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct PredictionErrorTotals {
    pub(crate) issued: u64,
    pub(crate) completed: [u64; 7],
    pub(crate) recurrence_squared_error: [[f64; 3]; 7],
    pub(crate) history_squared_error: [[f64; 3]; 7],
    pub(crate) mixed_squared_error: [[f64; 3]; 7],
}

#[derive(Clone, Copy)]
struct PendingHistory {
    step: Option<u64>,
    features: [f64; FEATURES],
    energy: [f32; 3],
}

#[derive(Clone)]
struct PendingComparison {
    step: Option<u64>,
    requested_frame: Option<u64>,
    recurrence: [[f32; 3]; FORECAST_LEN],
    history: [[f32; 3]; FORECAST_LEN],
    mixed: [[f32; 3]; FORECAST_LEN],
    history_weight: [[f32; 3]; 7],
    completed_before_issue: [u64; 7],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::temporal_history::{AuditoryHistory, HISTORY_AGES_SEC};

    #[test]
    fn readonly_comparison_matches_issue_without_registering_or_training() {
        let mut predictor = HistoryEnergyPrediction::new();
        let mut untouched = predictor.clone();
        let recurrence = [[0.1; 3]; FORECAST_LEN];
        let history = [[0.5; 3]; FORECAST_LEN];
        for step in 1..450 {
            let preview = predictor.preview_comparison(recurrence, history);
            assert_eq!(predictor.preview_comparison(recurrence, history), preview);
            assert_eq!(
                serde_json::to_value(&predictor.errors).unwrap(),
                serde_json::to_value(&untouched.errors).unwrap()
            );
            assert_eq!(
                predictor.issue_comparison(step, recurrence, history, None),
                preview
            );
            assert_eq!(
                untouched.issue_comparison(step, recurrence, history, None),
                preview
            );
            let target = [0.2 + (step % 3) as f32 * 0.1; 3];
            predictor.observe(step, target, |_| {});
            untouched.observe(step, target, |_| {});
        }
        assert_eq!(
            serde_json::to_value(predictor.take_errors()).unwrap(),
            serde_json::to_value(untouched.take_errors()).unwrap()
        );
    }

    #[test]
    fn matches_preserve_context_and_issued_support_across_ring_wraps() {
        let mut p = HistoryEnergyPrediction::new();
        let mut quiet = p.clone();
        let history = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        let energy_at = |step: u64| [0.01 * (1.0 + (step as f32 * 0.031).sin()); 3];
        let mut seen = [0; 7];
        for step in 0..950 {
            let energy = energy_at(step);
            p.observe(step, energy, |m| {
                let horizon = (m.target_step - m.issued_step) as usize;
                let index = SCORE_LEADS.iter().position(|h| 2 * h == horizon).unwrap();
                seen[index] += 1;
                assert_eq!(m.target_step, step);
                assert_eq!(m.observed, energy);
                assert_eq!(m.recurrence, [0.0; 3]);
                assert_eq!(m.history, [1.0; 3]);
                assert_eq!(m.mixed, m.history_weight);
                assert_eq!(
                    m.issued_features.unwrap()[1],
                    energy_at(m.issued_step - 1)[0].sqrt() as f64
                );
                assert_eq!(
                    m.completed_before_issue,
                    (m.issued_step - 1).saturating_sub(horizon as u64) / 10
                );
            });
            quiet.observe(step, energy, |_| {});
            p.issue(step + 1, energy, &history);
            quiet.issue(step + 1, energy, &history);
            if (step + 1) % 10 == 0 {
                assert_eq!(
                    p.issue_comparison(
                        step + 1,
                        [[0.0; 3]; FORECAST_LEN],
                        [[1.0; 3]; FORECAST_LEN],
                        None
                    ),
                    quiet.issue_comparison(
                        step + 1,
                        [[0.0; 3]; FORECAST_LEN],
                        [[1.0; 3]; FORECAST_LEN],
                        None
                    )
                );
            }
        }
        assert!(seen[6] > 0 && seen[0] > seen[6]);
        p.clear_pending(false);
        p.observe(1000, [100.0; 3], |_| panic!("cancelled target was emitted"));
        p.clear_pending(true);
        p.issue_comparison(
            1001,
            [[0.0; 3]; FORECAST_LEN],
            [[1.0; 3]; FORECAST_LEN],
            None,
        );
        p.observe(1001, [0.0; 3], |m| {
            assert!(m.issued_features.is_none());
            assert_eq!(m.completed_before_issue, 0);
        });
    }

    #[test]
    fn errors_score_the_frozen_mix_and_draining_does_not_train() {
        let mut p = HistoryEnergyPrediction::new();
        let a = [[0.0; 3]; FORECAST_LEN];
        let b = [[1.0; 3]; FORECAST_LEN];
        assert_eq!(p.issue_comparison(1, a, b, None).0, a);
        assert_eq!(p.take_errors().unwrap().issued, 1);
        p.observe(1, [1.0; 3], |_| {});
        let mut undrained = p.clone();
        let errors = p.take_errors().unwrap();
        assert_eq!(errors.completed, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(errors.mixed_squared_error[0], [1.0; 3]);
        assert_eq!(errors.history_squared_error[0], [0.0; 3]);
        assert!(p.take_errors().is_none());
        assert_eq!(
            p.issue_comparison(2, a, b, None),
            undrained.issue_comparison(2, a, b, None)
        );
        p.observe(2, [0.0; 3], |_| {});
        undrained.observe(2, [0.0; 3], |_| {});
        assert_eq!(
            p.issue_comparison(3, a, b, None),
            undrained.issue_comparison(3, a, b, None)
        );
        let second = p.take_errors().unwrap();
        assert_eq!(second.completed, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(second.mixed_squared_error[0], [1.0; 3]);
        assert_eq!(
            undrained.take_errors().unwrap().mixed_squared_error[0],
            [2.0; 3]
        );
        p.clear_pending(false);
        p.observe(401, [100.0; 3], |_| {});
        assert!(p.take_errors().is_none());
    }

    #[test]
    fn directed_age_contrasts_reverse_order_and_ignore_common_gain() {
        let mut snapshot = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        snapshot.known_band_rms_by_age[0] = [0.0, 0.0, 1.0];
        snapshot.known_band_rms_by_age[1] = [1.0, 0.0, 0.0];
        let before = HistoryEnergyPrediction::features([0.0; 3], &snapshot);
        assert!(before[37] > 0.49);
        snapshot.known_band_rms_by_age.swap(0, 1);
        let reverse = HistoryEnergyPrediction::features([0.0; 3], &snapshot);
        assert_eq!(reverse[37], -before[37]);
        for row in &mut snapshot.known_band_rms_by_age {
            *row = row.map(|v| 0.2 * v);
        }
        let scaled = HistoryEnergyPrediction::features([0.0; 3], &snapshot);
        for (a, b) in reverse[36..].iter().zip(&scaled[36..]) {
            assert!((a - b).abs() < 1e-10);
            assert!(b.abs() <= 0.5);
        }
        for (i, row) in snapshot.known_band_rms_by_age.iter_mut().enumerate() {
            *row = [1.0, 2.0, 3.0].map(|v| v * (i + 1) as f32);
        }
        assert!(
            HistoryEnergyPrediction::features([0.0; 3], &snapshot)[36..]
                .iter()
                .all(|v| *v == 0.0)
        );
    }

    #[test]
    fn comparison_uses_completed_targets_and_repeated_issue_is_idempotent() {
        let mut predictor = HistoryEnergyPrediction::new();
        let a = [[0.1; 3]; FORECAST_LEN];
        let b = [[0.5; 3]; FORECAST_LEN];
        let issued = predictor.issue_comparison(1, a, b, None);
        assert_eq!(issued.0, a);
        assert_eq!(issued.1, [[0.0; 3]; FORECAST_LEN]);
        for _ in 0..20 {
            assert_eq!(predictor.issue_comparison(1, a, b, None), issued);
        }
        predictor.observe(1, [0.3; 3], |_| {});
        let later = predictor.issue_comparison(2, a, b, None);
        assert!((later.1[0][0] - 0.5).abs() < 1e-6);
        assert_eq!(later.1[1], [0.0; 3]);
        predictor.observe(2, [0.1; 3], |_| {});
        predictor.observe(3, [0.4; 3], |_| {});
        let next = predictor.issue_comparison(4, a, b, None);
        assert!((next.1[0][0] - 0.25).abs() < 1e-6);
        assert!((next.1[1][0] - 0.75).abs() < 1e-6);
        predictor.clear_pending(false);
        predictor.observe(4, [100.0; 3], |_| {});
        assert_eq!(predictor.issue_comparison(5, a, b, None), next);
        predictor.clear_pending(true);
        assert_eq!(predictor.issue_comparison(6, a, b, None), issued);
    }

    #[test]
    fn delayed_prediction_matches_independent_batch_ridge_across_ring_wraps() {
        let mut predictor = HistoryEnergyPrediction::new();
        let mut history = AuditoryHistory::new(HISTORY_AGES_SEC);
        let mut inputs = Vec::new();
        let mut energies = Vec::new();
        let mut snapshot = history.snapshot();
        for step in 0..950 {
            let phase = step as f32 * 0.031;
            let energy = [
                0.05 * (1.0 + phase.sin()),
                0.03 * (1.0 + (phase * 1.7).cos()),
                if step % 71 < 20 { 0.02 } else { 0.0 },
            ];
            predictor.observe(step, energy, |_| {});
            history.advance(0.01, Some(energy.map(f32::sqrt)));
            snapshot = history.snapshot();
            predictor.issue(step + 1, energy, &snapshot);
            inputs.push(HistoryEnergyPrediction::features(energy, &snapshot));
            energies.push(energy);
        }
        let current = *energies.last().unwrap();
        let x = *inputs.last().unwrap();
        let actual = predictor.forecast(current, &snapshot);
        // Preserve the former three separate dot products bit for bit on trained state.
        let legacy_projection: [f64; FEATURES] =
            std::array::from_fn(|i| predictor.inverse[i].iter().zip(x).map(|(p, x)| p * x).sum());
        for (lead, bands) in actual.iter().enumerate() {
            for (band, &value) in bands.iter().enumerate() {
                let correction: f64 = predictor.response_cross[lead]
                    .iter()
                    .zip(legacy_projection)
                    .map(|(row, x)| row[band] * x)
                    .sum();
                let expected = (current[band] as f64 + correction).max(0.) as f32;
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "lead={lead} band={band}"
                );
            }
        }
        // Cholesky solve of the batch Gram, independent of inverse rank-one updates.
        let mut lower = [[0.0; FEATURES]; FEATURES];
        for i in 0..FEATURES {
            for j in 0..=i {
                let gram = inputs.iter().map(|x| x[i] * x[j]).sum::<f64>()
                    + if i == j { RIDGE } else { 0.0 };
                let remainder = gram - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    remainder.sqrt()
                } else {
                    remainder / lower[j][j]
                };
            }
        }
        let mut intermediate = [0.0; FEATURES];
        for i in 0..FEATURES {
            intermediate[i] =
                (x[i] - (0..i).map(|j| lower[i][j] * intermediate[j]).sum::<f64>()) / lower[i][i];
        }
        let mut projected = [0.0; FEATURES];
        for i in (0..FEATURES).rev() {
            projected[i] = (intermediate[i]
                - (i + 1..FEATURES)
                    .map(|j| lower[j][i] * projected[j])
                    .sum::<f64>())
                / lower[i][i];
        }
        for lead in [0, 1, 10, 50, 100, 200] {
            let delay = 1 + lead * FORECAST_STRIDE;
            for (band, current) in current.iter().enumerate() {
                let correction: f64 = (0..energies.len() - delay)
                    .map(|i| {
                        let projection = inputs[i]
                            .iter()
                            .zip(projected)
                            .map(|(x, p)| x * p)
                            .sum::<f64>();
                        projection * (energies[i + delay][band] as f64 - energies[i][band] as f64)
                    })
                    .sum();
                let expected = (*current as f64 + correction).max(0.0) as f32;
                assert!(
                    (actual[lead][band] - expected).abs() < 2e-6,
                    "lead={lead} band={band}: {} != {expected}",
                    actual[lead][band]
                );
            }
        }
    }

    #[test]
    fn pending_targets_gaps_and_queries_do_not_fabricate_training() {
        let mut predictor = HistoryEnergyPrediction::new();
        let mut history = AuditoryHistory::new(HISTORY_AGES_SEC);
        history.advance(0.2, Some([0.2; 3]));
        let snapshot = history.snapshot();
        predictor.issue(1, [0.04; 3], &snapshot);
        let initial = predictor.forecast([0.04; 3], &snapshot);
        assert_eq!(initial, [[0.04; 3]; FORECAST_LEN]);
        assert_eq!(predictor.forecast([0.04; 3], &snapshot), initial);
        predictor.observe(1, [0.08; 3], |_| {});
        let learned = predictor.forecast([0.04; 3], &snapshot);
        assert!(learned[0][0] > initial[0][0]);
        assert_eq!(learned[1], initial[1]);
        predictor.clear_pending(false);
        predictor.observe(3, [100.0; 3], |_| {});
        assert_eq!(predictor.forecast([0.04; 3], &snapshot), learned);
        assert_eq!(initial, [[0.04; 3]; FORECAST_LEN]);
        predictor.clear_pending(true);
        assert_eq!(predictor.forecast([0.04; 3], &snapshot), initial);
    }
}

/// Pending targets penalize corrections but never supply invented outcome labels.
#[derive(Clone)]
pub(crate) struct HistoryEnergyPrediction {
    inverse: [[f64; FEATURES]; FEATURES],
    response_cross: Box<[[[f64; 3]; FEATURES]]>,
    pending: Box<[PendingHistory]>,
    comparisons: Box<[PendingComparison]>,
    comparison_square: [[f64; 3]; FORECAST_LEN],
    comparison_cross: [[f64; 3]; FORECAST_LEN],
    comparison_completed: [u64; 7],
    errors: PredictionErrorTotals,
}

impl HistoryEnergyPrediction {
    pub(crate) fn new() -> Self {
        let mut inverse = [[0.0; FEATURES]; FEATURES];
        for (i, row) in inverse.iter_mut().enumerate() {
            row[i] = 1.0 / RIDGE;
        }
        Self {
            inverse,
            response_cross: vec![[[0.0; 3]; FEATURES]; FORECAST_LEN].into_boxed_slice(),
            pending: vec![
                PendingHistory {
                    step: None,
                    features: [0.0; FEATURES],
                    energy: [0.0; 3]
                };
                ENERGY_PENDING_LEN
            ]
            .into_boxed_slice(),
            comparisons: vec![
                PendingComparison {
                    step: None,
                    requested_frame: None,
                    recurrence: [[0.0; 3]; FORECAST_LEN],
                    history: [[0.0; 3]; FORECAST_LEN],
                    mixed: [[0.0; 3]; FORECAST_LEN],
                    history_weight: [[0.0; 3]; 7],
                    completed_before_issue: [0; 7],
                };
                ENERGY_PENDING_LEN
            ]
            .into_boxed_slice(),
            comparison_square: [[0.0; 3]; FORECAST_LEN],
            comparison_cross: [[0.0; 3]; FORECAST_LEN],
            comparison_completed: [0; 7],
            errors: PredictionErrorTotals::default(),
        }
    }

    pub(crate) fn clear_pending(&mut self, rewind: bool) {
        for p in &mut self.pending {
            p.step = None;
        }
        for p in &mut self.comparisons {
            p.step = None;
        }
        if rewind {
            self.comparison_square.fill([0.0; 3]);
            self.comparison_cross.fill([0.0; 3]);
            self.comparison_completed.fill(0);
            self.errors = PredictionErrorTotals::default();
            self.response_cross.fill([[0.0; 3]; FEATURES]);
            for (i, row) in self.inverse.iter_mut().enumerate() {
                row.fill(0.0);
                row[i] = 1.0 / RIDGE;
            }
        }
    }

    pub(crate) fn observe(
        &mut self,
        step: u64,
        energy: [f32; 3],
        mut emit: impl FnMut(&PredictionMatch<'_>),
    ) {
        for (lead, cross) in self.response_cross.iter_mut().enumerate() {
            let Some(issued_step) = step.checked_sub((lead * FORECAST_STRIDE) as u64) else {
                continue;
            };
            let comparison = &self.comparisons[(issued_step % ENERGY_PENDING_LEN as u64) as usize];
            if comparison.step == Some(issued_step) {
                if let Some(index) = SCORE_LEADS.iter().position(|&h| h == lead) {
                    self.errors.completed[index] += 1;
                    self.comparison_completed[index] += 1;
                    for (band, observed) in energy.into_iter().enumerate() {
                        let observed = observed as f64;
                        self.errors.recurrence_squared_error[index][band] +=
                            (comparison.recurrence[lead][band] as f64 - observed).powi(2);
                        self.errors.history_squared_error[index][band] +=
                            (comparison.history[lead][band] as f64 - observed).powi(2);
                        self.errors.mixed_squared_error[index][band] +=
                            (comparison.mixed[lead][band] as f64 - observed).powi(2);
                    }
                    // Borrow the issuance context before the raw-history ring advances.
                    let context = &self.pending[(issued_step % ENERGY_PENDING_LEN as u64) as usize];
                    emit(&PredictionMatch {
                        issued_step,
                        target_step: step,
                        requested_frame: comparison.requested_frame,
                        completed_before_issue: comparison.completed_before_issue[index],
                        issued_features: (context.step == Some(issued_step))
                            .then_some(context.features.as_slice()),
                        history_weight: comparison.history_weight[index],
                        recurrence: comparison.recurrence[lead],
                        history: comparison.history[lead],
                        mixed: comparison.mixed[lead],
                        observed: energy,
                    });
                }
                for (band, observed) in energy.into_iter().enumerate() {
                    let baseline = comparison.recurrence[lead][band] as f64;
                    let delta = comparison.history[lead][band] as f64 - baseline;
                    self.comparison_square[lead][band] += delta * delta;
                    self.comparison_cross[lead][band] += delta * (observed as f64 - baseline);
                }
            }
            let p = &self.pending[(issued_step % ENERGY_PENDING_LEN as u64) as usize];
            if p.step != Some(issued_step) {
                continue;
            }
            let residual: [f64; 3] = std::array::from_fn(|b| energy[b] as f64 - p.energy[b] as f64);
            for (row, x) in cross.iter_mut().zip(p.features) {
                for (b, value) in row.iter_mut().enumerate() {
                    *value += x * residual[b];
                }
            }
        }
    }

    fn features(energy: [f32; 3], history: &AuditoryHistorySnapshot) -> [f64; FEATURES] {
        let mut x = [0.0; FEATURES];
        x[0] = 1.0;
        for (target, value) in x[1..4].iter_mut().zip(energy) {
            *target = value.sqrt() as f64;
        }
        for (target, value) in x[4..28]
            .iter_mut()
            .zip(history.known_band_rms_by_age.iter().flatten())
        {
            *target = *value as f64;
        }
        for (target, value) in x[28..36].iter_mut().zip(history.known_coverage_by_age) {
            *target = value as f64;
        }
        let mut index = 36;
        for pair in history.known_band_rms_by_age.windows(2) {
            let younger = pair[0].map(f64::from);
            let older = pair[1].map(f64::from);
            let power: f64 = younger.iter().chain(older.iter()).map(|v| v * v).sum();
            for a in 0..3 {
                for b in a + 1..3 {
                    // Directed age contrasts distinguish order; the floor is numerical.
                    x[index] = (older[a] * younger[b] - older[b] * younger[a]) / (power + 1e-12);
                    index += 1;
                }
            }
        }
        x
    }

    pub(crate) fn issue(&mut self, step: u64, energy: [f32; 3], history: &AuditoryHistorySnapshot) {
        let p = &mut self.pending[(step % ENERGY_PENDING_LEN as u64) as usize];
        debug_assert_ne!(p.step, Some(step));
        let x = Self::features(energy, history);
        let projected: [f64; FEATURES] =
            std::array::from_fn(|i| self.inverse[i].iter().zip(x).map(|(p, x)| p * x).sum());
        let denominator = 1.0 + x.iter().zip(projected).map(|(x, p)| x * p).sum::<f64>();
        for (i, row) in self.inverse.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value -= projected[i] * projected[j] / denominator;
            }
        }
        *p = PendingHistory {
            step: Some(step),
            features: x,
            energy,
        };
    }

    pub(crate) fn forecast(
        &self,
        energy: [f32; 3],
        history: &AuditoryHistorySnapshot,
    ) -> [[f32; 3]; FORECAST_LEN] {
        let x = Self::features(energy, history);
        let projected: [f64; FEATURES] =
            std::array::from_fn(|i| self.inverse[i].iter().zip(x).map(|(p, x)| p * x).sum());
        std::array::from_fn(|lead| {
            let mut correction = [0.; 3];
            for (row, &x) in self.response_cross[lead].iter().zip(&projected) {
                for band in 0..3 {
                    correction[band] += row[band] * x;
                }
            }
            std::array::from_fn(|band| (energy[band] as f64 + correction[band]).max(0.) as f32)
        })
    }

    pub(crate) fn preview_comparison(
        &self,
        recurrence: [[f32; 3]; FORECAST_LEN],
        history: [[f32; 3]; FORECAST_LEN],
    ) -> ([[f32; 3]; FORECAST_LEN], [[f32; 3]; FORECAST_LEN]) {
        let weights = std::array::from_fn(|lead| {
            std::array::from_fn(|band| {
                let square = self.comparison_square[lead][band];
                if square > 0.0 {
                    (self.comparison_cross[lead][band] / square).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                }
            })
        });
        let prediction = std::array::from_fn(|lead| {
            std::array::from_fn(|band| {
                let w = weights[lead][band];
                (1.0 - w) * recurrence[lead][band] + w * history[lead][band]
            })
        });
        (prediction, weights)
    }

    /// Publish once per observed context; only later actual outcomes fit the mixture.
    pub(crate) fn issue_comparison(
        &mut self,
        step: u64,
        recurrence: [[f32; 3]; FORECAST_LEN],
        history: [[f32; 3]; FORECAST_LEN],
        requested_frame: Option<u64>,
    ) -> ([[f32; 3]; FORECAST_LEN], [[f32; 3]; FORECAST_LEN]) {
        let (prediction, weights) = self.preview_comparison(recurrence, history);
        let pending = &mut self.comparisons[(step % ENERGY_PENDING_LEN as u64) as usize];
        if pending.step == Some(step) {
            debug_assert_eq!(pending.recurrence, recurrence);
            debug_assert_eq!(pending.history, history);
            debug_assert_eq!(pending.mixed, prediction);
        } else {
            self.errors.issued += 1;
            *pending = PendingComparison {
                step: Some(step),
                requested_frame,
                recurrence,
                history,
                mixed: prediction,
                history_weight: SCORE_LEADS.map(|lead| weights[lead]),
                completed_before_issue: self.comparison_completed,
            };
        }
        (prediction, weights)
    }

    /// Diagnostics consume completed errors, never prediction or learning state.
    pub(crate) fn take_errors(&mut self) -> Option<PredictionErrorTotals> {
        (self.errors.issued > 0 || self.errors.completed.iter().any(|&n| n > 0))
            .then(|| std::mem::take(&mut self.errors))
    }
}
