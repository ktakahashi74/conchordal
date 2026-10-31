//! Matching-horizon energy regression from observed auditory history.

use crate::core::compensated::Sum;
use std::cell::{Cell, RefCell};

use crate::core::temporal_expectation::{ENERGY_PENDING_LEN, FORECAST_LEN, FORECAST_STRIDE};
use crate::core::temporal_history::AuditoryHistorySnapshot;

#[cfg(test)]
#[path = "prediction_reference.rs"]
pub(crate) mod reference;

const FEATURES: usize = 57;
const RIDGE: f64 = 0.01;
pub(crate) const SCORE_LEADS: [usize; 7] = [0, 5, 10, 25, 50, 100, 200];

const REFACTOR_INTERVAL: u64 = FEATURES.div_ceil(6) as u64;
fn next_observer() -> u64 {
    NEXT_OBSERVER.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    })
}

pub(crate) fn freeze_status() -> (u64, u64) {
    FREEZE_STATUS.with(Cell::get)
}

#[derive(Clone, PartialEq, Debug)]
struct Gram {
    sum: [[Sum; FEATURES]; FEATURES],
    lower: [[f64; FEATURES]; FEATURES],
}

impl Gram {
    fn new() -> Self {
        let mut result = Self {
            sum: [[Sum::ZERO; FEATURES]; FEATURES],
            lower: [[0.0; FEATURES]; FEATURES],
        };
        for i in 0..FEATURES {
            result.sum[i][i].hi = RIDGE;
            result.lower[i][i] = RIDGE.sqrt();
        }
        result
    }

    fn update(&mut self, mut x: [f64; FEATURES], refactor: bool) -> bool {
        for i in 0..FEATURES {
            for j in 0..=i {
                self.sum[i][j].add(x[i] * x[j]);
                if !self.sum[i][j].total().is_finite() {
                    return false;
                }
            }
        }
        // The lower triangle alone owns the symmetric Gram.
        for k in 0..FEATURES {
            let diagonal = self.lower[k][k];
            let r = diagonal.hypot(x[k]);
            if !(r.is_finite() && r > 0.0 && diagonal > 0.0) {
                return false;
            }
            let c = r / diagonal;
            let s = x[k] / diagonal;
            self.lower[k][k] = r;
            for j in k + 1..FEATURES {
                self.lower[j][k] = (self.lower[j][k] + s * x[j]) / c;
                x[j] = c * x[j] - s * self.lower[j][k];
                if !self.lower[j][k].is_finite() || !x[j].is_finite() {
                    return false;
                }
            }
        }
        if refactor {
            for i in 0..FEATURES {
                for j in 0..=i {
                    let mut value = self.sum[i][j].total();
                    for k in 0..j {
                        value -= self.lower[i][k] * self.lower[j][k];
                    }
                    if i == j {
                        if !(value > 0.0 && value.is_finite()) {
                            return false;
                        }
                        self.lower[i][j] = value.sqrt();
                    } else {
                        self.lower[i][j] = value / self.lower[j][j];
                        if !self.lower[i][j].is_finite() {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn solve(&self, mut x: [f64; FEATURES]) -> [f64; FEATURES] {
        for i in 0..FEATURES {
            for j in 0..i {
                x[i] -= self.lower[i][j] * x[j];
            }
            x[i] /= self.lower[i][i];
        }
        for i in (0..FEATURES).rev() {
            for j in i + 1..FEATURES {
                x[i] -= self.lower[j][i] * x[j];
            }
            x[i] /= self.lower[i][i];
        }
        x
    }
}

thread_local! {
    static NEXT_OBSERVER: Cell<u64> = const { Cell::new(0) };
    // Each runtime owns its estimator thread and its cumulative failure telemetry.
    static FREEZE_STATUS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    // One transactional buffer per estimator thread, initialized only on first use.
    static GRAM_SCRATCH: RefCell<Gram> = RefCell::new(Gram::new());
}

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
    pub(crate) frozen_at_issue: Option<u64>,
    pub(crate) freeze_count: u64,
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

/// Pending targets penalize corrections but never supply invented outcome labels.
pub(crate) struct HistoryEnergyPrediction {
    gram: Gram,
    issues: u64,
    phase: u64,
    frozen_at_issue: Option<u64>,
    response_cross: Box<[[[Sum; 3]; FEATURES]]>,
    pending: Box<[PendingHistory]>,
    comparisons: Vec<PendingComparison>,
    comparison_slots: [Option<usize>; ENERGY_PENDING_LEN],
    comparison_square: [[Sum; 3]; FORECAST_LEN],
    comparison_cross: [[Sum; 3]; FORECAST_LEN],
    comparison_completed: [u64; 7],
    errors: PredictionErrorTotals,
}

impl Clone for HistoryEnergyPrediction {
    fn clone(&self) -> Self {
        // Reserve every ring slot, but copy only records that have been issued.
        let mut comparisons = Vec::with_capacity(ENERGY_PENDING_LEN);
        comparisons.extend_from_slice(&self.comparisons);
        let result = Self {
            gram: self.gram.clone(),
            issues: self.issues,
            phase: next_observer(),
            frozen_at_issue: self.frozen_at_issue,
            response_cross: self.response_cross.clone(),
            pending: self.pending.clone(),
            comparisons,
            comparison_slots: self.comparison_slots,
            comparison_square: self.comparison_square,
            comparison_cross: self.comparison_cross,
            comparison_completed: self.comparison_completed,
            errors: self.errors,
        };
        #[cfg(test)]
        reference::clone_history(self.phase, result.phase);
        result
    }
}

impl HistoryEnergyPrediction {
    #[cfg(test)]
    pub(crate) fn align_refactor_phase(&mut self, other: &Self) {
        self.phase = other.phase;
    }

    pub(crate) fn new() -> Self {
        Self {
            gram: Gram::new(),
            issues: 0,
            phase: next_observer(),
            frozen_at_issue: None,
            response_cross: vec![[[Sum::ZERO; 3]; FEATURES]; FORECAST_LEN].into_boxed_slice(),
            pending: vec![
                PendingHistory {
                    step: None,
                    features: [0.0; FEATURES],
                    energy: [0.0; 3]
                };
                ENERGY_PENDING_LEN
            ]
            .into_boxed_slice(),
            comparisons: Vec::with_capacity(ENERGY_PENDING_LEN),
            comparison_slots: [None; ENERGY_PENDING_LEN],
            comparison_square: [[Sum::ZERO; 3]; FORECAST_LEN],
            comparison_cross: [[Sum::ZERO; 3]; FORECAST_LEN],
            comparison_completed: [0; 7],
            errors: PredictionErrorTotals::default(),
        }
    }

    pub(crate) fn clear_pending(&mut self, rewind: bool) {
        for p in &mut self.pending {
            p.step = None;
        }
        self.comparisons.clear();
        self.comparison_slots.fill(None);
        if rewind {
            #[cfg(test)]
            reference::reset(self.phase);
            self.comparison_square.fill([Sum::ZERO; 3]);
            self.comparison_cross.fill([Sum::ZERO; 3]);
            self.comparison_completed.fill(0);
            self.errors = PredictionErrorTotals::default();
            self.response_cross.fill([[Sum::ZERO; 3]; FEATURES]);
            self.gram = Gram::new();
            self.issues = 0;
            self.frozen_at_issue = None;
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
            let comparison = self.comparison_slots
                [(issued_step % ENERGY_PENDING_LEN as u64) as usize]
                .map(|slot| &self.comparisons[slot])
                .filter(|p| p.step == Some(issued_step));
            if let Some(comparison) = comparison {
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
                    #[cfg(test)]
                    reference::comparison(
                        self.phase,
                        lead,
                        band,
                        delta * delta,
                        delta * (observed as f64 - baseline),
                    );
                    self.comparison_square[lead][band].add(delta * delta);
                    self.comparison_cross[lead][band].add(delta * (observed as f64 - baseline));
                }
            }
            let p = &self.pending[(issued_step % ENERGY_PENDING_LEN as u64) as usize];
            if p.step != Some(issued_step) || self.frozen_at_issue.is_some() {
                continue;
            }
            let residual: [f64; 3] = std::array::from_fn(|b| energy[b] as f64 - p.energy[b] as f64);
            #[cfg(test)]
            reference::response(self.phase, lead, &p.features, residual);
            for (row, x) in cross.iter_mut().zip(p.features) {
                for (b, value) in row.iter_mut().enumerate() {
                    value.add(x * residual[b]);
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
        if self.frozen_at_issue.is_some() {
            return;
        }
        let p = &mut self.pending[(step % ENERGY_PENDING_LEN as u64) as usize];
        debug_assert_ne!(p.step, Some(step));
        let x = Self::features(energy, history);
        let next = self.issues + 1;
        let success = GRAM_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clone_from(&self.gram);
            if scratch.update(
                x,
                next % REFACTOR_INTERVAL == self.phase % REFACTOR_INTERVAL,
            ) {
                std::mem::swap(&mut self.gram, &mut scratch);
                true
            } else {
                false
            }
        });
        if !success {
            self.frozen_at_issue = Some(next);
            self.errors.frozen_at_issue = Some(next);
            self.errors.freeze_count += 1;
            FREEZE_STATUS.with(|status| status.set((status.get().0 + 1, next)));
            return;
        }
        self.issues = next;
        #[cfg(test)]
        reference::issue(self.phase, &x);
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
        let projected = self.gram.solve(x);
        let output = std::array::from_fn(|lead| {
            let mut correction = [0.; 3];
            for (row, &x) in self.response_cross[lead].iter().zip(&projected) {
                for band in 0..3 {
                    correction[band] += row[band].total() * x;
                }
            }
            std::array::from_fn(|band| (energy[band] as f64 + correction[band]).max(0.) as f32)
        });
        #[cfg(test)]
        reference::forecast(self.phase, &x, energy, &output);
        output
    }

    pub(crate) fn preview_comparison(
        &self,
        recurrence: [[f32; 3]; FORECAST_LEN],
        history: [[f32; 3]; FORECAST_LEN],
    ) -> ([[f32; 3]; FORECAST_LEN], [[f32; 3]; FORECAST_LEN]) {
        let weights = std::array::from_fn(|lead| {
            std::array::from_fn(|band| {
                let square = self.comparison_square[lead][band].total();
                if square > 0.0 {
                    (self.comparison_cross[lead][band].total() / square).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                }
            })
        });
        #[cfg(test)]
        reference::weights(self.phase, &weights);
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
        let slot = &mut self.comparison_slots[(step % ENERGY_PENDING_LEN as u64) as usize];
        if let Some(pending) = slot
            .map(|index| &self.comparisons[index])
            .filter(|p| p.step == Some(step))
        {
            debug_assert_eq!(pending.recurrence, recurrence);
            debug_assert_eq!(pending.history, history);
            debug_assert_eq!(pending.mixed, prediction);
        } else {
            self.errors.issued += 1;
            let pending = PendingComparison {
                step: Some(step),
                requested_frame,
                recurrence,
                history,
                mixed: prediction,
                history_weight: SCORE_LEADS.map(|lead| weights[lead]),
                completed_before_issue: self.comparison_completed,
            };
            if let Some(index) = *slot {
                self.comparisons[index] = pending;
            } else {
                *slot = Some(self.comparisons.len());
                self.comparisons.push(pending);
            }
        }
        (prediction, weights)
    }

    /// Diagnostics consume completed errors, never prediction or learning state.
    pub(crate) fn take_errors(&mut self) -> Option<PredictionErrorTotals> {
        (self.errors.freeze_count > 0
            || self.errors.issued > 0
            || self.errors.completed.iter().any(|&n| n > 0))
        .then(|| {
            let mut errors = std::mem::take(&mut self.errors);
            errors.frozen_at_issue = self.frozen_at_issue;
            errors
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::temporal_history::{AuditoryHistory, HISTORY_AGES_SEC};

    #[test]
    fn partial_update_failure_freezes_committed_pair_and_rewind_recovers() {
        let history = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        let mut predictor = HistoryEnergyPrediction::new();
        predictor.issue(1, [0.1; 3], &history);
        predictor.observe(1, [0.2; 3], |_| {});
        let gram = predictor.gram.clone();
        let response = predictor.response_cross.clone();
        let prediction = predictor.forecast([0.1; 3], &history);
        // G[0][0] is already changed in scratch when the next row fails.
        predictor.issue(2, [f32::NAN, 0.0, 0.0], &history);
        assert_eq!(predictor.frozen_at_issue, Some(2));
        assert_eq!(predictor.gram, gram);
        assert_eq!(predictor.response_cross, response);
        predictor.observe(2, [10.0; 3], |_| {});
        predictor.issue(3, [0.2; 3], &history);
        assert_eq!(predictor.gram, gram);
        assert_eq!(predictor.response_cross, response);
        assert_eq!(predictor.forecast([0.1; 3], &history), prediction);
        let status = predictor.take_errors().unwrap();
        assert_eq!(status.frozen_at_issue, Some(2));
        assert_eq!(status.freeze_count, 1);
        predictor.clear_pending(false);
        assert_eq!(predictor.frozen_at_issue, Some(2));
        predictor.clear_pending(true);
        assert_eq!(predictor.frozen_at_issue, None);
        assert_eq!(predictor.gram, Gram::new());
        predictor.issue(1, [0.1; 3], &history);
        assert_eq!(predictor.issues, 1);
    }

    #[test]
    fn cloned_factor_waits_for_its_own_phase() {
        let history = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        let mut parent = HistoryEnergyPrediction::new();
        parent.phase = 0;
        for step in 1..=9 {
            parent.issue(step, [0.1; 3], &history);
        }
        let mut child = parent.clone();
        child.phase = 9;
        assert_eq!(child.gram, parent.gram);
        assert_eq!(child.response_cross, parent.response_cross);
        let mut expected = parent.gram.clone();
        for step in 10..=19 {
            let x = HistoryEnergyPrediction::features([0.2; 3], &history);
            assert!(expected.update(x, step == 19));
            child.issue(step, [0.2; 3], &history);
            assert_eq!(child.gram, expected);
        }
    }

    #[test]
    fn numerical_reference_on_existing_history_fixture() {
        reference::start_selected(2, vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);
        let mut predictor = HistoryEnergyPrediction::new();
        let mut history = AuditoryHistory::new(HISTORY_AGES_SEC);
        for step in 0..1000 {
            let phase = step as f32 * 0.031;
            let energy = [
                0.05 * (1.0 + phase.sin()),
                0.03 * (1.0 + (phase * 1.7).cos()),
                if step % 71 < 20 { 0.02 } else { 0.0 },
            ];
            predictor.observe(step, energy, |_| {});
            history.advance(0.01, Some(energy.map(f32::sqrt)));
            predictor.issue(step + 1, energy, &history.snapshot());
            let candidate = predictor.forecast(energy, &history.snapshot());
            predictor.issue_comparison(step + 1, [[0.01; 3]; FORECAST_LEN], candidate, None);
            if step == 408 {
                predictor = predictor.clone();
            }
        }
        let counts = reference::counts();
        reference::stop();
        println!("numerical history fixture {counts:?}");
        assert!(counts.passed[0] > 0 && counts.passed[1] > 0);
        assert_eq!(counts.failed, [0; 3]);
        assert_eq!(counts.inconclusive, [0; 3]);
    }

    #[test]
    #[ignore = "release-only estimator cost estimate, not realtime acceptance"]
    fn estimator_release_cost() {
        let mut predictor = HistoryEnergyPrediction::new();
        let history = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        predictor.issue(1, [0.1; 3], &history);
        let start = std::time::Instant::now();
        for step in 2..=10001 {
            let energy = [0.01 * (1.0 + (step as f32 * 0.031).sin()); 3];
            predictor.observe(step, energy, |_| {});
            predictor.issue(step + 1, energy, &history);
            let candidate = predictor.forecast(energy, &history);
            std::hint::black_box(predictor.issue_comparison(
                step + 1,
                [[0.01; 3]; FORECAST_LEN],
                candidate,
                None,
            ));
        }
        println!(
            "estimator cost: {} us/observer issue+observe+forecast+comparison (10000 iterations)",
            start.elapsed().as_secs_f64() * 100.0
        );
    }

    #[test]
    fn sparse_comparisons_match_dense_ring_after_clone_wrap_and_clear() {
        let mut sparse = HistoryEnergyPrediction::new();
        let mut dense = sparse.clone();
        dense.comparisons.resize(
            ENERGY_PENDING_LEN,
            PendingComparison {
                step: None,
                requested_frame: None,
                recurrence: [[0.0; 3]; FORECAST_LEN],
                history: [[0.0; 3]; FORECAST_LEN],
                mixed: [[0.0; 3]; FORECAST_LEN],
                history_weight: [[0.0; 3]; 7],
                completed_before_issue: [0; 7],
            },
        );
        dense.comparison_slots = std::array::from_fn(Some);
        let mut history = AuditoryHistory::new(HISTORY_AGES_SEC);
        for step in 0..1000 {
            let energy = [0.01 * (1.0 + (step as f32 * 0.031).sin()); 3];
            let mut expected = Vec::new();
            dense.observe(step, energy, |m| {
                expected.push(serde_json::to_value(m).unwrap());
            });
            let mut actual = Vec::new();
            sparse.observe(step, energy, |m| {
                actual.push(serde_json::to_value(m).unwrap());
            });
            assert_eq!(actual, expected);
            history.advance(0.01, Some(energy.map(f32::sqrt)));
            let snapshot = history.snapshot();
            sparse.issue(step + 1, energy, &snapshot);
            dense.issue(step + 1, energy, &snapshot);
            let recurrence = [[step as f32 * 0.001; 3]; FORECAST_LEN];
            let candidate = sparse.forecast(energy, &snapshot);
            assert_eq!(candidate, dense.forecast(energy, &snapshot));
            if step % 7 == 0 || step % 19 == 0 {
                for _ in 0..2 {
                    assert_eq!(
                        sparse.issue_comparison(step + 1, recurrence, candidate, Some(step)),
                        dense.issue_comparison(step + 1, recurrence, candidate, Some(step))
                    );
                }
            }
            if step == 17 || step == 405 {
                sparse = sparse.clone();
            }
            if step == 510 || step == 610 {
                sparse.clear_pending(step == 610);
                dense.clear_pending(step == 610);
            }
            assert_eq!(
                serde_json::to_value(sparse.take_errors()).unwrap(),
                serde_json::to_value(dense.take_errors()).unwrap()
            );
        }
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn estimator_updates_and_refactors_reuse_thread_scratch() {
        let mut first = HistoryEnergyPrediction::new();
        let mut second = first.clone();
        let history = AuditoryHistory::new(HISTORY_AGES_SEC).snapshot();
        first.issue(1, [0.1; 3], &history);
        second.issue(1, [0.1; 3], &history);
        crate::runtime_profile::begin_allocations();
        for step in 2..32 {
            for predictor in [&mut first, &mut second] {
                predictor.observe(step, [0.2; 3], |_| {});
                predictor.issue(step + 1, [0.1; 3], &history);
                std::hint::black_box(predictor.forecast([0.1; 3], &history));
            }
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn cloned_comparison_ring_never_allocates_on_issue_or_reset() {
        let mut original = HistoryEnergyPrediction::new();
        original.issue_comparison(1, [[0.0; 3]; FORECAST_LEN], [[1.0; 3]; FORECAST_LEN], None);
        for mut predictor in [HistoryEnergyPrediction::new().clone(), original.clone()] {
            crate::runtime_profile::begin_allocations();
            for rewind in [false, true] {
                for step in 1..=2 * ENERGY_PENDING_LEN as u64 {
                    predictor.issue_comparison(
                        step,
                        [[0.0; 3]; FORECAST_LEN],
                        [[1.0; 3]; FORECAST_LEN],
                        None,
                    );
                }
                predictor.clear_pending(rewind);
            }
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((counts.count, counts.bytes), (0, 0));
        }
    }

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
                serde_json::to_value(predictor.errors).unwrap(),
                serde_json::to_value(untouched.errors).unwrap()
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
