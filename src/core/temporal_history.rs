//! Continuous auditory history; age precision is not a cognitive retention lifetime.

pub(crate) const HISTORY_AGES_SEC: [f32; 8] = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
const ORDER: usize = 12;
const STAGES: usize = ORDER + 1;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct AuditoryHistorySnapshot {
    pub(crate) ages_sec: [f32; 8],
    pub(crate) post_order: usize,
    pub(crate) known_band_rms_by_age: [[f32; 3]; 8],
    pub(crate) known_coverage_by_age: [f32; 8],
}

/// Equivalent cascade for K(d) = s^(k+1) d^k exp(-s d) / k!, s = k / age.
#[derive(Clone)]
pub(crate) struct AuditoryHistory {
    ages_sec: [f32; 8],
    state: [[[f32; 4]; STAGES]; 8],
    transition: [[f32; STAGES]; 8],
    input_gain: [[f32; STAGES]; 8],
    previous_dt: f64,
}

impl AuditoryHistory {
    pub(crate) fn new(ages_sec: [f32; 8]) -> Self {
        assert!(ages_sec.iter().all(|a| a.is_finite() && *a > 0.0));
        assert!(ages_sec.windows(2).all(|a| a[0] < a[1]));
        Self {
            ages_sec,
            state: [[[0.0; 4]; STAGES]; 8],
            transition: [[0.0; STAGES]; 8],
            input_gain: [[0.0; STAGES]; 8],
            previous_dt: 0.0,
        }
    }

    /// Unknown input contributes no invented signal and no observed kernel mass.
    pub(crate) fn advance(&mut self, dt_sec: f64, band_rms: Option<[f32; 3]>) {
        assert!(dt_sec.is_finite() && dt_sec > 0.0);
        assert!(band_rms.is_none_or(|v| v.iter().all(|x| x.is_finite() && *x >= 0.0)));
        if dt_sec != self.previous_dt {
            for (age, age_sec) in self.ages_sec.iter().enumerate() {
                let x = ORDER as f64 * dt_sec / *age_sec as f64;
                let mut term = (-x).exp();
                let mut sum = 0.0;
                for stage in 0..STAGES {
                    self.transition[age][stage] = term as f32;
                    sum += term;
                    // Positive Poisson tails avoid subtracting almost one from one.
                    let gain = if x < STAGES as f64 {
                        let mut tail = term * x / (stage + 1) as f64;
                        let mut total = tail;
                        let mut degree = stage + 2;
                        while tail > f64::EPSILON * total {
                            tail *= x / degree as f64;
                            total += tail;
                            degree += 1;
                        }
                        total
                    } else {
                        (1.0 - sum).max(0.0)
                    };
                    self.input_gain[age][stage] = gain as f32;
                    term = if x.is_finite() {
                        term * x / (stage + 1) as f64
                    } else {
                        0.0
                    };
                }
            }
            self.previous_dt = dt_sec;
        }
        let input = band_rms.map_or([0.0; 4], |[a, b, c]| [a, b, c, 1.0]);
        for age in 0..self.ages_sec.len() {
            // Descending updates retain the preceding endpoint in every dependency.
            for stage in (0..STAGES).rev() {
                for (channel, value) in input.iter().enumerate() {
                    let mut next = self.input_gain[age][stage] * value;
                    for earlier in 0..=stage {
                        next += self.transition[age][stage - earlier]
                            * self.state[age][earlier][channel];
                    }
                    self.state[age][stage][channel] = next;
                }
            }
        }
    }

    pub(crate) fn snapshot(&self) -> AuditoryHistorySnapshot {
        AuditoryHistorySnapshot {
            ages_sec: self.ages_sec,
            post_order: ORDER,
            known_band_rms_by_age: self.state.map(|s| [s[ORDER][0], s[ORDER][1], s[ORDER][2]]),
            known_coverage_by_age: self.state.map(|s| s[ORDER][3].clamp(0.0, 1.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_input_partition_and_joint_time_scaling_preserve_history() {
        let mut one = AuditoryHistory::new(HISTORY_AGES_SEC);
        one.advance(1.07, Some([0.1, 1.0, 0.0]));
        for scale in [0.01, 1.0, 100.0] {
            let mut split = AuditoryHistory::new(HISTORY_AGES_SEC.map(|a| a * scale as f32));
            for dt in [0.2, 0.31, 0.0001, 0.5599] {
                split.advance(dt * scale, Some([0.1, 1.0, 0.0]));
            }
            for (a, b) in one
                .state
                .iter()
                .flatten()
                .flatten()
                .zip(split.state.iter().flatten().flatten())
            {
                assert!((a - b).abs() < 2e-6, "{a} != {b}");
            }
        }
    }

    #[test]
    fn order_survives_equal_present_and_totals_and_missing_input_is_not_silence() {
        let mut ab = AuditoryHistory::new(HISTORY_AGES_SEC);
        let mut ba = ab.clone();
        ab.advance(0.2, Some([1.0, 0.0, 0.0]));
        ab.advance(0.2, Some([0.0, 1.0, 0.0]));
        ba.advance(0.2, Some([0.0, 1.0, 0.0]));
        ba.advance(0.2, Some([1.0, 0.0, 0.0]));
        for h in [&mut ab, &mut ba] {
            h.advance(0.1, Some([0.0; 3]));
        }
        let a = ab.snapshot();
        let b = ba.snapshot();
        assert!(a.known_band_rms_by_age[0][1] > a.known_band_rms_by_age[0][0]);
        assert!(a.known_band_rms_by_age[2][0] > a.known_band_rms_by_age[2][1]);
        for (a, b) in a.known_band_rms_by_age.iter().zip(b.known_band_rms_by_age) {
            assert_eq!(a[0], b[1]);
            assert_eq!(a[1], b[0]);
        }
        let mut missing = ab.clone();
        ab.advance(0.7, Some([0.0; 3]));
        missing.advance(0.7, None);
        let silent = ab.snapshot();
        let unknown = missing.snapshot();
        assert_eq!(silent.known_band_rms_by_age, unknown.known_band_rms_by_age);
        assert!(silent.known_coverage_by_age[2] > unknown.known_coverage_by_age[2]);
        assert!(unknown.known_band_rms_by_age[2][0] > 0.0);
    }

    #[test]
    fn step_response_matches_independent_gamma_integral() {
        let mut history = AuditoryHistory::new(HISTORY_AGES_SEC);
        history.advance(1.2, Some([1.0, 0.25, 0.0]));
        let snapshot = history.snapshot();
        for (i, age) in HISTORY_AGES_SEC.iter().enumerate() {
            // Simpson quadrature of the normalized kernel, independent of transitions.
            let rate = ORDER as f64 / *age as f64;
            let dx = 1.2 / 20_000.0;
            let factorial: f64 = (1..=ORDER).map(|n| n as f64).product();
            let mut integral = 0.0;
            for n in 0..=20_000 {
                let d = n as f64 * dx;
                let kernel =
                    rate.powi(STAGES as i32) * d.powi(ORDER as i32) * (-rate * d).exp() / factorial;
                let weight = if n == 0 || n == 20_000 {
                    1.0
                } else if n % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                integral += weight * kernel * dx / 3.0;
            }
            assert!((snapshot.known_band_rms_by_age[i][0] as f64 - integral).abs() < 2e-7);
            assert_eq!(
                snapshot.known_band_rms_by_age[i][0],
                snapshot.known_coverage_by_age[i]
            );
            assert_eq!(
                snapshot.known_band_rms_by_age[i][1],
                snapshot.known_band_rms_by_age[i][0] * 0.25
            );
        }
        history.advance(1e300, None);
        assert_eq!(history.snapshot().known_band_rms_by_age, [[0.0; 3]; 8]);
    }
}
