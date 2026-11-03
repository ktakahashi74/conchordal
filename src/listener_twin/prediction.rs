//! Presentation-only E0: empirical successors of mixed acoustic support.
//! Resource bounds and the categorical observation adapter are Author rules.
use crate::core::meter::MeterState;
use crate::viewdata::{ListenerCensored, ListenerForecast, ListenerOutcome, ListenerPrediction};

// Prototype resource capacity, not a memory half-life or a musical duration.
const CAPACITY: usize = 256;

#[derive(Clone, Copy, Default)]
struct Example {
    period: f32,
    tc: Option<f32>,
    recent: Option<usize>,
    outcome: usize,
    reference_audible: bool,
}

pub(super) struct Predictor {
    shapes: Vec<f32>,
    shape_sum: Vec<f32>,
    current: Vec<f32>,
    pending_shape: Vec<f32>,
    examples: [Example; CAPACITY],
    count: usize,
    next: usize,
    pending: Option<Example>,
    first: Option<(usize, u64)>,
    recent: Option<usize>,
    last_sample: Option<u64>,
    version: u64,
    pub(super) state: ListenerPrediction,
}

impl Default for Predictor {
    fn default() -> Self {
        Self {
            shapes: Vec::new(),
            shape_sum: Vec::new(),
            current: Vec::new(),
            pending_shape: Vec::new(),
            examples: [Example::default(); CAPACITY],
            count: 0,
            next: 0,
            pending: None,
            first: None,
            recent: None,
            last_sample: None,
            version: 0,
            state: ListenerPrediction::default(),
        }
    }
}

impl Predictor {
    pub(super) fn configure(&mut self, bins: usize) {
        self.shapes.resize(CAPACITY * bins, 0.0);
        self.shape_sum.resize(bins, 0.0);
        self.current.resize(bins, 0.0);
        self.pending_shape.resize(bins, 0.0);
        self.shape_sum.fill(0.0);
        self.count = 0;
        self.next = 0;
        self.gap();
        self.last_sample = None;
    }

    pub(super) fn gap(&mut self) {
        // Completed episodes remain retrievable; no pair spans missing evidence.
        self.pending = None;
        self.first = None;
        self.recent = None;
        self.state = ListenerPrediction::default();
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe(
        &mut self,
        end: u64,
        available: u64,
        hop: u64,
        fs: f32,
        mass: &[f32],
        peak: Option<usize>,
        tc: Option<Option<f64>>,
        meter: MeterState,
    ) -> ListenerPrediction {
        if self.last_sample.is_some_and(|last| end <= last) {
            return self.state;
        }
        if self.last_sample.is_some_and(|last| end != last + hop) {
            self.gap();
        }
        self.last_sample = Some(end);
        if mass.len() != self.current.len()
            || hop == 0
            || tc.is_some_and(|value| value.is_none_or(|x| !x.is_finite()))
        {
            self.gap();
            return self.state;
        }
        let total: f32 = mass.iter().sum();
        for (out, &x) in self.current.iter_mut().zip(mass) {
            *out = if peak.is_some() && total > 0.0 {
                (x / total).sqrt()
            } else {
                0.0
            };
        }
        if let Some(forecast) = self.state.forecast {
            // Delayed observations can invalidate the frozen reference before issuance.
            if end <= forecast.start_sample && peak != forecast.reference_bin {
                self.state.censored = Some(ListenerCensored {
                    forecast,
                    observed_sample: end,
                });
                self.state.forecast = None;
                self.pending = None;
                self.first = None;
                self.recent = None;
            }
        }
        if let Some(forecast) = self.state.forecast {
            // The interval begins at issuance availability, never in already heard audio.
            if end > forecast.start_sample {
                let outcome = match (forecast.reference_bin, peak) {
                    (a, b) if a == b => 0,
                    (None, Some(_)) => 1,
                    (Some(a), Some(b)) if b < a => 2,
                    (Some(_), Some(_)) => 3,
                    (Some(_), None) => 4,
                    (None, None) => 0,
                };
                if self.first.is_none() && outcome != 0 {
                    self.first = Some((outcome, end));
                }
            }
            if end >= forecast.end_sample {
                let (outcome, observed_sample) = self.first.unwrap_or((0, end));
                let p = forecast.q.map(|p| p[outcome]);
                self.state.completed = Some(ListenerOutcome {
                    forecast,
                    outcome,
                    observed_sample,
                    surprise_nats: p.map(|p| -p.ln()),
                    zero_likelihood: p == Some(0.0),
                });
                // Score the immutable forecast before inserting its observed successor.
                let mut example = self.pending.take().expect("issued window owns context");
                example.outcome = outcome;
                let bins = self.current.len();
                let slot = &mut self.shapes[self.next * bins..(self.next + 1) * bins];
                for ((old, sum), &new) in slot
                    .iter_mut()
                    .zip(&mut self.shape_sum)
                    .zip(&self.pending_shape)
                {
                    *sum += new - if self.count == CAPACITY { *old } else { 0.0 };
                    *old = new;
                }
                self.examples[self.next] = example;
                self.count = (self.count + 1).min(CAPACITY);
                self.next = (self.next + 1) % CAPACITY;
                self.recent = Some(outcome);
                self.state.forecast = None;
            }
        }
        if self.state.forecast.is_some() {
            return self.state;
        }
        let beat = meter.beat;
        if !(fs.is_finite()
            && fs > 0.0
            && beat.freq_hz.is_finite()
            && beat.freq_hz > 0.0
            && beat.confidence.is_finite()
            && beat.confidence > 0.0)
        {
            return self.state;
        }
        let period = 1.0 / beat.freq_hz;
        // Read out at complete analysis hops; the rounding is part of the target measure.
        let duration = ((period as f64 * fs as f64 / hop as f64).ceil() as u64).max(1) * hop;
        let start = available.max(end);
        let tc = tc.flatten().map(|v| v as f32);
        let context = Example {
            period,
            tc,
            recent: self.recent,
            reference_audible: peak.is_some(),
            outcome: 0,
        };
        self.version += 1;
        let mut forecast = ListenerForecast {
            version: self.version,
            start_sample: start,
            end_sample: start + duration,
            reference_sample: end,
            reference_bin: peak,
            horizon_sec: duration as f32 / fs,
            examples: self.count,
            ..ListenerForecast::default()
        };
        let examples = &self.examples[..self.count];
        let eligible = examples
            .iter()
            .filter(|e| e.reference_audible == context.reference_audible);
        let n = eligible.clone().count() as f32;
        if n > 0.0 {
            let norm = if context.reference_audible { n } else { 0.0 };
            let sum_sq = if context.reference_audible {
                self.shape_sum.iter().map(|x| x * x).sum::<f32>()
            } else {
                0.0
            };
            let shape_var = if n > 1.0 {
                (0.5 * (norm - sum_sq / n) / (n - 1.0)).max(0.0)
            } else {
                0.0
            };
            let period_mean = eligible.clone().map(|e| e.period).sum::<f32>() / n;
            let period_var = eligible
                .clone()
                .map(|e| (e.period - period_mean).powi(2))
                .sum::<f32>()
                / (n - 1.0).max(1.0);
            let tc_n = eligible.clone().filter(|e| e.tc.is_some()).count() as f32;
            let tc_mean = eligible.clone().filter_map(|e| e.tc).sum::<f32>() / tc_n.max(1.0);
            let tc_var = eligible
                .clone()
                .filter_map(|e| e.tc)
                .map(|x| (x - tc_mean).powi(2))
                .sum::<f32>()
                / (tc_n - 1.0).max(1.0);
            let mut weights = [0.0; 5];
            let mut marginal = [0.0; 5];
            for (i, e) in examples.iter().enumerate() {
                if e.reference_audible != context.reference_audible {
                    continue;
                }
                marginal[e.outcome] += 1.0;
                if e.recent != context.recent || e.tc.is_some() != tc.is_some() {
                    continue;
                }
                let shape = &self.shapes[i * self.current.len()..(i + 1) * self.current.len()];
                let distance = 0.5
                    * shape
                        .iter()
                        .zip(&self.current)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>();
                let mut z = standardized(distance, shape_var)
                    + standardized((e.period - period).powi(2), period_var);
                let mut dimensions = 2.0;
                if let (Some(a), Some(b)) = (e.tc, tc) {
                    z += standardized((a - b).powi(2), tc_var);
                    dimensions += 1.0;
                }
                // One joint feature distance, not a product of independent PCM evidence.
                weights[e.outcome] += (-0.5 * z / dimensions).exp() / n;
            }
            let marginal = marginal.map(|count| count / n);
            forecast.marginal = Some(marginal);
            let coverage: f32 = weights.iter().sum();
            forecast.coverage_score = coverage.clamp(0.0, 1.0);
            if coverage > 0.0 {
                let distribution = weights.map(|w| w / coverage);
                forecast.entropy_nats = Some(
                    -distribution
                        .iter()
                        .filter(|&&p| p > 0.0)
                        .map(|p| p * p.ln())
                        .sum::<f32>(),
                );
                forecast.p_known = Some(distribution);
            }
            // Freeze one whole point forecast, including the unknown-coverage backoff.
            let c = forecast.coverage_score;
            let known = forecast.p_known.unwrap_or([0.0; 5]);
            let q = std::array::from_fn(|i| c * known[i] + (1.0 - c) * marginal[i]);
            forecast.arrival_score = Some(q[1] + q[2] + q[3]);
            forecast.q = Some(q);
        }
        self.pending_shape.copy_from_slice(&self.current);
        self.pending = Some(context);
        self.first = None;
        self.state.forecast = Some(forecast);
        self.state
    }
}

fn standardized(distance_sq: f32, variance: f32) -> f32 {
    if distance_sq == 0.0 {
        0.0
    } else if variance > 0.0 {
        distance_sq / variance
    } else {
        f32::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> MeterState {
        let mut state = MeterState::default();
        state.beat.freq_hz = 2.0;
        state.beat.confidence = 0.8;
        state
    }

    fn step(p: &mut Predictor, t: u64, peak: Option<usize>) -> ListenerPrediction {
        let mut mass = [0.0; 3];
        if let Some(i) = peak {
            mass[i] = 1.0;
        }
        p.observe(t, t, 10, 100.0, &mass, peak, None, meter())
    }

    #[test]
    fn issued_distribution_is_frozen_and_scored_before_learning() {
        let mut p = Predictor::default();
        p.configure(3);
        for t in (10..=210).step_by(10) {
            step(&mut p, t, Some(1));
        }
        let issued = p.state.forecast.unwrap();
        assert_eq!(issued.p_known, Some([1.0, 0.0, 0.0, 0.0, 0.0]));
        for t in (220..260).step_by(10) {
            let state = step(&mut p, t, Some(2));
            assert_eq!(state.forecast, Some(issued));
        }
        let done = step(&mut p, 260, Some(2)).completed.unwrap();
        assert_eq!(done.forecast, issued);
        assert_eq!(done.outcome, 3);
        assert_eq!(done.observed_sample, 220);
        assert!(done.zero_likelihood);
        assert_eq!(done.surprise_nats, Some(f32::INFINITY));
        let n = p.count;
        let state = p.state;
        assert_eq!(step(&mut p, 260, None), state);
        assert_eq!(p.count, n);
    }

    #[test]
    fn gap_is_not_silence_and_unknown_is_not_zero_entropy() {
        let mut p = Predictor::default();
        p.configure(3);
        let cold = step(&mut p, 10, Some(1)).forecast.unwrap();
        assert_eq!(cold.entropy_nats, None);
        assert_eq!(cold.coverage_score, 0.0);
        step(&mut p, 20, None);
        // Missing observations censor the entire pending interval.
        step(&mut p, 50, None);
        assert_eq!(p.count, 0);
        assert_eq!(p.state.completed, None);
        for t in (60..=100).step_by(10) {
            step(&mut p, t, None);
        }
        assert_eq!(p.count, 1);
        assert_eq!(p.state.completed.unwrap().outcome, 0);
        for t in (110..=150).step_by(10) {
            step(&mut p, t, Some(1));
        }
        assert_eq!(p.state.completed.unwrap().outcome, 1);
        let state = p.observe(
            160,
            160,
            10,
            100.,
            &[0., 1., 0.],
            Some(1),
            Some(None),
            meter(),
        );
        assert_eq!(state, ListenerPrediction::default());
        assert_eq!(p.count, 2);
    }

    #[test]
    fn delayed_reference_changes_are_censored_without_learning() {
        for (reference, change_at, changed) in [
            (Some(1), 20, Some(2)),
            (None, 20, Some(1)),
            (Some(1), 30, Some(2)),
        ] {
            let mut p = Predictor::default();
            p.configure(3);
            let mass = if reference.is_some() {
                [0., 1., 0.]
            } else {
                [0.; 3]
            };
            let issued = p
                .observe(10, 30, 10, 100., &mass, reference, None, meter())
                .forecast
                .unwrap();
            assert_eq!((issued.start_sample, issued.end_sample), (30, 80));
            for t in (20..=80).step_by(10) {
                let peak = if t >= change_at { changed } else { reference };
                let mut mass = [0.; 3];
                if let Some(i) = peak {
                    mass[i] = 1.;
                }
                p.observe(
                    t,
                    30.max(t),
                    10,
                    100.,
                    &mass,
                    peak,
                    None,
                    MeterState::default(),
                );
            }
            assert_eq!(p.state.censored.unwrap().forecast, issued);
            assert_eq!(p.state.censored.unwrap().observed_sample, change_at);
            assert!(p.state.completed.is_none());
            assert!(p.state.forecast.is_none());
            assert_eq!(p.count, 0);
        }
    }

    #[test]
    fn confirmed_delayed_reference_scores_only_after_start() {
        let mut p = Predictor::default();
        p.configure(3);
        let issued = p
            .observe(10, 30, 10, 100., &[0., 1., 0.], Some(1), None, meter())
            .forecast
            .unwrap();
        for t in (20..=80).step_by(10) {
            step(&mut p, t, Some(if t <= 30 { 1 } else { 2 }));
        }
        let done = p.state.completed.unwrap();
        assert_eq!(done.forecast, issued);
        assert_eq!((done.outcome, done.observed_sample), (3, 40));
        assert!(p.state.censored.is_none());
        assert_eq!(p.count, 1);
    }

    #[test]
    fn reference_support_backoff_and_scoring_survive_completion_and_eviction() {
        let mut p = Predictor::default();
        p.configure(3);
        let mut seed = 37u64;
        let mut partial = 0;
        let mut fallback = 0;
        let mut scored = [0; 2];
        for t in (10..=20000).step_by(10) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let choice = (seed >> 32) % 4;
            let state = step(&mut p, t, (choice < 3).then_some(choice as usize));
            let f = state.forecast.unwrap();
            if let (Some(q), Some(b)) = (f.q, f.marginal) {
                let c = f.coverage_score;
                let known = f.p_known.unwrap_or([0.; 5]);
                for i in 0..5 {
                    assert_eq!(q[i], c * known[i] + (1. - c) * b[i]);
                }
                assert_eq!(f.arrival_score, Some(q[1] + q[2] + q[3]));
                partial += usize::from(c > 0. && c < 1.);
                fallback += usize::from(f.p_known.is_none());
                for distribution in [Some(q), Some(b), f.p_known].into_iter().flatten() {
                    assert!((distribution.iter().sum::<f32>() - 1.).abs() < 1e-5);
                    if f.reference_bin.is_some() {
                        assert_eq!(distribution[1], 0.);
                    } else {
                        assert_eq!(&distribution[2..], &[0.; 3]);
                    }
                }
            }
            if let Some(done) = state.completed {
                if let Some(q) = done.forecast.q {
                    assert_eq!(done.surprise_nats, Some(-q[done.outcome].ln()));
                    assert_eq!(done.zero_likelihood, q[done.outcome] == 0.);
                    scored[usize::from(done.forecast.reference_bin.is_some())] += 1;
                } else {
                    assert_eq!(done.surprise_nats, None);
                }
            }
        }
        assert!(partial > 0 && fallback > 0);
        assert!(scored.iter().all(|&n| n > 0));
        assert_eq!(p.count, CAPACITY);
    }

    #[test]
    fn all_unknown_and_unconfirmed_meter_remain_unavailable() {
        let mut p = Predictor::default();
        p.configure(3);
        let s = p.observe(
            10,
            10,
            10,
            100.,
            &[0., 1., 0.],
            Some(1),
            None,
            MeterState::default(),
        );
        assert_eq!(s.forecast, None);
        for t in (20..=220).step_by(10) {
            step(&mut p, t, Some(1));
        }
        for t in (230..=270).step_by(10) {
            step(&mut p, t, Some(2));
        }
        let f = p.state.forecast.unwrap();
        assert_eq!(f.coverage_score, 0.0);
        assert_eq!(f.p_known, None);
        assert_eq!(f.q, f.marginal);
        assert!(f.arrival_score.is_some());
        assert_eq!(f.entropy_nats, None);
    }

    #[test]
    fn tc_is_a_condition_not_a_probability_and_owners_are_independent() {
        let mut a = Predictor::default();
        a.configure(3);
        let mut b = Predictor::default();
        b.configure(3);
        for t in (10..=210).step_by(10) {
            let state = a.observe(
                t,
                t,
                10,
                100.,
                &[0., 1., 0.],
                Some(1),
                Some(Some(-0.7)),
                meter(),
            );
            step(&mut b, t, Some(2));
            if t == 210 {
                assert_eq!(state.forecast.unwrap().p_known, Some([1., 0., 0., 0., 0.]));
            }
        }
        assert_ne!(a.shapes, b.shapes);
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn prediction_retrieval_completion_eviction_and_gap_do_not_allocate() {
        let mut p = Predictor::default();
        p.configure(690);
        let mass = [1.0; 690];
        crate::runtime_profile::begin_allocations();
        for t in (10..=16000).step_by(10) {
            p.observe(t, t, 10, 100., &mass, Some(12), Some(Some(0.4)), meter());
            if t == 15000 {
                p.gap();
            }
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
        assert_eq!(p.count, CAPACITY);
        println!("GUI1_PREDICTION_ALLOC {counts:?}");
    }
}
