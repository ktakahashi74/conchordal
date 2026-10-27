//! von Hippel (2000), Eq. 1 and text-consistent stationary AR(1) variance.
//! Prefix estimation, execution support and generation coupling are author rules.
use super::own_sequence::{ExecutionKind, ExecutionRecord};
use std::collections::HashSet;

// Outward rounding certifies support; it is not a correlation clip or variance floor.
#[derive(Clone, Copy, Debug, Default)]
struct Bounded {
    value: f64,
    lo: f64,
    hi: f64,
}

impl Bounded {
    fn exact(value: f64) -> Self {
        Self {
            value,
            lo: value,
            hi: value,
        }
    }
    fn add(self, b: Self) -> Self {
        Self {
            value: self.value + b.value,
            lo: (self.lo + b.lo).next_down(),
            hi: (self.hi + b.hi).next_up(),
        }
    }
    fn sub(self, b: Self) -> Self {
        Self {
            value: self.value - b.value,
            lo: (self.lo - b.hi).next_down(),
            hi: (self.hi - b.lo).next_up(),
        }
    }
    fn mul(self, b: Self) -> Self {
        let products = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self {
            value: self.value * b.value,
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    fn div_positive(self, b: Self) -> Self {
        self.mul(Self {
            value: 1.0 / b.value,
            lo: (1.0 / b.hi).next_down(),
            hi: (1.0 / b.lo).next_up(),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Prefix {
    pub(crate) n: usize,
    origin: f64,
    last: f64,
    mean: Bounded,
    m2: Bounded,
    a: Bounded,
    b: Bounded,
    aa: Bounded,
    bb: Bounded,
    ab: Bounded,
}

impl Prefix {
    pub(crate) fn push(&mut self, pitch: f64) {
        if self.n == 0 {
            self.origin = pitch;
        }
        let p = Bounded::exact(pitch).sub(Bounded::exact(self.origin));
        self.n += 1;
        let delta = p.sub(self.mean);
        self.mean = self
            .mean
            .add(delta.div_positive(Bounded::exact(self.n as f64)));
        self.m2 = self.m2.add(delta.mul(p.sub(self.mean)));
        if self.n > 1 {
            let previous = Bounded::exact(self.last).sub(Bounded::exact(self.origin));
            let da = previous.sub(self.a);
            let db = p.sub(self.b);
            let count = Bounded::exact((self.n - 1) as f64);
            self.a = self.a.add(da.div_positive(count));
            self.b = self.b.add(db.div_positive(count));
            self.aa = self.aa.add(da.mul(previous.sub(self.a)));
            self.bb = self.bb.add(db.mul(p.sub(self.b)));
            self.ab = self.ab.add(da.mul(p.sub(self.b)));
        }
        self.last = pitch;
    }

    pub(crate) fn distribution(&self) -> Option<Distribution> {
        if self.n < 4 || self.m2.lo <= 0.0 || self.aa.lo <= 0.0 || self.bb.lo <= 0.0 {
            return None;
        }
        let product = self.aa.mul(self.bb);
        let residual = product.sub(self.ab.mul(self.ab));
        if !residual.lo.is_finite() || residual.lo <= 0.0 {
            return None;
        }
        let r = self.ab.value / product.value.sqrt();
        Distribution::new(
            self.origin + self.mean.value,
            self.m2.value / (self.n - 1) as f64,
            r,
            self.last,
        )
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Distribution {
    pub(crate) m: f64,
    pub(crate) s2: f64,
    pub(crate) r: f64,
    pub(crate) mu: f64,
    pub(crate) variance: f64,
}

impl Distribution {
    pub(crate) fn new(m: f64, s2: f64, r: f64, last: f64) -> Option<Self> {
        let mu = m + r * (last - m);
        let variance = s2 * (1.0 - r * r);
        (m.is_finite()
            && s2.is_finite()
            && s2 > 0.0
            && r.abs() < 1.0
            && mu.is_finite()
            && variance.is_finite()
            && variance > 0.0)
            .then_some(Self {
                m,
                s2,
                r,
                mu,
                variance,
            })
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub(crate) distribution: Distribution,
    pub(crate) bounds: (f32, f32),
    pub(crate) weight: f64,
    pub(crate) epoch: u64,
    pub(crate) n: usize,
}

impl Snapshot {
    pub(crate) fn contribution(&self, pitch: f32, baseline: f32) -> Option<f64> {
        let y = 12.0 * pitch as f64;
        let y0 = 12.0 * baseline as f64;
        let d = self.distribution;
        let term = self.weight * ((y0 - d.mu).powi(2) - (y - d.mu).powi(2))
            / (2.0 * d.variance * std::f64::consts::LN_2);
        term.is_finite().then_some(term)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Decision {
    pub(crate) at: u64,
    pub(crate) snapshot: Option<Snapshot>,
    pub(crate) reason: Option<&'static str>,
    pub(crate) old_terms_replaced: bool,
    pub(crate) selected_log2: f32,
    pub(crate) without_melodic_log2: f32,
    pub(crate) changed: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct MelodicContinuity {
    pub(crate) prefix: Prefix,
    pub(crate) epoch: u64,
    pub(crate) reason: Option<&'static str>,
    pub(crate) supported: bool,
    pub(crate) resource_stopped: bool,
    pub(crate) resident: bool,
    pub(crate) trace_enabled: bool,
    pub(crate) executions: Vec<ExecutionRecord>,
    pub(crate) decisions: Vec<Decision>,
    seen: HashSet<(u64, u64)>,
    active: HashSet<u64>,
    previous_sample: Option<u64>,
    bounds: (f32, f32),
    fs: f64,
    weight: f64,
    decision_count: usize,
}

impl MelodicContinuity {
    pub(crate) fn new(fs: f32, weight: f64, bounds: (f32, f32)) -> Self {
        let valid = fs.is_finite()
            && fs > 0.0
            && bounds.0.is_finite()
            && bounds.1.is_finite()
            && bounds.0 >= 20.0f32.log2()
            && bounds.1 <= 20000.0f32.log2()
            && bounds.0 <= bounds.1;
        if !valid {
            tracing::warn!(
                "melodic_continuity: unsupported frequency range or sample rate; unknown until reconfigured"
            );
        }
        Self {
            prefix: Prefix::default(),
            epoch: 0,
            reason: (!valid).then_some("unsupported_range"),
            supported: valid,
            resource_stopped: false,
            resident: true,
            trace_enabled: false,
            executions: Vec::new(),
            decisions: Vec::new(),
            seen: HashSet::with_capacity(4801),
            active: HashSet::with_capacity(4801),
            previous_sample: None,
            bounds,
            fs: fs as f64,
            weight,
            decision_count: 0,
        }
    }
    pub(crate) fn stop(&mut self, reason: &'static str) {
        self.resource_stopped = true;
        self.reason = Some(reason);
    }
    pub(crate) fn reject_residency(&mut self) {
        self.stop("resident_state_resource_limit");
        self.resident = false;
        self.seen.shrink_to_fit();
        self.active.shrink_to_fit();
    }
    pub(crate) fn close_epoch(&mut self, reason: &'static str) {
        self.epoch += 1;
        self.prefix = Prefix::default();
        self.previous_sample = None;
        self.reason = Some(reason);
    }
    pub(crate) fn reconfigure(&mut self, supported: bool, bounds: (f32, f32)) {
        self.close_epoch("reconfigured");
        self.bounds = bounds;
        self.supported = supported
            && bounds.0.is_finite()
            && bounds.1.is_finite()
            && bounds.0 >= 20.0f32.log2()
            && bounds.1 <= 20000.0f32.log2()
            && bounds.0 <= bounds.1;
    }
    pub(crate) fn observe(&mut self, record: &ExecutionRecord) {
        if self.trace_enabled {
            self.executions.push(record.clone());
        }
        if record.sample as f64 > 600.0 * self.fs {
            self.stop("duration_resource_limit");
        }
        if self.resource_stopped {
            return;
        }
        if record.available_at != record.sample {
            self.close_epoch("late_or_missing_execution");
            return;
        }
        match record.kind {
            ExecutionKind::Off => {
                self.active.remove(&record.tone_id);
            }
            ExecutionKind::PitchUpdate | ExecutionKind::Missing => {
                self.close_epoch("pitch_update_or_missing")
            }
            ExecutionKind::On => {
                let identity = (record.tone_id, record.sample);
                if self.seen.contains(&identity) {
                    return;
                }
                if self.seen.len() == 4801 {
                    self.stop("event_resource_limit");
                    return;
                }
                self.seen.insert(identity);
                self.active.insert(record.tone_id);
                if !self.supported {
                    return;
                }
                if self.active.len() > 1 {
                    self.close_epoch("unreleased_overlap");
                    return;
                }
                let Some(f) = record.frequency_hz.filter(|f| f.is_finite() && *f > 0.0) else {
                    self.close_epoch("unknown_frequency");
                    return;
                };
                if f.log2() < self.bounds.0 || f.log2() > self.bounds.1 {
                    self.close_epoch("execution_outside_range");
                    return;
                }
                if self.previous_sample.is_some_and(|at| record.sample <= at) {
                    self.close_epoch("nonincreasing_onset");
                    return;
                }
                self.prefix.push(12.0 * (f as f64).log2());
                self.previous_sample = Some(record.sample);
                self.reason = None;
            }
        }
    }
    pub(crate) fn predict(&mut self, at: u64) -> Option<Snapshot> {
        if self.resource_stopped || !self.supported {
            return None;
        }
        if self.decision_count == 38400 || at as f64 > 600.0 * self.fs {
            self.stop("decision_or_duration_resource_limit");
            return None;
        }
        self.decision_count += 1;
        let distribution = self.prefix.distribution();
        self.reason = distribution.is_none().then_some("unknown_statistics");
        Some(Snapshot {
            distribution: distribution?,
            bounds: self.bounds,
            weight: self.weight,
            epoch: self.epoch,
            n: self.prefix.n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn melodic_source_equations_and_independent_normal_bins() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/melodic_continuity/reference.json"
        ))
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 18);
        for c in cases {
            let f = |key: &str| c[key].as_f64().unwrap();
            let d = Distribution::new(f("m"), f("s2"), f("r"), f("last")).unwrap();
            assert!((d.mu - f("mu")).abs() <= 2.0 * f64::EPSILON * d.mu.abs());
            assert!((d.variance - f("variance")).abs() <= 16.0 * f64::EPSILON * f("s2"));
            // erfc keeps upper tails; this evaluator is not on the generation path.
            let scale = (2.0 * d.variance).sqrt();
            let q =
                0.5 * (libm::erfc((f("a") - d.mu) / scale) - libm::erfc((f("b") - d.mu) / scale));
            // Decimal boundaries are rounded to binary64 before standardization.
            let bound = 64.0 * f64::EPSILON * (1.0 + d.mu.abs() / d.variance.sqrt());
            assert!(
                (q / f("q") - 1.0).abs() <= bound,
                "{c} q={q} relative={}",
                q / f("q") - 1.0
            );
        }
    }

    pub(super) fn prefix(pitches: &[f64]) -> Prefix {
        let mut p = Prefix::default();
        for &x in pitches {
            p.push(x);
        }
        p
    }

    #[test]
    fn melodic_prefix_batch_reference_and_degenerate_support() {
        for xs in [
            vec![60., 60., 62., 60.],
            vec![64., 60., 62., 60.],
            (0..4801)
                .map(|i| 60.0 + ((i * 17) % 31) as f64 / 7.0)
                .collect(),
        ] {
            let p = prefix(&xs);
            let n = xs.len() as f64;
            let m = xs.iter().sum::<f64>() / n;
            let s2 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
            let a = xs[..xs.len() - 1].iter().sum::<f64>() / (n - 1.0);
            let b = xs[1..].iter().sum::<f64>() / (n - 1.0);
            let aa = xs[..xs.len() - 1]
                .iter()
                .map(|x| (x - a).powi(2))
                .sum::<f64>();
            let bb = xs[1..].iter().map(|x| (x - b).powi(2)).sum::<f64>();
            let ab = xs.windows(2).map(|w| (w[0] - a) * (w[1] - b)).sum::<f64>();
            let d = p.distribution().unwrap();
            let reference =
                Distribution::new(m, s2, ab / (aa * bb).sqrt(), *xs.last().unwrap()).unwrap();
            let budget = 64.0 * n * f64::EPSILON;
            assert!((d.mu - reference.mu).abs() < budget * reference.mu.abs());
            assert!((d.variance - reference.variance).abs() < budget * reference.variance);
        }
        for xs in [
            vec![60.],
            vec![60., 61., 63.],
            vec![60., 60., 60., 60.],
            vec![60., 61., 62., 63., 64.],
            vec![60., 62., 60., 62., 60., 62.],
        ] {
            assert!(prefix(&xs).distribution().is_none(), "{xs:?}");
        }
        let mut p = prefix(&[60., 61., 62.]);
        assert!(p.distribution().is_none());
        p.push(60.);
        assert!(p.distribution().unwrap().r < 0.0);
        // A resolvably near-perfect correlation is not clipped or floored.
        let d = prefix(&[60., 61., 62., 63., 64.01]).distribution().unwrap();
        assert!(d.r > 0.999 && d.r < 1.0);
    }

    fn record(kind: ExecutionKind, tone: u64, sample: u64, hz: f32) -> ExecutionRecord {
        ExecutionRecord {
            voice_id: 1,
            generation: 0,
            tone_id: tone,
            kind,
            sample,
            available_at: sample,
            execution_sequence: sample,
            frequency_hz: Some(hz),
            strength: None,
        }
    }

    #[test]
    fn melodic_support_loss_is_distinct_from_numerical_unknown() {
        for kind in [
            ExecutionKind::PitchUpdate,
            ExecutionKind::Missing,
            ExecutionKind::On,
        ] {
            for invalid in 0..4 {
                let mut me = MelodicContinuity::new(48000., 1., (5., 13.));
                for p in [96., 96., 98., 96.] {
                    me.prefix.push(p);
                }
                assert!(me.predict(0).is_some());
                me.previous_sample = Some(10);
                let mut e = record(kind, 7, 11, 256.);
                match invalid {
                    0 => e.available_at = 12,
                    1 => e.frequency_hz = Some(f32::NAN),
                    2 => e.frequency_hz = Some(1.0),
                    _ => {
                        e.sample = 10;
                        e.available_at = 10;
                    }
                }
                me.observe(&e);
                assert_eq!(me.epoch, 1);
                assert_eq!(me.prefix.n, 0);
                assert!(!me.resource_stopped);
            }
        }
        let mut me = MelodicContinuity::new(48000., 1., (5., 13.));
        for p in [96., 97., 98.] {
            me.prefix.push(p);
        }
        assert!(me.predict(0).is_none());
        assert_eq!((me.epoch, me.prefix.n), (0, 3));
        me.prefix.push(96.);
        assert!(me.predict(1).is_some());
    }

    #[test]
    fn melodic_execution_support_and_resource_boundaries() {
        let mut me = MelodicContinuity::new(48000., 1., (5., 13.));
        for (i, hz) in [256., 256., 288., 256.].into_iter().enumerate() {
            if i > 0 {
                me.observe(&record(ExecutionKind::Off, i as u64 - 1, 2 * i as u64, 0.));
            }
            let on = record(ExecutionKind::On, i as u64, 2 * i as u64 + 1, hz);
            me.observe(&on);
            me.observe(&on);
        }
        assert_eq!(me.prefix.n, 4);
        assert!(me.predict(10).is_some());
        let epoch = me.epoch;
        me.observe(&record(ExecutionKind::On, 99, 10, 256.));
        assert_eq!(me.reason, Some("unreleased_overlap"));
        assert_eq!(me.epoch, epoch + 1);
        me.reconfigure(false, (5., 13.));
        assert!(me.predict(11).is_none());
        me.reconfigure(true, (5., 13.));
        assert_eq!(me.prefix.n, 0);
        me.seen = (0..4801).map(|i| (i, i)).collect();
        me.observe(&record(ExecutionKind::On, 5000, 5000, 256.));
        assert!(me.resource_stopped);
        assert_eq!(me.seen.len(), 4801);
        me.reconfigure(true, (5., 13.));
        assert!(me.resource_stopped);
        let mut me = MelodicContinuity::new(48000., 1., (5., 13.));
        assert!(me.predict(600 * 48000 + 1).is_none());
        assert!(me.resource_stopped);
        let mut me = MelodicContinuity::new(48000., 1., (5., 13.));
        me.decision_count = 38400;
        assert!(me.predict(10).is_none());
        assert_eq!(me.reason, Some("decision_or_duration_resource_limit"));
    }
}
