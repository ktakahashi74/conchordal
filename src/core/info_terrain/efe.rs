//! Registration A-3: passive, cell-local Rao-Blackwellised particle inference.

use super::{Log2Space, Moments};
use crate::core::erb::{ErbCell, hz_to_erb, partition_erb_cells};
use serde::{Deserialize, Serialize};

const ETA: f64 = 0.99;
const PARTICLES: usize = 64;
const V0: f64 = 0.01;
const S0: f64 = 0.05;

fn s_floor() -> f64 {
    (std::f64::consts::LN_10 / 20.0).powi(2)
}

#[derive(Clone, Debug, PartialEq)]
struct Rng {
    state: u64,
    spare: Option<f64>,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        // 52-bit midpoints are exactly representable and strictly inside (0,1).
        ((self.next() >> 12) as f64 + 0.5) / (1_u64 << 52) as f64
    }
    fn normal(&mut self) -> f64 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = (9.0 * d).sqrt().recip();
        loop {
            let x = self.normal();
            let base = 1.0 + c * x;
            if base <= 0.0 {
                continue;
            }
            let v = base.powi(3);
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
    fn beta_transition(&mut self) -> f64 {
        let a = self.gamma(0.5 * ETA / (1.0 - ETA));
        let b = self.gamma(0.5);
        a / (a + b)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Particle {
    v: f64,
    s: f64,
    m: f64,
    w: f64,
    log_b: f64,
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn valid(p: Particle) -> bool {
    positive(p.v) && positive(p.s) && positive(p.w) && positive(p.w + p.v + p.s) && p.m.is_finite()
}
fn kalman(p: &mut Particle, o: f64) -> f64 {
    let prior_w = p.w + p.v;
    let prior_m = p.m;
    let alpha = prior_w / (prior_w + p.s);
    p.m += alpha * (o - p.m);
    p.w = (1.0 - alpha) * prior_w;
    // Posterior || prior, conditional on this particle's hypothesis.
    0.5 * ((prior_w / p.w).ln() + (p.w + (p.m - prior_m).powi(2)) / prior_w - 1.0)
}
fn normalize(particles: &mut [Particle]) -> Option<usize> {
    if particles
        .iter()
        .any(|p| p.log_b.is_nan() || p.log_b == f64::INFINITY)
    {
        return None;
    }
    let max = particles
        .iter()
        .map(|p| p.log_b)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    let mut vanished = 0;
    for p in particles.iter_mut() {
        if p.log_b.is_finite() && p.log_b <= max - 700.0 {
            p.log_b = f64::NEG_INFINITY;
            vanished += 1;
        }
    }
    let log_sum = particles
        .iter()
        .map(|p| (p.log_b - max).exp())
        .sum::<f64>()
        .ln();
    for p in particles {
        p.log_b = (p.log_b - max) - log_sum;
    }
    Some(vanished)
}
fn information_gain(particles: &[Particle], prior: &[f64]) -> Option<f64> {
    let mut n = 0.0;
    for (p, &pred) in particles.iter().zip(prior) {
        if p.log_b == f64::NEG_INFINITY {
            continue;
        }
        if !pred.is_finite() {
            return None;
        }
        n += p.log_b.exp() * (p.log_b - pred);
    }
    n.is_finite().then_some(n)
}
fn ess(particles: &[Particle]) -> f64 {
    particles
        .iter()
        .map(|p| (2.0 * p.log_b).exp())
        .sum::<f64>()
        .recip()
}
fn needs_resampling(ess: f64, len: usize) -> bool {
    ess / (len as f64) < 0.5
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Metrics {
    a: f64,
    n: f64,
    sx: f64,
    v: f64,
    s: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CellHop {
    metrics: Option<Metrics>,
    reason: Option<String>,
    energy: f64,
    ess: Option<f64>,
    max_weight: Option<f64>,
    min_pred_weight: Option<f64>,
    min_positive_pred_weight: Option<f64>,
    vanished: usize,
    zero_particles: usize,
    floor_fraction: Option<f64>,
    resampled: bool,
}
impl CellHop {
    fn missing(reason: &'static str) -> Self {
        Self {
            metrics: None,
            reason: Some(reason.into()),
            energy: 0.0,
            ess: None,
            max_weight: None,
            min_pred_weight: None,
            min_positive_pred_weight: None,
            vanished: 0,
            zero_particles: 0,
            floor_fraction: None,
            resampled: false,
        }
    }
}

struct Filter {
    particles: Vec<Particle>,
    scratch: Vec<Particle>,
    predicted: Vec<f64>,
    state_kl: Vec<f64>,
    transitions: Vec<[Rng; 2]>,
    resampling: Rng,
    initial: (f64, f64),
    initialized: bool,
}
impl Filter {
    fn new(seed: u64, stream: u64, len: usize, initial: (f64, f64)) -> Self {
        assert!(len > 0 && positive(initial.0) && positive(initial.1));
        // Separate stream, particle and role domains; RNGs stay at their slots on resampling.
        let mut seeds = Rng::new(seed ^ stream.wrapping_mul(0xd2b74407b1ce6e93));
        let p = Particle {
            v: initial.0,
            s: initial.1,
            m: 0.0,
            w: 1.0,
            log_b: -(len as f64).ln(),
        };
        let resampling = Rng::new(seeds.next());
        let transitions = (0..len)
            .map(|_| [Rng::new(seeds.next()), Rng::new(seeds.next())])
            .collect();
        Self {
            particles: vec![p; len],
            scratch: vec![p; len],
            predicted: vec![0.0; len],
            state_kl: vec![0.0; len],
            transitions,
            resampling,
            initial,
            initialized: false,
        }
    }
    fn initialize(&mut self, o: f64) {
        let log_b = -(self.particles.len() as f64).ln();
        self.particles.fill(Particle {
            v: self.initial.0,
            s: self.initial.1,
            m: o,
            w: 1.0,
            log_b,
        });
        self.initialized = true;
    }
    fn observe(&mut self, o: f64) -> CellHop {
        if !o.is_finite() {
            self.initialized = false;
            return CellHop::missing("numerical");
        }
        if !self.initialized {
            self.initialize(o);
            return CellHop::missing("first");
        }
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            if !valid(*p) {
                self.initialize(o);
                return CellHop::missing("numerical");
            }
            self.predicted[i] = p.log_b;
            // z' = z * beta / eta, hence v' = v * eta / beta (and likewise s).
            p.v *= ETA / self.transitions[i][0].beta_transition();
            p.s *= ETA / self.transitions[i][1].beta_transition();
            if !valid(*p) {
                self.initialize(o);
                return CellHop::missing("numerical");
            }
            let variance = p.w + p.v + p.s;
            p.log_b +=
                -0.5 * ((std::f64::consts::TAU * variance).ln() + (o - p.m).powi(2) / variance);
            self.state_kl[i] = kalman(p, o);
            if !valid(*p) {
                self.initialize(o);
                return CellHop::missing("numerical");
            }
        }
        let Some(vanished) = normalize(&mut self.particles) else {
            self.initialize(o);
            return CellHop::missing("numerical");
        };
        let Some(n) = information_gain(&self.particles, &self.predicted) else {
            return CellHop::missing("numerical");
        };
        let mut metrics = Metrics {
            a: 0.5 * (std::f64::consts::TAU * std::f64::consts::E).ln(),
            n,
            sx: 0.0,
            v: 0.0,
            s: 0.0,
        };
        for (p, kl) in self.particles.iter().zip(&self.state_kl) {
            if p.log_b == f64::NEG_INFINITY {
                continue;
            }
            let b = p.log_b.exp();
            metrics.a += 0.5 * b * p.s.max(s_floor()).ln();
            metrics.sx += b * kl;
            metrics.v += b * p.v;
            metrics.s += b * p.s;
        }
        if ![metrics.a, metrics.n, metrics.sx, metrics.v, metrics.s]
            .iter()
            .all(|x| x.is_finite())
        {
            return CellHop::missing("numerical");
        }
        let effective = ess(&self.particles);
        let resampled = needs_resampling(effective, self.particles.len());
        let row = CellHop {
            metrics: Some(metrics),
            reason: None,
            energy: 0.0,
            ess: Some(effective),
            max_weight: Some(
                self.particles
                    .iter()
                    .map(|p| p.log_b.exp())
                    .fold(0.0, f64::max),
            ),
            min_pred_weight: Some(
                self.predicted
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
                    .exp(),
            ),
            min_positive_pred_weight: self
                .predicted
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .min_by(f64::total_cmp)
                .map(f64::exp),
            vanished,
            zero_particles: self
                .particles
                .iter()
                .filter(|p| p.log_b == f64::NEG_INFINITY)
                .count(),
            floor_fraction: Some(
                self.particles.iter().filter(|p| p.s < s_floor()).count() as f64
                    / self.particles.len() as f64,
            ),
            resampled,
        };
        if resampled {
            self.resample();
        }
        row
    }
    fn resample(&mut self) {
        let len = self.particles.len();
        let start = self.resampling.uniform() / len as f64;
        let mut index = 0;
        let mut cumulative = self.particles[0].log_b.exp();
        for (i, target) in self.scratch.iter_mut().enumerate() {
            let u = start + i as f64 / len as f64;
            while index + 1 < len
                && (cumulative <= u || self.particles[index].log_b == f64::NEG_INFINITY)
            {
                index += 1;
                cumulative += self.particles[index].log_b.exp();
            }
            *target = self.particles[index];
            target.log_b = -(len as f64).ln();
        }
        std::mem::swap(&mut self.particles, &mut self.scratch);
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Hop {
    frame_id: u64,
    // Order: ambiguity, particle information gain, conditional state KL, reference mismatch, composite.
    values: [Option<f64>; 5],
    valid_cells: usize,
    sounding_cells: usize,
    cells: Vec<CellHop>,
}
fn aggregate(cells: &[CellHop]) -> (Option<[f64; 3]>, usize) {
    let mut sum = [0.0; 3];
    let mut energy = 0.0;
    let mut count = 0;
    for cell in cells {
        if let Some(m) = &cell.metrics {
            if ![m.a, m.n, m.sx].iter().all(|x| x.is_finite()) || !positive(cell.energy) {
                continue;
            }
            for (s, x) in sum.iter_mut().zip([m.a, m.n, m.sx]) {
                *s += cell.energy * x;
            }
            energy += cell.energy;
            count += 1;
        }
    }
    ((energy > 0.0).then(|| sum.map(|x| x / energy)), count)
}

struct Terrain {
    space: Log2Space,
    cells: Vec<ErbCell>,
    filters: Vec<Filter>,
    stream: u64,
}
impl Terrain {
    fn new(space: Log2Space, stream: u64) -> Self {
        let erb_scan: Vec<_> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
        let cells = partition_erb_cells(
            &space,
            &erb_scan,
            (
                space.fmin.min(space.centers_hz[0]),
                space.fmax.max(*space.centers_hz.last().unwrap()),
            ),
        );
        let filters = (0..cells.len())
            .map(|c| Filter::new(7001 + c as u64, stream, PARTICLES, (V0, S0)))
            .collect();
        Self {
            space,
            cells,
            filters,
            stream,
        }
    }
    fn reset(&mut self) {
        self.filters = (0..self.cells.len())
            .map(|c| Filter::new(7001 + c as u64, self.stream, PARTICLES, (V0, S0)))
            .collect();
    }
    fn observe(
        &mut self,
        frame_id: u64,
        power_scan: &[f32],
        field_level_scan: &[f32],
        maximum: Option<f64>,
        warming: bool,
    ) -> Hop {
        self.space
            .assert_scan_len_named(power_scan, "efe_power_scan");
        self.space
            .assert_scan_len_named(field_level_scan, "efe_field_level_scan");
        let unavailable = if warming {
            Some("refill")
        } else if maximum.is_none() {
            Some("startup")
        } else if maximum == Some(0.0) {
            Some("global_silence")
        } else {
            None
        };
        let epsilon = (1e-4 * maximum.unwrap_or(0.0)).powi(2);
        let mut rows = Vec::with_capacity(self.cells.len());
        let mut fields = Vec::with_capacity(self.cells.len());
        let mut sounding_energy = 0.0;
        let mut sounding_cells = 0;
        for (cell, filter) in self.cells.iter().zip(&mut self.filters) {
            let energy = cell
                .bins
                .clone()
                .map(|b| f64::from(power_scan[b]))
                .sum::<f64>();
            let field = cell
                .bins
                .clone()
                .map(|b| f64::from(field_level_scan[b]))
                .sum::<f64>()
                / cell.bins.len() as f64;
            fields.push(4.0 * field);
            let mut row = if let Some(reason) = unavailable {
                CellHop::missing(reason)
            } else if !energy.is_finite() || energy < 0.0 || !positive(epsilon) {
                CellHop::missing("numerical")
            } else if energy < epsilon {
                CellHop::missing("silence")
            } else {
                sounding_cells += 1;
                sounding_energy += energy;
                filter.observe(0.5 * (energy + epsilon).ln())
            };
            row.energy = if energy.is_finite() { energy } else { 0.0 };
            rows.push(row);
        }
        let (means, valid_cells) = aggregate(&rows);
        let mut values = [None; 5];
        if let Some([a, n, sx]) = means {
            values = [Some(a), Some(n), Some(sx), None, Some(n - a)];
        }
        if sounding_energy > 0.0 && fields.iter().all(|x| x.is_finite()) {
            let max = fields.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_z = max + fields.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
            let r = rows
                .iter()
                .zip(&fields)
                .filter(|(c, _)| c.energy >= epsilon && positive(c.energy))
                .map(|(c, f)| {
                    let p = c.energy / sounding_energy;
                    p * (p.ln() - f + log_z)
                })
                .sum::<f64>();
            values[3] = r.is_finite().then_some(r);
        }
        Hop {
            frame_id,
            values,
            valid_cells,
            sounding_cells,
            cells: rows,
        }
    }
}

pub(super) fn effective_settings(
    config: &crate::config::AppConfig,
    space: &Log2Space,
    fs: u32,
    hop: usize,
    nfft: usize,
) -> serde_json::Value {
    serde_json::json!({"sample_rate":fs,"hop":hop,"nfft":nfft,
        "analysis":config.analysis,"incoherent_power":config.psychoacoustics.use_incoherent_power,
        "log2space":{"fmin":space.fmin,"fmax":space.fmax,"bins_per_oct":space.bins_per_oct,"n_bins":space.n_bins()},
        "info_terrain":config.info_terrain, "terrain":config.psychoacoustics,
        "initial_habituation_state":"all zero"})
}

#[derive(Serialize)]
pub(super) struct Summary {
    effective_analysis_settings: serde_json::Value,
    means: [Option<f64>; 5],
    defined_hops: [u64; 5],
    missing_hops: [u64; 5],
    valid_cell_hops: u64,
    sounding_cell_hops: u64,
    numerical_missing_cell_hops: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<Vec<Hop>>,
}

pub(super) struct Diagnostic {
    settings: serde_json::Value,
    terrain: Terrain,
    moments: [Moments; 5],
    hops: u64,
    valid: u64,
    sounding: u64,
    numerical: u64,
    trace: Option<Vec<Hop>>,
    habituation: Option<crate::core::habituation::HabituationField>,
}
impl Diagnostic {
    pub(super) fn new(space: Log2Space, stream: u64, settings: serde_json::Value) -> Self {
        Self {
            settings,
            terrain: Terrain::new(space, stream),
            moments: [Moments::default(); 5],
            hops: 0,
            valid: 0,
            sounding: 0,
            numerical: 0,
            trace: Some(Vec::new()),
            habituation: None,
        }
    }
    pub(super) fn reset(&mut self) {
        self.terrain.reset();
        self.habituation = None;
    }
    pub(super) fn observe(
        &mut self,
        frame_id: u64,
        frame: &crate::core::landscape::Landscape,
        params: &crate::core::landscape::LandscapeParams,
        maximum: Option<f64>,
        warming: bool,
        dt: f32,
    ) {
        // A private reference terrain; nothing derived here reaches runtime consumers.
        let mut reference = frame.clone();
        reference.recompute_consonance(params);
        let hab = self.habituation.get_or_insert_with(|| {
            crate::core::habituation::HabituationField::new(
                &params.habituation,
                params.consonance_representation.theta,
                reference.space.n_bins(),
            )
        });
        if !warming {
            crate::core::habituation::drive_and_apply_habituation(&mut reference, hab, params, dt);
        }
        let row = self.terrain.observe(
            frame_id,
            &reference.nsgt_power,
            &reference.consonance_field_level_eff,
            maximum,
            warming,
        );
        self.hops += 1;
        self.valid += row.valid_cells as u64;
        self.sounding += row.sounding_cells as u64;
        self.numerical += row
            .cells
            .iter()
            .filter(|c| c.reason.as_deref() == Some("numerical"))
            .count() as u64;
        for (moment, value) in self.moments.iter_mut().zip(row.values) {
            if let Some(x) = value {
                moment.add(x);
            }
        }
        if let Some(trace) = &mut self.trace {
            trace.push(row);
        }
    }
    pub(super) fn finish(&mut self) -> Summary {
        let result = Summary {
            effective_analysis_settings: self.settings.clone(),
            means: self.moments.map(Moments::mean),
            defined_hops: self.moments.map(|m| m.count),
            missing_hops: self.moments.map(|m| self.hops - m.count),
            valid_cell_hops: self.valid,
            sounding_cell_hops: self.sounding,
            numerical_missing_cell_hops: self.numerical,
            trace: self.trace.as_mut().map(std::mem::take),
        };
        self.moments.fill(Moments::default());
        self.hops = 0;
        self.valid = 0;
        self.sounding = 0;
        self.numerical = 0;
        result
    }
}

#[cfg(test)]
mod evaluation;
#[cfg(test)]
mod tests;
