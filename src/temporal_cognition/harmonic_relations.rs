//! Author-rule temporal relations over heard support and executed source events.
//! The adapter, background worker and consumers preserve observation provenance.
#![allow(dead_code)]

mod adapter;
pub(crate) mod consumer;
mod context;
mod derived;
mod graph;
mod index;
mod matching;
pub(crate) mod runtime;

use context::Contexts;
use derived::Relations;
use graph::Graph;
use matching::Matcher;
pub(crate) use matching::Prediction;

pub(crate) const TMAX: f64 = 600.0;
pub(crate) const UMAX: usize = 38_400;
pub(crate) const SMAX: usize = 16;
pub(crate) const KMAX: usize = 16;
pub(crate) const HMAX: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Source {
    pub kind: u8,
    pub id: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Value {
    pub value: f64,
    pub error: f64,
    pub valid: bool,
    pub bound: bool,
}
impl Value {
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            valid: true,
            ..Self::default()
        }
    }
    fn checked(self) -> bool {
        !self.valid || (self.value.is_finite() && self.error.is_finite() && self.error >= 0.)
    }
    fn difference(self, previous: Self) -> Self {
        Self {
            value: self.value - previous.value,
            error: if self.bound || previous.bound {
                self.error + previous.error
            } else {
                self.error.hypot(previous.error)
            },
            valid: self.valid && previous.valid,
            bound: self.bound || previous.bound,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Bin {
    pub index: usize,
    pub mass: f32,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Support<'a> {
    pub bins: &'a [Bin],
    pub residual: f64,
    pub error: f64,
    pub bound: bool,
    pub valid: bool,
}
impl Default for Support<'_> {
    fn default() -> Self {
        Self {
            bins: &[],
            residual: 0.,
            error: 0.,
            bound: false,
            valid: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Acoustic {
    pub snapshot: u64,
    pub sampled_at: f64,
    pub coordinate_version: u64,
    pub mass: Value,
    pub focus: Value,
    pub competition: Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Trajectory {
    pub source: Source,
    /// Log2 Hz, in the support coordinate system.
    pub pitch: Value,
    pub amplitude: Value,
    pub timbre: Value,
    pub articulation: u64,
    pub sounding: bool,
}

pub(crate) const ONSET: u8 = 1;
pub(crate) const OFFSET: u8 = 2;
pub(crate) const OPPORTUNITY: u8 = 4;
pub(crate) const BOUNDARY: u8 = 8;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Event {
    pub source: Option<Source>,
    pub flags: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation<'a> {
    pub time: f64,
    pub timing_error: f64,
    pub hop: bool,
    pub acoustic: Acoustic,
    pub support: Support<'a>,
    pub trajectories: &'a [Trajectory],
    pub events: &'a [Event],
}

#[derive(Clone, Debug)]
pub(crate) struct Load {
    pub sample_rate: u32,
    pub hop: usize,
    pub sources: usize,
    pub coordinates: Vec<f64>,
    /// Producer-declared bound, including a possible dense shape.
    pub sparse_entries: usize,
}
#[derive(Clone, Copy, Debug)]
struct Limits {
    seconds: f64,
    records: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            seconds: TMAX,
            records: UMAX,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Resources {
    pub node_capacity: usize,
    pub edge_capacity: usize,
    pub anchor_capacity: usize,
    pub nodes: usize,
    pub edges: usize,
    pub anchors: usize,
    pub records: usize,
    pub retained_support_entries: usize,
    pub allocated_bytes: usize,
    pub dense_support_bytes: usize,
    pub reserved_dense_graph_bytes: usize,
    pub live_payload_bytes: usize,
    pub index_allocated_bytes: usize,
    pub dense_graph_categories: [usize; 8],
}

/// Removable centered moments; no tolerance is inserted at zero scale.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: usize,
    mean: f64,
    m2: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }
    fn remove(&mut self, x: f64) {
        if self.n <= 1 {
            *self = Self::default();
            return;
        }
        let old = self.mean;
        self.n -= 1;
        self.mean -= (x - old) / self.n as f64;
        self.m2 = (self.m2 - (x - old) * (x - self.mean)).max(0.);
    }
    fn variance(self) -> f64 {
        if self.n < 2 {
            0.
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kernel {
    sum: f64,
    valid: usize,
    requested: usize,
}
impl Kernel {
    fn scalar(&mut self, a: Value, b: Value, variance: f64) {
        self.requested += 1;
        if a.valid && b.valid {
            let residual = (a.value - b.value).powi(2);
            self.distance(residual, a.error.powi(2) + b.error.powi(2) + variance);
        }
    }
    fn distance(&mut self, squared: f64, scale_squared: f64) {
        self.valid += 1;
        self.sum += if scale_squared > 0. {
            squared / scale_squared
        } else if squared == 0. {
            0.
        } else {
            f64::INFINITY
        };
    }
    fn compatibility(self) -> f64 {
        if self.valid == 0 {
            0.
        } else {
            (-0.5 * self.sum / self.valid as f64).exp()
        }
    }
    fn coverage(self) -> f64 {
        if self.requested == 0 {
            0.
        } else {
            self.valid as f64 / self.requested as f64
        }
    }
    fn weight(self) -> f64 {
        self.coverage() * self.compatibility()
    }
}

pub(crate) struct Model {
    graph: Graph,
    matcher: Matcher,
    contexts: Contexts,
    relations: Relations,
    prediction: Prediction,
    local: Accumulator,
    global: Accumulator,
    version: u64,
    learning: bool,
    correspondence_enabled: bool,
    last_time: Option<f64>,
}
impl Model {
    pub fn new(load: Load) -> Result<Self, &'static str> {
        Self::with_limits(load, Limits::default())
    }
    fn with_limits(load: Load, limits: Limits) -> Result<Self, &'static str> {
        let b = load.coordinates.len();
        let graph = Graph::new(load, limits)?;
        let mut matcher = Matcher::new();
        matcher.prepare(b);
        Ok(Self {
            matcher,
            contexts: Contexts::new(limits.records),
            relations: Relations::new(graph.load.sources),
            prediction: Prediction::new(b, graph.load.sources),
            local: Accumulator::new(b, 0.1),
            global: Accumulator::new(b, 1.5),
            graph,
            version: 0,
            learning: true,
            correspondence_enabled: true,
            last_time: None,
        })
    }
    /// Querying never registers observations or updates statistics.
    pub fn predict(
        &mut self,
        query: &[Observation<'_>],
        query_version: u64,
    ) -> Result<&Prediction, &'static str> {
        if query.is_empty() || query.len() > SMAX {
            return Err("query anchor count");
        }
        for (i, o) in query.iter().enumerate() {
            self.graph.validate(o)?;
            if o.events.is_empty() || (i > 0 && query[i - 1].time >= o.time) {
                return Err("query must contain ordered structural anchors");
            }
        }
        let time = query.last().expect("nonempty query").time;
        if self.last_time.is_some_and(|last| time < last) {
            return Err("query precedes latest observation");
        }
        let first = self.graph.anchors.front().map(|a| a.id);
        self.graph.expire(time);
        if first != self.graph.anchors.front().map(|a| a.id) {
            self.contexts.expire(&self.graph);
            self.version += 1;
        }
        self.matcher
            .search(&self.graph, query, self.correspondence_enabled);
        self.matcher.publish(
            &self.graph,
            query,
            self.version,
            query_version,
            &mut self.prediction,
        );
        Ok(&self.prediction)
    }
    /// Background-worker query over actual retained anchors, including hop-only aging.
    /// A fresh acoustic hop does not fabricate a structural event or move an anchor.
    pub fn predict_latest(
        &mut self,
        now: f64,
        query_version: u64,
    ) -> Result<&Prediction, &'static str> {
        if !now.is_finite() || now < 0. || self.last_time.is_some_and(|last| now < last) {
            return Err("query precedes latest observation");
        }
        let first = self.graph.anchors.front().map(|a| a.id);
        self.graph.expire(now);
        if first != self.graph.anchors.front().map(|a| a.id) {
            self.contexts.expire(&self.graph);
            self.version += 1;
        }
        let Some(last) = self.graph.anchors.back() else {
            self.prediction.len = 0;
            self.prediction.unknown = 1.;
            self.prediction.query_version = query_version;
            self.prediction.model_version = self.version;
            self.prediction.issued_at = now;
            self.prediction.searched_endpoints = 0;
            return Ok(&self.prediction);
        };
        let mut query = [self.graph.observation(last); SMAX];
        let n = self
            .graph
            .chunks
            .back()
            .map_or(1, |chunk| (chunk.last - chunk.first + 1) as usize)
            .min(SMAX);
        for (out, anchor) in query[..n]
            .iter_mut()
            .zip(self.graph.anchors.iter().skip(self.graph.anchors.len() - n))
        {
            *out = self.graph.observation(anchor);
        }
        self.matcher
            .search(&self.graph, &query[..n], self.correspondence_enabled);
        self.matcher.publish(
            &self.graph,
            &query[..n],
            self.version,
            query_version,
            &mut self.prediction,
        );
        self.prediction.issued_at = now;
        Ok(&self.prediction)
    }

    pub fn observe(&mut self, o: Observation<'_>) -> Result<(), &'static str> {
        self.graph.validate(&o)?;
        if self.last_time.is_some_and(|t| o.time <= t) {
            return Err("merge simultaneous event records before observation");
        }
        let dt = self.last_time.map_or(0., |t| o.time - t);
        // Score the frozen prediction before any observation changes statistics.
        self.relations
            .observe(&self.graph, &self.prediction, self.version, &o);
        self.local.advance(&o, dt);
        self.global.advance(&o, dt);
        let previous_first = self.graph.anchors.front().map(|a| a.id);
        let previous_coordinate = self.graph.coordinate_version;
        self.graph.insert(&o, self.learning)?;
        let expired = previous_first != self.graph.anchors.front().map(|a| a.id);
        if expired {
            self.contexts.expire(&self.graph);
        }
        if self.learning && self.correspondence_enabled && !o.events.is_empty() {
            let last = self.graph.anchors.back().expect("committed anchor");
            let mut query = [self.graph.observation(last); SMAX];
            let n = self
                .graph
                .chunks
                .back()
                .map_or(1, |chunk| (chunk.last - chunk.first + 1) as usize)
                .min(SMAX);
            for (out, anchor) in query[..n]
                .iter_mut()
                .zip(self.graph.anchors.iter().skip(self.graph.anchors.len() - n))
            {
                *out = self.graph.observation(anchor);
            }
            self.matcher.search(&self.graph, &query[..n], true);
            self.contexts.complete(&self.graph, &self.matcher);
        } else if !o.events.is_empty() {
            self.contexts.evidence.fill(0.);
            self.contexts.evidence_ids.fill(0);
        }
        if !o.events.is_empty() {
            let anchor = *self.graph.anchors.back().expect("committed anchor");
            self.graph.register(anchor, false);
        }
        if !o.events.is_empty() || expired {
            self.relations.contexts(&self.contexts, o.time);
        }
        self.last_time = Some(o.time);
        if !o.events.is_empty() || expired || previous_coordinate != self.graph.coordinate_version {
            self.version += 1;
        }
        Ok(())
    }
    pub fn relations(&self) -> &Relations {
        &self.relations
    }
    pub fn contexts(&self) -> &[Option<context::Context>; KMAX] {
        &self.contexts.active
    }
    pub fn support_context(&self, global: bool, normalized: &mut [f64]) -> (f64, f64, bool) {
        let c = if global { &self.global } else { &self.local };
        assert_eq!(normalized.len(), c.values.len());
        let known = c.normalized(normalized);
        (c.mass(), c.unknown_mass, known && c.input_valid)
    }
    pub fn freeze_learning(&mut self) {
        self.learning = false;
    }
    pub fn remove_correspondences(&mut self) {
        self.correspondence_enabled = false;
        self.contexts.evidence.fill(0.);
        self.contexts.evidence_ids.fill(0);
        self.contexts.unknown = 1.;
        for context in self.contexts.active.iter_mut().flatten() {
            if context.responsibility != 0. {
                context.version += 1;
            }
            context.responsibility = 0.;
        }
        self.prediction.len = 0;
        self.prediction.unknown = 1.;
        self.version += 1;
        if let Some(time) = self.last_time {
            self.relations.contexts(&self.contexts, time);
        }
    }
    pub fn resources(&self) -> Resources {
        let mut r = self.graph.resources();
        r.allocated_bytes += self.matcher.bytes()
            + self.contexts.bytes()
            + self.prediction.bytes()
            + self.relations.bytes()
            + (self.local.values.capacity() + self.global.values.capacity()) * size_of::<f64>();
        r
    }
}

struct Accumulator {
    input_valid: bool,
    values: Vec<f64>,
    half_decay: f64,
    unknown_mass: f64,
    coordinate_version: Option<u64>,
}
impl Accumulator {
    fn new(b: usize, half_decay: f64) -> Self {
        Self {
            values: vec![0.; b],
            half_decay,
            input_valid: false,
            unknown_mass: 0.,
            coordinate_version: None,
        }
    }
    fn advance(&mut self, o: &Observation<'_>, dt: f64) {
        if self.coordinate_version != Some(o.acoustic.coordinate_version) {
            self.values.fill(0.);
            self.unknown_mass = 0.;
            self.coordinate_version = Some(o.acoustic.coordinate_version);
        }
        self.input_valid =
            o.acoustic.mass.valid && (o.acoustic.mass.value == 0. || o.support.valid);
        let a = 2_f64.powf(-dt / self.half_decay);
        for x in &mut self.values {
            *x *= a;
        }
        self.unknown_mass *= a;
        if o.acoustic.mass.valid && o.acoustic.mass.value > 0. {
            let mass = (1. - a) * o.acoustic.mass.value;
            if o.support.valid {
                for b in o.support.bins {
                    self.values[b.index] += mass * f64::from(b.mass);
                }
                self.unknown_mass += mass * o.support.residual;
            } else {
                self.unknown_mass += mass;
            }
        }
    }
    fn mass(&self) -> f64 {
        self.values.iter().sum::<f64>() + self.unknown_mass
    }
    fn normalized(&self, out: &mut [f64]) -> bool {
        let mass = self.mass();
        if mass == 0. {
            out.fill(0.);
            return false;
        }
        for (out, value) in out.iter_mut().zip(&self.values) {
            *out = value / mass;
        }
        true
    }
}

#[cfg(test)]
mod tests;
