use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeKind {
    Order,
    Continuation(Source),
    Simultaneous(Source, Source),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EdgeId {
    pub from: u64,
    pub to: u64,
    pub kind: EdgeKind,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    pub acoustic: Acoustic,
    pub timing_error: f64,
    pub trajectories: usize,
    pub events: usize,
    pub chunk: u64,
    pub episode: u64,
    pub anchor: Option<u64>,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Anchor {
    pub id: u64,
    pub node: u64,
    pub slot: usize,
    pub len: usize,
    pub residual: f64,
    pub error: f64,
    pub bound: bool,
    pub valid: bool,
    pub learned: bool,
    pub duration: Value,
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Motion {
    pub displacement: Value,
    pub speed: Value,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Chunk {
    pub id: u64,
    pub episode: u64,
    pub first: u64,
    pub last: u64,
    pub closed: bool,
    pub learned: bool,
}

pub(super) struct Graph {
    pub load: Load,
    limits: Limits,
    pub nodes: VecDeque<Interval>,
    pub anchors: VecDeque<Anchor>,
    pub chunks: VecDeque<Chunk>,
    trajectories: VecDeque<Box<[Trajectory]>>,
    events: VecDeque<Box<[Event]>>,
    support: VecDeque<Box<[Bin]>>,
    motions: VecDeque<Box<[Motion]>>,
    index: super::index::Index,
    live_payload_bytes: usize,
    pub statistics: [Moments; 9],
    shape_sum: Vec<f64>,
    shape_mean: Vec<f64>,
    shape_m2: f64,
    shape_count: usize,
    pub coordinate_version: Option<u64>,
    node_capacity: usize,
    edge_capacity: usize,
    records: usize,
    next_node: u64,
    next_anchor: u64,
    chunk: u64,
    episode: u64,
    chunk_anchors: usize,
}
impl Graph {
    pub fn new(load: Load, limits: Limits) -> Result<Self, &'static str> {
        let b = load.coordinates.len();
        if load.sample_rate == 0
            || load.hop == 0
            || b == 0
            || load.sparse_entries == 0
            || load.sparse_entries > b
            || !load.coordinates.iter().all(|v| v.is_finite())
            || !load.coordinates.windows(2).all(|w| w[0] < w[1])
            || limits.records < SMAX
            || !limits.seconds.is_finite()
            || limits.seconds <= 0.
        {
            return Err("invalid declared load");
        }
        let nodes = 1
            + (limits.seconds * f64::from(load.sample_rate) / load.hop as f64).ceil() as usize
            + limits.records;
        let s = load.sources;
        let edges_per = 1usize
            .checked_add(s)
            .and_then(|n| {
                s.checked_mul(s.saturating_sub(1))
                    .and_then(|pairs| n.checked_add(pairs / 2))
            })
            .ok_or("edge capacity overflow")?;
        let edges = nodes
            .checked_mul(edges_per)
            .ok_or("edge capacity overflow")?;
        nodes.checked_mul(s).ok_or("trajectory capacity overflow")?;
        nodes.checked_mul(s + 1).ok_or("event capacity overflow")?;
        limits
            .records
            .checked_mul(load.sparse_entries)
            .ok_or("support capacity overflow")?;
        let index = super::index::Index::new(&load.coordinates);
        Ok(Self {
            trajectories: VecDeque::new(),
            events: VecDeque::new(),
            support: VecDeque::new(),
            motions: VecDeque::new(),
            nodes: VecDeque::new(),
            anchors: VecDeque::new(),
            chunks: VecDeque::new(),
            index,
            live_payload_bytes: 0,
            statistics: [Moments::default(); 9],
            shape_sum: vec![0.; b],
            shape_mean: vec![0.; b],
            shape_m2: 0.,
            shape_count: 0,
            load,
            limits,
            coordinate_version: None,
            node_capacity: nodes,
            edge_capacity: edges,
            records: 0,
            next_node: 1,
            next_anchor: 1,
            chunk: 1,
            episode: 1,
            chunk_anchors: 0,
        })
    }
    pub fn validate(&self, o: &Observation<'_>) -> Result<(), &'static str> {
        if !o.time.is_finite()
            || o.time < 0.
            || !o.timing_error.is_finite()
            || o.timing_error < 0.
            || !o.acoustic.sampled_at.is_finite()
            || o.acoustic.sampled_at > o.time
            || ![o.acoustic.mass, o.acoustic.focus, o.acoustic.competition]
                .iter()
                .all(|x| x.checked())
            || (o.acoustic.mass.valid && o.acoustic.mass.value < 0.)
        {
            return Err("invalid observation");
        }
        if !o.hop && o.events.is_empty() {
            return Err("not a partition boundary");
        }
        if o.trajectories.len() > self.load.sources || o.events.len() > self.load.sources + 1 {
            return Err("source load exceeded");
        }
        for (i, t) in o.trajectories.iter().enumerate() {
            if ![t.pitch, t.amplitude, t.timbre].iter().all(|v| v.checked())
                || o.trajectories[..i].iter().any(|p| p.source == t.source)
            {
                return Err("invalid or duplicate trajectory");
            }
        }
        for (i, e) in o.events.iter().enumerate() {
            if e.flags == 0
                || e.flags & !(ONSET | OFFSET | OPPORTUNITY | BOUNDARY) != 0
                || o.events[..i].iter().any(|p| p.source == e.source)
            {
                return Err("merge simultaneous flags per source");
            }
        }
        let p = o.support;
        if p.bins.len() > self.load.sparse_entries
            || !p.residual.is_finite()
            || !(0. ..=1.).contains(&p.residual)
            || !p.error.is_finite()
            || p.error < 0.
        {
            return Err("invalid sparse support");
        }
        let mut sum = p.residual;
        for (i, b) in p.bins.iter().enumerate() {
            if b.index >= self.load.coordinates.len()
                || !b.mass.is_finite()
                || b.mass < 0.
                || (i > 0 && p.bins[i - 1].index >= b.index)
            {
                return Err("invalid support coordinates");
            }
            sum += f64::from(b.mass);
        }
        // This is a representation-rounding bound, never a matching tolerance.
        let rounding = (p.bins.len() + 1) as f64 * f64::from(f32::EPSILON);
        if p.valid
            && (!o.acoustic.mass.valid
                || o.acoustic.mass.value == 0.
                || (sum - 1.).abs() > rounding)
        {
            return Err("undefined or unnormalized support");
        }
        Ok(())
    }
    pub fn node(&self, id: u64) -> Option<&Interval> {
        let first = self.nodes.front()?.id;
        self.nodes
            .get(usize::try_from(id.checked_sub(first)?).ok()?)
            .filter(|n| n.id == id)
    }
    pub fn anchor(&self, id: u64) -> Option<&Anchor> {
        let first = self.anchors.front()?.id;
        self.anchors
            .get(usize::try_from(id.checked_sub(first)?).ok()?)
            .filter(|a| a.id == id)
    }
    pub fn chunk(&self, id: u64) -> Option<&Chunk> {
        self.chunks
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .and_then(|i| self.chunks.get(i))
    }
    pub fn trajectories(&self, n: &Interval) -> &[Trajectory] {
        &self.trajectories[(n.id - self.nodes.front().expect("live node").id) as usize]
    }
    pub fn events(&self, n: &Interval) -> &[Event] {
        &self.events[(n.id - self.nodes.front().expect("live node").id) as usize]
    }
    pub fn bins(&self, a: &Anchor) -> &[Bin] {
        &self.support[(a.id - self.anchors.front().expect("live anchor").id) as usize]
    }
    pub fn observation(&self, a: &Anchor) -> Observation<'_> {
        let n = self.node(a.node).expect("live anchor node");
        Observation {
            time: n.start,
            timing_error: n.timing_error,
            hop: true,
            acoustic: n.acoustic,
            support: Support {
                bins: self.bins(a),
                residual: a.residual,
                error: a.error,
                bound: a.bound,
                valid: a.valid,
            },
            trajectories: self.trajectories(n),
            events: self.events(n),
        }
    }
    #[cfg(test)]
    pub fn shape_count_for_test(&self) -> usize {
        self.shape_count
    }
    pub fn shape_variance(&self) -> f64 {
        if self.shape_count < 2 {
            return 0.;
        }
        // Centered evaluation of n*(1-||S/n||^2)/(2*(n-1)) avoids
        // inventing variance through cancellation for identical support shapes.
        self.shape_m2 / (2. * (self.shape_count - 1) as f64)
    }
    pub(super) fn register(&mut self, a: Anchor, remove: bool) {
        if !a.learned {
            return;
        }
        let n = *self.node(a.node).expect("statistics node");
        let scalar = [
            n.acoustic.mass,
            n.acoustic.focus,
            n.acoustic.competition,
            a.duration,
        ];
        for (i, v) in scalar.into_iter().enumerate() {
            if v.valid {
                if remove {
                    self.statistics[i].remove(v.value)
                } else {
                    self.statistics[i].add(v.value)
                }
            }
        }
        let node_offset = (n.id - self.nodes.front().expect("live node").id) as usize;
        let anchor_offset = (a.id - self.anchors.front().expect("live anchor").id) as usize;
        for i in 0..n.trajectories {
            let t = self.trajectories[node_offset][i];
            let m = self.motions[anchor_offset][i];
            for (j, v) in [
                (4, m.displacement),
                (5, t.pitch),
                (6, m.speed),
                (7, t.timbre),
                (8, t.amplitude),
            ] {
                if v.valid {
                    if remove {
                        self.statistics[j].remove(v.value)
                    } else {
                        self.statistics[j].add(v.value)
                    }
                }
            }
        }
        // Residual mass has no known coordinate and cannot enter a unit embedding.
        if a.valid && a.residual == 0. {
            let old_count = self.shape_count;
            if remove {
                self.shape_count -= 1;
            } else {
                self.shape_count += 1;
            }
            let total = self.support[anchor_offset]
                .iter()
                .map(|b| f64::from(b.mass))
                .sum::<f64>();
            let mut sparse = 0;
            for index in 0..self.shape_sum.len() {
                let u = if sparse < a.len && self.support[anchor_offset][sparse].index == index {
                    let u = (f64::from(self.support[anchor_offset][sparse].mass) / total).sqrt();
                    sparse += 1;
                    u
                } else {
                    0.
                };
                self.shape_sum[index] += if remove { -u } else { u };
                let old_mean = self.shape_mean[index];
                if self.shape_count > 0 {
                    if remove {
                        self.shape_mean[index] -= (u - old_mean) / self.shape_count as f64;
                        self.shape_m2 -= (u - old_mean) * (u - self.shape_mean[index]);
                    } else {
                        self.shape_mean[index] += (u - old_mean) / self.shape_count as f64;
                        self.shape_m2 += (u - old_mean) * (u - self.shape_mean[index]);
                    }
                }
            }
            self.shape_m2 = self.shape_m2.max(0.);
            if self.shape_count < 2 {
                self.shape_m2 = 0.;
            }
            if self.shape_count == 0 {
                self.shape_sum.fill(0.);
                self.shape_mean.fill(0.);
            }
            debug_assert!(old_count.abs_diff(self.shape_count) == 1);
        }
    }

    fn remove_front(&mut self) {
        let Some(n) = self.nodes.front().copied() else {
            return;
        };
        if let Some(id) = n.anchor {
            let a = *self.anchors.front().expect("anchor expiration order");
            assert_eq!(a.id, id);
            self.register(a, true);
            self.anchors.pop_front();
            self.live_payload_bytes -= self.support.pop_front().unwrap().len() * size_of::<Bin>();
            self.live_payload_bytes -=
                self.motions.pop_front().unwrap().len() * size_of::<Motion>();
        }
        self.records -= n.events;
        self.live_payload_bytes -=
            self.trajectories.pop_front().unwrap().len() * size_of::<Trajectory>();
        self.live_payload_bytes -= self.events.pop_front().unwrap().len() * size_of::<Event>();
        self.nodes.pop_front();
    }
    fn clean_references(&mut self) {
        let anchor = self.anchors.front().map_or(self.next_anchor, |a| a.id);
        while self.chunks.front().is_some_and(|c| c.last < anchor) {
            self.chunks.pop_front();
        }
        if let Some(c) = self.chunks.front_mut() {
            c.first = c.first.max(anchor);
        }
    }
    pub fn expire(&mut self, now: f64) {
        while self
            .nodes
            .front()
            .is_some_and(|n| n.start < now - self.limits.seconds)
        {
            self.remove_front();
        }
        self.clean_references();
        self.index
            .expire_before(self.anchors.front().map_or(self.next_anchor, |a| a.id));
    }
    fn evict_chunk(&mut self) {
        let Some(chunk) = self.nodes.front().map(|n| n.chunk) else {
            return;
        };
        if chunk == self.chunk {
            self.close_chunk(false);
        }
        while self.nodes.front().is_some_and(|n| n.chunk == chunk) {
            self.remove_front();
        }
        self.clean_references();
    }
    fn close_chunk(&mut self, boundary: bool) {
        if let Some(c) = self.chunks.back_mut().filter(|c| c.id == self.chunk) {
            c.closed = true;
        }
        self.chunk += 1;
        self.chunk_anchors = 0;
        if boundary {
            self.episode += 1;
        }
    }
    pub fn insert(&mut self, o: &Observation<'_>, learned: bool) -> Result<(), &'static str> {
        self.expire(o.time);
        if self
            .coordinate_version
            .is_some_and(|v| v != o.acoustic.coordinate_version)
        {
            while !self.nodes.is_empty() {
                self.remove_front();
            }
            self.clean_references();
            self.close_chunk(true);
        }
        self.coordinate_version = Some(o.acoustic.coordinate_version);
        if let Some(last) = self.nodes.back().copied()
            && o.events.is_empty()
            && last.events == 0
            && last.acoustic == o.acoustic
            && self.trajectories(&last) == o.trajectories
        {
            self.nodes.back_mut().unwrap().end = o.time;
            return Ok(());
        }
        while self.nodes.len() == self.node_capacity
            || self.records + o.events.len() > self.limits.records
        {
            self.evict_chunk();
        }
        // Capacity/coordinate eviction can remove several chunks. Update the
        // neighborhood once after their complete ordered statistics removal.
        self.index
            .expire_before(self.anchors.front().map_or(self.next_anchor, |a| a.id));
        let id = self.next_node;
        self.next_node += 1;
        self.trajectories.push_back(o.trajectories.into());
        self.events.push_back(o.events.into());
        self.live_payload_bytes +=
            std::mem::size_of_val(o.trajectories) + std::mem::size_of_val(o.events);
        if let Some(previous) = self.nodes.back().copied() {
            self.nodes.back_mut().unwrap().end = o.time;
            debug_assert_eq!(previous.id + 1, id);
        }
        let anchor_id = (!o.events.is_empty()).then_some(self.next_anchor);
        self.nodes.push_back(Interval {
            id,
            start: o.time,
            end: o.time,
            acoustic: o.acoustic,
            timing_error: o.timing_error,
            trajectories: o.trajectories.len(),
            events: o.events.len(),
            chunk: self.chunk,
            episode: self.episode,
            anchor: anchor_id,
        });
        self.records += o.events.len();
        if let Some(aid) = anchor_id {
            self.next_anchor += 1;
            let aslot = aid as usize % self.limits.records;
            let mut duration = Value::default();
            let mut motions = vec![Motion::default(); o.trajectories.len()].into_boxed_slice();
            if let Some(previous) = self.anchors.back().copied() {
                let pn = *self.node(previous.node).unwrap();
                if pn.episode == self.episode {
                    duration = Value {
                        value: o.time - pn.start,
                        error: o.timing_error.hypot(pn.timing_error),
                        valid: true,
                        bound: false,
                    };
                    for (i, t) in o.trajectories.iter().enumerate() {
                        if let Some(p) =
                            self.trajectories(&pn).iter().find(|p| p.source == t.source)
                        {
                            let displacement = t.pitch.difference(p.pitch);
                            let speed = if duration.value > 0. {
                                Value {
                                    value: displacement.value / duration.value,
                                    error: displacement.error / duration.value
                                        + displacement.value.abs() * duration.error
                                            / duration.value.powi(2),
                                    valid: displacement.valid,
                                    bound: true,
                                }
                            } else {
                                Value::default()
                            };
                            motions[i] = Motion {
                                displacement,
                                speed,
                            };
                        }
                    }
                }
            }
            self.live_payload_bytes +=
                std::mem::size_of_val(o.support.bins) + motions.len() * size_of::<Motion>();
            self.support.push_back(o.support.bins.into());
            self.motions.push_back(motions);
            let a = Anchor {
                id: aid,
                node: id,
                slot: aslot,
                len: o.support.bins.len(),
                residual: o.support.residual,
                error: o.support.error,
                bound: o.support.bound,
                valid: o.support.valid,
                learned,
                duration,
            };
            self.anchors.push_back(a);
            // Context evidence can use a committed singleton endpoint. Prediction
            // separately requires an actual observed successor before publishing.
            if learned {
                self.index
                    .insert(aid, o.time, self.index.project(o.support));
            }
            self.chunk_anchors += 1;
            if let Some(c) = self.chunks.back_mut().filter(|c| c.id == self.chunk) {
                c.last = aid;
                c.learned &= learned;
            } else {
                self.chunks.push_back(Chunk {
                    id: self.chunk,
                    episode: self.episode,
                    first: aid,
                    last: aid,
                    closed: false,
                    learned,
                });
            }
            if o.events.iter().any(|e| e.flags & BOUNDARY != 0) {
                self.close_chunk(true);
            } else if self.chunk_anchors == SMAX {
                // Finalize with this chunk's evidence, before a later query replaces it.
                self.close_chunk(false);
            }
        }
        Ok(())
    }
    pub fn successor(&self, a: &Anchor) -> Option<&Anchor> {
        let next = self.anchor(a.id + 1)?;
        (next.learned && self.node(next.node)?.episode == self.node(a.node)?.episode)
            .then_some(next)
    }
    pub fn nearby(&self, query: Support<'_>, before: f64, limit: usize, out: &mut Vec<u64>) {
        self.index.nearby(query, before, limit, out);
    }
    pub fn resources(&self) -> Resources {
        let dense_graph_categories = [
            self.node_capacity * size_of::<Interval>(),
            self.limits.records * size_of::<Anchor>(),
            self.limits.records * size_of::<Chunk>(),
            self.node_capacity * self.load.sources * size_of::<Trajectory>(),
            self.node_capacity * (self.load.sources + 1) * size_of::<Event>(),
            self.limits.records * self.load.sparse_entries * size_of::<Bin>(),
            self.limits.records * self.load.sources * size_of::<Motion>(),
            (self.shape_sum.capacity()
                + self.shape_mean.capacity()
                + self.load.coordinates.capacity())
                * size_of::<f64>(),
        ];
        Resources {
            reserved_dense_graph_bytes: dense_graph_categories.iter().sum(),
            dense_graph_categories,
            live_payload_bytes: self.live_payload_bytes,
            index_allocated_bytes: self.index.bytes(),
            node_capacity: self.node_capacity,
            edge_capacity: self.edge_capacity,
            anchor_capacity: self.limits.records,
            nodes: self.nodes.len(),
            edges: self
                .nodes
                .iter()
                .map(|n| {
                    let s = n.trajectories;
                    s * s.saturating_sub(1) / 2
                        + self.node(n.id - 1).map_or(0, |previous| {
                            1 + self
                                .trajectories(n)
                                .iter()
                                .filter(|t| {
                                    self.trajectories(previous)
                                        .iter()
                                        .any(|p| p.source == t.source)
                                })
                                .count()
                        })
                })
                .sum(),
            anchors: self.anchors.len(),
            records: self.records,
            retained_support_entries: self.anchors.iter().map(|a| a.len).sum(),
            dense_support_bytes: self.limits.records
                * self.load.coordinates.len()
                * size_of::<Bin>(),
            allocated_bytes: self.nodes.capacity() * size_of::<Interval>()
                + self.anchors.capacity() * size_of::<Anchor>()
                + self.chunks.capacity() * size_of::<Chunk>()
                + self.trajectories.capacity() * size_of::<Box<[Trajectory]>>()
                + self.events.capacity() * size_of::<Box<[Event]>>()
                + self.support.capacity() * size_of::<Box<[Bin]>>()
                + self.motions.capacity() * size_of::<Box<[Motion]>>()
                + self.live_payload_bytes
                + self.index.bytes()
                + (self.shape_sum.capacity() + self.shape_mean.capacity()) * size_of::<f64>()
                + self.load.coordinates.capacity() * size_of::<f64>(),
        }
    }
}

/// Attribute views borrow the graph's shared trajectory records instead of copying them per edge.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EdgeAttributes {
    pub relative_time: Value,
    pub overlap: Value,
    pub relative_pitch: Value,
    pub absolute_register: [Value; 2],
    pub absolute_speed: Value,
    pub articulation_changed: Option<bool>,
    pub timbre: [Value; 2],
    pub source_continuity: Option<Source>,
}
impl Graph {
    /// All edges are determined by adjacent nodes and their shared source records.
    pub fn edges(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.nodes.iter().flat_map(move |n| {
            let previous = self.node(n.id - 1);
            let trajectories = self.trajectories(n);
            let order = previous.into_iter().map(move |p| EdgeId {
                from: p.id,
                to: n.id,
                kind: EdgeKind::Order,
            });
            let continuation = previous.into_iter().flat_map(move |p| {
                trajectories
                    .iter()
                    .filter(move |t| self.trajectories(p).iter().any(|v| v.source == t.source))
                    .map(move |t| EdgeId {
                        from: p.id,
                        to: n.id,
                        kind: EdgeKind::Continuation(t.source),
                    })
            });
            let simultaneous = trajectories.iter().enumerate().flat_map(move |(i, a)| {
                trajectories[i + 1..].iter().map(move |b| EdgeId {
                    from: n.id,
                    to: n.id,
                    kind: EdgeKind::Simultaneous(a.source, b.source),
                })
            });
            order.chain(continuation).chain(simultaneous)
        })
    }

    pub fn edge_attributes(&self, edge: EdgeId) -> Option<EdgeAttributes> {
        let from = self.node(edge.from)?;
        let to = self.node(edge.to)?;
        match edge.kind {
            EdgeKind::Order | EdgeKind::Continuation(_) if edge.from + 1 != edge.to => return None,
            EdgeKind::Simultaneous(a, b) => {
                if edge.from != edge.to {
                    return None;
                }
                let source = self.trajectories(to);
                let i = source.iter().position(|t| t.source == a)?;
                let j = source.iter().position(|t| t.source == b)?;
                if i >= j {
                    return None;
                }
            }
            _ => {}
        }
        let mut attributes = EdgeAttributes {
            relative_time: Value {
                value: to.start - from.start,
                error: from.timing_error.hypot(to.timing_error),
                valid: true,
                bound: false,
            },
            ..EdgeAttributes::default()
        };
        let (a, b) = match edge.kind {
            EdgeKind::Order => return Some(attributes),
            EdgeKind::Continuation(source) => {
                attributes.source_continuity = Some(source);
                (
                    self.trajectories(from)
                        .iter()
                        .find(|t| t.source == source)?,
                    self.trajectories(to).iter().find(|t| t.source == source)?,
                )
            }
            EdgeKind::Simultaneous(a, b) => (
                self.trajectories(to).iter().find(|t| t.source == a)?,
                self.trajectories(to).iter().find(|t| t.source == b)?,
            ),
        };
        attributes.overlap = Value::exact(if a.sounding && b.sounding {
            (from.end.min(to.end) - from.start.max(to.start)).max(0.)
        } else {
            0.
        });
        attributes.relative_pitch = b.pitch.difference(a.pitch);
        attributes.absolute_register = [a.pitch, b.pitch];
        attributes.timbre = [a.timbre, b.timbre];
        attributes.articulation_changed = Some(a.articulation != b.articulation);
        if attributes.source_continuity.is_some() && attributes.relative_time.value > 0. {
            let d = attributes.relative_pitch;
            let t = attributes.relative_time;
            attributes.absolute_speed = Value {
                value: d.value / t.value,
                error: d.error / t.value + d.value.abs() * t.error / t.value.powi(2),
                valid: d.valid,
                bound: true,
            };
        }
        Some(attributes)
    }
}

#[cfg(test)]
mod storage_tests {
    use super::super::tests::{Input, model, observe};
    use super::*;

    #[test]
    fn harmonic_relations_sparse_graph_keeps_payload_identity_and_expiry() {
        let mut model = model();
        let empty = model.graph.resources();
        assert_eq!(empty.live_payload_bytes, 0);
        assert!(empty.allocated_bytes < empty.reserved_dense_graph_bytes);
        let first = Input::new(0., 1);
        let second = Input::new(1., 2);
        observe(&mut model, &first);
        observe(&mut model, &second);
        for (id, input) in [(1, &first), (2, &second)] {
            let anchor = model.graph.anchor(id).unwrap();
            let actual = model.graph.observation(anchor);
            let expected = input.observation();
            assert_eq!(actual.acoustic, expected.acoustic);
            assert_eq!(actual.support.bins, expected.support.bins);
            assert_eq!(actual.trajectories, expected.trajectories);
            assert_eq!(actual.events, expected.events);
        }
        let expected_payload = 2
            * (size_of::<Trajectory>()
                + size_of::<Event>()
                + size_of::<Bin>()
                + size_of::<Motion>());
        assert_eq!(model.graph.resources().live_payload_bytes, expected_payload);
        let mut nearby = Vec::with_capacity(HMAX);
        model
            .graph
            .nearby(Support::default(), 2., HMAX, &mut nearby);
        assert_eq!(nearby, vec![2, 1]);
        model.graph.expire(602.);
        model
            .graph
            .nearby(Support::default(), 603., HMAX, &mut nearby);
        assert!(nearby.is_empty());
        assert_eq!(model.graph.resources().live_payload_bytes, 0);
        assert_eq!(model.graph.shape_count, 0);
    }

    #[test]
    fn harmonic_relations_graph_storage_categories_are_measured_without_dense_allocation() {
        let graph = Graph::new(
            Load {
                sample_rate: 48000,
                hop: 512,
                sources: 36,
                coordinates: (0..690).map(|i| i as f64 / 100.).collect(),
                sparse_entries: 690,
            },
            Limits::default(),
        )
        .unwrap();
        let resources = graph.resources();
        assert_eq!(resources.node_capacity, 94651);
        assert_eq!(resources.anchor_capacity, UMAX);
        assert_eq!(resources.live_payload_bytes, 0);
        assert_eq!(
            resources.reserved_dense_graph_bytes,
            resources.dense_graph_categories.iter().sum::<usize>()
        );
        eprintln!(
            "HV7_GRAPH_STORAGE categories=intervals,anchors,chunks,trajectories,events,support,motions,shape_and_coordinates declared={:?} dense_total={} allocated={} index={} live_payload={}",
            resources.dense_graph_categories,
            resources.reserved_dense_graph_bytes,
            resources.allocated_bytes,
            resources.index_allocated_bytes,
            resources.live_payload_bytes
        );
    }
}
