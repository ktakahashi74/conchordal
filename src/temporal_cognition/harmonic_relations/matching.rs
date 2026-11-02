use super::*;
use graph::{Anchor, Graph};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Pair {
    pub query: u8,
    pub history: u8,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mapping {
    pub endpoint: u64,
    pub pairs: [Pair; SMAX],
    pub len: usize,
    pub omitted_query: u16,
    pub omitted_history: u16,
    pub query_source: Option<Source>,
    pub history_source: Option<Source>,
    pub transformed: bool,
    pub translation: Value,
    pub time_scale: Value,
    pub kernel: Kernel,
    compatibility_weight: f64,
    pub alternatives: f64,
    pub historical_start: u64,
    pub source_episode: u64,
    pub historical_nodes: [u64; SMAX],
    /// Current interval identity is (prediction query version, pair.query).
    pub current_snapshots: [u64; SMAX],
    /// Compact references to consecutive order/source edges along retained paths.
    pub history_paths: [Option<(u64, u64)>; SMAX],
    query_times: [u64; SMAX],
    translation_fit: Fit,
    time_fit: Fit,
    previous_time: f64,
    previous_timing_error: f64,
    previous_trajectory: Option<Trajectory>,
}
impl Default for Mapping {
    fn default() -> Self {
        Self {
            endpoint: 0,
            pairs: [Pair::default(); SMAX],
            len: 0,
            omitted_query: 0,
            omitted_history: 0,
            query_source: None,
            history_source: None,
            transformed: false,
            translation: Value::exact(0.),
            time_scale: Value::exact(1.),
            kernel: Kernel::default(),
            compatibility_weight: 0.,
            alternatives: 1.,
            historical_start: 0,
            source_episode: 0,
            historical_nodes: [0; SMAX],
            current_snapshots: [0; SMAX],
            history_paths: [None; SMAX],
            query_times: [0; SMAX],
            translation_fit: Fit::default(),
            time_fit: Fit::default(),
            previous_time: 0.,
            previous_timing_error: 0.,
            previous_trajectory: None,
        }
    }
}
impl Mapping {
    pub(super) fn weight(&self) -> f64 {
        self.compatibility_weight / self.alternatives
    }
}

#[derive(Clone, Copy)]
struct Cell {
    mappings: [Mapping; HMAX],
    len: usize,
    count: f64,
    endpoint_counts: [(u64, usize); 6 * HMAX],
    endpoints: usize,
}
impl Default for Cell {
    fn default() -> Self {
        Self {
            mappings: [Mapping::default(); HMAX],
            len: 0,
            count: 0.,
            endpoint_counts: [(0, 0); 6 * HMAX],
            endpoints: 0,
        }
    }
}
impl Cell {
    fn propose(&mut self, endpoint: u64) {
        self.count += 1.;
        if let Some((_, count)) = self.endpoint_counts[..self.endpoints]
            .iter_mut()
            .find(|(id, _)| *id == endpoint)
        {
            *count += 1;
        } else {
            self.endpoint_counts[self.endpoints] = (endpoint, 1);
            self.endpoints += 1;
        }
    }
    fn alternatives(&self, endpoint: u64) -> f64 {
        self.endpoint_counts[..self.endpoints]
            .iter()
            .find(|(id, _)| *id == endpoint)
            .unwrap()
            .1 as f64
    }
    fn insert(&mut self, m: Mapping) {
        self.propose(m.endpoint);
        if self.mappings[..self.len].iter().any(|p| {
            p.endpoint == m.endpoint
                && p.query_source == m.query_source
                && p.history_source == m.history_source
                && p.transformed == m.transformed
                && p.len == m.len
                && p.historical_nodes[..p.len] == m.historical_nodes[..m.len]
                && p.current_snapshots[..p.len] == m.current_snapshots[..m.len]
                && p.query_times[..p.len] == m.query_times[..m.len]
                && p.omitted_query == m.omitted_query
                && p.omitted_history == m.omitted_history
        }) {
            return;
        }
        let pos = self.mappings[..self.len].partition_point(|p| better(p, &m));
        if pos >= HMAX {
            return;
        }
        self.len = (self.len + 1).min(HMAX);
        for i in (pos + 1..self.len).rev() {
            self.mappings[i] = self.mappings[i - 1];
        }
        self.mappings[pos] = m;
    }
}
fn better(a: &Mapping, b: &Mapping) -> bool {
    b.compatibility_weight
        .total_cmp(&a.compatibility_weight)
        .then_with(|| a.endpoint.cmp(&b.endpoint))
        .then_with(|| a.query_source.cmp(&b.query_source))
        .then_with(|| a.history_source.cmp(&b.history_source))
        .then_with(|| a.transformed.cmp(&b.transformed))
        .then_with(|| a.pairs[..a.len].cmp(&b.pairs[..b.len]))
        .is_le()
}

#[derive(Clone, Debug)]
pub(crate) struct Atom {
    pub weight: f64,
    pub mapping: Mapping,
    pub successor: u64,
    pub acoustic: Acoustic,
    pub time: Value,
    pub bins: Vec<Bin>,
    pub residual: f64,
    pub shape_error: f64,
    pub shape_bound: bool,
    pub shape_valid: bool,
    pub absolute_trajectories: Vec<Trajectory>,
    pub trajectories: Vec<Trajectory>,
    pub source_version: u64,
    pub absolute_acoustic: Acoustic,
    pub absolute_time: f64,
}
#[derive(Clone, Debug)]
pub(crate) struct Prediction {
    pub atoms: Vec<Atom>,
    pub len: usize,
    pub unknown: f64,
    pub model_version: u64,
    pub query_version: u64,
    pub issued_at: f64,
    pub coordinate_version: u64,
    pub searched_endpoints: usize,
    pub variances: [f64; 9],
    pub shape_variance: f64,
    pub arrival_variances: [f64; 4],
    pub arrival_shape_variance: f64,
    pub arrival_pitch_variances: Vec<(Source, f64)>,
    pub arrival_expires_at: f64,
}
impl Prediction {
    pub fn new(b: usize, sources: usize) -> Self {
        Self {
            atoms: (0..HMAX)
                .map(|_| Atom {
                    weight: 0.,
                    mapping: Mapping::default(),
                    successor: 0,
                    acoustic: Acoustic::default(),
                    time: Value::default(),
                    bins: Vec::with_capacity(b),
                    residual: 0.,
                    shape_error: 0.,
                    shape_bound: false,
                    shape_valid: false,
                    absolute_trajectories: Vec::with_capacity(sources),
                    trajectories: Vec::with_capacity(sources),
                    source_version: 0,
                    absolute_acoustic: Acoustic::default(),
                    absolute_time: 0.,
                })
                .collect(),
            len: 0,
            unknown: 1.,
            model_version: 0,
            query_version: 0,
            issued_at: 0.,
            coordinate_version: 0,
            searched_endpoints: 0,
            variances: [0.; 9],
            shape_variance: 0.,
            arrival_variances: [0.; 4],
            arrival_shape_variance: 0.,
            arrival_pitch_variances: Vec::with_capacity(sources),
            arrival_expires_at: f64::INFINITY,
        }
    }
    pub fn for_query(&self, version: u64) -> Option<&Self> {
        (self.query_version == version).then_some(self)
    }
    pub fn bytes(&self) -> usize {
        self.atoms.capacity() * size_of::<Atom>()
            + self.arrival_pitch_variances.capacity() * size_of::<(Source, f64)>()
            + self
                .atoms
                .iter()
                .map(|a| {
                    a.bins.capacity() * size_of::<Bin>()
                        + (a.trajectories.capacity() + a.absolute_trajectories.capacity())
                            * size_of::<Trajectory>()
                })
                .sum::<usize>()
    }
}

pub(super) struct Matcher {
    pub best: [Mapping; HMAX],
    pub len: usize,
    pub endpoints: usize,
    transported: Vec<f64>,
    nearby: Vec<u64>,
    query_keys: [(u64, u64); SMAX],
    query_len: usize,
    coordinate_version: Option<u64>,
    #[cfg(test)]
    extensions: usize,
}
impl Matcher {
    pub fn new() -> Self {
        Self {
            best: [Mapping::default(); HMAX],
            len: 0,
            endpoints: 0,
            transported: Vec::new(),
            nearby: Vec::with_capacity(HMAX),
            query_keys: [(0, 0); SMAX],
            query_len: 0,
            coordinate_version: None,
            #[cfg(test)]
            extensions: 0,
        }
    }
    pub fn prepare(&mut self, b: usize) {
        self.transported = vec![0.; b];
    }
    pub fn bytes(&self) -> usize {
        self.transported.capacity() * size_of::<f64>()
            + self.nearby.capacity() * size_of::<u64>()
            + size_of::<Self>()
    }
    pub fn search(&mut self, g: &Graph, q: &[Observation<'_>], enabled: bool) {
        if q.is_empty()
            || !enabled
            || g.coordinate_version != Some(q[q.len() - 1].acoustic.coordinate_version)
        {
            self.len = 0;
            self.endpoints = 0;
            self.query_len = 0;
            return;
        }
        let q = &q[q.len().saturating_sub(SMAX)..];
        let mut keys = [(0, 0); SMAX];
        for (key, observation) in keys.iter_mut().zip(q) {
            *key = (observation.time.to_bits(), observation.acoustic.snapshot);
        }
        let same_coordinate = self.coordinate_version == g.coordinate_version;
        let overlap = if same_coordinate {
            (1..=self.query_len.min(q.len()))
                .rev()
                .find(|&n| self.query_keys[self.query_len - n..self.query_len] == keys[..n])
                .unwrap_or(0)
        } else {
            0
        };
        if overlap == 0 {
            self.len = 0;
            self.endpoints = 0;
        }
        // Reindex references only. Previously scored pairs and fitted statistics stay fixed.
        let mut retained = 0;
        for i in 0..self.len {
            let mut m = self.best[i];
            let mut omitted = 0;
            for (old, key) in self.query_keys[..self.query_len].iter().enumerate() {
                if m.omitted_query & (1 << old) != 0
                    && let Some(index) = keys[..q.len()].iter().position(|k| k == key)
                {
                    omitted |= 1 << index;
                }
            }
            m.omitted_query = omitted;
            if g.anchor(m.endpoint).is_none() {
                continue;
            }
            let mut valid = true;
            for j in 0..m.len {
                if let Some(index) = keys[..q.len()].iter().position(|&(time, snapshot)| {
                    time == m.query_times[j] && snapshot == m.current_snapshots[j]
                }) {
                    m.pairs[j].query = index as u8;
                } else {
                    valid = false;
                    break;
                }
                if g.node(m.historical_nodes[j]).is_none() {
                    valid = false;
                    break;
                }
            }
            if valid {
                self.best[retained] = m;
                retained += 1;
            }
        }
        self.len = retained;
        for (index, observation) in q.iter().enumerate().skip(overlap) {
            self.advance(g, observation, index, q[0].time);
        }
        self.query_keys = keys;
        self.query_len = q.len();
        self.coordinate_version = g.coordinate_version;
    }
    fn advance(&mut self, g: &Graph, query: &Observation<'_>, index: usize, before: f64) {
        #[cfg(test)]
        {
            self.extensions += 1;
        }
        let mut bank = Cell::default();
        for old in self.best[..self.len].iter().copied() {
            let Some(end) = g.anchor(old.endpoint) else {
                continue;
            };
            let mut omitted = old;
            omitted.omitted_query |= 1 << index;
            omitted.kernel.requested += 4 + usize::from(old.query_source.is_some()) * 4;
            omitted.compatibility_weight = if old.compatibility_weight == 0. {
                0.
            } else {
                omitted.kernel.weight()
            };
            bank.insert(omitted);
            if old.len == SMAX {
                continue;
            }
            if let Some(next) = g.successor(end)
                && g.observation(next).time < before
            {
                let mut matched = old;
                if append(g, query, index, next, &mut matched) {
                    bank.insert(matched);
                } else {
                    bank.propose(next.id);
                }
                if let Some(skipped) = g.successor(next)
                    && g.observation(skipped).time < before
                {
                    let mut matched = old;
                    matched.omitted_history |= 1 << old.len;
                    if append(g, query, index, skipped, &mut matched) {
                        bank.insert(matched);
                    } else {
                        bank.propose(skipped.id);
                    }
                }
            }
        }
        g.nearby(query.support, before, HMAX, &mut self.nearby);
        let mut bindings = [[(None, None); HMAX]; HMAX];
        let mut counts = [0; HMAX];
        let mut seed = |end: &Anchor, qs, hs, transformed| {
            let mut m = Mapping {
                query_source: qs,
                history_source: hs,
                transformed,
                time_scale: if transformed {
                    Value::default()
                } else {
                    Value::exact(1.)
                },
                historical_start: end.id,
                ..Mapping::default()
            };
            m.omitted_query = ((1_u32 << index) - 1) as u16;
            m.kernel.requested = index * (4 + usize::from(qs.is_some()) * 4);
            if append(g, query, index, end, &mut m) {
                bank.insert(m);
            } else {
                bank.propose(end.id);
            }
        };
        // Every retrieved endpoint gets a joint proposal before any source variants.
        for (i, id) in self.nearby.iter().copied().enumerate() {
            let Some(end) = g.anchor(id) else {
                continue;
            };
            seed(end, None, None, false);
            (bindings[i], counts[i]) = source_bindings(query, &g.observation(end));
        }
        let mut seeded = 0;
        'sources: for round in 1..HMAX {
            for (i, id) in self.nearby.iter().copied().enumerate() {
                if round >= counts[i] {
                    continue;
                }
                let Some(end) = g.anchor(id) else {
                    continue;
                };
                let (qs, hs) = bindings[i][round];
                for transformed in [false, true] {
                    seed(end, qs, hs, transformed);
                }
                seeded += 1;
                if seeded == HMAX {
                    break 'sources;
                }
            }
        }
        // Endpoint-local proposal counts preserve empirical priors and unknown mass.
        self.endpoints = bank.endpoints;
        self.len = bank.len;
        for (out, mut m) in self.best.iter_mut().zip(bank.mappings).take(bank.len) {
            m.alternatives = bank.alternatives(m.endpoint);
            *out = m;
        }
    }
    /// A bounded aligned seed, also used by private definition tests.
    pub fn endpoint(&mut self, g: &Graph, q: &[Observation<'_>], end: &Anchor) -> f64 {
        if q.is_empty() {
            return 0.;
        }
        let q = &q[q.len().saturating_sub(SMAX)..];
        let episode = g.node(end.node).unwrap().episode;
        let mut history = [None; SMAX];
        let mut count = 0;
        for offset in 0..q.len() {
            let Some(anchor) = end
                .id
                .checked_sub(offset as u64)
                .and_then(|id| g.anchor(id))
            else {
                break;
            };
            if !anchor.learned || g.node(anchor.node).unwrap().episode != episode {
                break;
            }
            history[count] = Some(anchor);
            count += 1;
        }
        history[..count].reverse();
        let (bindings, sources) = source_bindings(q.last().unwrap(), &g.observation(end));
        let mut bank = Cell::default();
        for &(qs, hs) in &bindings[..sources] {
            for transformed in [false, true] {
                if transformed && (qs.is_none() || count < 2) {
                    continue;
                }
                let mut m = Mapping {
                    query_source: qs,
                    history_source: hs,
                    transformed,
                    time_scale: if transformed {
                        Value::default()
                    } else {
                        Value::exact(1.)
                    },
                    historical_start: history[0].unwrap().id,
                    ..Mapping::default()
                };
                let offset = q.len() - count;
                m.omitted_query = ((1_u32 << offset) - 1) as u16;
                m.kernel.requested = offset * (4 + usize::from(qs.is_some()) * 4);
                let mut valid = true;
                for (j, anchor) in history[..count].iter().enumerate() {
                    if !append(g, &q[offset + j], offset + j, anchor.unwrap(), &mut m) {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    bank.insert(m);
                } else {
                    bank.propose(end.id);
                }
            }
        }
        let mut likelihood = 0.;
        for mut m in bank.mappings[..bank.len].iter().copied() {
            m.alternatives = bank.count.max(1.);
            likelihood += m.compatibility_weight / m.alternatives;
            let pos = self.best[..self.len].partition_point(|p| better(p, &m));
            if pos < HMAX {
                self.len = (self.len + 1).min(HMAX);
                for k in (pos + 1..self.len).rev() {
                    self.best[k] = self.best[k - 1];
                }
                self.best[pos] = m;
            }
        }
        likelihood
    }
    pub fn publish(
        &mut self,
        g: &Graph,
        q: &[Observation<'_>],
        version: u64,
        query_version: u64,
        p: &mut Prediction,
    ) {
        p.len = 0;
        p.unknown = 1.;
        p.model_version = version;
        p.query_version = query_version;
        p.issued_at = q.last().unwrap().time;
        p.coordinate_version = q.last().unwrap().acoustic.coordinate_version;
        p.searched_endpoints = self.endpoints;
        p.variances = g.statistics.map(Moments::variance);
        p.shape_variance = g.shape_variance();
        p.arrival_variances = [0.; 4];
        p.arrival_shape_variance = 0.;
        p.arrival_pitch_variances.clear();
        p.arrival_expires_at = g.nodes.front().map_or(f64::INFINITY, |n| n.start + TMAX);
        if self.endpoints == 0 {
            return;
        }
        for m in self.best[..self.len].iter().copied() {
            let Some(end) = g.anchor(m.endpoint) else {
                continue;
            };
            let Some(successor) = g.successor(end) else {
                continue;
            };
            if m.kernel.valid == 0 || m.compatibility_weight == 0. {
                continue;
            }
            let observed = g.observation(successor);
            let historical = g.observation(end);
            let current = q[usize::from(m.pairs[m.len - 1].query)];
            let a = &mut p.atoms[p.len];
            a.bins.clear();
            a.trajectories.clear();
            a.absolute_trajectories.clear();
            a.absolute_trajectories
                .extend_from_slice(observed.trajectories);
            a.shape_valid = observed.support.valid;
            a.mapping = m;
            a.successor = successor.id;
            a.source_version = version;
            a.acoustic = observed.acoustic;
            a.absolute_acoustic = observed.acoustic;
            a.absolute_time = observed.time;
            let wait = observed.time - historical.time;
            a.time = Value {
                value: current.time + wait * m.time_scale.value,
                error: current.timing_error
                    + (observed.timing_error + historical.timing_error) * m.time_scale.value
                    + wait * m.time_scale.error,
                valid: true,
                bound: true,
            };
            a.residual = if observed.support.valid {
                observed.support.residual
            } else {
                0.
            };
            let min_step = g
                .load
                .coordinates
                .windows(2)
                .map(|w| w[1] - w[0])
                .fold(f64::INFINITY, f64::min);
            a.shape_error =
                observed.support.error + (m.translation.error / min_step).min(1.).sqrt();
            a.shape_bound = observed.support.bound || m.transformed;
            self.transported.fill(0.);
            if observed.support.valid {
                for bin in observed.support.bins {
                    let x = g.load.coordinates[bin.index] + m.translation.value;
                    let right = g.load.coordinates.partition_point(|v| *v < x);
                    if right < g.load.coordinates.len() && g.load.coordinates[right] == x {
                        self.transported[right] += f64::from(bin.mass);
                    } else if right == 0 || right == g.load.coordinates.len() {
                        a.residual += f64::from(bin.mass);
                    } else {
                        let left = right - 1;
                        let t = (x - g.load.coordinates[left])
                            / (g.load.coordinates[right] - g.load.coordinates[left]);
                        self.transported[left] += f64::from(bin.mass) * (1. - t);
                        self.transported[right] += f64::from(bin.mass) * t;
                    }
                }
            }
            for (i, mass) in self.transported.iter().copied().enumerate() {
                if mass > 0. {
                    a.bins.push(Bin {
                        index: i,
                        mass: mass as f32,
                    });
                }
            }
            if a.residual > 0. && a.residual < 1. {
                let retained = a.bins.iter().map(|b| f64::from(b.mass)).sum::<f64>();
                if retained > 0. {
                    for b in &mut a.bins {
                        b.mass = (f64::from(b.mass) / retained) as f32;
                    }
                }
            }
            if m.transformed {
                a.acoustic.focus.valid = false;
                a.acoustic.competition.valid = false;
                // Template mass is not certified invariant under the transformation either.
                a.acoustic.mass.valid = false;
            }
            for mut t in observed.trajectories.iter().copied() {
                if m.history_source.is_some_and(|s| s != t.source) {
                    continue;
                }
                if let Some(s) = m.query_source {
                    t.source = s;
                }
                t.pitch.value += m.translation.value;
                t.pitch.error += m.translation.error;
                if m.transformed {
                    t.timbre.valid = false;
                }
                a.trajectories.push(t);
            }
            // Lost support mass is explicitly unknown in the predictive mixture.
            a.weight = m.compatibility_weight / m.alternatives / self.endpoints as f64
                * (1. - a.residual).clamp(0., 1.);
            p.unknown -= a.weight;
            p.len += 1;
        }
        p.unknown = p.unknown.clamp(0., 1.);
        p.arrival_uncertainty(q.last().unwrap().trajectories);
    }
}

impl Prediction {
    fn arrival_uncertainty(&mut self, current: &[Trajectory]) {
        let atoms = &self.atoms[..self.len];
        for (i, variance) in self.arrival_variances.iter_mut().enumerate() {
            *variance = conditional_variance(atoms, |a, b| {
                let x = [
                    a.acoustic.mass,
                    a.acoustic.focus,
                    a.acoustic.competition,
                    a.time,
                ][i];
                let y = [
                    b.acoustic.mass,
                    b.acoustic.focus,
                    b.acoustic.competition,
                    b.time,
                ][i];
                (x.valid && y.valid).then_some((x.value - y.value).powi(2))
            });
        }
        self.arrival_shape_variance = conditional_variance(atoms, |a, b| {
            (a.shape_valid && b.shape_valid && a.residual == 0. && b.residual == 0.)
                .then(|| hellinger(&a.bins, &b.bins))
        });
        self.arrival_pitch_variances.clear();
        for trajectory in current {
            let source = trajectory.source;
            let variance = conditional_variance(atoms, |a, b| {
                let x = a.trajectories.iter().find(|t| t.source == source)?.pitch;
                let y = b.trajectories.iter().find(|t| t.source == source)?.pitch;
                (x.valid && y.valid).then_some((x.value - y.value).powi(2))
            });
            self.arrival_pitch_variances.push((source, variance));
        }
    }
}

// Pairwise sample variance of the conditional empirical successors. Alternative
// mappings of one observation cannot supply a second independent sample.
fn conditional_variance(atoms: &[Atom], distance: impl Fn(&Atom, &Atom) -> Option<f64>) -> f64 {
    let scale = atoms.iter().map(|a| a.weight).fold(0., f64::max);
    if scale == 0. {
        return 0.;
    }
    let (mut squared_residual, mut weight) = (0., 0.);
    for (i, a) in atoms.iter().enumerate() {
        for b in &atoms[..i] {
            if a.successor == b.successor {
                continue;
            }
            if let Some(d) = distance(a, b) {
                let w = (a.weight / scale) * (b.weight / scale);
                squared_residual += w * d;
                weight += w;
            }
        }
    }
    if weight > 0. {
        squared_residual / (2. * weight)
    } else {
        0.
    }
}
// A source interval can continue across another source's structural event.
fn types(o: &Observation<'_>, source: Option<Source>) -> u8 {
    o.events
        .iter()
        .filter(|e| source.is_none() || e.source.is_none() || e.source == source)
        .fold(
            if source.is_some_and(|s| o.trajectories.iter().any(|t| t.source == s)) {
                16
            } else {
                0
            },
            |a, e| a | e.flags,
        )
}
fn trajectory<'a>(o: &Observation<'a>, source: Option<Source>) -> Option<&'a Trajectory> {
    o.trajectories.iter().find(|t| Some(t.source) == source)
}
#[derive(Clone, Copy, Debug, Default)]
struct Fit {
    weighted: f64,
    precision: f64,
    exact: Option<f64>,
    conflict: bool,
    bounded: bool,
    error_sum: f64,
}
impl Fit {
    fn push(&mut self, value: f64, error: f64, bound: bool) {
        self.bounded |= bound;
        if error == 0. {
            self.conflict |= self.exact.is_some_and(|old| old != value);
            self.exact = Some(value);
        } else {
            let weight = 1. / error.powi(2);
            self.weighted += weight * value;
            self.precision += weight;
            self.error_sum += weight * error;
        }
    }
    fn value(self) -> Value {
        if self.conflict {
            Value::default()
        } else if let Some(value) = self.exact {
            Value::exact(value)
        } else if self.precision > 0. {
            Value {
                value: self.weighted / self.precision,
                error: if self.bounded {
                    self.error_sum / self.precision
                } else {
                    self.precision.sqrt().recip()
                },
                valid: true,
                bound: self.bounded,
            }
        } else {
            Value::default()
        }
    }
}

fn source_bindings(
    a: &Observation<'_>,
    b: &Observation<'_>,
) -> ([(Option<Source>, Option<Source>); HMAX], usize) {
    let mut bindings = [(None, None); HMAX];
    let mut len = 1;
    for query in a.trajectories.iter().take(HMAX - 1) {
        let same = b.trajectories.iter().find(|t| t.source == query.source);
        let nearest = b
            .trajectories
            .iter()
            .filter(|t| query.pitch.valid && t.pitch.valid)
            .min_by(|x, y| {
                (x.pitch.value - query.pitch.value)
                    .abs()
                    .total_cmp(&(y.pitch.value - query.pitch.value).abs())
                    .then(x.source.cmp(&y.source))
            });
        for historical in [same, nearest].into_iter().flatten() {
            let pair = (Some(query.source), Some(historical.source));
            if bindings[..len].contains(&pair) {
                continue;
            }
            if len == HMAX {
                return (bindings, len);
            }
            bindings[len] = pair;
            len += 1;
        }
    }
    (bindings, len)
}
fn acoustic_terms(g: &Graph, a: &Observation<'_>, b: &Observation<'_>) -> [Kernel; 4] {
    let mut terms = [Kernel::default(); 4];
    let same = a.acoustic.coordinate_version == b.acoustic.coordinate_version;
    terms[0].requested = 1;
    if same
        && a.support.valid
        && b.support.valid
        && a.support.residual == 0.
        && b.support.residual == 0.
    {
        terms[0].distance(
            hellinger(a.support.bins, b.support.bins),
            a.support.error.powi(2) + b.support.error.powi(2) + g.shape_variance(),
        );
    }
    for (i, (x, mut y)) in [
        (a.acoustic.mass, b.acoustic.mass),
        (a.acoustic.focus, b.acoustic.focus),
        (a.acoustic.competition, b.acoustic.competition),
    ]
    .into_iter()
    .enumerate()
    {
        y.valid &= same;
        terms[i + 1].scalar(x, y, g.statistics[i].variance());
    }
    terms
}

fn append(g: &Graph, a: &Observation<'_>, index: usize, end: &Anchor, m: &mut Mapping) -> bool {
    if m.len == SMAX {
        return false;
    }
    let b = g.observation(end);
    if types(a, m.query_source) & types(&b, m.history_source) == 0 {
        return false;
    }
    let previous = if m.len > 0 {
        let Some(anchor) = g.anchor(m.endpoint) else {
            return false;
        };
        Some(g.observation(anchor))
    } else {
        None
    };
    let x = trajectory(a, m.query_source).copied();
    let y = trajectory(&b, m.history_source).copied();
    if m.transformed {
        if let (Some(x), Some(y)) = (x, y)
            && x.pitch.valid
            && y.pitch.valid
        {
            m.translation_fit.push(
                x.pitch.value - y.pitch.value,
                if x.pitch.bound || y.pitch.bound {
                    x.pitch.error + y.pitch.error
                } else {
                    x.pitch.error.hypot(y.pitch.error)
                },
                x.pitch.bound || y.pitch.bound,
            );
        }
        m.translation = m.translation_fit.value();
        if let Some(previous) = previous {
            let dx = a.time - m.previous_time;
            let dy = b.time - previous.time;
            if dx > 0. && dy > 0. {
                m.time_fit.push(
                    (dx / dy).ln(),
                    (a.timing_error.hypot(m.previous_timing_error) / dx)
                        .hypot(b.timing_error.hypot(previous.timing_error) / dy),
                    false,
                );
            }
            let time = m.time_fit.value();
            m.time_scale = Value {
                value: time.value.exp(),
                error: time.error * time.value.exp(),
                valid: time.valid,
                bound: time.bound,
            };
        }
    }
    // Only the newly observed pair and edge are scored, using the frozen prefix fit.
    let mut kernel = m.kernel;
    if m.transformed {
        kernel.requested += 4;
    } else {
        for term in acoustic_terms(g, a, &b) {
            kernel.requested += term.requested;
            kernel.valid += term.valid;
            kernel.sum += term.sum;
        }
    }
    if m.query_source.is_some() {
        if let (Some(x), Some(y)) = (x, y) {
            let mut pitch = y.pitch;
            pitch.value += m.translation.value;
            pitch.error += m.translation.error;
            pitch.valid &= m.translation.valid;
            kernel.scalar(x.pitch, pitch, g.statistics[5].variance());
            kernel.scalar(x.amplitude, y.amplitude, g.statistics[8].variance());
            kernel.scalar(
                x.timbre,
                if m.transformed {
                    Value::default()
                } else {
                    y.timbre
                },
                g.statistics[7].variance(),
            );
            kernel.scalar(
                Value::exact(f64::from(x.sounding)),
                Value::exact(f64::from(y.sounding)),
                0.,
            );
            if let Some(previous) = previous {
                if let (Some(px), Some(py)) = (
                    m.previous_trajectory,
                    trajectory(&previous, m.history_source),
                ) {
                    kernel.scalar(
                        x.pitch.difference(px.pitch),
                        y.pitch.difference(py.pitch),
                        g.statistics[4].variance(),
                    );
                    kernel.scalar(
                        Value::exact(f64::from(x.articulation != px.articulation)),
                        Value::exact(f64::from(y.articulation != py.articulation)),
                        0.,
                    );
                    let dx = a.time - m.previous_time;
                    let dy = b.time - previous.time;
                    if dx > 0. && dy > 0. && m.time_scale.valid {
                        let mut xs = x.pitch.difference(px.pitch);
                        let mut ys = y.pitch.difference(py.pitch);
                        xs.value /= dx;
                        xs.error /= dx;
                        ys.value /= dy * m.time_scale.value;
                        ys.error /= dy * m.time_scale.value;
                        kernel.scalar(xs, ys, g.statistics[6].variance());
                    } else {
                        kernel.requested += 1;
                    }
                } else {
                    kernel.requested += 3;
                }
            }
        } else {
            kernel.requested += if previous.is_none() { 4 } else { 7 };
        }
    }
    if let Some(previous) = previous {
        kernel.scalar(
            Value {
                value: a.time - m.previous_time,
                error: a.timing_error.hypot(m.previous_timing_error),
                valid: true,
                bound: false,
            },
            Value {
                value: (b.time - previous.time) * m.time_scale.value,
                error: b.timing_error.hypot(previous.timing_error) * m.time_scale.value
                    + (b.time - previous.time) * m.time_scale.error,
                valid: m.time_scale.valid,
                bound: true,
            },
            g.statistics[3].variance(),
        );
    }
    m.endpoint = end.id;
    m.source_episode = g.node(end.node).unwrap().episode;
    m.pairs[m.len] = Pair {
        query: index as u8,
        history: end
            .id
            .saturating_sub(m.historical_start)
            .min(u8::MAX as u64) as u8,
    };
    m.historical_nodes[m.len] = end.node;
    m.current_snapshots[m.len] = a.acoustic.snapshot;
    m.query_times[m.len] = a.time.to_bits();
    m.history_paths[m.len] = (m.len > 0).then(|| (m.historical_nodes[m.len - 1], end.node));
    m.len += 1;
    m.previous_time = a.time;
    m.previous_timing_error = a.timing_error;
    m.previous_trajectory = x;
    m.kernel = kernel;
    m.compatibility_weight = if m.transformed && (!m.translation.valid || !m.time_scale.valid) {
        0.
    } else {
        kernel.weight()
    };
    true
}
pub(super) fn hellinger(a: &[Bin], b: &[Bin]) -> f64 {
    let (mut i, mut j, mut sum) = (0, 0, 0.);
    let am = a.iter().map(|b| f64::from(b.mass)).sum::<f64>();
    let bm = b.iter().map(|b| f64::from(b.mass)).sum::<f64>();
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i].index < b[j].index) {
            sum += f64::from(a[i].mass) / am;
            i += 1;
        } else if i == a.len() || b[j].index < a[i].index {
            sum += f64::from(b[j].mass) / bm;
            j += 1;
        } else {
            sum +=
                ((f64::from(a[i].mass) / am).sqrt() - (f64::from(b[j].mass) / bm).sqrt()).powi(2);
            i += 1;
            j += 1;
        }
    }
    0.5 * sum
}

#[cfg(test)]
#[test]
fn incremental_queries_reuse_scored_prefix_and_match_bounded_replay() {
    let mut model = super::tests::model();
    let inputs: Vec<_> = (0..12)
        .map(|i| super::tests::Input::new(i as f64, i % 4))
        .collect();
    for input in &inputs[..8] {
        model.observe(input.observation()).unwrap();
    }
    let query: Vec<_> = inputs[8..].iter().map(|i| i.observation()).collect();
    let mut incremental = Matcher::new();
    let mut replay = Matcher::new();
    for n in 1..=query.len() {
        incremental.search(&model.graph, &query[..n], true);
    }
    replay.search(&model.graph, &query, true);
    assert_eq!(incremental.extensions, query.len());
    assert_eq!(replay.extensions, query.len());
    assert_eq!(incremental.len, replay.len);
    assert!(incremental.len > 0 && incremental.len <= HMAX);
    for mapping in &incremental.best[..incremental.len] {
        let retained_here = incremental.best[..incremental.len]
            .iter()
            .filter(|m| m.endpoint == mapping.endpoint)
            .count();
        assert!(retained_here as f64 <= mapping.alternatives);
        assert!((0. ..=1.).contains(&mapping.compatibility_weight));
    }
    for (a, b) in incremental.best[..incremental.len]
        .iter()
        .zip(&replay.best[..replay.len])
    {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
    let before = format!("{:?}", incremental.best);
    incremental.search(&model.graph, &query, true);
    assert_eq!(incremental.extensions, query.len());
    assert_eq!(before, format!("{:?}", incremental.best));
}

#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn conditional_residual_variance_counts_observations_not_mappings() {
        let mut p = Prediction::new(2, 0);
        p.len = 3;
        let values = [1., 2., 6.];
        let mut moments = Moments::default();
        for (i, a) in p.atoms[..3].iter_mut().enumerate() {
            a.successor = i as u64;
            a.weight = 0.1;
            a.time = Value::exact(values[i]);
            moments.add(values[i]);
        }
        p.arrival_uncertainty(&[]);
        let expected = moments.variance();
        assert!((p.arrival_variances[3] - expected).abs() < 1e-14);
        assert_eq!(p.arrival_variances[..3], [0.; 3]);
        // Splitting one empirical observation into equivalent mappings adds no data.
        p.atoms[0].weight *= 0.5;
        p.atoms[3] = p.atoms[0].clone();
        p.len = 4;
        p.arrival_uncertainty(&[]);
        assert!((p.arrival_variances[3] - expected).abs() < 1e-14);
        p.atoms[1].weight = 0.;
        p.atoms[2].weight = 0.;
        p.arrival_uncertainty(&[]);
        assert_eq!(p.arrival_variances[3], 0.);
    }

    #[test]
    fn conditional_shape_uses_embedding_dispersion_and_preserves_unknown() {
        let mut p = Prediction::new(2, 0);
        p.len = 2;
        p.unknown = 0.7;
        for (i, a) in p.atoms[..2].iter_mut().enumerate() {
            a.successor = i as u64;
            a.weight = 0.15;
            a.shape_valid = true;
            a.bins.push(Bin { index: i, mass: 1. });
        }
        p.arrival_uncertainty(&[]);
        // Orthogonal unit embeddings: half the pairwise Hellinger square is 1/2.
        assert_eq!(p.arrival_shape_variance, 0.5);
        assert_eq!(p.unknown, 0.7);
        p.atoms[1].residual = 0.1;
        p.arrival_uncertainty(&[]);
        assert_eq!(p.arrival_shape_variance, 0.);
    }
}
