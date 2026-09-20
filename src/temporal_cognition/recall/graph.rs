//! Heard-episode correspondence graph with original query support and bounded edges.

use super::super::section::commitment::{ContextSupport, Evidence};
use super::*;

const EDGES: usize = 16;
const PENDING: usize = 256;
// One completion can arrive before the current phrase inventory is reconciled.
const OWNERS: usize = super::super::section::cue::MAX_RETAINED_PREFIXES + 1;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Edge {
    pub target: u64,
    pub target_generation: u64,
    pub query_id: u64,
    pub query_start: u64,
    pub query_end: u64,
    pub received_at: u64,
    pub target_end: u64,
    pub target_available: u64,
    pub cost: f64,
    pub transformation: [Option<f64>; 2],
    pub ambiguous: bool,
}

impl Edge {
    // Retain the strongest witnessed subspan; a later cue need not match its continuation.
    fn quality_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ambiguous
            .cmp(&other.ambiguous)
            .then(self.cost.total_cmp(&other.cost))
            .then_with(|| {
                (other.query_end - other.query_start).cmp(&(self.query_end - self.query_start))
            })
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Source {
    pub episode: u64,
    pub generation: u64,
    pub occurrence: u64,
    pub credit: u64,
    pub group: Handle,
    pub start: u64,
    pub end: u64,
    pub available: u64,
    pub support: f64,
    pub contexts: [Option<ContextSupport>; 16],
}

struct Node {
    source: Source,
    edges: Box<[Option<Edge>]>,
}

#[derive(serde::Serialize)]
pub(super) struct PendingQuery {
    group: Handle,
    occurrence: u64,
    start: u64,
    end: u64,
    edges: Box<[Option<Edge>]>,
    attached: bool,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Snapshot {
    pub end_sample: u64,
    pub nodes: usize,
    pub edges: usize,
    pub pending_queries: usize,
    pub retained_prefixes: usize,
    pub prefix_edges: usize,
    pub prefix_metadata_updates: u64,
    pub pruned_prefix_edges: u64,
    pub lost_prefixes: u64,
    pub retired_unattached_prefixes: u64,
    pub metadata_updates: u64,
    pub pruned_edges: u64,
    pub lost_pending_queries: u64,
    pub expired_unattached_queries: u64,
    pub retired_edges: u64,
    pub allocated_bytes: usize,
    pub edge_capacity: usize,
    pub latest_source: Option<Source>,
    pub latest_edges: [Option<Edge>; EDGES],
}

pub(super) struct Graph {
    bus: u8,
    epoch: u64,
    rate: u32,
    capacity: usize,
    edge_capacity: usize,
    free_edges: Vec<Box<[Option<Edge>]>>,
    recent_keys: Vec<(u64, u64, u64)>,
    nodes: Vec<Node>,
    pending: Vec<PendingQuery>,
    owners: Vec<PendingQuery>,
    summary: Snapshot,
}

// An edge supports its recorded subspan, never an unobserved remainder of a source.
fn attach(
    node: &mut Node,
    pending: &PendingQuery,
    summary: &mut Snapshot,
    recent: &[(u64, u64, u64)],
) -> bool {
    let s = node.source;
    if s.group != pending.group
        || (s.credit != pending.occurrence && s.occurrence != pending.occurrence)
    {
        return false;
    }
    let mut used = false;
    for &edge in pending
        .edges
        .iter()
        .flatten()
        .filter(|e| s.start <= e.query_start && e.query_end <= s.end && e.target != s.episode)
        .filter(|e| {
            recent
                .binary_search(&(e.query_id, e.target, e.target_generation))
                .is_err()
        })
    {
        used = true;
        if let Some(old) = node
            .edges
            .iter_mut()
            .flatten()
            .find(|e| e.target == edge.target && e.target_generation == edge.target_generation)
        {
            if edge.quality_cmp(old).is_lt() {
                *old = edge;
                summary.metadata_updates += 1;
            }
        } else if let Some(slot) = node.edges.iter_mut().find(|e| e.is_none()) {
            *slot = Some(edge);
            summary.edges += 1;
            summary.metadata_updates += 1;
        } else {
            let worst = node
                .edges
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.unwrap().quality_cmp(&b.unwrap()))
                .unwrap()
                .0;
            summary.pruned_edges += 1;
            if edge.quality_cmp(&node.edges[worst].unwrap()).is_lt() {
                node.edges[worst] = Some(edge);
                summary.metadata_updates += 1;
            }
        }
    }
    used
}

impl Graph {
    pub fn new(bus: u8, epoch: u64, rate: u32, capacity: usize, edge_capacity: usize) -> Self {
        assert!((1..=memory::MAX_CANDIDATES).contains(&edge_capacity));
        let buffers = PENDING + OWNERS + 2;
        let mut free_edges = Vec::with_capacity(capacity + buffers);
        free_edges.extend((0..buffers).map(|_| vec![None; edge_capacity].into_boxed_slice()));
        Self {
            bus,
            epoch,
            rate,
            capacity,
            edge_capacity,
            free_edges,
            recent_keys: Vec::with_capacity(PENDING * edge_capacity),
            nodes: Vec::with_capacity(capacity),
            pending: Vec::with_capacity(PENDING),
            owners: Vec::with_capacity(OWNERS),
            summary: Snapshot {
                allocated_bytes: std::mem::size_of::<Self>()
                    + capacity * std::mem::size_of::<Node>()
                    + (PENDING + OWNERS) * std::mem::size_of::<PendingQuery>()
                    + (capacity + buffers) * std::mem::size_of::<Box<[Option<Edge>]>>()
                    + PENDING * edge_capacity * std::mem::size_of::<(u64, u64, u64)>()
                    + buffers * edge_capacity * std::mem::size_of::<Option<Edge>>(),
                edge_capacity,
                ..Snapshot::default()
            },
        }
    }

    fn take_edges(&mut self) -> Box<[Option<Edge>]> {
        if let Some(mut edges) = self.free_edges.pop() {
            edges.fill(None);
            edges
        } else {
            self.summary.allocated_bytes +=
                self.edge_capacity * std::mem::size_of::<Option<Edge>>();
            vec![None; self.edge_capacity].into_boxed_slice()
        }
    }

    pub fn register(
        &mut self,
        id: transport::Identity,
        e: Evidence,
        cut: u64,
    ) -> Result<(), &'static str> {
        if e.group.epoch != self.epoch
            || e.group.bus != self.bus
            || e.start >= e.end
            || e.end > e.sealed_at
            || e.sealed_at > cut
            || cut < self.summary.end_sample
            || self.nodes.len() >= self.capacity
            || self.nodes.last().is_some_and(|n| n.source.episode >= id.id)
        {
            return Err(
                "graph requires ordered retained episode identities and sealed original support",
            );
        }
        let mut node = Node {
            source: Source {
                episode: id.id,
                generation: id.generation,
                occurrence: e.occurrence_id,
                credit: e.ongoing_credit,
                group: e.group,
                start: e.start,
                end: e.end,
                available: e.sealed_at,
                support: e.support,
                contexts: e.contexts,
            },
            edges: self.take_edges(),
        };
        for pending in &mut self.pending {
            pending.attached |= attach(&mut node, pending, &mut self.summary, &[]);
        }
        let source = node.source;
        self.recent_keys.clear();
        self.recent_keys.extend(
            self.pending
                .iter()
                .filter(|p| {
                    p.group == source.group
                        && (p.occurrence == source.credit || p.occurrence == source.occurrence)
                })
                .flat_map(|p| {
                    p.edges
                        .iter()
                        .flatten()
                        .map(|e| (e.query_id, e.target, e.target_generation))
                }),
        );
        self.recent_keys.sort_unstable();
        self.recent_keys.dedup();
        for owner in self.owners.iter_mut().filter(|owner| {
            owner.group == source.group
                && owner.start == source.start
                && (owner.occurrence == source.credit || owner.occurrence == source.occurrence)
        }) {
            owner.attached |= self.pending.iter().any(|p| {
                p.group == owner.group
                    && p.occurrence == owner.occurrence
                    && p.start == owner.start
                    && p.attached
            });
            // Recent receipts already supplied the exact endpoint-specific evidence above.
            owner.attached |= attach(&mut node, owner, &mut self.summary, &self.recent_keys);
        }
        self.summary.end_sample = cut;
        self.summary.latest_source = Some(node.source);
        self.summary.latest_edges = std::array::from_fn(|i| node.edges.get(i).copied().flatten());
        self.nodes.push(node);
        self.summary.nodes = self.nodes.len();
        Ok(())
    }

    pub fn receive(
        &mut self,
        q: ResultSnapshot,
        matches: &[[Option<MatchSnapshot>; 4]],
    ) -> Result<(), &'static str> {
        let Some(cue) = q.cue else {
            return Ok(());
        };
        if q.group.epoch != self.epoch
            || matches.len() > self.edge_capacity
            || q.group.bus != self.bus
            || q.received_at < self.summary.end_sample
            || !(q.support_start_sample < q.support_end_sample
                && q.support_end_sample <= q.available_at
                && q.available_at <= q.issued_at
                && q.issued_at <= q.completed_at
                && q.completed_at <= q.received_at
                && q.received_at <= q.deadline)
            || cue.start_sample != q.support_start_sample
            || cue.support_end_sample != q.support_end_sample
        {
            return Err("graph requires the original completed query support and epoch");
        }
        let mut pending = PendingQuery {
            group: q.group,
            occurrence: cue.occurrence_id,
            start: cue.start_sample,
            end: q.support_end_sample,
            edges: self.take_edges(),
            attached: false,
        };
        for (slot, row) in matches.iter().enumerate() {
            let Some(m) = row
                .iter()
                .flatten()
                .filter(|m| m.cost.is_finite())
                // A band-limited alternative cannot displace a supported refinement.
                .min_by(|a, b| {
                    a.ambiguous
                        .cmp(&b.ambiguous)
                        .then(a.cost.total_cmp(&b.cost))
                })
            else {
                continue;
            };
            let Ok(target) = self
                .nodes
                .binary_search_by_key(&m.episode_id, |n| n.source.episode)
            else {
                continue;
            };
            if m.available_at > q.issued_at
                || m.support_end_sample > q.support_start_sample
                || self.nodes[target].source.end != m.support_end_sample
                || self.nodes[target].source.generation != m.episode_generation
            {
                self.free_edges.push(pending.edges);
                return Err("graph target must be retained earlier nonoverlapping heard material");
            }
            pending.edges[slot] = Some(Edge {
                target: m.episode_id,
                target_generation: m.episode_generation,
                query_id: q.query_id,
                query_start: q.support_start_sample,
                query_end: q.support_end_sample,
                received_at: q.received_at,
                target_end: m.support_end_sample,
                target_available: m.available_at,
                cost: m.cost,
                transformation: m.transformation,
                ambiguous: m.ambiguous || q.cutoff_tie,
            });
        }
        for node in &mut self.nodes {
            if attach(node, &pending, &mut self.summary, &[]) {
                pending.attached = true;
                self.summary.latest_source = Some(node.source);
                self.summary.latest_edges =
                    std::array::from_fn(|i| node.edges.get(i).copied().flatten());
            }
        }
        if pending.edges.iter().any(Option::is_some) {
            if let Some(owner) = self.owners.iter_mut().find(|p| {
                p.group == pending.group
                    && p.occurrence == pending.occurrence
                    && p.start == pending.start
            }) {
                owner.end = owner.end.max(pending.end);
                owner.attached |= pending.attached;
                for edge in pending.edges.iter().flatten() {
                    if let Some(old) = owner.edges.iter_mut().flatten().find(|old| {
                        old.target == edge.target && old.target_generation == edge.target_generation
                    }) {
                        if edge.quality_cmp(old).is_lt() {
                            *old = *edge;
                            self.summary.prefix_metadata_updates += 1;
                        }
                    } else if let Some(slot) = owner.edges.iter_mut().find(|e| e.is_none()) {
                        *slot = Some(*edge);
                        self.summary.prefix_metadata_updates += 1;
                    } else {
                        self.summary.pruned_prefix_edges += 1;
                        let worst = owner
                            .edges
                            .iter_mut()
                            .flatten()
                            .max_by(|a, b| a.quality_cmp(b))
                            .unwrap();
                        if edge.quality_cmp(worst).is_lt() {
                            *worst = *edge;
                            self.summary.prefix_metadata_updates += 1;
                        }
                    }
                }
            } else if self.owners.len() < OWNERS {
                let mut edges = self.take_edges();
                edges.copy_from_slice(&pending.edges);
                self.owners.push(PendingQuery {
                    group: pending.group,
                    occurrence: pending.occurrence,
                    start: pending.start,
                    end: pending.end,
                    attached: pending.attached,
                    edges,
                });
                self.summary.prefix_metadata_updates +=
                    pending.edges.iter().flatten().count() as u64;
            } else {
                self.summary.lost_prefixes += 1;
            }
        }
        if self.pending.len() == PENDING {
            let old = self.pending.remove(0);
            self.summary.lost_pending_queries += u64::from(!old.attached);
            self.free_edges.push(old.edges);
        }
        self.pending.push(pending);
        self.summary.end_sample = q.received_at;
        self.summary.pending_queries = self.pending.len();
        Ok(())
    }

    pub fn retain_sources(
        &mut self,
        cut: u64,
        mut retained: impl FnMut(Handle, u64, u64) -> bool,
    ) -> Result<(), &'static str> {
        if cut < self.summary.end_sample {
            return Err("graph source ownership requires an ordered observation cut");
        }
        self.owners.retain_mut(|p| {
            let keep = retained(p.group, p.occurrence, p.start);
            if !keep && !p.attached {
                self.summary.retired_unattached_prefixes += 1;
            }
            if !keep {
                self.free_edges
                    .push(std::mem::replace(&mut p.edges, Box::new([])));
            }
            keep
        });
        self.pending.retain_mut(|p| {
            let keep = p.end.saturating_add(2 * u64::from(self.rate)) >= cut;
            if !keep && !p.attached {
                self.summary.expired_unattached_queries += 1;
            }
            if !keep {
                self.free_edges
                    .push(std::mem::replace(&mut p.edges, Box::new([])));
            }
            keep
        });
        self.summary.pending_queries = self.pending.len();
        self.summary.end_sample = cut;
        Ok(())
    }

    pub fn retire(&mut self, id: transport::Identity) {
        if let Ok(i) = self
            .nodes
            .binary_search_by_key(&id.id, |n| n.source.episode)
        {
            if self.nodes[i].source.generation != id.generation {
                return;
            }
            let old = self.nodes.remove(i);
            let count = old.edges.iter().flatten().count();
            self.summary.edges -= count;
            self.summary.retired_edges += count as u64;
            self.free_edges.push(old.edges);
        }
        for n in &mut self.nodes {
            for e in &mut n.edges {
                if e.is_some_and(|e| e.target == id.id && e.target_generation == id.generation) {
                    *e = None;
                    self.summary.edges -= 1;
                    self.summary.retired_edges += 1;
                }
            }
        }
        for p in self.pending.iter_mut().chain(&mut self.owners) {
            for e in &mut p.edges {
                if e.is_some_and(|e| e.target == id.id && e.target_generation == id.generation) {
                    *e = None;
                }
            }
        }
        self.summary.nodes = self.nodes.len();
        if let Some(source) = self.summary.latest_source {
            if let Ok(i) = self
                .nodes
                .binary_search_by_key(&source.episode, |n| n.source.episode)
            {
                self.summary.latest_edges =
                    std::array::from_fn(|j| self.nodes[i].edges.get(j).copied().flatten());
            } else {
                self.summary.latest_source = None;
                self.summary.latest_edges = [None; EDGES];
            }
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            retained_prefixes: self.owners.len(),
            prefix_edges: self
                .owners
                .iter()
                .map(|p| p.edges.iter().flatten().count())
                .sum(),
            ..self.summary
        }
    }

    #[cfg(test)]
    pub(super) fn archived_nodes(&self) -> impl Iterator<Item = (&Source, &[Option<Edge>])> {
        self.nodes.iter().map(|n| (&n.source, n.edges.as_ref()))
    }

    #[cfg(test)]
    pub(super) fn archived_prefixes(&self) -> impl Iterator<Item = &PendingQuery> {
        self.owners.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: u64, credit: u64, start: u64, end: u64) -> Evidence {
        Evidence {
            occurrence_id: (1 << 63) + id,
            sequence: id,
            ongoing_credit: credit,
            group: Handle {
                bus: 1,
                epoch: 0,
                generation: 7,
            },
            start,
            end,
            deadline: end + 500,
            sealed_at: end + 500,
            assignment_seconds: (end - start) as f64 / 1000.,
            retained_mass: 0.5,
            support: 0.5,
            lag_known_seconds: 0.5,
            lag_missing_seconds: 0.,
            ending: None,
            contexts: [None; 16],
        }
    }

    fn query(id: u64, credit: u64, start: u64, end: u64, received: u64) -> ResultSnapshot {
        ResultSnapshot {
            query_id: id,
            cue: Some(super::super::super::section::cue::Selection {
                occurrence_id: credit,
                start_sample: start,
                support_end_sample: end,
                selected_at: end,
                last_observed_sample: end,
                window_start_sample: end.saturating_sub(500) as f64,
                weighted_seconds: (end - start) as f64 / 1000.,
            }),
            group: Handle {
                bus: 1,
                epoch: 0,
                generation: 7,
            },
            support_start_sample: start,
            support_end_sample: end,
            source_start_sample: start,
            supporting_audio_end: Some(end),
            available_at: end,
            issued_at: received - 50,
            completed_at: received - 50,
            received_at: received,
            deadline: received + 100,
            candidates: 1,
            search_covered: true,
            cutoff_tie: false,
            pruned_candidates: 0,
            pruned_ties: 0,
            dp_cells: 0,
            reconstruction_error: 0.,
            best: None,
            prediction: None,
        }
    }

    fn matches(id: u64, cost: f64) -> [[Option<MatchSnapshot>; 4]; 16] {
        let mut result = [[None; 4]; 16];
        result[0][0] = Some(MatchSnapshot {
            episode_id: id,
            episode_generation: id,
            support_start_sample: 10,
            support_end_sample: 100,
            source_start_sample: 10,
            available_at: 600,
            cost,
            transformation: [Some(0.); 2],
            ambiguous: false,
            path_steps: 10,
            anchor: Some(0),
            residuals: None,
        });
        result
    }

    #[test]
    fn wide_graph_retains_edges_beyond_the_preview_and_reuses_query_buffers() {
        let mut graph = Graph::new(1, 0, 1000, 128, 128);
        let mut rows = vec![[None; 4]; 80];
        for id in 1..=80 {
            graph
                .register(
                    transport::Identity { id, generation: id },
                    evidence(id, id, 10, 100),
                    600,
                )
                .unwrap();
            rows[id as usize - 1] = matches(id, id as f64 / 100.)[0];
        }
        graph
            .receive(query(1, 200, 1000, 1200, 1250), &rows)
            .unwrap();
        graph
            .register(
                transport::Identity {
                    id: 81,
                    generation: 81,
                },
                evidence(81, 200, 1000, 1400),
                1900,
            )
            .unwrap();
        assert_eq!(graph.snapshot().edges, 80);
        assert_eq!(
            graph.snapshot().latest_edges.iter().flatten().count(),
            EDGES
        );
        let node = graph.nodes.last().unwrap();
        assert_eq!(node.edges.iter().flatten().count(), 80);
        assert_eq!(node.edges[79].unwrap().target, 80);
        let before = serde_json::to_value(graph.snapshot()).unwrap();
        rows[79][0].as_mut().unwrap().episode_generation = 999;
        assert!(
            graph
                .receive(query(2, 200, 1000, 1300, 1950), &rows)
                .is_err()
        );
        assert_eq!(serde_json::to_value(graph.snapshot()).unwrap(), before);
        rows[79][0].as_mut().unwrap().episode_generation = 80;
        let allocated = graph.snapshot().allocated_bytes;
        for id in 2..50 {
            let cut = id * 3000;
            graph.retain_sources(cut, |_, _, _| false).unwrap();
            graph
                .receive(query(id, 200, 1000, 1300, cut), &rows)
                .unwrap();
            assert_eq!(graph.snapshot().allocated_bytes, allocated);
        }
        graph.retire(transport::Identity {
            id: 80,
            generation: 80,
        });
        assert_eq!(graph.snapshot().edges, 79);
        assert!(
            graph
                .nodes
                .last()
                .unwrap()
                .edges
                .iter()
                .flatten()
                .all(|e| e.target != 80)
        );
        assert_eq!(graph.snapshot().pruned_edges, 0);
    }

    #[test]
    fn later_cues_cannot_erase_a_supported_subspan_before_or_after_sealing() {
        let mut graph = Graph::new(1, 0, 1000, 8, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        graph
            .receive(query(1, 20, 1000, 1200, 1250), &matches(1, 0.2))
            .unwrap();
        let original = graph.owners[0].edges[0].unwrap();
        let mut ambiguous = matches(1, 0.01);
        ambiguous[0][0].as_mut().unwrap().ambiguous = true;
        graph
            .receive(query(2, 20, 1000, 1500, 1550), &ambiguous)
            .unwrap();
        graph
            .receive(query(3, 20, 1000, 2000, 2050), &matches(1, 2.))
            .unwrap();
        assert_eq!(graph.owners[0].edges[0].unwrap().query_id, 1);
        graph.retain_sources(4500, |_, _, _| true).unwrap();
        assert!(graph.pending.is_empty());
        graph
            .register(
                transport::Identity {
                    id: 2,
                    generation: 2,
                },
                evidence(2, 20, 1000, 5000),
                5500,
            )
            .unwrap();
        graph
            .receive(query(4, 20, 1000, 3000, 5550), &ambiguous)
            .unwrap();
        graph
            .receive(query(5, 20, 1000, 3500, 5600), &matches(1, 2.))
            .unwrap();
        let edge = graph.nodes[1].edges[0].unwrap();
        assert_eq!(
            serde_json::to_value(edge).unwrap(),
            serde_json::to_value(original).unwrap()
        );
        assert_eq!(graph.nodes[1].source.end, 5000);
        graph
            .receive(query(6, 20, 1000, 1150, 5650), &matches(1, 0.1))
            .unwrap();
        assert_eq!(graph.nodes[1].edges[0].unwrap().query_end, 1150);
        graph
            .receive(query(7, 20, 1000, 4000, 5700), &matches(1, 0.1))
            .unwrap();
        assert_eq!(graph.nodes[1].edges[0].unwrap().query_id, 7);
        assert_eq!(graph.snapshot().edges, 1);
    }

    #[test]
    fn edge_pressure_cannot_prefer_an_ambiguous_target_over_a_supported_one() {
        let mut graph = Graph::new(1, 0, 1000, 4, 1);
        for id in 1..=2 {
            graph
                .register(
                    transport::Identity { id, generation: id },
                    evidence(id, id, 10, 100),
                    600,
                )
                .unwrap();
        }
        let mut ambiguous = matches(1, 0.01);
        ambiguous[0][0].as_mut().unwrap().ambiguous = true;
        graph
            .receive(query(1, 20, 1000, 1200, 1250), &ambiguous[..1])
            .unwrap();
        graph
            .receive(query(2, 20, 1000, 1500, 1550), &matches(2, 0.2)[..1])
            .unwrap();
        assert_eq!(graph.owners[0].edges[0].unwrap().target, 2);
        graph
            .register(
                transport::Identity {
                    id: 3,
                    generation: 3,
                },
                evidence(3, 20, 1000, 2000),
                2500,
            )
            .unwrap();
        assert_eq!(graph.nodes[2].edges[0].unwrap().target, 2);
        graph
            .receive(query(3, 20, 1000, 1700, 2550), &ambiguous[..1])
            .unwrap();
        assert_eq!(graph.nodes[2].edges[0].unwrap().target, 2);
        assert_eq!(graph.owners[0].edges[0].unwrap().target, 2);
        assert_eq!(graph.snapshot().edges, 1);
        assert_eq!(graph.snapshot().pruned_edges, 2);
        assert_eq!(graph.snapshot().pruned_prefix_edges, 2);
    }

    #[test]
    fn supported_refinement_survives_a_cheaper_band_limited_alternative() {
        let mut graph = Graph::new(1, 0, 1000, 8, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        let mut alternatives = matches(1, 0.1);
        alternatives[0][0].as_mut().unwrap().ambiguous = true;
        let mut supported = matches(1, 0.2)[0][0].unwrap();
        supported.transformation = [Some(0.25), Some(0.)];
        alternatives[0][1] = Some(supported);
        graph
            .receive(query(1, 20, 1000, 1200, 1250), &alternatives)
            .unwrap();
        graph
            .register(
                transport::Identity {
                    id: 2,
                    generation: 2,
                },
                evidence(2, 20, 1000, 1400),
                1900,
            )
            .unwrap();
        let edge = graph.snapshot().latest_edges[0].unwrap();
        assert!(!edge.ambiguous);
        assert_eq!(edge.cost, 0.2);
        assert_eq!(edge.transformation, supported.transformation);
        assert_eq!(
            (edge.query_id, edge.query_start, edge.query_end),
            (1, 1000, 1200)
        );

        alternatives[0][1] = None;
        graph
            .receive(query(2, 30, 2000, 2200, 2250), &alternatives)
            .unwrap();
        graph
            .register(
                transport::Identity {
                    id: 3,
                    generation: 3,
                },
                evidence(3, 30, 2000, 2400),
                2900,
            )
            .unwrap();
        assert!(graph.snapshot().latest_edges[0].unwrap().ambiguous);
    }

    #[test]
    fn pending_query_attaches_only_to_its_original_heard_subspan_and_identity() {
        let mut graph = Graph::new(1, 0, 1000, 8, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        graph
            .receive(query(1, 20, 1000, 1200, 1250), &matches(1, 0.2))
            .unwrap();
        assert_eq!(graph.snapshot().edges, 0);
        let mut source = evidence(2, 20, 1000, 1400);
        source.support = 0.125;
        source.contexts[0] = Some(ContextSupport {
            context_id: 17,
            mass: 0.2,
        });
        graph
            .register(
                transport::Identity {
                    id: 2,
                    generation: 2,
                },
                source,
                1900,
            )
            .unwrap();
        let edge = graph.snapshot().latest_edges[0].unwrap();
        assert_eq!(
            (edge.query_start, edge.query_end, edge.received_at),
            (1000, 1200, 1250)
        );
        assert_eq!((graph.snapshot().nodes, graph.snapshot().edges), (2, 1));
        graph
            .register(
                transport::Identity {
                    id: 3,
                    generation: 3,
                },
                evidence(3, 20, 1000, 1100),
                1900,
            )
            .unwrap();
        assert!(graph.snapshot().latest_edges.iter().all(Option::is_none));
        graph
            .receive(query(2, (1 << 63) + 2, 1000, 1400, 2000), &matches(1, 0.3))
            .unwrap();
        let node = &graph.nodes[1];
        assert_eq!(
            (node.source.start, node.source.end, node.source.available),
            (1000, 1400, 1900)
        );
        assert_eq!(node.edges[0].unwrap().query_end, 1200);
        assert_eq!(node.source.support, 0.125);
        assert_eq!(node.source.contexts[0].unwrap().context_id, 17);
        assert_eq!(node.source.contexts[0].unwrap().mass, 0.2);
        assert_eq!(graph.snapshot().edges, 1);
        graph.retire(transport::Identity {
            id: 1,
            generation: 99,
        });
        assert_eq!(graph.snapshot().edges, 1);
        graph.retire(transport::Identity {
            id: 1,
            generation: 1,
        });
        assert_eq!((graph.snapshot().nodes, graph.snapshot().edges), (2, 0));
        assert!(
            graph
                .pending
                .iter()
                .chain(&graph.owners)
                .all(|p| p.edges.iter().all(Option::is_none))
        );
    }

    #[test]
    fn future_cross_bus_and_stale_generation_queries_leave_graph_unchanged() {
        let mut graph = Graph::new(1, 0, 1000, 4, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        let before = serde_json::to_value(graph.snapshot()).unwrap();
        let mut wrong = matches(1, 0.2);
        wrong[0][0].as_mut().unwrap().episode_generation = 2;
        assert!(
            graph
                .receive(query(1, 20, 1000, 1200, 1250), &wrong)
                .is_err()
        );
        let mut q = query(1, 20, 1000, 1200, 1250);
        q.group.bus = 0;
        assert!(graph.receive(q, &matches(1, 0.2)).is_err());
        let mut q = query(1, 20, 50, 80, 650);
        q.available_at = 600;
        assert!(graph.receive(q, &matches(1, 0.2)).is_err());
        assert!(
            graph
                .register(
                    transport::Identity {
                        id: 2,
                        generation: 2
                    },
                    evidence(2, 20, 1000, 1400),
                    1899
                )
                .is_err()
        );
        assert_eq!(serde_json::to_value(graph.snapshot()).unwrap(), before);
    }

    #[test]
    fn a_long_source_keeps_early_correspondence_until_its_endpoint_is_sealed() {
        let mut graph = Graph::new(1, 0, 1000, 8, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        graph
            .receive(query(1, 20, 1000, 1200, 1250), &matches(1, 0.2))
            .unwrap();
        graph
            .retain_sources(1250, |_, credit, start| credit == 20 && start == 1000)
            .unwrap();
        // Other strands keep producing queries while the original strand continues.
        graph
            .receive(query(2, 30, 28000, 29000, 29050), &matches(1, 0.3))
            .unwrap();
        graph
            .retain_sources(29050, |_, credit, start| credit == 20 && start == 1000)
            .unwrap();
        assert!(graph.pending.iter().all(|p| p.occurrence != 20));
        assert_eq!(graph.snapshot().retained_prefixes, 1);
        graph
            .register(
                transport::Identity {
                    id: 2,
                    generation: 2,
                },
                evidence(2, 20, 1000, 31000),
                31500,
            )
            .unwrap();
        let edge = graph.snapshot().latest_edges[0].expect("early correspondence was lost");
        assert_eq!(
            (edge.query_id, edge.query_start, edge.query_end),
            (1, 1000, 1200)
        );
        assert_eq!(graph.snapshot().latest_source.unwrap().end, 31000);
        graph.retain_sources(31500, |_, _, _| false).unwrap();
        assert_eq!(graph.snapshot().retained_prefixes, 0);
        assert_eq!(graph.snapshot().edges, 1);
        assert_eq!(graph.snapshot().retired_unattached_prefixes, 1);
        let before = serde_json::to_value(graph.snapshot()).unwrap();
        assert!(graph.retain_sources(31499, |_, _, _| false).is_err());
        assert_eq!(serde_json::to_value(graph.snapshot()).unwrap(), before);
    }

    #[test]
    fn prefix_pressure_is_reported_without_reallocating_or_reusing_foreign_ownership() {
        let mut graph = Graph::new(1, 0, 1000, 2, EDGES);
        graph
            .register(
                transport::Identity {
                    id: 1,
                    generation: 1,
                },
                evidence(1, 10, 10, 100),
                600,
            )
            .unwrap();
        let owner_ptr = graph.owners.as_ptr();
        for i in 0..=OWNERS as u64 {
            graph
                .receive(query(i + 1, i + 20, 1000, 1200, 1250), &matches(1, 0.2))
                .unwrap();
        }
        assert_eq!(graph.snapshot().retained_prefixes, OWNERS);
        assert_eq!(graph.snapshot().lost_prefixes, 1);
        assert_eq!(owner_ptr, graph.owners.as_ptr());
        graph
            .retain_sources(5000, |g, _, _| g.generation == 8)
            .unwrap();
        assert_eq!(graph.snapshot().retained_prefixes, 0);
        assert_eq!(graph.snapshot().pending_queries, 0);
        assert_eq!(graph.snapshot().retired_unattached_prefixes, OWNERS as u64);
        graph
            .register(
                transport::Identity {
                    id: 2,
                    generation: 2,
                },
                evidence(2, 20, 1000, 6000),
                6500,
            )
            .unwrap();
        assert_eq!(graph.snapshot().edges, 0);
        assert_eq!(owner_ptr, graph.owners.as_ptr());
    }

    #[test]
    fn edge_pressure_is_visible_and_storage_is_bounded() {
        let mut graph = Graph::new(1, 0, 1000, 32, EDGES);
        for id in 1..=17 {
            graph
                .register(
                    transport::Identity { id, generation: id },
                    evidence(id, id, 10, 100),
                    600,
                )
                .unwrap();
        }
        for id in 1..=17 {
            graph
                .receive(
                    query(id, 100, 1000, 1200, 1250),
                    &matches(id, id as f64 / 100.),
                )
                .unwrap();
        }
        graph
            .register(
                transport::Identity {
                    id: 18,
                    generation: 18,
                },
                evidence(18, 100, 1000, 1400),
                1900,
            )
            .unwrap();
        assert_eq!(graph.snapshot().edges, 16);
        assert_eq!(graph.snapshot().pruned_edges, 1);
        assert_eq!(graph.snapshot().pruned_prefix_edges, 1);
        assert_eq!(
            graph
                .snapshot()
                .latest_edges
                .iter()
                .flatten()
                .map(|e| e.target)
                .collect::<Vec<_>>(),
            (1..=16).collect::<Vec<_>>()
        );
        let node_ptr = graph.nodes.as_ptr();
        let pending_ptr = graph.pending.as_ptr();
        for i in 0..300 {
            graph
                .receive(query(100 + i, 200, 2000, 2200, 2250), &matches(1, 0.1))
                .unwrap();
        }
        assert_eq!(graph.pending.len(), PENDING);
        assert!(graph.snapshot().lost_pending_queries > 0);
        assert_eq!(node_ptr, graph.nodes.as_ptr());
        assert_eq!(pending_ptr, graph.pending.as_ptr());
    }
}
