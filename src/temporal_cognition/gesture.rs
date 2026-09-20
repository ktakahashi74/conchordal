//! Single-context, uncalibrated articulation paths and the registered pairwise gesture view.

use super::{features::RawDescriptor, proposals::frontend, ridge};
use crate::config::TemporalGestureConfig;
use serde::Serialize;
use std::collections::VecDeque;

pub(in crate::temporal_cognition) mod articulation;
mod projection;
pub(crate) use projection::Projection;
pub(in crate::temporal_cognition) use projection::ProjectionCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Attack,
    Continuation,
    Release,
    Gap,
}
const STATES: [State; 4] = [
    State::Attack,
    State::Continuation,
    State::Release,
    State::Gap,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct Evidence {
    pub start: u64,
    pub end: u64,
    pub source_start: u64,
    pub source_end: u64,
    pub available: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct Run {
    pub attack: Option<Evidence>,
    pub release: Option<Evidence>,
    pub gap: Option<Evidence>,
    pub observed_end: u64,
    pub censored: bool,
}

#[derive(Clone, Copy, Debug)]
struct Path {
    state: State,
    entered: u64,
    run: Run,
    mass: f64,
}

#[derive(Clone, Copy)]
struct Support {
    start: u64,
    end: u64,
    alpha: f64,
}

struct Group {
    handle: ridge::Handle,
    born: u64,
    end: u64,
    paths: Vec<Path>,
    scratch: Vec<Path>,
    unknown: f64,
    support: VecDeque<Support>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct Member {
    pub group: ridge::Handle,
    pub run_index: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Candidate {
    pub members: [Option<Member>; 2],
    pub mass: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct GroupSnapshot {
    pub group: ridge::Handle,
    pub states: [f64; 4],
    pub unknown: f64,
    pub support_fraction: f64,
    pub coverage: f64,
    pub lifetime_start: u64,
    pub window_start: u64,
    pub incoming_union_mass: f64,
    pub outgoing_union_mass: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Snapshot {
    pub end_sample: u64,
    pub refreshed_at: u64,
    pub groups: [Option<GroupSnapshot>; 8],
    pub candidates: [Option<Candidate>; 16],
    pub runs: [Option<Run>; 32],
    pub unresolved: f64,
    pub discarded_mass: f64,
    pub family_count: usize,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            end_sample: 0,
            refreshed_at: 0,
            groups: [None; 8],
            candidates: [None; 16],
            runs: [None; 32],
            unresolved: 1.,
            discarded_mass: 0.,
            family_count: 0,
        }
    }
}

pub(crate) struct Gesture {
    bus: u8,
    epoch: u64,
    rate: u32,
    hop: u64,
    config: TemporalGestureConfig,
    groups: [Option<Group>; 8],
    scratch: Vec<Candidate>,
    last: Option<u64>,
    snapshot: Snapshot,
}

// Rates are constant within the supplied physical interval. Gap admission precedes normalization.
pub(super) fn transition(state: State, rates: [f64; 4], dt: f64, gap_allowed: bool) -> [f64; 4] {
    let total: f64 = rates
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != state as usize)
        .map(|(_, v)| v)
        .sum();
    let exit = -(-total * dt).exp_m1();
    let mut out = std::array::from_fn(|i| {
        if i == state as usize {
            (-total * dt).exp()
        } else if total > 0. {
            exit * rates[i] / total
        } else {
            0.
        }
    });
    if state != State::Gap && !gap_allowed {
        out[3] = 0.;
    }
    let sum: f64 = out.iter().sum();
    if sum > 0. {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

// Keep raw known scores for linear updates and the log survival for tuple composition.
pub(super) struct ArticulationProposals {
    pub scores: [f64; 4],
    pub log_keep: f64,
    pub unknown: f64,
}

pub(super) fn articulation_proposals(
    parent: Option<State>,
    rates: [f64; 4],
    dt: f64,
    observed: bool,
    low: bool,
) -> Result<ArticulationProposals, &'static str> {
    if !dt.is_finite() || dt < 0. || rates.iter().any(|r| !r.is_finite() || *r < 0.) {
        return Err("invalid articulation proposal rates or duration");
    }
    if parent.is_some_and(|state| {
        !rates
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != state as usize)
            .map(|(_, r)| r)
            .sum::<f64>()
            .is_finite()
    }) {
        return Err("articulation competing-rate overflow");
    }
    let scores = parent.map_or_else(
        || std::array::from_fn(|i| f64::from(observed && (i != 3 || low))),
        |state| transition(state, rates, dt, observed && low),
    );
    let log_keep = if scores.iter().any(|v| *v > 0.) {
        -dt / 10.
    } else {
        f64::NEG_INFINITY
    };
    let keep = log_keep.exp();
    // Preserve ordinary branch arithmetic; expm1 rescues leaks below one ulp at one.
    let unknown = if keep == 1. {
        -log_keep.exp_m1()
    } else {
        1. - keep
    };
    Ok(ArticulationProposals {
        scores,
        log_keep,
        unknown,
    })
}

// Frame features are shared by every path; elapsed time is filled per path below.
fn rate_features(values: [Option<f64>; 4], cfg: &TemporalGestureConfig) -> [f64; 11] {
    let mut x = [0.; 11];
    x[0] = 1.;
    for i in 0..4 {
        x[1 + i] = values[i].map_or(0., |v| (v - cfg.means[i]) / cfg.deviations[i].max(1e-6));
        x[6 + i] = f64::from(values[i].is_none());
    }
    x
}

// Shared by observed and conditional state paths; hypothetical paths never admit unknown parents.
fn rates(
    state: State,
    x: &[f64; 11],
    log_elapsed: f64,
    cfg: &TemporalGestureConfig,
) -> Result<[f64; 4], &'static str> {
    let elapsed = (log_elapsed - cfg.means[4]) / cfg.deviations[4].max(1e-6);
    let mut rates = [0.; 4];
    for (i, r) in rates.iter_mut().enumerate() {
        if i == state as usize {
            continue;
        }
        let z: f64 = cfg.coefficients[state as usize][i]
            .iter()
            .enumerate()
            .map(|(j, a)| a * if j == 5 { elapsed } else { x[j] })
            .sum();
        if !z.is_finite() {
            return Err("nonfinite articulation rate input");
        }
        *r = z.max(0.) + (-z.abs()).exp().ln_1p();
    }
    if !rates.iter().sum::<f64>().is_finite() {
        return Err("articulation rate overflow");
    }
    Ok(rates)
}

impl Group {
    fn advance(
        &mut self,
        raw: Option<&RawDescriptor>,
        end: u64,
        motion: Option<f64>,
        cfg: TemporalGestureConfig,
        rate: u32,
    ) -> Result<(), &'static str> {
        use articulation::{Articulation, Input};
        let input = Input::new(self.handle, raw, motion, [self.end, end], rate, &cfg)?;
        self.scratch.clear();
        let admission = articulation_proposals(None, [0.; 4], input.dt, input.observed, input.low)?;
        let keep = admission.log_keep.exp();
        let n: f64 = admission.scores.iter().sum();
        let mut unknown = self.unknown * admission.unknown;
        for (state, _) in STATES
            .into_iter()
            .zip(admission.scores)
            .filter(|(_, s)| *s > 0.)
        {
            let child = input.child(None, state);
            self.scratch.push(Path {
                state: child.state,
                entered: child.entered,
                run: child.run,
                mass: self.unknown * keep / n,
            });
        }
        for path in &self.paths {
            let parent = Articulation {
                state: path.state,
                entered: path.entered,
                run: path.run,
            };
            let rates = input.rates(Some(parent), &cfg)?;
            let proposals = articulation_proposals(
                Some(path.state),
                rates,
                input.dt,
                input.observed,
                input.low,
            )?;
            let keep = proposals.log_keep.exp();
            unknown += path.mass * proposals.unknown;
            for (i, score) in proposals
                .scores
                .into_iter()
                .enumerate()
                .filter(|(_, s)| *s > 0.)
            {
                let child = input.child(Some(parent), STATES[i]);
                self.scratch.push(Path {
                    state: child.state,
                    entered: child.entered,
                    run: child.run,
                    mass: path.mass * keep * score,
                });
            }
        }
        self.scratch.sort_by_key(|p| (p.state, p.entered, p.run));
        self.paths.clear();
        for path in self.scratch.drain(..) {
            if let Some(last) = self
                .paths
                .last_mut()
                .filter(|p| (p.state, p.entered, p.run) == (path.state, path.entered, path.run))
            {
                last.mass += path.mass;
            } else {
                self.paths.push(path);
            }
        }
        self.paths.sort_by(|a, b| {
            b.mass
                .total_cmp(&a.mass)
                .then((a.state, a.entered, a.run).cmp(&(b.state, b.entered, b.run)))
        });
        if self.paths.len() > 15 {
            unknown += self.paths.drain(15..).map(|p| p.mass).sum::<f64>();
        }
        self.unknown = unknown;
        self.end = end;
        Ok(())
    }
}

// None preserves endpoint uncertainty; false means a supported separate-run pair.
fn handoff(a: Run, b: Run, rate: u32) -> Option<bool> {
    if a.censored || b.censored {
        return None;
    }
    let (aa, ba) = (a.attack?, b.attack?);
    if aa.start == ba.start {
        return Some(false);
    }
    let (early, late) = if aa.start < ba.start {
        (a, ba)
    } else {
        (b, aa)
    };
    let Some(release) = early.release else {
        return Some(false);
    };
    let lead = 2 * u64::from(rate);
    if late.end < release.start.saturating_sub(lead) {
        return Some(false);
    }
    let lead_ok = late.start >= release.end.saturating_sub(lead);
    if lead_ok && late.end <= early.observed_end {
        return Some(true);
    }
    if let Some(gap) = early.gap {
        let allowance = u64::from(rate);
        if lead_ok && late.end <= gap.start.saturating_add(allowance) {
            return Some(true);
        }
        if late.start > gap.end.saturating_add(allowance) {
            return Some(false);
        }
    }
    None
}

impl Gesture {
    pub(crate) fn new(
        bus: u8,
        epoch: u64,
        rate: u32,
        hop: u64,
        config: TemporalGestureConfig,
    ) -> Result<Self, &'static str> {
        if bus > 1
            || rate == 0
            || hop == 0
            || !config.rms_reference.is_finite()
            || config.rms_reference < 1e-6
            || config.means.iter().any(|v| !v.is_finite())
            || config.deviations.iter().any(|v| !v.is_finite() || *v < 0.)
            || config
                .coefficients
                .iter()
                .flatten()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("invalid gesture diagnostic reference or rate coefficients");
        }
        Ok(Self {
            bus,
            epoch,
            rate,
            hop,
            config,
            groups: std::array::from_fn(|_| None),
            scratch: Vec::with_capacity(10880),
            last: None,
            snapshot: Snapshot::default(),
        })
    }

    pub(in crate::temporal_cognition) fn rms_reference(&self) -> f64 {
        self.config.rms_reference
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        self.snapshot
    }

    pub(crate) fn advance(
        &mut self,
        acoustic: &frontend::Snapshot,
        ridges: &ridge::Update,
        end: u64,
    ) -> Result<(), &'static str> {
        if self.last.is_some_and(|old| end <= old) || !end.is_multiple_of(self.hop) {
            return Err("gesture observations must be ordered canonical hops");
        }
        for (i, raw) in acoustic
            .features
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.map(|f| (i, f.raw)))
        {
            if raw.group.bus != self.bus
                || raw.group.epoch != self.epoch
                || raw.end != end
                || raw.available_end > end
                || raw.source_end > raw.available_end
                || raw.end > raw.source_end
                || raw.source_start > raw.start
                || raw.start >= raw.end
                || raw.known_samples > raw.end - raw.start
                || raw.values.iter().flatten().any(|v| !v.is_finite())
            {
                return Err("invalid gesture raw support or identity");
            }
            if acoustic.group_handles[i] != Some(raw.group) {
                return Err("gesture feature owner differs from pre-update assignment");
            }
        }
        // Match handles rather than slot positions: births/splits/merges never inherit another run.
        let mut retired = false;
        for group in &mut self.groups {
            if group
                .as_ref()
                .is_some_and(|g| !acoustic.group_handles.contains(&Some(g.handle)))
            {
                *group = None;
                retired = true;
            }
        }
        let denominator: f64 = acoustic
            .assignment
            .rows
            .iter()
            .flatten()
            .flat_map(|r| r.weights)
            .sum();
        for i in 0..8 {
            let Some(raw) = acoustic.features[i].as_ref().map(|f| &f.raw) else {
                continue;
            };
            let slot = if let Some(slot) = self
                .groups
                .iter()
                .position(|g| g.as_ref().is_some_and(|g| g.handle == raw.group))
            {
                slot
            } else {
                let slot = self
                    .groups
                    .iter()
                    .position(Option::is_none)
                    .ok_or("gesture group capacity exhausted")?;
                self.groups[slot] = Some(Group {
                    handle: raw.group,
                    born: raw.start,
                    end: raw.start,
                    paths: Vec::with_capacity(64),
                    scratch: Vec::with_capacity(64),
                    unknown: 1.,
                    support: VecDeque::with_capacity(
                        (8 * u64::from(self.rate)).div_ceil(self.hop) as usize + 1,
                    ),
                });
                slot
            };
            let group = self.groups[slot].as_mut().unwrap();
            if raw.start > group.end {
                group.advance(None, raw.start, None, self.config, self.rate)?;
            }
            let mut motion = 0.;
            let mut weight = 0.;
            for row in acoustic.assignment.rows.iter().flatten() {
                let w = row.weights[i];
                if let Some(r) = ridges
                    .current
                    .iter()
                    .flatten()
                    .find(|r| r.handle == row.trajectory)
                    && let Some(slope) = r.slopes[0]
                {
                    motion += w * slope.abs();
                    weight += w;
                }
            }
            group.advance(
                acoustic.eligible[i].then_some(raw),
                end,
                (weight > 0.).then(|| motion / weight),
                self.config,
                self.rate,
            )?;
            let valid = raw.known_samples == self.hop
                && raw.end - raw.start == self.hop
                && acoustic.eligible[i]
                && acoustic.spectral_shape_supported
                && denominator > 0.;
            if valid {
                let alpha = acoustic
                    .assignment
                    .rows
                    .iter()
                    .flatten()
                    .map(|r| r.weights[i])
                    .sum::<f64>()
                    / denominator;
                group.support.push_back(Support {
                    start: raw.start,
                    end,
                    alpha,
                });
            }
            let window = end.saturating_sub(8 * u64::from(self.rate));
            while group.support.front().is_some_and(|s| s.end <= window) {
                group.support.pop_front();
            }
        }
        for group in self.groups.iter_mut().flatten() {
            if group.end < end {
                group.advance(None, end, None, self.config, self.rate)?;
            }
        }
        self.last = Some(end);
        self.snapshot.end_sample = end;
        if retired
            || end.saturating_sub(self.snapshot.refreshed_at) >= u64::from(self.rate).div_ceil(10)
        {
            self.refresh(end);
        }
        Ok(())
    }

    fn refresh(&mut self, end: u64) {
        self.snapshot = Snapshot {
            end_sample: end,
            refreshed_at: end,
            ..Snapshot::default()
        };
        let mut b = [0.; 8];
        let mut coverage = [0.; 8];
        for (i, g) in self
            .groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.as_ref().map(|g| (i, g)))
        {
            let start = g.born.max(end.saturating_sub(8 * u64::from(self.rate)));
            let mut known = 0.;
            for s in &g.support {
                let n = s.end.saturating_sub(s.start.max(start)) as f64;
                known += n;
                b[i] += n * s.alpha;
            }
            coverage[i] = if end > start {
                known / (end - start) as f64
            } else {
                0.
            };
        }
        let sum: f64 = b.iter().sum();
        if sum > 0. {
            for v in &mut b {
                *v /= sum;
            }
        }
        let mut norm = b.iter().sum::<f64>();
        for i in 0..8 {
            for j in i + 1..8 {
                if self.groups[i]
                    .as_ref()
                    .is_some_and(|g| g.handle.generation != 1)
                    && self.groups[j]
                        .as_ref()
                        .is_some_and(|g| g.handle.generation != 1)
                {
                    norm += (b[i] * b[j]).sqrt();
                }
            }
        }
        self.scratch.clear();
        for (i, g) in self
            .groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.as_ref().map(|g| (i, g)))
        {
            let mut states = [0.; 4];
            for p in &g.paths {
                states[p.state as usize] += p.mass;
            }
            self.snapshot.groups[i] = Some(GroupSnapshot {
                group: g.handle,
                states,
                unknown: g.unknown,
                support_fraction: b[i],
                coverage: coverage[i],
                lifetime_start: g.born,
                window_start: g.born.max(end.saturating_sub(8 * u64::from(self.rate))),
                incoming_union_mass: 0.,
                outgoing_union_mass: 0.,
            });
            if norm == 0. || b[i] == 0. {
                continue;
            }
            self.snapshot.family_count += 1;
            if coverage[i] < 0.9 {
                continue;
            }
            for p in &g.paths {
                if p.run.attack.is_some()
                    && !p.run.censored
                    && p.run.observed_end >= end.saturating_sub(8 * u64::from(self.rate))
                {
                    self.scratch.push(Candidate {
                        members: [
                            Some(Member {
                                group: g.handle,
                                run_index: g.paths.iter().position(|q| q.run == p.run).unwrap(),
                            }),
                            None,
                        ],
                        mass: b[i] / norm * p.mass,
                    });
                }
            }
        }
        for i in 0..8 {
            for j in i + 1..8 {
                let (Some(g), Some(h)) = (&self.groups[i], &self.groups[j]) else {
                    continue;
                };
                if g.handle.generation == 1
                    || h.handle.generation == 1
                    || b[i] * b[j] == 0.
                    || norm == 0.
                {
                    continue;
                }
                self.snapshot.family_count += 1;
                if coverage[i] < 0.9 || coverage[j] < 0.9 {
                    continue;
                }
                for p in &g.paths {
                    for q in &h.paths {
                        if p.run.observed_end < end.saturating_sub(8 * u64::from(self.rate))
                            || q.run.observed_end < end.saturating_sub(8 * u64::from(self.rate))
                        {
                            continue;
                        }
                        let mass = (b[i] * b[j]).sqrt() / norm * p.mass * q.mass;
                        let members = [
                            Some(Member {
                                group: g.handle,
                                run_index: g.paths.iter().position(|r| r.run == p.run).unwrap(),
                            }),
                            Some(Member {
                                group: h.handle,
                                run_index: h.paths.iter().position(|r| r.run == q.run).unwrap(),
                            }),
                        ];
                        match handoff(p.run, q.run, self.rate) {
                            Some(true) => self.scratch.push(Candidate { members, mass }),
                            Some(false) => {
                                for k in 0..2 {
                                    self.scratch.push(Candidate {
                                        members: [members[k], None],
                                        mass: mass
                                            * if k == 0 {
                                                b[i] / (b[i] + b[j])
                                            } else {
                                                b[j] / (b[i] + b[j])
                                            },
                                    });
                                }
                            }
                            None => (),
                        }
                    }
                }
            }
        }
        self.scratch.sort_by_key(|c| c.members);
        let mut n = 0;
        for i in 0..self.scratch.len() {
            let c = self.scratch[i];
            if n > 0 && self.scratch[n - 1].members == c.members {
                self.scratch[n - 1].mass += c.mass;
            } else {
                self.scratch[n] = c;
                n += 1;
            }
        }
        self.scratch.truncate(n);
        self.scratch
            .sort_by(|a, b| b.mass.total_cmp(&a.mass).then(a.members.cmp(&b.members)));
        let mut retained = 0.;
        let mut run_keys = [None; 32];
        let mut run_count = 0;
        for (i, c) in self.scratch.iter().enumerate() {
            if i >= 16 {
                self.snapshot.discarded_mass += c.mass;
                continue;
            }
            let mut published = *c;
            for member in published.members.iter_mut().flatten() {
                let index = if let Some(index) = run_keys[..run_count]
                    .iter()
                    .position(|key| *key == Some(*member))
                {
                    index
                } else {
                    let index = run_count;
                    run_count += 1;
                    run_keys[index] = Some(*member);
                    let g = self
                        .groups
                        .iter()
                        .flatten()
                        .find(|g| g.handle == member.group)
                        .unwrap();
                    self.snapshot.runs[index] = Some(g.paths[member.run_index].run);
                    index
                };
                member.run_index = index;
            }
            self.snapshot.candidates[i] = Some(published);
            retained += c.mass;
            if let [Some(a), Some(b)] = published.members {
                let (early, late) = if self.snapshot.runs[a.run_index]
                    .unwrap()
                    .attack
                    .unwrap()
                    .start
                    < self.snapshot.runs[b.run_index]
                        .unwrap()
                        .attack
                        .unwrap()
                        .start
                {
                    (a, b)
                } else {
                    (b, a)
                };
                for g in self.snapshot.groups.iter_mut().flatten() {
                    if g.group == early.group {
                        g.outgoing_union_mass += c.mass;
                    }
                    if g.group == late.group {
                        g.incoming_union_mass += c.mass;
                    }
                }
            }
        }
        self.snapshot.unresolved = (1. - retained).clamp(0., 1.);
    }

    pub(crate) fn finish(&mut self, end: u64) -> Result<(), &'static str> {
        for group in self.groups.iter_mut().flatten() {
            if end > group.end {
                group.advance(None, end, None, self.config, self.rate)?;
            }
        }
        self.refresh(end);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
