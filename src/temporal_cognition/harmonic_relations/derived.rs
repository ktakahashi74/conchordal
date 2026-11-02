use super::*;
use context::Contexts;
use graph::Graph;
use matching::{Prediction, hellinger};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Arrival {
    pub categorical: bool,
    pub compatibility: f64,
    pub weight: f64,
    pub successor: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompetitionReduction {
    pub difference: Value,
    pub before_mass: Value,
    pub after_mass: Value,
    pub removed: usize,
    pub added: usize,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Continuity {
    pub source: Source,
    pub retained: bool,
    pub rearticulated: bool,
    pub silence: bool,
    pub timbre_changed: bool,
    pub handover: Option<Source>,
    pub displacement: Value,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cadence {
    pub arrival: bool,
    pub retention: bool,
    pub articulation: bool,
    pub boundary: bool,
    pub inactivity: bool,
}
impl Cadence {
    pub fn candidate(self) -> bool {
        self.arrival && (self.retention || self.articulation || self.boundary || self.inactivity)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrnamentRole {
    Passing,
    Return,
    DelayedApproach,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ornament {
    pub source: Source,
    pub intermediate: u64,
    pub role: OrnamentRole,
    pub residence: f64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Substitution {
    pub alternative: u64,
    pub predecessor: f64,
    pub successor: Option<f64>,
    pub observed_successor: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Pivot {
    pub state: u64,
    pub contexts: [u64; KMAX],
    pub count: usize,
    pub from: u64,
    pub to: u64,
    pub start: f64,
    pub transition_at: Option<f64>,
    pub transition_end: Option<f64>,
    pub persistence: f64,
}

pub(crate) struct Relations {
    pub arrivals: [Arrival; HMAX],
    pub arrival_count: usize,
    pub competition: CompetitionReduction,
    pub continuity: Vec<Continuity>,
    pub cadence: Cadence,
    pub ornaments: Vec<Ornament>,
    pub substitutions: [Option<Substitution>; HMAX],
    pub pivot: Option<Pivot>,
    pub leading_context: u64,
    last_arrival: bool,
    last_snapshot: u64,
    last_time: f64,
}
impl Relations {
    pub fn new(sources: usize) -> Self {
        Self {
            arrivals: [Arrival::default(); HMAX],
            arrival_count: 0,
            competition: CompetitionReduction::default(),
            continuity: Vec::with_capacity(sources),
            cadence: Cadence::default(),
            ornaments: Vec::with_capacity(sources),
            substitutions: [None; HMAX],
            pivot: None,
            leading_context: 0,
            last_arrival: false,
            last_snapshot: 0,
            last_time: 0.,
        }
    }
    pub fn bytes(&self) -> usize {
        self.continuity.capacity() * size_of::<Continuity>()
            + self.ornaments.capacity() * size_of::<Ornament>()
    }
    pub(super) fn observe(
        &mut self,
        g: &Graph,
        p: &Prediction,
        model_version: u64,
        o: &Observation<'_>,
    ) {
        self.arrival_count = 0;
        self.continuity.clear();
        self.ornaments.clear();
        self.substitutions.fill(None);
        let usable = p.model_version == model_version
            && p.issued_at <= o.time
            && p.coordinate_version == o.acoustic.coordinate_version;
        if usable {
            for atom in &p.atoms[..p.len] {
                if atom.weight == 0.
                    || !g
                        .anchor(atom.successor)
                        .is_some_and(|a| g.observation(a).time >= o.time - TMAX)
                {
                    continue;
                }
                let mut k = state_kernel(
                    g,
                    o.acoustic,
                    o.support,
                    atom.acoustic,
                    Support {
                        bins: &atom.bins,
                        residual: atom.residual,
                        error: atom.shape_error,
                        bound: atom.shape_bound,
                        valid: atom.shape_valid && atom.residual == 0.,
                    },
                );
                k.scalar(
                    Value {
                        value: o.time,
                        error: o.timing_error,
                        valid: true,
                        bound: false,
                    },
                    atom.time,
                    g.statistics[3].variance(),
                );
                let mut envelope =
                    (o.time - atom.time.value).abs() <= o.timing_error + atom.time.error;
                let mut valid = 0;
                for (a, b) in [
                    (o.acoustic.mass, atom.acoustic.mass),
                    (o.acoustic.focus, atom.acoustic.focus),
                    (o.acoustic.competition, atom.acoustic.competition),
                ] {
                    if a.valid && b.valid {
                        valid += 1;
                        envelope &= (a.value - b.value).abs() <= a.error + b.error;
                    }
                }
                if o.support.valid
                    && o.support.residual == 0.
                    && atom.shape_valid
                    && atom.residual == 0.
                {
                    valid += 1;
                    envelope &= hellinger(o.support.bins, &atom.bins).sqrt()
                        <= o.support.error + atom.shape_error;
                }
                for expected in &atom.trajectories {
                    if let Some(actual) =
                        o.trajectories.iter().find(|t| t.source == expected.source)
                    {
                        if actual.pitch.valid && expected.pitch.valid {
                            valid += 1;
                            envelope &= (actual.pitch.value - expected.pitch.value).abs()
                                <= actual.pitch.error + expected.pitch.error;
                            k.scalar(actual.pitch, expected.pitch, g.statistics[5].variance());
                        }
                    } else {
                        envelope = false;
                    }
                }
                self.arrivals[self.arrival_count] = Arrival {
                    categorical: envelope && valid > 0,
                    compatibility: k.compatibility(),
                    weight: atom.weight,
                    successor: atom.successor,
                };
                self.arrival_count += 1;
            }
        }
        let arrival = self.arrivals[..self.arrival_count]
            .iter()
            .any(|a| a.categorical);
        self.cadence = Cadence {
            arrival: arrival || self.last_arrival,
            boundary: o.events.iter().any(|e| e.flags & BOUNDARY != 0),
            inactivity: o.trajectories.iter().all(|t| !t.sounding),
            ..Cadence::default()
        };
        if let Some(previous) = g.anchors.back() {
            let before = g.observation(previous);
            self.competition = CompetitionReduction {
                difference: before
                    .acoustic
                    .competition
                    .difference(o.acoustic.competition),
                before_mass: before.acoustic.mass,
                after_mass: o.acoustic.mass,
                removed: before
                    .trajectories
                    .iter()
                    .filter(|t| !o.trajectories.iter().any(|n| n.source == t.source))
                    .count(),
                added: o
                    .trajectories
                    .iter()
                    .filter(|t| !before.trajectories.iter().any(|n| n.source == t.source))
                    .count(),
            };
            for old in before.trajectories {
                if let Some(new) = o.trajectories.iter().find(|t| t.source == old.source) {
                    let d = new.pitch.difference(old.pitch);
                    let retained = old.sounding
                        && new.sounding
                        && d.valid
                        && d.value.abs() <= old.pitch.error + new.pitch.error;
                    let rearticulated = old.articulation != new.articulation
                        || o.events
                            .iter()
                            .any(|e| e.source == Some(new.source) && e.flags & ONSET != 0);
                    self.continuity.push(Continuity {
                        source: old.source,
                        retained,
                        rearticulated,
                        silence: !new.sounding,
                        timbre_changed: old.timbre.valid
                            && new.timbre.valid
                            && old.timbre.value != new.timbre.value,
                        displacement: d,
                        handover: None,
                    });
                    self.cadence.retention |= retained;
                    self.cadence.articulation |= rearticulated;
                } else {
                    let handover = o
                        .trajectories
                        .iter()
                        .find(|n| {
                            !before.trajectories.iter().any(|b| b.source == n.source)
                                && n.pitch.valid
                                && old.pitch.valid
                                && (n.pitch.value - old.pitch.value).abs()
                                    <= n.pitch.error + old.pitch.error
                        })
                        .map(|n| n.source);
                    self.continuity.push(Continuity {
                        source: old.source,
                        silence: true,
                        handover,
                        ..Continuity::default()
                    });
                }
            }
            if !o.events.is_empty() {
                if let Some(prior) = previous.id.checked_sub(1).and_then(|id| g.anchor(id)) {
                    let earlier = g.observation(prior);
                    if g.node(prior.node).unwrap().episode == g.node(previous.node).unwrap().episode
                    {
                        for middle in before.trajectories {
                            let (Some(first), Some(last)) = (
                                earlier
                                    .trajectories
                                    .iter()
                                    .find(|t| t.source == middle.source),
                                o.trajectories.iter().find(|t| t.source == middle.source),
                            ) else {
                                continue;
                            };
                            if !first.pitch.valid || !middle.pitch.valid || !last.pitch.valid {
                                continue;
                            }
                            let a = middle.pitch.value - first.pitch.value;
                            let b = last.pitch.value - middle.pitch.value;
                            let ae = middle.pitch.error + first.pitch.error;
                            let be = last.pitch.error + middle.pitch.error;
                            let role = if (a > ae && b > be) || (a < -ae && b < -be) {
                                Some(OrnamentRole::Passing)
                            } else if (last.pitch.value - first.pitch.value).abs()
                                <= last.pitch.error + first.pitch.error
                                && a.abs() > ae
                            {
                                Some(OrnamentRole::Return)
                            } else if a.abs() <= ae && b.abs() > be {
                                Some(OrnamentRole::DelayedApproach)
                            } else {
                                None
                            };
                            if let Some(role) = role {
                                self.ornaments.push(Ornament {
                                    source: middle.source,
                                    intermediate: previous.id,
                                    role,
                                    residence: o.time - before.time,
                                });
                            }
                        }
                        self.substitute(g, &earlier, &before, Some(o), p, usable);
                    }
                }
                if self.substitutions.iter().all(Option::is_none) {
                    self.substitute(g, &before, o, None, p, usable);
                }
            }
        } else {
            self.competition = CompetitionReduction::default();
        }
        self.last_arrival = arrival;
        self.last_snapshot = o.acoustic.snapshot;
        self.last_time = o.time;
    }
    fn substitute(
        &mut self,
        g: &Graph,
        predecessor: &Observation<'_>,
        middle: &Observation<'_>,
        successor: Option<&Observation<'_>>,
        p: &Prediction,
        usable: bool,
    ) {
        let mut count = 0;
        for alternative in &g.anchors {
            if count == HMAX || g.observation(alternative).time >= predecessor.time {
                break;
            }
            let Some(before) = alternative.id.checked_sub(1).and_then(|id| g.anchor(id)) else {
                continue;
            };
            if g.node(before.node).unwrap().episode != g.node(alternative.node).unwrap().episode {
                continue;
            }
            let old = g.observation(alternative);
            let before = g.observation(before);
            let distinct = old.acoustic.mass != middle.acoustic.mass
                || old.acoustic.focus != middle.acoustic.focus
                || old.acoustic.competition != middle.acoustic.competition
                || old.support.bins != middle.support.bins;
            if !distinct {
                continue;
            }
            let pred = state_kernel(
                g,
                predecessor.acoustic,
                predecessor.support,
                before.acoustic,
                before.support,
            )
            .weight();
            if pred == 0. {
                continue;
            }
            let next = g.successor(alternative).map(|a| g.observation(a));
            let succ = match (successor, next) {
                (Some(a), Some(b)) => {
                    Some(state_kernel(g, a.acoustic, a.support, b.acoustic, b.support).weight())
                }
                (None, Some(b)) if usable && p.atoms[..p.len].iter().any(|a| a.weight > 0.) => {
                    Some(
                        p.atoms[..p.len]
                            .iter()
                            .map(|a| {
                                a.weight
                                    * state_kernel(
                                        g,
                                        a.acoustic,
                                        Support {
                                            bins: &a.bins,
                                            residual: a.residual,
                                            error: a.shape_error,
                                            bound: a.shape_bound,
                                            valid: a.shape_valid && a.residual == 0.,
                                        },
                                        b.acoustic,
                                        b.support,
                                    )
                                    .weight()
                            })
                            .sum(),
                    )
                }
                _ => None,
            };
            self.substitutions[count] = Some(Substitution {
                alternative: alternative.id,
                predecessor: pred,
                successor: succ,
                observed_successor: successor.is_some() && next.is_some(),
            });
            count += 1;
        }
    }
    pub(super) fn contexts(&mut self, contexts: &Contexts, time: f64) {
        let lead = contexts
            .active
            .iter()
            .flatten()
            .filter(|c| c.responsibility > 0.)
            .max_by(|a, b| {
                a.responsibility
                    .total_cmp(&b.responsibility)
                    .then(b.id.cmp(&a.id))
            })
            .map_or(0, |c| c.id);
        if self.pivot.is_some_and(|p| {
            time - p.start > TMAX
                || p.contexts[..p.count]
                    .iter()
                    .all(|id| !contexts.active.iter().flatten().any(|c| c.id == *id))
        }) {
            self.pivot = None;
        }
        if let Some(p) = self.pivot.as_mut() {
            if p.transition_at.is_none()
                && lead != 0
                && lead != p.from
                && p.contexts[..p.count].contains(&lead)
                && time > p.start
            {
                p.to = lead;
                p.transition_at = Some(time);
            }
            if let Some(at) = p.transition_at
                && p.transition_end.is_none()
            {
                p.persistence = time - at;
                if lead != p.to {
                    p.transition_end = Some(time);
                }
            }
        }
        let count = contexts.evidence.iter().filter(|w| **w > 0.).count();
        if count > 1 && self.pivot.is_none_or(|p| p.transition_end.is_some()) {
            let mut pivot = Pivot {
                state: self.last_snapshot,
                from: self.leading_context,
                start: time,
                ..Pivot::default()
            };
            for (i, w) in contexts.evidence.iter().enumerate() {
                if *w > 0. {
                    pivot.contexts[pivot.count] = contexts.evidence_ids[i];
                    pivot.count += 1;
                }
            }
            self.pivot = Some(pivot);
        }
        self.leading_context = lead;
    }
}
pub(super) fn state_kernel(
    g: &Graph,
    a: Acoustic,
    ap: Support<'_>,
    b: Acoustic,
    bp: Support<'_>,
) -> Kernel {
    let mut k = Kernel::default();
    for (i, (x, y)) in [
        (a.mass, b.mass),
        (a.focus, b.focus),
        (a.competition, b.competition),
    ]
    .into_iter()
    .enumerate()
    {
        k.scalar(x, y, g.statistics[i].variance());
    }
    k.requested += 1;
    if a.coordinate_version == b.coordinate_version
        && ap.valid
        && bp.valid
        && ap.residual == 0.
        && bp.residual == 0.
    {
        k.distance(
            hellinger(ap.bins, bp.bins),
            ap.error.powi(2) + bp.error.powi(2) + g.shape_variance(),
        );
    }
    k
}
