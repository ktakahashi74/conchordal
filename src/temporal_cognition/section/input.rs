//! Acoustic Activity and phrase-owned accumulation shared by marginal and joint paths.

use super::{Activity, Assignment, Commit, Record, Status};
use crate::config::TemporalSectionConfig;
use crate::temporal_cognition::{
    accents::Delivery, features, observables::WindowDescriptor, phrase::Foreground,
    proposals::frontend, ridge::Handle,
};

#[derive(Clone, Copy, PartialEq)]
pub(in crate::temporal_cognition) struct Observation {
    pub group: Handle,
    pub interval: [u64; 2],
    pub rate: u32,
    pub raw: Option<features::Update>,
    pub delivery: Option<Delivery>,
    pub observed: bool,
    pub delta: Activity,
}

impl Observation {
    pub fn new(
        acoustic: &frontend::Snapshot,
        group: Handle,
        interval: [u64; 2],
        rate: u32,
        grouped: Option<bool>,
    ) -> Result<Self, &'static str> {
        if rate == 0 || interval[0] >= interval[1] {
            return Err("invalid section observation clock");
        }
        let end = interval[1];
        let sample_rate = rate;
        let rate = f64::from(rate);
        let acoustic_index = acoustic
            .group_handles
            .iter()
            .position(|h| *h == Some(group));
        let raw = acoustic_index.and_then(|i| acoustic.features[i]);
        let observed = raw.filter(|u| {
            acoustic.eligible[acoustic_index.unwrap()]
                && u.raw.group == group
                && u.raw.start >= interval[0]
                && u.raw.start < end
                && u.raw.end == end
                && u.raw.available_end <= end
                && u.raw.source_end <= end
                && u.raw.known_samples == u.raw.end - u.raw.start
        });
        let mut delta = Activity {
            window: [interval[0] as f64 / rate, end as f64 / rate],
            numerators: [0.; 9],
            denominators: [0.; 9],
            physical_valid_seconds: [0.; 9],
            assignment_seconds: 0.,
            physical_window_seconds: (end - interval[0]) as f64 / rate,
        };
        if let Some(u) = observed {
            let i = acoustic_index.unwrap();
            let rows = acoustic.assignment.rows.iter().flatten().count();
            let alpha = if rows > 0 {
                acoustic
                    .assignment
                    .rows
                    .iter()
                    .flatten()
                    .map(|r| r.weights[i])
                    .sum::<f64>()
                    / rows as f64
            } else {
                0.
            };
            let duration = (u.raw.end - u.raw.start) as f64 / rate;
            let assigned = duration * alpha;
            delta.assignment_seconds = assigned;
            delta.denominators[4] = assigned;
            delta.physical_valid_seconds[4] = duration;
            if let Some(energy) = acoustic.energy {
                let bus: f64 = energy.iter().sum();
                let others: f64 = energy[..7]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i && acoustic.group_handles[*j].is_some())
                    .map(|(_, v)| *v)
                    .sum();
                delta.numerators[8] = assigned
                    * f64::from(bus > 0. && energy[i] / bus >= 0.01 && others / bus >= 0.01);
                delta.denominators[8] = assigned;
                delta.physical_valid_seconds[8] = duration;
            }
        }

        let mut output = Self {
            group,
            interval,
            rate: sample_rate,
            raw,
            delivery: None,
            observed: observed.is_some(),
            delta,
        };
        output.condition_grouping(grouped);
        Ok(output)
    }

    pub fn condition_grouping(&mut self, grouped: Option<bool>) {
        self.delta.numerators[5] = 0.;
        self.delta.denominators[5] = 0.;
        self.delta.physical_valid_seconds[5] = 0.;
        if let Some(grouped) = grouped.filter(|_| self.observed) {
            self.delta.numerators[5] = self.delta.assignment_seconds * f64::from(grouped);
            self.delta.denominators[5] = self.delta.assignment_seconds;
            let u = self.raw.unwrap();
            self.delta.physical_valid_seconds[5] =
                (u.raw.end - u.raw.start) as f64 / f64::from(self.rate);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition) struct Span {
    span: Record,
    foreground: Option<Foreground>,
    adjacency: f64,
    pub sequence: u64,
}

impl Default for Span {
    fn default() -> Self {
        Self {
            span: Record::default(),
            foreground: None,
            adjacency: 1.,
            sequence: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Phrase {
    pub foreground: Option<Foreground>,
    pub completed_foreground: Option<Foreground>,
    pub completed_ending: Option<WindowDescriptor>,
}

impl Span {
    pub fn advance(
        &mut self,
        observation: &Observation,
        child: Phrase,
        initial: bool,
        config: &TemporalSectionConfig,
    ) -> Result<Option<(Commit, Activity, u64, f64)>, &'static str> {
        let rate = f64::from(observation.rate);
        let end = observation.interval[1];
        let delta = observation.delta;
        let delivery = observation.delivery;
        let observed = observation.raw.filter(|_| observation.observed);
        let mut completion = None;
        if let Some(closed) = child.completed_foreground {
            if let Some(previous) = self.foreground {
                if previous != closed {
                    return Err("closed section span differs from its parent support");
                }
            } else if initial && closed.start == observation.interval[0] && closed.heard_end == end
            {
                self.span
                    .add_activity(delta, 1., delta.physical_window_seconds)?;
            }
            let ending = child
                .completed_ending
                .map(|e| {
                    std::array::from_fn(|j| {
                        e.raw_values[j].map(|v| {
                            (v - config.ending_means[j]) / config.ending_deviations[j].max(1e-6)
                        })
                    })
                })
                .unwrap_or([None; 6]);
            // A nearest cached match is not a path-assigned correspondence.
            let assignment = Assignment {
                status: Status::Unresolved,
                cost: None,
                supported: false,
                search_completed: false,
                search_covered: false,
                search_nonempty: false,
                frequency_shift_log2: None,
                tempo_shift_log2: None,
                bound_hit: false,
                ambiguous_cutoff: false,
            };
            let mut activity = Activity {
                window: [closed.start as f64 / rate, closed.heard_end as f64 / rate],
                numerators: [0.; 9],
                denominators: [0.; 9],
                physical_valid_seconds: [0.; 9],
                assignment_seconds: self.span.statistics[52],
                physical_window_seconds: self.span.statistics[53],
            };
            for j in 0usize..9 {
                activity.numerators[j] = self.span.statistics[31 + j];
                activity.denominators[j] = self.span.statistics[40 + j.saturating_sub(3)];
                activity.physical_valid_seconds[j] = self.span.statistics[46 + j.saturating_sub(3)];
            }
            let sequence = self
                .sequence
                .checked_add(1)
                .ok_or("section sequence exhausted")?;
            let record = Commit {
                epoch: observation.group.epoch,
                occurrence_id: closed.credit,
                start: activity.window[0],
                support_end: activity.window[1],
                assignment_seconds: activity.assignment_seconds,
                membership: 1.,
                ordering_known: true,
                ending_descriptor: ending,
                assignment,
                ending_generation: Some(observation.group.generation),
            };
            completion = Some((record, activity, sequence, self.adjacency));
            if activity.assignment_seconds > 0. {
                self.sequence = sequence;
            }
            self.span = Record::default();
            self.foreground = None;
            self.adjacency = child.foreground.map_or(1., |f| {
                if f.start <= closed.heard_end {
                    1.
                } else {
                    observed
                        .map_or(0., |u| {
                            (u.raw.end - u.raw.start) as f64 / (f.start - closed.heard_end) as f64
                        })
                        .min(1.)
                }
            });
        }
        if let Some(foreground) = child.foreground {
            let start = self.foreground.map_or(foreground.start, |f| f.heard_end);
            if let Some(u) = observed.filter(|_| foreground.heard_end > start) {
                let mut owned = delta;
                let lo = u.raw.start.max(start).max(foreground.start);
                let hi = u.raw.end.min(foreground.heard_end);
                let fraction = hi.saturating_sub(lo) as f64 / (u.raw.end - u.raw.start) as f64;
                for v in owned
                    .numerators
                    .iter_mut()
                    .chain(&mut owned.denominators)
                    .chain(&mut owned.physical_valid_seconds)
                {
                    *v *= fraction;
                }
                owned.assignment_seconds *= fraction;
                owned.physical_window_seconds = (foreground.heard_end - start) as f64 / rate;
                if let Some(d) = delivery.filter(|d| {
                    d.accent.event_end > foreground.start && d.accent.at_cut(foreground.heard_end)
                }) {
                    owned.numerators[4] = d.accent.weight;
                }
                self.span
                    .add_activity(owned, 1., owned.physical_window_seconds)?;
            }
            self.foreground = Some(foreground);
        }
        Ok(completion)
    }
}
