//! Conditional frozen-body diagnostics, isolated from command execution and learning.

use super::{BodyState, Class, Input, OnsetOpportunity};
use crate::life::self_prediction::{CoherentWindow, ScheduledRelease, ToneEnergy};
use crate::temporal_cognition::action_profiles::consumer;
use crossbeam_channel::{Receiver, Sender, bounded};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

mod ratios;
mod release_trace;

const CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Window {
    pub start: u64,
    pub end: u64,
    pub points: [u64; 16],
    pub energies: [Option<f64>; 16],
    pub incoherent_energies: [Option<f64>; 16],
    pub coherent_energies: [Option<f64>; 16],
    pub mean: Option<f64>,
    pub default_mean: Option<f64>,
    pub difference: Option<f64>,
}

struct Body<'a> {
    retained: &'a [(u64, [bool; 2], ToneEnergy)],
    added: Option<([bool; 2], ToneEnergy)>,
    at: u64,
    intervention: (Option<ScheduledRelease>, Option<u64>),
}

impl Body<'_> {
    fn tones(
        &self,
        bus: usize,
    ) -> impl Iterator<Item = (Option<u64>, ToneEnergy, Option<ScheduledRelease>)> + '_ {
        self.retained
            .iter()
            .filter_map(move |(id, routed, tone)| {
                if !routed[bus]
                    || self.intervention.1.is_some_and(|until| {
                        self.at <= tone.envelope.onset && tone.envelope.onset < until
                    })
                {
                    return None;
                }
                let envelope = tone
                    .scheduled_release
                    .map_or(tone.envelope, |p| tone.envelope.with_release(p.off_sample));
                let release = (envelope.onset <= self.at && self.at < envelope.release_end)
                    .then_some(self.intervention.0)
                    .flatten();
                Some((Some(*id), *tone, release))
            })
            .chain(
                self.added
                    .into_iter()
                    .filter_map(move |(routed, tone)| routed[bus].then_some((None, tone, None))),
            )
    }

    fn support_after(&self, after: u64, bus: usize) -> Option<[u64; 2]> {
        self.tones(bus)
            .filter_map(|(_, tone, release)| tone.support_after(after, release))
            .reduce(|[a, b], [c, d]| [a.min(c), b.max(d)])
    }

    fn known_on(&self, interval: [u64; 2], bus: usize) -> bool {
        self.tones(bus).all(|(_, tone, release)| {
            let Some([start, end]) = tone.support_after(interval[0], release) else {
                return true;
            };
            let end = end.min(interval[1]);
            start >= end
                || (tone.amplitude.is_finite()
                    && tone
                        .control_for(release)
                        .is_some_and(|c| c.known_on([start, end])))
        })
    }

    fn point(&self, tick: u64, left: u64, right: u64, bus: usize) -> (Option<f64>, Option<f64>) {
        let mut energy = Some(0.);
        let mut coherent = CoherentWindow::new();
        for (_, tone, release) in self.tones(bus) {
            let projected = tone.at(tick, release);
            energy = energy.zip(projected).map(|(a, b)| a + b);
            coherent.add(tone.sine_point_from_energy(tick, release, projected));
        }
        (energy, coherent.mean(left, right, tick))
    }
}

pub(crate) fn project_window(
    retained: &[(u64, [bool; 2], ToneEnergy)],
    added: Option<([bool; 2], ToneEnergy)>,
    action_at: u64,
    intervention: (Option<ScheduledRelease>, Option<u64>),
    bus: usize,
    interval: [u64; 2],
    use_coherent: bool,
) -> Option<Window> {
    let [start, end] = interval;
    let width = end.checked_sub(start)?;
    if bus > 1 || width < 16 || retained.len() > CAPACITY {
        return None;
    }
    let mut window = Window {
        start,
        end,
        points: [0; 16],
        energies: [None; 16],
        incoherent_energies: [None; 16],
        coherent_energies: [None; 16],
        mean: None,
        default_mean: None,
        difference: None,
    };
    let mut total = Some(0.);
    for k in 0..16 {
        let [left, right] =
            [k, k + 1].map(|edge| start + (u128::from(width) * edge as u128).div_ceil(16) as u64);
        let tick = left + (right - left) / 2;
        window.points[k] = tick;
        let body = Body {
            retained,
            added,
            at: action_at,
            intervention,
        };
        let (mut energy, coherent) = body.point(tick, left, right, bus);
        window.incoherent_energies[k] = energy;
        window.coherent_energies[k] = coherent;
        if use_coherent {
            energy = window.coherent_energies[k].or(energy);
        }
        window.energies[k] = energy;
        total = total
            .zip(energy)
            .map(|(sum, e)| sum + e * (right - left) as f64 / width as f64);
    }
    window.mean = total;
    Some(window)
}

#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: Option<u32>,
    pub tone_id: u64,
    pub issued_at: u64,
    pub sample_rate: u32,
    pub hop: u64,
    pub opportunity: OnsetOpportunity,
    pub routed: [bool; 2],
    pub recipe: ToneEnergy,
}

#[derive(Clone, Copy)]
pub(crate) struct ScheduledRequest {
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: u32,
    pub issued_at: u64,
    pub sample_rate: u32,
    pub hop: u64,
    pub period: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct DefaultSchedule {
    pub accepted_transitions: usize,
    pub first_transition: Option<Input>,
    pub scheduled_tones: usize,
    pub queued_tones: usize,
    pub unsupported_control_tones: usize,
    pub amplitude_smoothing_tones: usize,
    pub scheduled_amplitude_tones: usize,
}

pub(crate) struct Packet {
    pub request: Option<Request>,
    pub scheduled: Option<ScheduledRequest>,
    pub retained: Vec<(u64, [bool; 2], ToneEnergy)>,
    pub trace: Option<crate::life::participation_trace::Frozen>,
    pub release_trace: Option<crate::life::participation_trace::Frozen>,
    pub shared: [Option<Arc<consumer::Publication>>; 2],
    pub bindings: [Option<consumer::Binding>; 2],
    pub default_schedule: Option<DefaultSchedule>,
    pub external: Option<crate::core::temporal_expectation::TemporalForecast>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Summary {
    pub interval: [u64; 2],
    pub mean: Option<f64>,
    pub default_mean: Option<f64>,
    pub difference: Option<f64>,
    pub supported_bins: u8,
    pub coherent_bins: u8,
}

#[derive(Clone, Copy)]
struct Projection {
    buses: [[Option<Summary>; 4]; 2],
}

#[derive(Debug, Serialize)]
pub(crate) struct Candidate {
    pub input: Input,
    pub buses: [[Option<Summary>; 4]; 2],
    pub onset_trace: Option<crate::life::participation_trace::TimingCandidate>,
    pub release_trace: Vec<release_trace::EndPair>,
    pub shared: [consumer::Pair; 2],
    #[serde(skip)]
    added: Option<([bool; 2], ToneEnergy)>,
    #[serde(skip)]
    intervention: Option<ScheduledRelease>,
    pub energy_ratio: Option<ratios::CandidateRatio>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Record {
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: Option<u32>,
    pub tone_id: Option<u64>,
    pub issued_at: u64,
    pub decision_at: u64,
    pub scope: &'static str,
    pub sample_rate: u32,
    pub candidate_times: Vec<u64>,
    pub retained_tones: usize,
    pub default_input: Input,
    pub default_basis: &'static str,
    pub default_routed: [bool; 2],
    pub default_schedule: Option<DefaultSchedule>,
    pub onset_trace_origin: Option<crate::life::participation_trace::Origin>,
    pub release_trace_context: Option<release_trace::Context>,
    pub onset_trace_default_sample: Option<u64>,
    pub shared_usage: &'static str,
    pub shared_origins: [Option<consumer::Key>; 2],
    pub shared_bytes: [usize; 2],
    pub shared_bindings: [Option<consumer::Binding>; 2],
    pub energy_ratio_context: Option<ratios::Context>,
    pub candidates: Vec<Candidate>,
    pub processing_us: u64,
}

fn evaluate(
    request: Request,
    retained: &[(u64, [bool; 2], ToneEnergy)],
    trace: Option<&crate::life::participation_trace::Frozen>,
) -> Option<Record> {
    let started = std::time::Instant::now();
    let decision = request.opportunity.at;
    if request.sample_rate == 0
        || request.hop == 0
        || request.opportunity.issued_at != request.issued_at
        || decision < request.issued_at
        || decision != request.recipe.envelope.onset
        || request
            .recipe
            .scheduled_release
            .is_some_and(|r| r.off_sample < decision)
        || retained.len() > CAPACITY
    {
        return None;
    }
    let horizon = u64::from(request.sample_rate) * 4;
    let trace = trace.filter(|t| {
        t.head == crate::temporal_cognition::private_trace::Head::Onset
            && t.origin.source_id == request.source_id
            && t.origin.source_generation == request.source_generation
            && t.origin.issued_at_sample == request.issued_at
            && t.origin.bus == 0
            && request.routed[0]
    });
    let trace_default = decision.checked_add(1);
    let span = request
        .opportunity
        .intrinsic_period_ticks
        .map(|p| p.saturating_mul(2).min(horizon));
    let (times, count) = super::times(decision, span, decision, [None; 3])?;
    let active = |at| {
        retained.iter().any(|(_, _, t)| {
            let e = t
                .scheduled_release
                .map_or(t.envelope, |p| t.envelope.with_release(p.off_sample));
            e.onset <= at && at < e.release_end
        })
    };
    let widths = [
        u64::from(request.sample_rate).div_ceil(4),
        u64::from(request.sample_rate),
        u64::from(request.sample_rate) * 2,
        horizon,
    ];
    let mut default: Option<[[Option<f64>; 4]; 2]> = None;
    // Only the physical windows are shared; candidate semantics stay distinct.
    let mut unchanged: Option<Projection> = None;
    let mut releases: [Option<Projection>; 16] = [None; 16];
    let mut candidates = Vec::with_capacity(7 * count);
    for class in [
        Class::OnsetNow,
        Class::DelayedOnset,
        Class::Wait,
        Class::Skip,
        Class::Continue,
        Class::Release,
        Class::Gap,
    ] {
        for (time_index, &at) in times[..count].iter().enumerate() {
            let Some(input) = class.input(
                decision,
                at,
                request.opportunity.intrinsic_period_ticks,
                request.sample_rate,
                BodyState {
                    permits_action: Some(true),
                    active_at_candidate: Some(active(at)),
                    pending_opportunity: true,
                    due_unconsumed: request.opportunity.intrinsic_due_at == Some(decision),
                },
            ) else {
                continue;
            };
            let added = if input.excitation_at.is_some() {
                let shift = at - decision;
                let mut tone = request.recipe;
                tone.envelope.onset = tone.envelope.onset.checked_add(shift)?;
                tone.envelope.hold_end = tone.envelope.hold_end.checked_add(shift)?;
                tone.envelope.release_end = tone.envelope.release_end.checked_add(shift)?;
                tone.sine = tone.sine.map(|mut s| {
                    if shift > 0 {
                        return s.for_new_onset(
                            at,
                            crate::life::schedule_renderer::modal_phase_seed(
                                request.source_id,
                                at,
                                request.tone_id,
                            ),
                        );
                    }
                    s.first_sample = at;
                    s
                });
                if let Some(control) = tone.control.as_mut() {
                    control.issued_at =
                        request.issued_at + (at - request.issued_at) / request.hop * request.hop;
                    control.starts_at = match control.starts_at {
                        Some(x) => Some(x.checked_add(shift)?),
                        None => None,
                    };
                    control.kick_at = match control.kick_at {
                        Some(x) => Some(x.checked_add(shift)?),
                        None => None,
                    };
                }
                tone.scheduled_release = match request.recipe.scheduled_release {
                    Some(release) => {
                        let off = release.off_sample.checked_add(shift)?;
                        Some(ScheduledRelease {
                            apply_at_sample: request.issued_at
                                + (off - request.issued_at) / request.hop * request.hop,
                            off_sample: off,
                        })
                    }
                    None => None,
                };
                Some((request.routed, tone))
            } else {
                None
            };
            let intervention = input.release_at.map(|off| ScheduledRelease {
                apply_at_sample: request.issued_at
                    + (off - request.issued_at) / request.hop * request.hop,
                off_sample: off,
            });
            let withheld = input.withhold_until.is_some_and(|until| {
                retained
                    .iter()
                    .any(|(_, _, tone)| at <= tone.envelope.onset && tone.envelope.onset < until)
            });
            let cached = if added.is_none() && !withheld {
                if input.release_at.is_some() {
                    releases[time_index]
                } else {
                    unchanged
                }
            } else {
                None
            };
            let projection = if let Some(projection) = cached {
                projection
            } else {
                let mut buses = [[None; 4]; 2];
                for bus in 0..2 {
                    for (i, width) in widths.into_iter().enumerate() {
                        let mut window = project_window(
                            retained,
                            added,
                            at,
                            (intervention, input.withhold_until),
                            bus,
                            [decision, decision.checked_add(width)?],
                            true,
                        )?;
                        window.default_mean = default.map_or(window.mean, |d| d[bus][i]);
                        window.difference =
                            window.mean.zip(window.default_mean).map(|(a, b)| a - b);
                        buses[bus][i] = Some(Summary {
                            interval: [window.start, window.end],
                            mean: window.mean,
                            default_mean: window.default_mean,
                            difference: window.difference,
                            supported_bins: window.energies.iter().flatten().count() as u8,
                            coherent_bins: window.coherent_energies.iter().flatten().count() as u8,
                        });
                    }
                }
                let projection = Projection { buses };
                if added.is_none() && !withheld {
                    if input.release_at.is_some() {
                        releases[time_index] = Some(projection);
                    } else {
                        unchanged = Some(projection);
                    }
                }
                projection
            };
            let buses = projection.buses;
            if class == Class::OnsetNow {
                default = Some(buses.map(|windows| windows.map(|w| w.and_then(|v| v.mean))));
            }
            let onset_trace = trace
                .zip(trace_default)
                .zip(input.excitation_at.and_then(|at| at.checked_add(1)))
                .map(|((trace, default), event_sample)| {
                    crate::life::participation_trace::TimingCandidate {
                        event_sample,
                        fits: trace.fit(request.sample_rate, event_sample, default),
                    }
                });
            candidates.push(Candidate {
                input,
                buses,
                onset_trace,
                release_trace: Vec::new(),
                shared: [consumer::Pair::NoTable; 2],
                added,
                intervention,
                energy_ratio: None,
            });
        }
    }
    let default_input = candidates.first()?.input;
    Some(Record {
        source_id: request.source_id,
        source_generation: request.source_generation,
        body_generation: request.body_generation,
        tone_id: Some(request.tone_id),
        issued_at: request.issued_at,
        decision_at: decision,
        scope: "Conditional on the first eligible accepted recipe for the sampled source after all issued-hop batches, other owned accepted/queued tones and known releases. No unissued future policy commands. The focused default_input is distinct from the first transition in default_schedule. Diagnostic energy approximation; not full future policy, calibrated consequence or learning evidence.",
        sample_rate: request.sample_rate,
        candidate_times: times[..count].to_vec(),
        retained_tones: retained.len(),
        default_input,
        default_basis: "accepted_onset_recipe",
        default_routed: std::array::from_fn(|bus| {
            request.routed[bus] || retained.iter().any(|(_, routed, _)| routed[bus])
        }),
        default_schedule: None,
        onset_trace_origin: trace.map(|t| t.origin),
        onset_trace_default_sample: trace.and(trace_default),
        release_trace_context: None,
        shared_usage: "Prototype base-head diagnostics (closure/continuation and optional groove/desire). Descriptor association is not validated body transfer; no ordinal action pressure or calibrated consequence.",
        shared_origins: [None; 2],
        shared_bytes: [0; 2],
        shared_bindings: [None; 2],
        energy_ratio_context: None,
        candidates,
        processing_us: started.elapsed().as_micros() as u64,
    })
}

fn evaluate_scheduled(
    request: ScheduledRequest,
    retained: &[(u64, [bool; 2], ToneEnergy)],
) -> Option<Record> {
    let started = std::time::Instant::now();
    if request.sample_rate == 0
        || request.hop == 0
        || retained.is_empty()
        || retained.len() > CAPACITY
    {
        return None;
    }
    let active = |at| {
        retained.iter().any(|(_, _, t)| {
            let e = t
                .scheduled_release
                .map_or(t.envelope, |p| t.envelope.with_release(p.off_sample));
            e.onset <= at && at < e.release_end
        })
    };
    let (default_class, default_at) = if active(request.issued_at) {
        (Class::Continue, request.issued_at)
    } else {
        let onset = retained
            .iter()
            .filter_map(|(_, _, t)| {
                let onset = t.envelope.onset;
                (onset > request.issued_at
                    && t.renderer_end_after(request.issued_at, None)
                        .is_some_and(|end| onset < end))
                .then_some(onset)
            })
            .min()?;
        (Class::DelayedOnset, onset)
    };
    let horizon = u64::from(request.sample_rate) * 4;
    let span = request.period.map(|p| p.saturating_mul(2).min(horizon));
    let (times, count) = super::times(request.issued_at, span, default_at, [None; 3])?;
    let widths = [
        u64::from(request.sample_rate).div_ceil(4),
        u64::from(request.sample_rate),
        u64::from(request.sample_rate) * 2,
        horizon,
    ];
    let mut default: Option<[[Option<f64>; 4]; 2]> = None;
    // Only the physical windows are shared; candidate semantics stay distinct.
    let mut unchanged: Option<Projection> = None;
    let mut releases: [Option<Projection>; 16] = [None; 16];
    let mut candidates = Vec::with_capacity(3 * count);
    for class in [default_class, Class::Release, Class::Gap] {
        for (time_index, &at) in times[..count].iter().enumerate() {
            // A queued default describes the retained schedule, not a new excitation.
            if class == default_class && at != default_at {
                continue;
            }
            if default_class == Class::DelayedOnset
                && class != default_class
                && span.is_none_or(|span| at - request.issued_at > span)
            {
                continue;
            }
            let Some(input) = class.input(
                request.issued_at,
                at,
                request.period,
                request.sample_rate,
                BodyState {
                    permits_action: Some(true),
                    active_at_candidate: Some(active(at)),
                    pending_opportunity: false,
                    due_unconsumed: false,
                },
            ) else {
                continue;
            };
            let intervention = input.release_at.map(|off| ScheduledRelease {
                apply_at_sample: request.issued_at
                    + (off - request.issued_at) / request.hop * request.hop,
                off_sample: off,
            });
            let withheld = input.withhold_until.is_some_and(|until| {
                retained
                    .iter()
                    .any(|(_, _, tone)| at <= tone.envelope.onset && tone.envelope.onset < until)
            });
            let cached = if withheld {
                None
            } else if input.release_at.is_some() {
                releases[time_index]
            } else {
                unchanged
            };
            let projection = if let Some(projection) = cached {
                projection
            } else {
                let mut buses = [[None; 4]; 2];
                for bus in 0..2 {
                    for (i, width) in widths.into_iter().enumerate() {
                        let window = project_window(
                            retained,
                            None,
                            at,
                            (intervention, input.withhold_until),
                            bus,
                            [request.issued_at, request.issued_at.checked_add(width)?],
                            true,
                        )?;
                        let default_mean = default.map_or(window.mean, |d| d[bus][i]);
                        buses[bus][i] = Some(Summary {
                            interval: [window.start, window.end],
                            mean: window.mean,
                            default_mean,
                            difference: window.mean.zip(default_mean).map(|(a, b)| a - b),
                            supported_bins: window.energies.iter().flatten().count() as u8,
                            coherent_bins: window.coherent_energies.iter().flatten().count() as u8,
                        });
                    }
                }
                let projection = Projection { buses };
                if !withheld && input.release_at.is_some() {
                    releases[time_index] = Some(projection);
                } else if !withheld {
                    unchanged = Some(projection);
                }
                projection
            };
            let buses = projection.buses;
            if class == default_class {
                default = Some(buses.map(|windows| windows.map(|w| w.and_then(|v| v.mean))));
            }
            candidates.push(Candidate {
                input,
                buses,
                onset_trace: None,
                release_trace: Vec::new(),
                shared: [consumer::Pair::NoTable; 2],
                added: None,
                intervention,
                energy_ratio: None,
            });
        }
    }
    let default_input = candidates.first()?.input;
    Some(Record {
        source_id: request.source_id,
        source_generation: request.source_generation,
        body_generation: Some(request.body_generation),
        tone_id: None,
        issued_at: request.issued_at,
        decision_at: request.issued_at,
        scope: if default_class == Class::Continue {
            "Conditional on the active owned body after all issued-hop batches, queued onsets and known releases, with no unissued future policy commands. Continue follows this post-command schedule; first_transition records the actual accepted default separately. Diagnostic energy approximation, not full future policy, calibrated consequence or learning evidence."
        } else {
            "Conditional on the idle owned body with accepted queued onsets after all issued-hop batches and known releases. The delayed_onset default describes the existing schedule and adds no recipe; it is not an unconsumed opportunity. Windows start at issue, including pre-onset silence. No unissued future policy commands, calibrated consequence or learning evidence."
        },
        sample_rate: request.sample_rate,
        candidate_times: times[..count].to_vec(),
        retained_tones: retained.len(),
        default_input,
        default_basis: if default_class == Class::Continue {
            "active_body"
        } else {
            "accepted_queued_onset"
        },
        default_routed: std::array::from_fn(|bus| {
            retained.iter().any(|(_, routed, _)| routed[bus])
        }),
        default_schedule: None,
        onset_trace_origin: None,
        onset_trace_default_sample: None,
        release_trace_context: None,
        shared_usage: "Prototype base-head diagnostics (closure/continuation and optional groove/desire). Descriptor association is not validated body transfer; no ordinal action pressure or calibrated consequence.",
        shared_origins: [None; 2],
        shared_bytes: [0; 2],
        shared_bindings: [None; 2],
        energy_ratio_context: None,
        candidates,
        processing_us: started.elapsed().as_micros() as u64,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Stats {
    pub submitted: u64,
    pub capacity_dropped: u64,
    pub unsupported: u64,
    pub completed: u64,
    pub worker_unsupported: u64,
    pub output_dropped: u64,
    pub max_processing_us: u64,
    pub worker_failed: bool,
}

pub(crate) struct Worker {
    free: Receiver<Box<Packet>>,
    recycle: Sender<Box<Packet>>,
    input: Option<Sender<Box<Packet>>>,
    output: Receiver<Record>,
    handle: Option<JoinHandle<()>>,
    counters: Arc<[AtomicU64; 4]>,
    pub stats: Stats,
}

impl Worker {
    pub fn new() -> Self {
        let (recycle, free) = bounded(CAPACITY);
        let (input, rx) = bounded::<Box<Packet>>(CAPACITY);
        let (tx, output) = bounded(CAPACITY);
        for _ in 0..CAPACITY {
            recycle
                .send(Box::new(Packet {
                    request: None,
                    scheduled: None,
                    retained: Vec::with_capacity(CAPACITY),
                    trace: None,
                    release_trace: None,
                    shared: [None, None],
                    bindings: [None; 2],
                    default_schedule: None,
                    external: None,
                }))
                .unwrap();
        }
        let counters = Arc::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let shared = Arc::clone(&counters);
        let returned = recycle.clone();
        let handle = std::thread::spawn(move || {
            for mut packet in rx {
                let evaluated = match (packet.request, packet.scheduled) {
                    (Some(r), None) => evaluate(r, &packet.retained, packet.trace.as_ref()),
                    (None, Some(r)) => evaluate_scheduled(r, &packet.retained),
                    _ => None,
                };
                if let Some(mut record) = evaluated {
                    let started = std::time::Instant::now();
                    ratios::attach(&mut record, &packet.retained, packet.external.as_ref());
                    release_trace::attach(
                        &mut record,
                        &packet.retained,
                        packet.release_trace.as_ref(),
                    );
                    record.shared_bindings = packet.bindings;
                    record.default_schedule = packet.default_schedule;
                    for bus in 0..2 {
                        if let Some(table) = &packet.shared[bus] {
                            record.shared_origins[bus] = Some(table.key);
                            record.shared_bytes[bus] = table.bytes();
                            for candidate in &mut record.candidates {
                                candidate.shared[bus] = if record.default_basis
                                    == "accepted_queued_onset"
                                {
                                    consumer::Pair::QueuedDefault
                                } else if record.default_schedule.is_some_and(|s| {
                                    s.accepted_transitions > 1
                                        || (s.accepted_transitions > 0 && record.tone_id.is_none())
                                        || s.queued_tones
                                            > usize::from(record.decision_at > record.issued_at)
                                        || s.unsupported_control_tones > 0
                                        || s.amplitude_smoothing_tones > 0
                                        || s.scheduled_amplitude_tones > 0
                                }) {
                                    consumer::Pair::CompositeDefault
                                } else if !record.default_routed[bus] {
                                    consumer::Pair::UnroutedDefault
                                } else {
                                    table.pair(
                                        packet.bindings[bus],
                                        (
                                            record.source_id,
                                            record.source_generation,
                                            record.body_generation,
                                        ),
                                        bus,
                                        record.issued_at,
                                        candidate.input,
                                        record.default_input,
                                    )
                                };
                            }
                        }
                    }
                    record.processing_us += started.elapsed().as_micros() as u64;
                    shared[0].fetch_add(1, Ordering::Relaxed);
                    shared[3].fetch_max(record.processing_us, Ordering::Relaxed);
                    if tx.try_send(record).is_err() {
                        shared[1].fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    shared[2].fetch_add(1, Ordering::Relaxed);
                }
                packet.request = None;
                packet.scheduled = None;
                packet.retained.clear();
                packet.trace = None;
                packet.release_trace = None;
                packet.shared = [None, None];
                packet.bindings = [None; 2];
                packet.default_schedule = None;
                packet.external = None;
                if returned.send(packet).is_err() {
                    break;
                }
            }
        });
        Self {
            free,
            recycle,
            input: Some(input),
            output,
            handle: Some(handle),
            counters,
            stats: Stats::default(),
        }
    }
    pub fn acquire(&mut self) -> Option<Box<Packet>> {
        match self.free.try_recv() {
            Ok(mut packet) => {
                packet.request = None;
                packet.scheduled = None;
                packet.retained.clear();
                packet.trace = None;
                packet.release_trace = None;
                packet.shared = [None, None];
                packet.bindings = [None; 2];
                packet.default_schedule = None;
                packet.external = None;
                Some(packet)
            }
            Err(_) => {
                self.stats.capacity_dropped += 1;
                None
            }
        }
    }
    pub fn submit(&mut self, packet: Box<Packet>, supported: bool) {
        if supported && let Some(tx) = &self.input {
            match tx.try_send(packet) {
                Ok(()) => self.stats.submitted += 1,
                Err(e) => {
                    self.stats.capacity_dropped += 1;
                    let _ = self.recycle.try_send(e.into_inner());
                }
            }
        } else {
            self.stats.unsupported += 1;
            let _ = self.recycle.try_send(packet);
        }
    }
    pub fn poll(&mut self) {
        self.stats.completed = self.counters[0].load(Ordering::Relaxed);
        self.stats.worker_unsupported = self.counters[2].load(Ordering::Relaxed);
        self.stats.output_dropped = self.counters[1].load(Ordering::Relaxed);
        self.stats.max_processing_us = self.counters[3].load(Ordering::Relaxed);
    }
    pub fn drain(&mut self) -> impl Iterator<Item = Record> + '_ {
        self.output.try_iter()
    }
    pub fn finish(&mut self) {
        self.input.take();
        if let Some(handle) = self.handle.take() {
            self.stats.worker_failed = handle.join().is_err();
        }
        self.poll();
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::{
        control_forecast::{AmplitudeModel, ControlForecast},
        envelope::Envelope,
    };

    pub(super) fn tone(onset: u64) -> ToneEnergy {
        ToneEnergy {
            amplitude: 0.5,
            envelope: Envelope {
                onset,
                hold_end: 100_000,
                release_end: 100_000,
                attack_ticks: 0,
                decay_ticks: 0,
                sustain_level: 1.,
                decay_lambda: 0.,
                release_ticks: 0,
            },
            control: Some(ControlForecast {
                issued_at: 0,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 8000.,
                starts_at: Some(onset),
                kick_at: None,
                model: AmplitudeModel::Unmodulated { gain: 1. },
            }),
            scheduled_release: None,
            sine: None,
        }
    }

    #[test]
    fn candidate_energy_ratios_keep_a_common_context_and_supported_default() {
        use crate::core::temporal_expectation::TemporalForecast;
        let request = Request {
            source_id: 2,
            source_generation: 7,
            body_generation: Some(3),
            tone_id: 1,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            opportunity: OnsetOpportunity {
                issued_at: 800,
                at: 800,
                gate: 0,
                intrinsic_due_at: Some(800),
                intrinsic_period_ticks: Some(4000),
                planned_release_at: None,
            },
            routed: [true, false],
            recipe: tone(800),
        };
        let external = TemporalForecast::energy_fixture(8000, 800, |_| [0.375, 0., 0.]);
        let mut record = evaluate(request, &[], None).unwrap();
        ratios::attach(&mut record, &[], Some(&external));
        let context = record.energy_ratio_context.as_ref().unwrap();
        assert_eq!(context.bus, 0);
        assert_eq!(context.external_origin, Some(800));
        assert_eq!(context.common_horizon, [800, 32800]);
        let default = record.candidates[0].energy_ratio.as_ref().unwrap();
        assert_eq!(
            default.external.unwrap().horizon_intersection,
            Some([800, 32800])
        );
        assert_eq!(default.status, "supported");
        assert!((default.overlap.unwrap() - 0.375 / (0.5 + 1e-12)).abs() < 1e-12);
        assert!((default.audibility.unwrap() - 0.125 / (0.5 + 1e-12)).abs() < 1e-12);
        assert_eq!(default.overlap_difference, Some(0.));
        assert_eq!(default.audibility_difference, Some(0.));
        let mut supported_changes = 0;
        for candidate in &record.candidates {
            let r = candidate.energy_ratio.as_ref().unwrap();
            assert_eq!(r.default_overlap, default.overlap);
            assert_eq!(r.default_audibility, default.audibility);
            assert_eq!(
                r.overlap_difference,
                r.overlap.zip(default.overlap).map(|(a, b)| a - b)
            );
            if candidate.input.class == Class::Wait || candidate.input.class == Class::Skip {
                assert_eq!(r.status, "known_silent_model");
                assert_eq!(r.overlap, Some(0.));
                assert_eq!(r.audibility, None);
                assert_eq!(r.overlap_difference, default.overlap.map(|v| -v));
            }
            supported_changes += usize::from(r.overlap_difference.is_some_and(|d| d != 0.));
        }
        assert!(supported_changes > 0);
        let mut unknown = request;
        unknown.recipe.control = None;
        let mut record = evaluate(unknown, &[(8, [true, false], tone(0))], None).unwrap();
        ratios::attach(&mut record, &[(8, [true, false], tone(0))], Some(&external));
        assert_eq!(
            record.candidates[0].energy_ratio.as_ref().unwrap().status,
            "unsupported"
        );
        let wait = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Wait)
            .unwrap();
        let ratio = wait.energy_ratio.as_ref().unwrap();
        assert_eq!(ratio.status, "supported");
        assert_eq!(ratio.default_overlap, None);
        assert_eq!(ratio.overlap_difference, None);
        assert_eq!(ratio.audibility_difference, None);
    }

    #[test]
    fn candidate_energy_ratios_report_clipped_support_and_hidden_external_gaps() {
        use crate::core::temporal_expectation::TemporalForecast;
        for start in [800, 1_u64 << 55] {
            let request = ScheduledRequest {
                source_id: 2,
                source_generation: 7,
                body_generation: 3,
                issued_at: start,
                sample_rate: 8000,
                hop: 400,
                period: None,
            };
            let mut t = tone(start);
            t.envelope.hold_end = start + 100_000;
            t.envelope.release_end = start + 100_000;
            let retained = [(1, [true, false], t)];
            let external = TemporalForecast::energy_fixture(8000, start - 800, |_| [0.; 3]);
            let mut record = evaluate_scheduled(request, &retained).unwrap();
            ratios::attach(&mut record, &retained, Some(&external));
            let context = record.energy_ratio_context.unwrap();
            assert_eq!(context.common_horizon, [start, start + 31200]);
            let ratio = record.candidates[0].energy_ratio.as_ref().unwrap();
            let footprint = ratio.external.unwrap();
            assert_eq!(footprint.requested, [start, start + 100_000]);
            assert_eq!(footprint.horizon_intersection, Some([start, start + 31200]));
            assert_eq!(footprint.omitted_tail_samples, Some(68800));
            assert_eq!(ratio.status, "supported");
            assert_eq!(ratio.overlap, Some(0.));
            assert_eq!(ratio.excluded_energy_mass, None);
            let broken = TemporalForecast::energy_fixture(8000, start, |t| {
                if (t - 0.02).abs() < 1e-8 {
                    [f32::NAN; 3]
                } else {
                    [0.; 3]
                }
            });
            let mut record = evaluate_scheduled(request, &retained).unwrap();
            ratios::attach(&mut record, &retained, Some(&broken));
            let footprint = record.candidates[0]
                .energy_ratio
                .as_ref()
                .unwrap()
                .external
                .unwrap();
            assert_eq!(footprint.continuous_support, Some(false));
            assert!(
                footprint
                    .points
                    .iter()
                    .flatten()
                    .all(|p| p.band_energy_sum.is_some())
            );
            assert!(record.candidates.iter().all(|c| {
                let r = c.energy_ratio.as_ref().unwrap();
                (r.status == "unsupported" && r.overlap.is_none() && r.audibility.is_none())
                    || (r.status == "known_silent_model" && r.body_support.is_none())
            }));
            let mut record = evaluate_scheduled(request, &retained).unwrap();
            ratios::attach(&mut record, &retained, None);
            assert!(record.candidates.iter().all(|c| {
                let r = c.energy_ratio.as_ref().unwrap();
                r.status == "external_unavailable"
                    || (r.status == "known_silent_model" && r.body_support.is_none())
            }));
        }
    }

    #[test]
    fn ordinary_onset_candidates_match_their_fresh_rendered_carrier_phase() {
        use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
        use crate::life::{
            action_candidates::PolicySnapshot,
            action_observation::Observer,
            phonation_engine::{OnsetKick, ToneCmd},
            schedule_renderer::ScheduleRenderer,
            sound::{BodyKind, BodySnapshot, RenderModulatorSpec, ToneAdsr},
            voice::{PhonationBatch, ToneSpec},
        };
        let time = Timebase { fs: 8000., hop: 64 };
        let rhythms = NeuralRhythms::default();
        let mut compared = 0;
        let mut maximum_error = 0.0f64;
        for freq_hz in [250., 5000.] {
            let mut live = ScheduleRenderer::new(time);
            let mut observer = Observer::new(8000);
            observer.enable_predictions();
            live.action_observer = Some(Box::new(observer));
            let mut batch = PhonationBatch {
                source_id: 2,
                source_generation: 7,
                tones: vec![ToneSpec {
                    opportunity: None,
                    tone_id: 10,
                    onset: 0,
                    hold_ticks: Some(48000),
                    freq_hz,
                    amp: 0.25,
                    smoothing_tau_sec: 0.,
                    body: BodySnapshot {
                        kind: BodyKind::Sine,
                        amp_scale: 1.,
                        brightness: 0.4,
                        inharmonic: 0.,
                        spread: 0.,
                        unison: 1,
                        motion: 0.,
                        ratios: None,
                    },
                    render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 10. },
                    adsr: Some(ToneAdsr {
                        attack_sec: 0.,
                        decay_sec: 0.,
                        sustain_level: 1.,
                        release_sec: 0.,
                    }),
                }],
                cmds: vec![ToneCmd::On {
                    tone_id: 10,
                    kick: OnsetKick { strength: 1. },
                }],
                ..Default::default()
            };
            for now in (0..512).step_by(64) {
                live.render(
                    if now == 0 {
                        std::slice::from_ref(&batch)
                    } else {
                        &[]
                    },
                    now,
                    &rhythms,
                );
            }
            let frozen = live.fork_source(2);
            batch.tones[0].tone_id = 11;
            batch.tones[0].onset = 512;
            batch.tones[0].opportunity = Some(OnsetOpportunity {
                issued_at: 512,
                at: 512,
                gate: 1,
                intrinsic_due_at: Some(512),
                intrinsic_period_ticks: Some(4000),
                planned_release_at: None,
            });
            batch.cmds[0] = ToneCmd::On {
                tone_id: 11,
                kick: OnsetKick { strength: 1. },
            };
            batch.body_policy = Some(PolicySnapshot {
                at: 512,
                is_alive: true,
                gate_allows_onset: true,
            });
            batch.intrinsic_period_sec = Some(0.5);
            live.render(std::slice::from_ref(&batch), 512, &rhythms);
            let observer = live.action_observer.as_mut().unwrap();
            observer.finish();
            assert_eq!(observer.snapshot.commands, 2);
            let records: Vec<_> = observer.drain_candidate_energy().collect();
            assert_eq!(records.len(), 1);
            let onset_candidates: Vec<_> = records[0]
                .candidates
                .iter()
                .filter(|c| c.input.excitation_at.is_some())
                .collect();
            assert_eq!(onset_candidates.len(), 12);
            for candidate in onset_candidates {
                let at = candidate.input.at;
                let predicted = candidate.added.unwrap().1;
                let mut actual = frozen.fork_source(2);
                let mut retained = frozen.fork_source(2);
                let mut action = batch.clone();
                action.body_policy = None;
                action.tones[0].onset = at;
                action.tones[0].opportunity = None;
                for now in (512..at + 64).step_by(64) {
                    let sent = if now <= at && at < now + 64 {
                        std::slice::from_ref(&action)
                    } else {
                        &[]
                    };
                    let rendered = actual.render(sent, now, &rhythms);
                    let reference = retained.render(&[], now, &rhythms);
                    for offset in 0..64 {
                        let tick = now + offset as u64;
                        if tick < at || tick >= at + 64 {
                            continue;
                        }
                        let estimate = predicted.sine_point(tick, None).unwrap().0[1];
                        for (mixed, old) in [
                            (rendered.habitat[offset], reference.habitat[offset]),
                            (
                                rendered.presentation[offset],
                                reference.presentation[offset],
                            ),
                        ] {
                            let observed = f64::from(mixed) - f64::from(old);
                            let error = (estimate - observed).abs();
                            maximum_error = maximum_error.max(error);
                            assert!(
                                error < 0.002,
                                "freq={freq_hz} at={at} tick={tick} estimate={estimate} observed={observed}"
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "ordinary candidate carrier comparisons={compared}, max absolute error={maximum_error}"
        );
    }

    #[test]
    fn delayed_onset_keeps_control_update_absolute_and_release_keeps_known_silence() {
        let mut recipe = tone(800);
        recipe.control.as_mut().unwrap().valid_until = Some(3000);
        let record = evaluate(
            Request {
                source_id: 2,
                source_generation: 7,
                body_generation: Some(3),
                tone_id: 1,
                issued_at: 800,
                sample_rate: 8000,
                hop: 400,
                opportunity: OnsetOpportunity {
                    issued_at: 800,
                    at: 800,
                    gate: 0,
                    intrinsic_due_at: Some(800),
                    intrinsic_period_ticks: Some(4000),
                    planned_release_at: None,
                },
                routed: [true; 2],
                recipe,
            },
            &[],
            None,
        )
        .unwrap();
        assert_eq!(record.candidates[0].buses[0][0].unwrap().mean, Some(0.125));
        let delayed = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::DelayedOnset && c.input.at >= 3000)
            .unwrap();
        let (_, added) = delayed.added.unwrap();
        assert_eq!(added.control.unwrap().valid_until, Some(3000));
        assert_eq!(added.at(delayed.input.at - 1, None), Some(0.));
        assert_eq!(added.at(delayed.input.at, None), None);
        assert!(delayed.buses.iter().all(|b| b[3].unwrap().mean.is_none()));

        let retained = [(1, [true; 2], recipe)];
        let body = Body {
            retained: &retained,
            added: None,
            at: 800,
            intervention: (None, None),
        };
        assert!(body.known_on([800, 3000], 0));
        assert!(!body.known_on([800, 3001], 0));
        let released = Body {
            intervention: (
                Some(ScheduledRelease {
                    apply_at_sample: 2800,
                    off_sample: 2800,
                }),
                None,
            ),
            ..body
        };
        for bus in 0..2 {
            assert!(released.known_on([800, 32800], bus));
            assert_eq!(released.point(3000, 3000, 3001, bus).0, Some(0.));
        }
    }

    #[test]
    fn physical_windows_preserve_known_zero_unknown_and_release_selection() {
        let mut unknown = tone(0);
        unknown.control = None;
        let retained = [(1, [true, false], tone(0)), (2, [false, true], unknown)];
        let h = project_window(&retained, None, 0, (None, None), 0, [0, 160], true).unwrap();
        assert_eq!(h.mean, Some(0.125));
        assert!(h.coherent_energies.iter().all(Option::is_none));
        let p = project_window(&retained, None, 0, (None, None), 1, [0, 160], true).unwrap();
        assert_eq!(p.mean, None);
        let release = Some(ScheduledRelease {
            apply_at_sample: 80,
            off_sample: 80,
        });
        let h = project_window(&retained, None, 0, (release, None), 0, [0, 160], true).unwrap();
        assert_eq!(h.mean, Some(0.0625));
        let p = project_window(&retained, None, 0, (release, None), 1, [80, 160], true).unwrap();
        assert_eq!(p.mean, Some(0.));
        let queued = [(1, [true, false], tone(80))];
        assert_eq!(
            project_window(&queued, None, 0, (release, None), 0, [0, 160], false)
                .unwrap()
                .mean,
            Some(0.0625)
        );
        assert!(project_window(&[], None, 0, (None, None), 2, [0, 160], true).is_none());
        assert!(project_window(&[], None, 0, (None, None), 0, [160, 0], true).is_none());
        assert!(project_window(&[], None, 0, (None, None), 0, [0, 15], true).is_none());
        assert!(
            project_window(
                &[(1, [true; 2], tone(0)); 65],
                None,
                0,
                (None, None),
                0,
                [0, 160],
                true
            )
            .is_none()
        );
    }

    #[test]
    fn reused_windows_match_direct_projection_across_inputs_and_requests() {
        let request = Request {
            source_id: 2,
            source_generation: 7,
            body_generation: Some(3),
            tone_id: 3,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            opportunity: OnsetOpportunity {
                issued_at: 800,
                at: 800,
                gate: 9,
                intrinsic_due_at: Some(800),
                intrinsic_period_ticks: Some(4000),
                planned_release_at: None,
            },
            routed: [true, false],
            recipe: tone(800),
        };
        for scheduled in [false, true] {
            let mut left = tone(0);
            left.envelope.release_ticks = 400;
            if scheduled {
                left.scheduled_release = Some(ScheduledRelease {
                    apply_at_sample: 1600,
                    off_sample: 1800,
                });
            }
            let mut right = left;
            right.control = None;
            let retained = [(1, [true, false], left), (2, [false, true], right)];
            let records = [
                evaluate(request, &retained, None).unwrap(),
                evaluate_scheduled(
                    ScheduledRequest {
                        source_id: 2,
                        source_generation: 7,
                        body_generation: 3,
                        issued_at: 800,
                        sample_rate: 8000,
                        hop: 400,
                        period: Some(4000),
                    },
                    &retained,
                )
                .unwrap(),
            ];
            for record in records {
                let baseline = record.candidates[0].buses;
                let mut nontrivial = false;
                for c in record
                    .candidates
                    .iter()
                    .filter(|c| c.input.excitation_at.is_none())
                {
                    let intervention = c.input.release_at.map(|off| ScheduledRelease {
                        apply_at_sample: 800 + (off - 800) / 400 * 400,
                        off_sample: off,
                    });
                    for (bus, windows) in c.buses.iter().enumerate() {
                        for (i, actual) in windows.iter().enumerate() {
                            let actual = actual.unwrap();
                            let direct = project_window(
                                &retained,
                                None,
                                c.input.at,
                                (intervention, c.input.withhold_until),
                                bus,
                                actual.interval,
                                true,
                            )
                            .unwrap();
                            assert_eq!(actual.mean, direct.mean);
                            assert_eq!(
                                actual.supported_bins,
                                direct.energies.iter().flatten().count() as u8
                            );
                            assert_eq!(
                                actual.coherent_bins,
                                direct.coherent_energies.iter().flatten().count() as u8
                            );
                            assert_eq!(actual.default_mean, baseline[bus][i].unwrap().mean);
                            assert_eq!(
                                actual.difference,
                                direct.mean.zip(actual.default_mean).map(|(a, b)| a - b)
                            );
                            nontrivial |= actual.difference.is_some_and(|d| d != 0.);
                        }
                    }
                }
                assert!(nontrivial);
                if scheduled {
                    assert!(
                        record
                            .candidates
                            .iter()
                            .any(|c| c.input.class == Class::Gap && c.input.release_at.is_none())
                    );
                }
                assert!(
                    record
                        .candidates
                        .iter()
                        .any(|c| c.buses[1].iter().flatten().any(|w| w.mean.is_none()))
                );
            }
        }
    }

    #[test]
    fn gap_withholds_queued_onsets_in_half_open_interval_and_keeps_release_tail() {
        let mut active = tone(0);
        active.envelope.release_ticks = 400;
        active.envelope.release_end += 400;
        let retained = [
            (1, [true, false], active),
            (2, [false, true], tone(800)),
            (3, [false, true], tone(1600)),
            (4, [false, true], tone(2400)),
        ];
        let release = Some(ScheduledRelease {
            apply_at_sample: 800,
            off_sample: 800,
        });
        let gap = (release, Some(2400));
        let baseline =
            project_window(&retained, None, 800, (None, None), 1, [800, 2400], true).unwrap();
        assert_eq!(baseline.mean, Some(0.1875));
        let withheld = project_window(&retained, None, 800, gap, 1, [800, 2400], true).unwrap();
        assert_eq!(withheld.mean, Some(0.));
        let at_end = project_window(&retained, None, 800, gap, 1, [2400, 2560], true).unwrap();
        assert_eq!(at_end.mean, Some(0.125));
        let tail = project_window(&retained, None, 800, gap, 0, [800, 960], true).unwrap();
        let release_only =
            project_window(&retained, None, 800, (release, None), 0, [800, 960], true).unwrap();
        assert!(tail.mean.unwrap() > 0.);
        assert_eq!(tail.energies, release_only.energies);
        let ended = project_window(&retained, None, 800, gap, 0, [1200, 1360], true).unwrap();
        assert_eq!(ended.mean, Some(0.));
        let queued = [retained[2]];
        assert_eq!(
            project_window(&queued, None, 800, (release, None), 1, [800, 2400], true)
                .unwrap()
                .mean,
            Some(0.0625)
        );
        let mut unknown = tone(1600);
        unknown.control = None;
        let queued = [(5, [false, true], unknown)];
        assert_eq!(
            project_window(&queued, None, 800, (None, None), 1, [800, 1600], true)
                .unwrap()
                .mean,
            Some(0.)
        );
        assert_eq!(
            project_window(&queued, None, 800, (None, None), 1, [1600, 2400], true)
                .unwrap()
                .mean,
            None
        );
        assert_eq!(
            project_window(&queued, None, 800, gap, 1, [800, 2400], true)
                .unwrap()
                .mean,
            Some(0.)
        );
    }

    #[test]
    fn queued_default_is_preserved_and_gap_does_not_reuse_release_windows() {
        let retained = [(1, [true, false], tone(0)), (2, [false, true], tone(1600))];
        let record = evaluate_scheduled(
            ScheduledRequest {
                source_id: 2,
                source_generation: 7,
                body_generation: 3,
                issued_at: 800,
                sample_rate: 8000,
                hop: 400,
                period: Some(1600),
            },
            &retained,
        )
        .unwrap();
        assert_eq!(record.default_routed, [true; 2]);
        let default = &record.candidates[0];
        assert_eq!(default.buses[0][0].unwrap().mean, Some(0.125));
        assert_eq!(default.buses[1][0].unwrap().mean, Some(0.078125));
        let release = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Release && c.input.at == 800)
            .unwrap();
        let gap = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 800)
            .unwrap();
        assert_eq!(release.buses[1][0].unwrap().mean, Some(0.078125));
        assert_eq!(release.buses[1][0].unwrap().difference, Some(0.));
        assert_eq!(gap.buses[1][0].unwrap().mean, Some(0.));
        assert_eq!(gap.buses[1][0].unwrap().default_mean, Some(0.078125));
        assert_eq!(gap.buses[1][0].unwrap().difference, Some(-0.078125));
        // The onset path shares the same cache, while retaining its focused recipe.
        let onset = evaluate(
            Request {
                source_id: 2,
                source_generation: 7,
                body_generation: Some(3),
                tone_id: 3,
                issued_at: 800,
                sample_rate: 8000,
                hop: 400,
                routed: [true, false],
                recipe: tone(800),
                opportunity: OnsetOpportunity {
                    issued_at: 800,
                    at: 800,
                    gate: 1,
                    intrinsic_due_at: Some(800),
                    intrinsic_period_ticks: Some(1600),
                    planned_release_at: None,
                },
            },
            &retained,
            None,
        )
        .unwrap();
        assert_eq!(onset.candidates[0].buses[0][0].unwrap().mean, Some(0.25));
        let release = onset
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Release && c.input.at == 800)
            .unwrap();
        let gap = onset
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 800)
            .unwrap();
        assert_eq!(release.buses[1][0].unwrap().mean, Some(0.078125));
        assert_eq!(gap.buses[1][0].unwrap().mean, Some(0.));
    }

    #[test]
    fn active_body_keeps_continue_default_known_release_and_unknown_separate() {
        let request = ScheduledRequest {
            source_id: 2,
            source_generation: 7,
            body_generation: 3,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            period: Some(4000),
        };
        let mut known = tone(0);
        known.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 1600,
            off_sample: 1800,
        });
        let mut unknown = known;
        unknown.control = None;
        let retained = [(1, [true, false], known), (2, [false, true], unknown)];
        let record = evaluate_scheduled(request, &retained).unwrap();
        assert_eq!(record.tone_id, None);
        assert_eq!(record.default_input.class, Class::Continue);
        assert!(record.onset_trace_origin.is_none());
        let default = &record.candidates[0];
        assert_eq!(default.buses[0][0].unwrap().mean, Some(0.0625));
        assert_eq!(default.buses[0][0].unwrap().difference, Some(0.));
        assert_eq!(default.buses[1][0].unwrap().mean, None);
        for c in &record.candidates {
            assert!(matches!(
                c.input.class,
                Class::Continue | Class::Release | Class::Gap
            ));
            assert!(c.input.excitation_at.is_none() && c.onset_trace.is_none());
            if c.input.class == Class::Release {
                assert!(c.input.at < 1800);
            }
            if c.input.class == Class::Gap {
                assert_eq!(c.input.withhold_until, Some(c.input.at + 4000));
            }
        }
        let release = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Release && c.input.at == 800)
            .unwrap();
        assert_eq!(release.buses[0][0].unwrap().mean, Some(0.));
        assert_eq!(release.buses[0][0].unwrap().difference, Some(-0.0625));
        assert_eq!(release.buses[1][0].unwrap().mean, Some(0.));
        assert_eq!(release.buses[1][0].unwrap().difference, None);
        let no_period = evaluate_scheduled(
            ScheduledRequest {
                period: None,
                ..request
            },
            &retained,
        )
        .unwrap();
        assert_eq!(no_period.candidates.len(), 2);
        assert!(
            evaluate_scheduled(
                ScheduledRequest {
                    issued_at: 1800,
                    ..request
                },
                &retained
            )
            .is_none()
        );
        assert!(evaluate_scheduled(request, &[]).is_none());
        assert!(evaluate_scheduled(request, &[(1, [true; 2], tone(800))]).is_some());
        assert_eq!(
            evaluate_scheduled(request, &[(1, [true; 2], tone(801))])
                .unwrap()
                .default_input
                .class,
            Class::DelayedOnset
        );
        assert!(evaluate_scheduled(request, &[(1, [true; 2], tone(0)); 65]).is_none());
        let mut worker = Worker::new();
        let mut packet = worker.acquire().unwrap();
        packet.scheduled = Some(request);
        packet.retained.extend(retained);
        worker.submit(packet, true);
        worker.finish();
        let records: Vec<_> = worker.drain().collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].default_input.class, Class::Continue);
        assert_eq!(worker.stats.worker_unsupported, 0);
    }

    #[test]
    fn queued_default_retains_schedule_without_duplicate_excitation_or_opportunity() {
        let request = ScheduledRequest {
            source_id: 2,
            source_generation: 7,
            body_generation: 3,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            period: Some(4000),
        };
        // The third onset lies exactly at the initial gap's open upper boundary.
        let retained = [
            (1, [true, false], tone(1600)),
            (2, [false, true], tone(2400)),
            (3, [true, false], tone(4800)),
        ];
        let mut record = evaluate_scheduled(request, &retained).unwrap();
        assert_eq!(record.default_basis, "accepted_queued_onset");
        assert_eq!(record.default_input.class, Class::DelayedOnset);
        assert_eq!(record.default_input.at, 1600);
        assert_eq!(record.decision_at, 800);
        assert_eq!(
            record
                .candidates
                .iter()
                .filter(|c| c.input.class == Class::DelayedOnset)
                .count(),
            1
        );
        assert!(record.candidate_times.len() <= 13);
        for c in &record.candidates {
            assert!(c.added.is_none() && c.onset_trace.is_none());
            assert!(!c.input.consumes_due_opportunity && c.input.reconsider_at.is_none());
            assert!(!matches!(
                c.input.class,
                Class::OnsetNow | Class::Continue | Class::Wait | Class::Skip
            ));
            if c.input.class == Class::Release {
                assert!(c.input.at >= 1600);
            }
        }
        let default = &record.candidates[0];
        for bus in 0..2 {
            let reference =
                project_window(&retained, None, 1600, (None, None), bus, [800, 2800], true)
                    .unwrap();
            assert_eq!(default.buses[bus][0].unwrap().mean, reference.mean);
            assert_eq!(default.buses[bus][0].unwrap().difference, Some(0.));
        }
        let initial = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 800)
            .unwrap();
        assert_eq!(initial.input.release_at, None);
        assert_eq!(initial.buses[0][0].unwrap().mean, Some(0.));
        assert_eq!(initial.buses[1][0].unwrap().mean, Some(0.));
        let body = Body {
            retained: &retained,
            added: None,
            at: 800,
            intervention: (None, Some(4800)),
        };
        assert_eq!(
            body.tones(0).map(|(id, _, _)| id).collect::<Vec<_>>(),
            vec![Some(3)]
        );
        release_trace::attach(&mut record, &retained, None);
        let initial = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 800)
            .unwrap();
        assert_eq!(initial.release_trace.len(), 2);
        assert_eq!(initial.release_trace[0].status, "missing_event_pair");
        assert_eq!(initial.release_trace[1].candidate_end_sample, Some(100_000));

        let mut unknown = tone(1600);
        unknown.control = None;
        let r = evaluate_scheduled(request, &[(1, [true, false], unknown)]).unwrap();
        assert_eq!(r.candidates[0].buses[0][0].unwrap().mean, None);
        assert_eq!(r.candidates[0].buses[1][0].unwrap().mean, Some(0.));
        let gap = r
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 800)
            .unwrap();
        assert_eq!(gap.buses[0][0].unwrap().mean, Some(0.));
    }

    #[test]
    fn queued_default_keeps_outside_horizon_but_rejects_cancelled_onsets() {
        let request = ScheduledRequest {
            source_id: 2,
            source_generation: 7,
            body_generation: 3,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            period: Some(4000),
        };
        let mut cancelled = tone(1600);
        cancelled.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 800,
            off_sample: 1200,
        });
        assert!(evaluate_scheduled(request, &[(1, [true; 2], cancelled)]).is_none());
        let mut r = evaluate_scheduled(
            request,
            &[(1, [true; 2], cancelled), (2, [true; 2], tone(40_000))],
        )
        .unwrap();
        assert_eq!(r.default_input.at, 40_000);
        assert_eq!(r.candidates[0].buses[0][3].unwrap().mean, Some(0.));
        assert!(r.candidates.iter().skip(1).all(|c| c.input.at <= 8800));
        let external =
            crate::core::temporal_expectation::TemporalForecast::energy_fixture(8000, 800, |_| {
                [0.1, 0., 0.]
            });
        ratios::attach(
            &mut r,
            &[(1, [true; 2], cancelled), (2, [true; 2], tone(40_000))],
            Some(&external),
        );
        assert_eq!(
            r.candidates[0].energy_ratio.as_ref().unwrap().status,
            "empty_intersection"
        );
        let no_period = evaluate_scheduled(
            ScheduledRequest {
                period: None,
                ..request
            },
            &[(2, [true; 2], tone(1600))],
        )
        .unwrap();
        assert_eq!(no_period.candidates.len(), 1);
        assert_eq!(no_period.default_input.at, 1600);
    }

    #[test]
    fn bounded_worker_drains_without_consumers_and_preserves_conditional_classes() {
        let request = Request {
            source_id: 2,
            source_generation: 7,
            body_generation: Some(3),
            tone_id: 2,
            issued_at: 800,
            sample_rate: 8000,
            hop: 400,
            opportunity: OnsetOpportunity {
                issued_at: 800,
                at: 800,
                gate: 9,
                intrinsic_due_at: Some(800),
                intrinsic_period_ticks: Some(4000),
                planned_release_at: None,
            },
            routed: [true, false],
            recipe: tone(800),
        };
        let mut worker = Worker::new();
        let mut packets: Vec<_> = (0..64).map(|_| worker.acquire().unwrap()).collect();
        assert!(worker.acquire().is_none());
        let mut packet = packets.pop().unwrap();
        packet.request = Some(request);
        packet.retained.push((1, [false, true], tone(0)));
        worker.submit(packet, true);
        let mut invalid = packets.pop().unwrap();
        invalid.request = Some(Request { hop: 0, ..request });
        worker.submit(invalid, true);
        for packet in packets {
            worker.submit(packet, false);
        }
        worker.finish();
        assert_eq!(worker.stats.submitted, 2);
        assert_eq!(worker.stats.completed, 1);
        assert_eq!(worker.stats.worker_unsupported, 1);
        assert_eq!(worker.stats.unsupported, 62);
        assert_eq!(worker.stats.capacity_dropped, 1);
        assert_eq!(worker.stats.output_dropped, 0);
        assert!(!worker.stats.worker_failed);
        let records: Vec<_> = worker.drain().collect();
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(
            (
                r.source_id,
                r.source_generation,
                r.body_generation,
                r.tone_id
            ),
            (2, 7, Some(3), Some(2))
        );
        for class in [
            Class::OnsetNow,
            Class::DelayedOnset,
            Class::Wait,
            Class::Skip,
            Class::Continue,
            Class::Release,
            Class::Gap,
        ] {
            assert!(r.candidates.iter().any(|c| c.input.class == class));
        }
        for c in &r.candidates {
            for w in c.buses.iter().flatten().flatten() {
                assert_eq!(w.supported_bins, 16);
                assert_eq!(
                    w.difference,
                    Some(w.mean.unwrap() - w.default_mean.unwrap())
                );
            }
        }
        assert!(
            r.candidates[0]
                .buses
                .iter()
                .flatten()
                .flatten()
                .all(|w| w.difference == Some(0.))
        );
        let skip = r
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Skip)
            .unwrap();
        assert_eq!(skip.buses[0][0].unwrap().mean, Some(0.));
        assert_eq!(skip.buses[1][0].unwrap().mean, Some(0.125));
        assert!(
            evaluate(
                Request {
                    opportunity: OnsetOpportunity {
                        issued_at: 0,
                        ..request.opportunity
                    },
                    ..request
                },
                &[],
                None
            )
            .is_none()
        );
        assert!(evaluate(request, &[(1, [true; 2], tone(800))], None).is_some());
    }
}
