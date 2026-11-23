use super::*;
use crate::life::arrival_cost::{ArrivalProvenance, ArrivalSet};
use crate::life::onset_footprint as footprint;

/// One grid candidate of a traced decision: every cost term and the inputs that fed it.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct CandidateTerms {
    pub(crate) offset: i8,
    pub(crate) at: f64,
    pub(crate) displacement_sq: f32,
    /// Hellinger distance before `coupling`; `None` when the context term was not applied.
    pub(crate) context_distance: Option<f32>,
    pub(crate) pred_external_band_energy: Option<[[f32; 3]; 2]>,
    /// Overlap sum before `coupling * 6 * s / norm`; `None` when the term was not applied.
    pub(crate) overlap: Option<f32>,
    /// `sustained_energy_at(at + delay_k)` for each footprint bin.
    pub(crate) external_energy: [Option<[f32; 3]>; 16],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_cost: Option<f32>,
    pub(crate) cost: f32,
}

/// I11-1 §4.6 report of one configured decision, carrying what an independent reference
/// needs to recompute every term (§5.1), swap the footprint source (§5.2) and check the
/// deadlines (§5.3).
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct ParticipationDecision {
    pub(crate) now: u64,
    pub(crate) due_frame: f64,
    pub(crate) period_frames: f64,
    pub(crate) width: f64,
    pub(crate) earliest: f64,
    pub(crate) coupling: f32,
    pub(crate) overlap_sensitivity: f32,
    pub(crate) onset_allowed: bool,
    pub(crate) memory: Option<[[f32; 3]; 2]>,
    pub(crate) own_band_energy: [f32; 3],
    pub(crate) sound_hold_sec: f32,
    /// Attack, decay, sustain level and release of the proxy envelope.
    pub(crate) sound_adsr: Option<[f32; 4]>,
    pub(crate) footprint_source: &'static str,
    pub(crate) footprint_identity: Option<footprint::Identity>,
    /// The identity of the recipe the Voice would emit at this hop.
    pub(crate) current_identity: Option<footprint::Identity>,
    pub(crate) footprint_requested_at: Option<u64>,
    pub(crate) footprint_received_at: Option<u64>,
    pub(crate) footprint_d_samples: f64,
    pub(crate) footprint_truncated: bool,
    pub(crate) footprint_delay: [f64; 16],
    pub(crate) footprint_power: [f32; 16],
    pub(crate) forecast_observed_frame: Option<u64>,
    pub(crate) forecast_available_through_frame: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_groups: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_group_exclusions: Option<[Option<&'static str>; 7]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_self_group_known: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_self_group_state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_provenance: Option<Box<ArrivalProvenance>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) selected_cost_without_arrival: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) minimum_cost_without_arrival: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arrival_weight: Option<f32>,
    /// Offsets `-2..=20` in grid order; `None` below `earliest`.
    pub(crate) candidates: [Option<CandidateTerms>; 23],
    pub(crate) reference_cost: f32,
    pub(crate) selected_offset: i8,
    pub(crate) selected_at: f64,
    pub(crate) selected_cost: f32,
    /// Cycles skipped before this decision; the skip threshold is `1 + skipped_cycles`.
    pub(crate) skipped_cycles: u32,
    pub(crate) skipped: bool,
}

/// What the supplier could deliver for the recipe the Voice would emit now (§4.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FootprintState {
    /// The powers belong to the current identity.
    Body,
    /// The identity moved on; only the span of the old body record survives (§4.3).
    Stale,
    /// The worker answered that the recipe has no supported projection.
    Unsupported,
}

/// Own-sound energy over the bounded onset window, supplied by the worker record.
/// `d_samples == 0` marks a record that carries no supported span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BodyFootprint {
    pub identity: footprint::Identity,
    pub d_samples: u64,
    pub power: [f64; 16],
    pub truncated: bool,
    pub state: FootprintState,
    pub requested_at: u64,
    pub received_at: u64,
}

impl BodyFootprint {
    fn from_record(record: &footprint::Record) -> Self {
        let supported = matches!(
            record.state,
            footprint::State::Body | footprint::State::BodySilent
        );
        Self {
            identity: record.identity,
            // An unsupported projection is not a body record, so it lends no span (§4.3).
            d_samples: if supported { record.d_samples } else { 0 },
            power: record.power,
            truncated: record.truncated,
            state: if supported {
                FootprintState::Body
            } else {
                FootprintState::Unsupported
            },
            requested_at: record.requested_at,
            // Only `Worker::drain_footprints` builds the records that reach a Voice.
            received_at: record.received_at.expect("a drained footprint has arrived"),
        }
    }
}

/// I11-1 §4.2: one outstanding request per Voice, and identity-keyed reuse of the
/// record it returns. The bookkeeping is per-Voice; the computation is not.
#[derive(Default)]
pub(crate) struct FootprintTracker {
    /// The identity of the recipe the Voice would emit now.
    identity: Option<footprint::Identity>,
    current: Option<(footprint::Identity, BodyFootprint)>,
    outstanding: Option<footprint::Identity>,
    superseded: bool,
    last_sent: Option<footprint::Identity>,
}

impl FootprintTracker {
    /// Adopts `identity` and sends at most one request through `send`.
    /// Returns true when this identity had already been handed to the supplier.
    pub(crate) fn hop(
        &mut self,
        identity: footprint::Identity,
        now: u64,
        recipe: &footprint::Recipe,
        send: impl FnOnce(footprint::Request) -> bool,
    ) -> bool {
        self.identity = Some(identity);
        if let Some(outstanding) = self.outstanding {
            // The reply must arrive before the new identity is asked for.
            self.superseded |= outstanding != identity;
            return false;
        }
        // A stale record never suppresses a request; a matching one always does.
        if self
            .current
            .as_ref()
            .is_some_and(|(known, _)| *known == identity)
        {
            return false;
        }
        let resent = self.last_sent == Some(identity);
        self.last_sent = Some(identity);
        if send(footprint::Request {
            identity,
            requested_at: now,
            recipe: recipe.clone(),
        }) {
            self.outstanding = Some(identity);
        }
        resent
    }

    pub(crate) fn identity(&self) -> Option<footprint::Identity> {
        self.identity
    }

    /// The reply to `identity` was dropped: clear the outstanding request so the next hop
    /// resends. Returns true when this tracker was waiting for it.
    pub(crate) fn release(&mut self, identity: footprint::Identity) -> bool {
        if self.outstanding != Some(identity) {
            return false;
        }
        self.outstanding = None;
        self.superseded = false;
        true
    }

    /// Returns true when the record was discarded because its request was superseded.
    pub(crate) fn receive(&mut self, record: &footprint::Record) -> bool {
        let Some(outstanding) = self.outstanding.take() else {
            return false;
        };
        if self.superseded || outstanding != record.identity {
            self.superseded = false;
            return true;
        }
        self.current = Some((record.identity, BodyFootprint::from_record(record)));
        false
    }

    /// The §4.6 state of the held record against the identity of this hop.
    pub(crate) fn selected(&self) -> Option<BodyFootprint> {
        let (known, footprint) = self.current.as_ref()?;
        let stale = self.identity != Some(*known);
        Some(match (stale, footprint.state) {
            (false, _) => *footprint,
            (true, FootprintState::Body) => BodyFootprint {
                state: FootprintState::Stale,
                ..*footprint
            },
            // An expired unsupported record lends neither powers nor span.
            (true, _) => BodyFootprint {
                d_samples: 0,
                state: FootprintState::Stale,
                ..*footprint
            },
        })
    }
}

#[derive(Default)]
pub(super) struct OnsetComparison {
    onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    body_footprint: Option<BodyFootprint>,
    current_identity: Option<footprint::Identity>,
    trace_decisions: bool,
    decisions: Vec<ParticipationDecision>,
}

pub(super) struct FootprintTrace(f64);

impl FootprintTrace {
    pub(super) fn new(samples: f64) -> Self {
        Self(samples)
    }
}

impl OnsetComparison {
    pub(super) fn footprint_clocks(&self) -> (Option<u64>, Option<u64>) {
        (
            self.body_footprint.map(|record| record.requested_at),
            self.body_footprint.map(|record| record.received_at),
        )
    }

    pub(super) fn fill_footprint(
        &self,
        owner: &TemporalParticipation,
        points: &mut [(f64, f32); 64],
        now: u64,
        duration: f32,
        release: f32,
    ) -> Option<(u8, &'static str, bool, u64, f64)> {
        let config = self.onset_comparison?;
        let record = self.body_footprint;
        // I11-1 §5.3: no decision may rest on an observation from its future.
        debug_assert!(
            record.is_none_or(|record| record.received_at <= now),
            "a footprint that arrives after the decision cannot be used"
        );
        let body = config.footprint == crate::config::FootprintSource::Body;
        // A retained span keeps body and proxy comparable bin for bin.
        let span = record.filter(|record| record.d_samples > 0);
        let (d_samples, truncated) = match span {
            Some(record) => (record.d_samples as f64, record.truncated),
            None => (f64::from(duration.min(4.0)) * owner.fs, duration > 4.0),
        };
        for (k, point) in points[..16].iter_mut().enumerate() {
            let delay = (k as f64 + 0.5) * d_samples / 16.0;
            let power = match record {
                Some(record) if body && record.state == FootprintState::Body => {
                    record.power[k] as f32
                }
                _ => {
                    let gain =
                        owner.footprint_proxy_gain((delay / owner.fs) as f32, duration, release);
                    gain * gain
                }
            };
            *point = (delay, power);
        }
        let source = match (body, record.map(|record| record.state)) {
            (false, _) => "proxy(setting)",
            (true, None) => "proxy(absent)",
            (true, Some(FootprintState::Body)) => "body",
            (true, Some(FootprintState::Stale)) => "proxy(stale)",
            (true, Some(FootprintState::Unsupported)) => "proxy(unsupported)",
        };
        Some((16u8, source, truncated, d_samples.ceil() as u64, d_samples))
    }

    pub(super) fn record_decision(
        &mut self,
        decision: DecisionComparison<'_>,
        skipped: bool,
        reference_cost: f32,
        (best_offset, best, best_cost): (i8, f64, f32),
        costs_without_arrival: (f32, f32),
    ) {
        let costs_without_arrival = decision.arrival_enabled.then_some(costs_without_arrival);
        if let Some(mut trace) = decision.trace {
            trace.skipped = skipped;
            trace.reference_cost = reference_cost;
            trace.selected_offset = best_offset;
            trace.selected_at = best;
            trace.selected_cost = best_cost;
            trace.selected_cost_without_arrival = costs_without_arrival.map(|cost| cost.0);
            trace.minimum_cost_without_arrival = costs_without_arrival.map(|cost| cost.1);
            self.decisions.push(trace);
        }
    }
}

impl TemporalParticipation {
    pub(crate) fn set_onset_comparison(
        &mut self,
        config: Option<crate::config::TemporalOnsetComparisonConfig>,
    ) {
        self.comparison.onset_comparison = config;
    }

    /// Identity and staleness are the supplier's concern; absence falls back to the proxy.
    pub(crate) fn set_body_footprint(&mut self, footprint: Option<BodyFootprint>) {
        self.comparison.body_footprint = footprint;
    }

    pub(crate) fn set_current_identity(&mut self, identity: Option<footprint::Identity>) {
        self.comparison.current_identity = identity;
    }

    /// Configured decisions are traced only while a reporter can take them.
    pub(crate) fn set_decision_trace(&mut self, enabled: bool) {
        self.comparison.trace_decisions = enabled;
    }

    pub(crate) fn drain_decisions(&mut self) -> std::vec::Drain<'_, ParticipationDecision> {
        self.comparison.decisions.drain(..)
    }

    #[allow(clippy::too_many_arguments)]
    fn decision_trace(
        &self,
        now: u64,
        width: f64,
        earliest: f64,
        onset_allowed: bool,
        footprint: &DecisionFootprint,
        forecast: Option<&TemporalForecast>,
        arrival: Option<&ArrivalContext<'_>>,
        arrival_set: Option<&ArrivalSet<'_>>,
    ) -> Option<ParticipationDecision> {
        let arrival_enabled = self
            .comparison
            .onset_comparison
            .is_some_and(|config| config.arrival);
        let arrival_state =
            arrival_enabled.then_some(arrival_set.map_or("no_snapshot", ArrivalSet::state));
        (self.comparison.trace_decisions && self.comparison.onset_comparison.is_some()).then(|| {
            ParticipationDecision {
                now,
                due_frame: self.due_frame,
                period_frames: self.period_frames,
                width,
                earliest,
                coupling: self.coupling,
                overlap_sensitivity: self.overlap_sensitivity,
                onset_allowed,
                memory: self.memory,
                own_band_energy: self.own_band_energy,
                sound_hold_sec: self.sound_hold_sec,
                sound_adsr: self
                    .sound_adsr
                    .map(|a| [a.attack_sec, a.decay_sec, a.sustain_level, a.release_sec]),
                footprint_source: footprint.footprint_source,
                footprint_identity: self.comparison.body_footprint.map(|record| record.identity),
                current_identity: self.comparison.current_identity,
                footprint_requested_at: self
                    .comparison
                    .body_footprint
                    .map(|record| record.requested_at),
                footprint_received_at: self
                    .comparison
                    .body_footprint
                    .map(|record| record.received_at),
                footprint_d_samples: footprint.footprint_trace.0,
                footprint_truncated: footprint.footprint_truncated,
                footprint_delay: std::array::from_fn(|k| footprint.points[k].0),
                footprint_power: std::array::from_fn(|k| footprint.points[k].1),
                forecast_observed_frame: forecast.map(TemporalForecast::observed_frame),
                forecast_available_through_frame: forecast
                    .map(TemporalForecast::available_through_frame),
                arrival_state,
                arrival_groups: arrival_enabled.then_some(arrival_set.map_or(0, ArrivalSet::len)),
                arrival_group_exclusions: arrival_set.map(ArrivalSet::exclusions),
                arrival_self_group_known: arrival_enabled.then_some(
                    arrival.is_some_and(|context| context.evidence.self_group.is_some()),
                ),
                arrival_self_group_state: arrival_enabled.then_some(
                    arrival.map_or("no_snapshot", |context| context.evidence.self_group_state),
                ),
                arrival_provenance: if arrival_enabled {
                    arrival
                        .zip(arrival_set)
                        .map(|(context, set)| ArrivalProvenance::capture(context, set))
                } else {
                    None
                },
                selected_cost_without_arrival: None,
                minimum_cost_without_arrival: None,
                arrival_weight: arrival_enabled.then(|| {
                    self.comparison
                        .onset_comparison
                        .unwrap()
                        .arrival_weight
                        .unwrap_or(1.0)
                }),
                candidates: [None; 23],
                reference_cost: f32::INFINITY,
                selected_offset: 0,
                selected_at: 0.0,
                selected_cost: f32::INFINITY,
                skipped_cycles: self.skipped_cycles,
                skipped: false,
            }
        })
    }
}

#[derive(Default)]
pub(super) struct CandidateTrace {
    context_distance: Option<f32>,
    overlap: Option<f32>,
    external_energy: [Option<[f32; 3]>; 16],
}

impl CandidateTrace {
    pub(super) fn distance(&mut self, distance: f32) {
        self.context_distance = Some(distance);
    }

    pub(super) fn overlap(&mut self, overlap: f32) {
        self.overlap = Some(overlap);
    }

    pub(super) fn energy(&mut self, k: usize, energy: [f32; 3], tracing: bool) {
        if tracing {
            self.external_energy[k] = Some(energy);
        }
    }
}

pub(super) struct DecisionComparison<'a> {
    arrival_enabled: bool,
    arrival_set: Option<ArrivalSet<'a>>,
    arrival_weight: Option<f32>,
    trace: Option<ParticipationDecision>,
}

impl<'a> DecisionComparison<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        owner: &TemporalParticipation,
        now: u64,
        width: f64,
        earliest: f64,
        onset_allowed: bool,
        footprint: &DecisionFootprint,
        forecast: Option<&TemporalForecast>,
        arrival: Option<&ArrivalContext<'a>>,
        candidate_at: &[Option<f64>; 23],
    ) -> Self {
        let arrival_enabled = owner
            .comparison
            .onset_comparison
            .is_some_and(|config| config.arrival);
        let arrival_set = if arrival_enabled {
            arrival.map(|context| {
                ArrivalSet::select(
                    context.evidence.snapshot,
                    now,
                    candidate_at,
                    width,
                    context.evidence.self_group,
                    owner.fs as u32,
                )
            })
        } else {
            None
        };
        let trace = owner.decision_trace(
            now,
            width,
            earliest,
            onset_allowed,
            footprint,
            forecast,
            arrival,
            arrival_set.as_ref(),
        );
        let arrival_weight = owner
            .comparison
            .onset_comparison
            .map(|config| config.arrival_weight.unwrap_or(1.0));
        Self {
            arrival_enabled,
            arrival_set,
            arrival_weight,
            trace,
        }
    }

    pub(super) fn tracing(&self) -> bool {
        self.trace.is_some()
    }

    pub(super) fn rejects_adoption(
        &self,
        index: usize,
        at: f64,
        width: f64,
        candidate_at: &[Option<f64>; 23],
    ) -> bool {
        index == 23
            && self.arrival_enabled
            && self.arrival_set.as_ref().is_some_and(|set| {
                set.probability(at, width).is_none()
                    && candidate_at
                        .iter()
                        .flatten()
                        .any(|t| set.probability(*t, width).is_some())
            })
    }

    pub(super) fn arrival_cost(
        &self,
        at: f64,
        width: f64,
        coupling: f32,
    ) -> (Option<f64>, Option<f32>) {
        let probability = if self.arrival_enabled {
            self.arrival_set
                .as_ref()
                .and_then(|set| set.probability(at, width))
        } else {
            None
        };
        let cost = probability.map(|p| coupling * self.arrival_weight.unwrap() * (1.0 - p) as f32);
        (probability, cost)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_candidate(
        &mut self,
        index: usize,
        offset: i8,
        at: f64,
        displacement: f64,
        context: Option<ParticipationContextPrediction>,
        candidate: CandidateTrace,
        arrival_probability: Option<f64>,
        arrival_cost: Option<f32>,
        cost: f32,
    ) {
        if let Some(trace) = self.trace.as_mut().filter(|_| index < 23) {
            trace.candidates[index] = Some(CandidateTerms {
                offset,
                at,
                displacement_sq: (displacement * displacement) as f32,
                context_distance: candidate.context_distance,
                pred_external_band_energy: context.map(|context| context.pred_external_band_energy),
                overlap: candidate.overlap,
                external_energy: candidate.external_energy,
                arrival_probability,
                arrival_cost,
                cost,
            });
        }
    }
}
