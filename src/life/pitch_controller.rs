use super::pitch_core::{AnyPitchCore, PitchCore, fill_occupancy_scan};
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::life::adaptation::{AdaptationContext, FeaturesNow};
use crate::scenario::control::{MAX_FREQ_HZ, MIN_FREQ_HZ, PitchControl, PitchMode};
use rand::rngs::SmallRng;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct DwellRecord {
    pub(crate) at: Option<u64>,
    pub(crate) event: &'static str,
    pub(crate) elapsed_sec: f32,
    pub(crate) phase: f32,
    pub(crate) remainder: f32,
    pub(crate) level: f32,
    pub(crate) source: &'static str,
    pub(crate) freq_hz: f32,
    pub(crate) weight: f32,
}

/// One `respond` report record: a spent control step or a proposal.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct RespondRecord {
    pub(crate) at: Option<u64>,
    pub(crate) event: &'static str,
    pub(crate) weight: f32,
    pub(crate) extra_sec: f64,
    pub(crate) rise_sum: f64,
    pub(crate) pending: f64,
    pub(crate) listening_bins: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cycle_extra_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) moved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) protected: Option<bool>,
}

/// Author rule (`respond_to_onsets`): new attacks heard in the Voice's own
/// self-excluding observer advance its proposal clock. A rise rho owes
/// `w*tau*rho` clock seconds, spent at most `w*dt` per control step. Nothing is
/// owed or spent until one ordinary proposal threshold of real time has passed
/// since the last pitch change, so a new note keeps its ordinary cycle.
#[derive(Clone, Debug)]
pub(crate) struct OnsetResponse {
    weight: f32,
    tau: f64,
    prev_scan: Vec<f32>,
    prev_end: u64,
    primed: bool,
    pub(crate) pending: f64,
    pub(crate) rise_sum: f64,
    pub(crate) extra_sec: f64,
    cycle_extra: f64,
    since_move_sec: f32,
    protected: bool,
    proposed: bool,
    listening_bins: u32,
}

impl OnsetResponse {
    pub(crate) fn new(weight: f32, bins: usize) -> Self {
        let mut response = Self {
            weight,
            tau: 0.0,
            prev_scan: vec![0.0; bins],
            prev_end: 0,
            primed: false,
            pending: 0.0,
            rise_sum: 0.0,
            extra_sec: 0.0,
            cycle_extra: 0.0,
            since_move_sec: 0.0,
            protected: true,
            proposed: false,
            listening_bins: 0,
        };
        response.configure(crate::core::habituation::HabituationParams::default());
        response
    }

    /// The time constant is the existing satiation time: `satiation_sec / ln 10`.
    pub(crate) fn configure(&mut self, params: crate::core::habituation::HabituationParams) {
        self.tau = f64::from(params.satiation_sec) / std::f64::consts::LN_10;
    }

    fn record(&self, at: Option<u64>, event: &'static str) -> RespondRecord {
        RespondRecord {
            at,
            event,
            weight: self.weight,
            extra_sec: self.extra_sec,
            rise_sum: self.rise_sum,
            pending: self.pending,
            listening_bins: self.listening_bins,
            cycle_extra_sec: None,
            moved: None,
            protected: None,
        }
    }

    #[cfg(all(test, feature = "native", feature = "loo-diagnostic"))]
    pub(crate) fn oracle_step(&mut self, frame: &crate::core::landscape::Landscape, dt: f32) {
        let mut input = super::pitch_core::LooInput::default();
        input.onset_habituation_state_scan =
            Some(frame.acoustic.onset_habituation_state_scan.clone());
        input.onset_habituation_end = frame.acoustic.onset_habituation_end;
        input.onset_habituation_known = frame.acoustic.onset_habituation_known;
        self.step(
            &input,
            Some(input.onset_habituation_end),
            &frame.acoustic.space,
            0..frame.acoustic.space.n_bins(),
            dt,
            0.5,
        );
    }

    /// One control step: reads a new own observation inside `bins` and returns the
    /// clock seconds spent. Unknown, future or regridded observations re-prime
    /// without contribution; the shared mixed history is never used.
    fn step(
        &mut self,
        input: &super::pitch_core::LooInput,
        now: Option<u64>,
        space: &Log2Space,
        bins: std::ops::Range<usize>,
        dt: f32,
        threshold: f32,
    ) -> f64 {
        self.since_move_sec += dt;
        self.protected = self.since_move_sec < threshold;
        let end = input.onset_habituation_end;
        match (&input.onset_habituation_state_scan, now) {
            (Some(scan), Some(now)) if input.onset_habituation_known && end <= now => {
                space.assert_scan_len_named(scan, "onset_habituation_state_scan");
                if self.prev_scan.len() != scan.len() {
                    self.prev_scan.clear();
                    self.prev_scan.resize(scan.len(), 0.0);
                    self.primed = false;
                }
                if !self.primed || end != self.prev_end {
                    if self.primed && end > self.prev_end {
                        let rise = bins
                            .clone()
                            .map(|i| scan[i] - self.prev_scan[i])
                            .fold(0.0_f32, f32::max);
                        self.listening_bins = bins.len() as u32;
                        self.rise_sum += f64::from(rise);
                        if !self.protected {
                            self.pending += f64::from(self.weight) * self.tau * f64::from(rise);
                        }
                    }
                    self.prev_scan.copy_from_slice(scan);
                    self.prev_end = end;
                    self.primed = true;
                }
            }
            _ => self.primed = false,
        }
        let spend = if self.protected {
            self.pending = 0.0;
            0.0
        } else {
            let spend = self.pending.min(f64::from(self.weight) * f64::from(dt));
            self.pending -= spend;
            spend
        };
        self.extra_sec += spend;
        self.cycle_extra += spend;
        spend
    }
}

#[derive(Debug)]
pub struct PitchController {
    pub(crate) harmonic_gate: Option<bool>,
    pub(crate) harmonic_next_tick: Option<u64>,
    pub(crate) harmonic_evaluated: bool,
    #[cfg(test)]
    #[cfg(feature = "research")]
    pub(crate) trace_owner: Option<(u64, u32)>,
    core: AnyPitchCore,
    pub(crate) melodic: Option<crate::life::melodic_continuity::MelodicContinuity>,
    pub(crate) own_sequence: Option<crate::life::own_sequence::OwnSequence>,
    pub(crate) sequence_decisions: Vec<crate::life::own_sequence::SequenceDecision>,
    adaptation: AdaptationContext,
    target_pitch_log2: f32,
    integration_window: f32,
    accumulated_time: f32,
    proposal_phase: f32,
    pub(crate) dwell_state: Option<DwellRecord>,
    pub(crate) dwell_trace: Vec<DwellRecord>,
    pub(crate) respond: Option<OnsetResponse>,
    pub(crate) respond_trace: Vec<RespondRecord>,
    pub(crate) tonal_trace: Vec<crate::core::tonal_context::Record>,
    pub(crate) pitch_set_trace: Vec<crate::core::tonal_context::pitch_set::Decision>,
    pub(crate) dwell_trace_enabled: bool,
    pub(crate) control_tick: Option<u64>,
    last_theta_phase: f32,
    theta_phase_initialized: bool,
    last_target_salience: f32,
    rng: SmallRng,
    adaptation_enabled: bool,
    /// Reusable buffer for the predictive fundamental-occupancy scan (Stage 1.5).
    occupancy_scan: Vec<f32>,
}

impl PitchController {
    #[cfg(test)]
    pub(crate) fn rng_state_for_test(&self) -> String {
        format!("{:?}", self.rng)
    }

    pub(super) fn target_bounds(
        pitch: &PitchControl,
        current_pitch_log2: f32,
        landscape: &Landscape,
    ) -> (f32, f32) {
        let (fmin, fmax) = landscape.acoustic.space.voice_pitch_bounds_log2();
        let center_log2 = if pitch.freq.is_finite() && pitch.freq > 0.0 {
            pitch.freq.log2()
        } else {
            current_pitch_log2
        };
        let range_oct = pitch.range_oct.clamp(0.0, 6.0);
        let (min_range, max_range) = if range_oct <= 0.0 {
            (center_log2, center_log2)
        } else {
            let half = range_oct * 0.5;
            (center_log2 - half, center_log2 + half)
        };
        (fmin.max(min_range), fmax.min(max_range))
    }

    pub(crate) fn next_proposal_tick(
        &self,
        at: u64,
        pitch: &PitchControl,
        rhythms: &NeuralRhythms,
        fs: f32,
        step: u64,
    ) -> Option<u64> {
        if self.harmonic_gate.is_some() {
            return self.harmonic_next_tick.filter(|&next| next > at);
        }
        if pitch.dwell > 0. || self.respond.is_some() || !fs.is_finite() || fs <= 0. {
            return None;
        }
        let interval = pitch.proposal_interval_sec.filter(|v| *v > 0.);
        if interval.is_none() && rhythms.theta.mag > 0. {
            // Adaptive theta has not published its next proposal sample.
            return None;
        }
        let threshold = interval.unwrap_or(self.integration_window);
        let dt = step.max(1) as f32 / fs;
        if !threshold.is_finite() || threshold > dt * (1u64 << f32::MANTISSA_DIGITS) as f32 {
            return None;
        }
        let mut phase = self.accumulated_time;
        let mut next = at;
        loop {
            let advanced = phase + dt;
            if !advanced.is_finite() || advanced == phase {
                return None;
            }
            next = next.checked_add(step.max(1))?;
            if advanced >= threshold {
                return Some(next);
            }
            phase = advanced;
        }
    }

    pub(crate) fn integration_window_for_freq(freq_hz: f32) -> f32 {
        2.0 + 10.0 / freq_hz.max(1.0)
    }

    pub fn new(
        core: AnyPitchCore,
        adaptation: AdaptationContext,
        target_pitch_log2: f32,
        integration_window: f32,
        rng: SmallRng,
    ) -> Self {
        Self {
            harmonic_gate: None,
            harmonic_next_tick: None,
            harmonic_evaluated: false,
            #[cfg(test)]
            #[cfg(feature = "research")]
            trace_owner: None,
            core,
            own_sequence: None,
            melodic: None,
            sequence_decisions: Vec::new(),
            adaptation,
            target_pitch_log2,
            integration_window,
            accumulated_time: 0.0,
            proposal_phase: 0.0,
            dwell_state: None,
            dwell_trace: Vec::new(),
            respond: None,
            respond_trace: Vec::new(),
            tonal_trace: Vec::new(),
            pitch_set_trace: Vec::new(),
            dwell_trace_enabled: false,
            control_tick: None,
            last_theta_phase: 0.0,
            theta_phase_initialized: false,
            last_target_salience: 0.0,
            rng,
            adaptation_enabled: true,
            occupancy_scan: Vec::new(),
        }
    }

    /// Build the predictive fundamental-occupancy scan from visible neighbors'
    /// fundamentals, using the same shared kernel and width as the crowding
    /// penalty (`occupancy_contribution`, Gaussian of width `sigma_cents` with
    /// pairwise split bias). Self is already excluded upstream, so this is the
    /// leave-self-out spatial slice of the occupancy field `F`. Crowding and this
    /// feed therefore share one occupancy definition.
    /// See docs/design-notes/voice-movement-redesign.md.
    fn build_neighbor_occupancy(
        &mut self,
        space: &Log2Space,
        neighbors: &[f32],
        neighbor_salience: &[f32],
        sigma_cents: f32,
    ) {
        fill_occupancy_scan(
            &mut self.occupancy_scan,
            space,
            neighbors,
            neighbor_salience,
            sigma_cents,
        );
    }

    pub fn target_pitch_log2(&self) -> f32 {
        self.target_pitch_log2
    }

    pub fn integration_window(&self) -> f32 {
        self.integration_window
    }

    pub fn last_target_salience(&self) -> f32 {
        self.last_target_salience
    }

    pub(crate) fn environment_score(&mut self, landscape: &Landscape, freq_hz: f32) -> f32 {
        self.core.environment_score(landscape, freq_hz)
    }

    pub(crate) fn loo_input_mut(&mut self) -> &mut super::pitch_core::LooInput {
        self.core.loo_input_mut()
    }

    pub(crate) fn apply_control(&mut self, pitch: &PitchControl) {
        self.core.apply_control(pitch);
    }

    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
        self.adaptation.set_enabled(enabled);
    }

    pub fn force_set_target_pitch_log2(&mut self, log_freq: f32) {
        self.target_pitch_log2 = log_freq.max(0.0);
        self.accumulated_time = 0.0;
        self.proposal_phase = 0.0;
        if let Some(state) = &mut self.dwell_state {
            state.phase = 0.0;
            state.remainder = 0.0;
            state.elapsed_sec = 0.0;
        }
        if let Some(response) = &mut self.respond {
            response.since_move_sec = 0.0;
            response.pending = 0.0;
            response.cycle_extra = 0.0;
        }
        self.last_theta_phase = 0.0;
        self.theta_phase_initialized = false;
        self.last_target_salience = 0.0;
    }

    /// Update pitch targets at control rate (hop-sized steps).
    #[allow(clippy::too_many_arguments)]
    pub fn update_pitch_target(
        &mut self,
        current_freq_hz: f32,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        pitch: &PitchControl,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        temperature_bonus: f32,
    ) -> bool {
        let moved = self.update_pitch_target_inner(
            current_freq_hz,
            rhythms,
            dt_sec,
            landscape,
            pitch,
            neighbor_pitch_log2,
            neighbor_salience,
            temperature_bonus,
        );
        if let Some(response) = &mut self.respond
            && std::mem::take(&mut response.proposed)
        {
            if self.dwell_trace_enabled {
                self.respond_trace.push(RespondRecord {
                    cycle_extra_sec: Some(response.cycle_extra),
                    moved: Some(moved),
                    protected: Some(response.protected),
                    ..response.record(self.control_tick, "proposal")
                });
            }
            response.cycle_extra = 0.0;
            if moved {
                response.since_move_sec = 0.0;
            }
        }
        moved
    }

    #[allow(clippy::too_many_arguments)]
    fn update_pitch_target_inner(
        &mut self,
        current_freq_hz: f32,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        pitch: &PitchControl,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        temperature_bonus: f32,
    ) -> bool {
        self.harmonic_evaluated = false;
        let previous_target = self.target_pitch_log2;
        let dt_sec = dt_sec.max(0.0);
        let current_freq = current_freq_hz.max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let mut target_pitch_log2 = if self.target_pitch_log2 <= 0.0 {
            current_pitch_log2
        } else {
            self.target_pitch_log2
        };
        if let AnyPitchCore::PitchHillClimb(core) = &mut self.core {
            if landscape.ecology.pitch_set_weight > 0. {
                let (lo, hi) = landscape.acoustic.space.voice_pitch_bounds_log2();
                let center = if pitch.freq.is_finite() && pitch.freq > 0. {
                    pitch.freq.log2()
                } else {
                    current_pitch_log2
                };
                let half = pitch.range_oct.clamp(0., 6.) * 0.5;
                let bounds = ((center - half).max(lo), (center + half).min(hi));
                core.pitch_set =
                    (bounds.0 <= bounds.1).then(|| crate::core::tonal_context::pitch_set::Use {
                        snapshot: landscape
                            .ecology
                            .pitch_set
                            .clone()
                            .filter(|s| self.control_tick.is_some_and(|at| s.available_at <= at)),
                        weight: landscape.ecology.pitch_set_weight,
                        bounds,
                        trace: self.dwell_trace_enabled,
                        decision: Default::default(),
                        added: Default::default(),
                        tc_failed_added: Default::default(),
                    });
            } else {
                core.pitch_set = None;
            }
            if pitch.follow_tonal_context > 0. {
                let snapshot = core
                    .loo_input
                    .tonal_context
                    .clone()
                    .filter(|s| self.control_tick.is_some_and(|at| s.end <= at));
                let (fmin, fmax) = landscape.acoustic.space.voice_pitch_bounds_log2();
                let center = if pitch.freq.is_finite() && pitch.freq > 0. {
                    pitch.freq.log2()
                } else {
                    current_pitch_log2
                };
                let half = pitch.range_oct.clamp(0., 6.) * 0.5;
                let bounds = ((center - half).max(fmin), (center + half).min(fmax));
                let bounds = if bounds.0 <= bounds.1 {
                    bounds
                } else {
                    (fmin, fmax)
                };
                core.tonal
                    .get_or_insert_with(crate::core::tonal_context::Use::new)
                    .pin(snapshot, pitch.follow_tonal_context, bounds);
            } else {
                core.tonal = None;
            }
        }
        self.integration_window = Self::integration_window_for_freq(current_freq);
        self.accumulated_time += dt_sec;
        let dwelling = pitch.dwell > 0.0;
        let spend = match &mut self.respond {
            Some(response) => {
                let (lo, hi) = Self::target_bounds(pitch, current_pitch_log2, landscape);
                let (lo, hi) = if lo <= hi {
                    (lo, hi)
                } else {
                    landscape.acoustic.space.voice_pitch_bounds_log2()
                };
                let centers = &landscape.acoustic.space.centers_log2;
                let bins =
                    centers.partition_point(|&x| x < lo)..centers.partition_point(|&x| x <= hi);
                let threshold = pitch
                    .proposal_interval_sec
                    .filter(|v| *v > 0.0)
                    .unwrap_or(self.integration_window);
                let spend = response.step(
                    self.core.loo_input(),
                    self.control_tick,
                    &landscape.acoustic.space,
                    bins,
                    dt_sec,
                    threshold,
                );
                if spend > 0.0 && self.dwell_trace_enabled {
                    self.respond_trace
                        .push(response.record(self.control_tick, "spend"));
                }
                spend as f32
            }
            None => 0.0,
        };
        let phased = dwelling || self.respond.is_some();
        if dwelling {
            let (level, excluded) =
                self.core
                    .dwell_level(landscape, current_freq, pitch.leave_self_out);
            // At extreme weights, retain a finite pending phase instead of overflowing.
            self.proposal_phase = (self.proposal_phase
                + dt_sec * (1.0 + pitch.dwell * (1.0 - 2.0 * level)).max(0.0))
            .min(f32::MAX);
            if spend > 0.0 {
                self.proposal_phase = (self.proposal_phase + spend).min(f32::MAX);
            }
            self.dwell_state = Some(DwellRecord {
                at: self.control_tick,
                event: "state",
                elapsed_sec: self.accumulated_time,
                phase: self.proposal_phase,
                remainder: self.proposal_phase,
                level,
                source: if excluded { "self_excluded" } else { "mixed" },
                freq_hz: current_freq,
                weight: pitch.dwell,
            });
        } else if phased {
            self.proposal_phase = (self.proposal_phase + dt_sec + spend).min(f32::MAX);
        }
        let proposal_clock = if phased {
            self.proposal_phase
        } else {
            self.accumulated_time
        };
        // Listener pressure feeds the search temperature: the per-core default
        // plus the tension-driven bonus.
        let base_temperature = pitch
            .temperature
            .unwrap_or_else(|| pitch.core_kind.default_temperature());
        let bonus = if temperature_bonus.is_finite() {
            temperature_bonus.max(0.0)
        } else {
            0.0
        };
        self.core.set_temperature(base_temperature + bonus);
        let proposal_interval_sec = if pitch.proposal_interval_sec.is_some_and(|v| v > 0.0) {
            pitch.proposal_interval_sec
        } else {
            None
        };
        self.core.set_proposal_interval_sec(proposal_interval_sec);

        // Detect theta wrap to avoid missing zero-crossings at control rate.
        let theta_phase = rhythms.theta.phase;
        let theta_cross = if theta_phase.is_finite() && self.last_theta_phase.is_finite() {
            let wrapped = self.theta_phase_initialized && theta_phase < self.last_theta_phase;
            wrapped && rhythms.theta.mag.is_finite() && rhythms.theta.mag > 0.0
        } else {
            false
        };
        self.last_theta_phase = if theta_phase.is_finite() {
            self.theta_phase_initialized = true;
            theta_phase
        } else {
            self.theta_phase_initialized = false;
            0.0
        };

        // Pitch proposal runs when timing conditions are met.
        // When theta rhythm is present, sync proposals to theta zero-crossings.
        // When no rhythm is detected (e.g. drone agents), fall back to time-based triggering.
        let has_rhythm = rhythms.theta.mag.is_finite() && rhythms.theta.mag > 0.0;
        let should_propose = if let Some(opportunity) = self.harmonic_gate {
            opportunity
                && proposal_clock >= proposal_interval_sec.unwrap_or(self.integration_window)
        } else if let Some(interval_sec) = proposal_interval_sec {
            proposal_clock >= interval_sec
        } else if has_rhythm {
            theta_cross && proposal_clock >= self.integration_window
        } else {
            proposal_clock >= self.integration_window
        };
        self.harmonic_evaluated = should_propose;
        if self.harmonic_gate.is_some() && !should_propose {
            return false;
        }
        if should_propose {
            let elapsed = self.accumulated_time;
            if phased {
                self.proposal_phase = proposal_interval_sec
                    .map_or(0.0, |interval| (self.proposal_phase - interval).max(0.0));
            }
            if let Some(response) = &mut self.respond {
                response.proposed = true;
            }
            if dwelling {
                let state = self
                    .dwell_state
                    .as_mut()
                    .expect("positive dwell has a state");
                state.event = "proposal";
                state.remainder = self.proposal_phase;
                if self.dwell_trace_enabled {
                    self.dwell_trace.push(*state);
                }
                state.event = "state";
                state.phase = self.proposal_phase;
                state.elapsed_sec = 0.0;
            }
            self.accumulated_time = if dwelling {
                0.0
            } else if let Some(interval_sec) = proposal_interval_sec {
                (self.accumulated_time - interval_sec).max(0.0)
            } else {
                0.0
            };
            if !dwelling && self.dwell_trace_enabled {
                let (level, excluded) =
                    self.core
                        .dwell_level(landscape, current_freq, pitch.leave_self_out);
                self.dwell_trace.push(DwellRecord {
                    at: self.control_tick,
                    event: "proposal",
                    elapsed_sec: elapsed,
                    phase: proposal_clock,
                    remainder: self.accumulated_time,
                    level,
                    source: if excluded { "self_excluded" } else { "mixed" },
                    freq_hz: current_freq,
                    weight: 0.0,
                });
            }
            if landscape.harmonic.relations_weight != 0.
                || landscape.harmonic.continuation_weight != 0.
            {
                let sample = self.control_tick.unwrap_or(landscape.harmonic.now_sample);
                let input = self.core.loo_input_mut();
                crate::temporal_cognition::harmonic_relations::runtime::record_opportunity(
                    landscape,
                    Some(input.relation_source),
                    sample,
                );
                if let Some(relations) = &input.relations {
                    relations.set_sample(sample);
                    // A future adaptive gate is unknown until its owning clock publishes it.
                    let next = if self.harmonic_gate.is_some() {
                        self.harmonic_next_tick.unwrap_or(sample)
                    } else if phased {
                        sample
                    } else if has_rhythm && proposal_interval_sec.is_none() {
                        if rhythms.theta.freq_hz.is_finite() && rhythms.theta.freq_hz > 0. {
                            relations.sample_after(f64::from(rhythms.theta.freq_hz.recip()))
                        } else {
                            sample
                        }
                    } else {
                        relations.sample_after(f64::from(
                            proposal_interval_sec.unwrap_or(self.integration_window),
                        ))
                    };
                    if !relations.allow_now(next) {
                        self.harmonic_evaluated = false;
                        return false;
                    }
                }
            }
            // Stage 1.5/2 of the occupancy redesign: adaptation's environment term is
            // fed by the predictive fundamental-occupancy of visible neighbors
            // (f0-based, zero-latency, self-excluded), built with the same shared
            // kernel and width as the crowding penalty so the two share one
            // occupancy definition. See docs/design-notes/voice-movement-redesign.md.
            self.build_neighbor_occupancy(
                &landscape.acoustic.space,
                neighbor_pitch_log2,
                neighbor_salience,
                pitch.crowding_sigma_cents,
            );
            let features =
                FeaturesNow::from_occupancy_scan(&landscape.acoustic.space, &self.occupancy_scan);
            self.adaptation.ensure_len(features.density.len());
            if let AnyPitchCore::PitchHillClimb(core) = &mut self.core {
                core.melodic = self
                    .control_tick
                    .and_then(|at| self.melodic.as_mut().and_then(|me| me.predict(at)));
                core.melodic_trace = self.melodic.as_ref().is_some_and(|me| me.trace_enabled);
                core.sequence_enabled = self.own_sequence.is_some();
                core.sequence_trace = self.own_sequence.as_ref().is_some_and(|s| s.trace_enabled);
                core.sequence = self
                    .control_tick
                    .and_then(|at| self.own_sequence.as_mut().and_then(|seq| seq.predict(at)));
            }
            #[cfg(feature = "loo-diagnostic")]
            super::pitch_core::diagnostic::replay(
                &self.core,
                &self.rng,
                current_pitch_log2,
                target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.adaptation,
                &features,
                neighbor_pitch_log2,
                neighbor_salience,
            );
            let proposal = self.core.propose_target_with_crowding_salience(
                current_pitch_log2,
                target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.adaptation,
                &features,
                neighbor_pitch_log2,
                neighbor_salience,
                &mut self.rng,
            );
            if let AnyPitchCore::PitchHillClimb(core) = &self.core
                && let Some(mut record) = core.sequence_decision.borrow_mut().take()
            {
                record.at = self.control_tick.unwrap_or(0);
                if let Some(seq) = &mut self.own_sequence {
                    if let Some(reason) = record.reason {
                        seq.reason = Some(reason);
                    }
                    if record.reason == Some("candidate_resource_limit") {
                        seq.resource_stopped = true;
                    }
                    if seq.trace_enabled {
                        self.sequence_decisions.push(record);
                    }
                }
            }
            if let AnyPitchCore::PitchHillClimb(core) = &self.core
                && let Some(me) = &mut self.melodic
            {
                let record = core.melodic_decision.borrow_mut().take();
                if let Some(mut record) = record {
                    record.at = self.control_tick.unwrap_or(0);
                    if record.reason == Some("candidate_resource_limit") {
                        me.stop("candidate_resource_limit");
                    }
                    if record.snapshot.is_none() {
                        record.reason = me.reason;
                    }
                    if me.trace_enabled {
                        me.decisions.push(record);
                    }
                }
            }
            target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if self.adaptation_enabled
                && landscape
                    .acoustic
                    .space
                    .index_of_log2(target_pitch_log2)
                    .is_some()
            {
                self.adaptation.update(&features, elapsed);
            }
        }

        if matches!(pitch.mode, PitchMode::Lock) {
            let freq = if pitch.freq.is_finite() && pitch.freq > 0.0 {
                pitch.freq
            } else {
                current_freq
            };
            // Anchored pitches belong to the sound body's range, independently
            // of which frequencies the current landscape can evaluate.
            self.target_pitch_log2 = freq.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ).log2();
            self.last_target_salience = 1.0;
            return false;
        }
        let (fmin, fmax) = landscape.acoustic.space.voice_pitch_bounds_log2();
        let (min, max) = Self::target_bounds(pitch, current_pitch_log2, landscape);
        let clamped = if min <= max {
            target_pitch_log2.clamp(min, max)
        } else {
            target_pitch_log2.clamp(fmin, fmax)
        };
        target_pitch_log2 = clamped;
        if should_propose
            && let Some(record) = self.melodic.as_mut().and_then(|me| me.decisions.last_mut())
            && Some(record.at) == self.control_tick
        {
            record.selected_log2 = clamped;
            record.without_melodic_log2 = if min <= max {
                record.without_melodic_log2.clamp(min, max)
            } else {
                record.without_melodic_log2.clamp(fmin, fmax)
            };
            record.changed =
                record.selected_log2.to_bits() != record.without_melodic_log2.to_bits();
        }
        self.target_pitch_log2 = target_pitch_log2;
        if should_propose
            && let AnyPitchCore::PitchHillClimb(core) = &self.core
            && let Some(ps) = &core.pitch_set
            && let Some(mut record) = ps.decision.borrow_mut().take()
        {
            record.at = self.control_tick;
            self.pitch_set_trace.push(record);
        }
        if self.dwell_trace_enabled
            && let AnyPitchCore::PitchHillClimb(core) = &self.core
            && let Some(tonal) = &core.tonal
        {
            self.tonal_trace.push(tonal.record(
                self.control_tick,
                current_freq,
                target_pitch_log2.exp2(),
            ));
        }
        if should_propose
            && previous_target > 0.0
            && target_pitch_log2 != previous_target
            && let (Some(sequence), Some(at)) = (&mut self.own_sequence, self.control_tick)
        {
            sequence.bind_target(at, target_pitch_log2);
        }
        #[cfg(all(test, feature = "research"))]
        if should_propose && let Some(owner) = self.trace_owner {
            crate::life::body_metabolism::assay::log_move(owner, target_pitch_log2);
        }
        should_propose && previous_target > 0.0 && target_pitch_log2 != previous_target
    }
}

#[cfg(test)]
impl PitchController {
    pub(crate) fn core_for_test(&self) -> &AnyPitchCore {
        &self.core
    }

    pub(crate) fn set_accumulated_time_for_test(&mut self, value: f32) {
        self.accumulated_time = value;
    }

    pub(crate) fn accumulated_time_for_test(&self) -> f32 {
        self.accumulated_time
    }

    pub(crate) fn set_theta_phase_state_for_test(&mut self, last_phase: f32, initialized: bool) {
        self.last_theta_phase = last_phase;
        self.theta_phase_initialized = initialized;
    }
}

#[cfg(test)]
mod dwell_tests {
    use super::*;
    use crate::life::adaptation::AdaptationConfig;
    use rand::SeedableRng;

    fn controller(pitch: &PitchControl, space: &Log2Space) -> PitchController {
        let mut ctl = PitchController::new(
            AnyPitchCore::from_control(pitch, 220.0_f32.log2()),
            AdaptationContext::from_config(&AdaptationConfig::default(), space.n_bins()),
            220.0_f32.log2(),
            2.0,
            SmallRng::seed_from_u64(21),
        );
        ctl.dwell_trace_enabled = true;
        ctl
    }

    #[test]
    fn harmonic_relation_timing_holds_then_acts_only_at_published_opportunities() {
        use crate::temporal_cognition::harmonic_relations::consumer::{test_landscape, test_use};
        let pitch = PitchControl {
            proposal_interval_sec: Some(0.1),
            ..Default::default()
        };
        let (landscape, _) = test_landscape(1., true);
        let mut ctl = controller(&pitch, &landscape.acoustic.space);
        ctl.loo_input_mut().relations = Some(test_use(1., true).0);
        ctl.loo_input_mut().relations.as_mut().unwrap().continuation = 1.;
        ctl.harmonic_gate = Some(true);
        ctl.harmonic_next_tick = Some(48000);
        ctl.control_tick = Some(512);
        ctl.update_pitch_target(
            220.,
            &NeuralRhythms::default(),
            1.,
            &landscape,
            &pitch,
            &[],
            &[],
            0.,
        );
        assert!(!ctl.harmonic_evaluated);
        ctl.harmonic_gate = Some(false);
        ctl.control_tick = Some(24000);
        ctl.update_pitch_target(
            220.,
            &NeuralRhythms::default(),
            0.5,
            &landscape,
            &pitch,
            &[],
            &[],
            0.,
        );
        assert!(!ctl.harmonic_evaluated);
        ctl.harmonic_gate = Some(true);
        ctl.harmonic_next_tick = Some(96000);
        ctl.control_tick = Some(48000);
        ctl.update_pitch_target(
            220.,
            &NeuralRhythms::default(),
            0.5,
            &landscape,
            &pitch,
            &[],
            &[],
            0.,
        );
        assert!(ctl.harmonic_evaluated);
    }

    #[test]
    fn dwell_clock_integrates_each_control_and_uses_first_legal_crossing() {
        for branch in 0..3 {
            for changing in [false, true] {
                let pitch = PitchControl {
                    dwell: 0.75,
                    proposal_interval_sec: (branch == 0).then_some(0.31),
                    ..Default::default()
                };
                let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
                let mut ctl = controller(&pitch, &landscape.acoustic.space);
                let mut phase = 0.0_f64;
                let mut elapsed = 0.0_f32;
                let mut previous_phase = 0.0;
                let mut rhythms = NeuralRhythms::default();
                let mut proposals = 0;
                for step in 0..180 {
                    let raw = if changing && step >= 70 { -0.7 } else { 0.6 };
                    landscape.acoustic.consonance_field_score_eff.fill(raw);
                    let level = landscape.acoustic.consonance_representation.level(raw);
                    let theta = (step % 9) as f32 / 9.0;
                    rhythms.theta.phase = theta;
                    rhythms.theta.mag = if branch == 1 { 1.0 } else { 0.0 };
                    let dt = 0.073_f32;
                    phase += f64::from(dt) * (1.0 + 0.75 * (1.0 - 2.0 * f64::from(level)));
                    elapsed += dt;
                    let threshold = pitch
                        .proposal_interval_sec
                        .unwrap_or(PitchController::integration_window_for_freq(220.0));
                    let expected = phase >= f64::from(threshold)
                        && (branch != 1 || (step > 0 && theta < previous_phase));
                    let before = ctl.dwell_trace.len();
                    ctl.update_pitch_target(220.0, &rhythms, dt, &landscape, &pitch, &[], &[], 0.0);
                    assert_eq!(
                        ctl.dwell_trace.len() > before,
                        expected,
                        "branch={branch} step={step}"
                    );
                    if expected {
                        let record = ctl.dwell_trace.last().unwrap();
                        assert!((record.elapsed_sec - elapsed).abs() < 1e-5);
                        assert!((f64::from(record.phase) - phase).abs() < 1e-5);
                        phase = if branch == 0 {
                            phase - f64::from(threshold)
                        } else {
                            0.0
                        };
                        elapsed = 0.0;
                        proposals += 1;
                    }
                    assert!((f64::from(ctl.proposal_phase) - phase).abs() < 1e-5);
                    assert_eq!(ctl.accumulated_time, elapsed);
                    previous_phase = theta;
                }
                assert!(proposals > 0);
            }
        }
    }

    #[test]
    fn dwell_adaptation_receives_real_seconds_and_cost_window_is_unchanged() {
        let pitch = PitchControl {
            dwell: 1.0,
            proposal_interval_sec: Some(0.4),
            ..Default::default()
        };
        let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
        landscape.acoustic.consonance_field_score_eff.fill(0.7);
        let mut ctl = controller(&pitch, &landscape.acoustic.space);
        let mut reference = AdaptationContext::from_config(
            &AdaptationConfig::default(),
            landscape.acoustic.space.n_bins(),
        );
        let neighbors = [330.0_f32.log2()];
        ctl.build_neighbor_occupancy(
            &landscape.acoustic.space,
            &neighbors,
            &[],
            pitch.crowding_sigma_cents,
        );
        let features =
            FeaturesNow::from_occupancy_scan(&landscape.acoustic.space, &ctl.occupancy_scan);
        let mut real = 0.0;
        for _ in 0..100 {
            real += 0.13;
            let before = ctl.dwell_trace.len();
            ctl.update_pitch_target(
                220.0,
                &NeuralRhythms::default(),
                0.13,
                &landscape,
                &pitch,
                &neighbors,
                &[],
                0.0,
            );
            if ctl.dwell_trace.len() > before {
                reference.update(&features, real);
                real = 0.0;
                for i in 0..landscape.acoustic.space.n_bins() {
                    assert_eq!(
                        reference.score_adjustment(i),
                        ctl.adaptation.score_adjustment(i)
                    );
                }
            }
            assert_eq!(
                ctl.integration_window,
                PitchController::integration_window_for_freq(220.0)
            );
            assert_eq!(ctl.core.proposal_interval_sec_for_test(), Some(0.4));
        }
        assert!(!ctl.dwell_trace.is_empty());
    }

    #[test]
    fn dwell_stops_resumes_resets_and_reporting_does_not_change_rng() {
        use rand::Rng;
        let pitch = PitchControl {
            dwell: 1.0,
            proposal_interval_sec: Some(0.25),
            ..Default::default()
        };
        let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
        let mut traced = controller(&pitch, &landscape.acoustic.space);
        let mut quiet = controller(&pitch, &landscape.acoustic.space);
        quiet.dwell_trace_enabled = false;
        for step in 0..80 {
            landscape
                .acoustic
                .consonance_field_score_eff
                .fill(if step < 40 { 100.0 } else { -100.0 });
            for ctl in [&mut traced, &mut quiet] {
                ctl.update_pitch_target(
                    220.0,
                    &NeuralRhythms::default(),
                    0.1,
                    &landscape,
                    &pitch,
                    &[],
                    &[],
                    0.0,
                );
            }
            assert_eq!(traced.target_pitch_log2, quiet.target_pitch_log2);
            assert_eq!(traced.proposal_phase, quiet.proposal_phase);
            if step < 40 {
                assert_eq!(traced.proposal_phase, 0.0);
                assert!(traced.dwell_trace.is_empty());
            }
        }
        assert!(!traced.dwell_trace.is_empty());
        assert!(quiet.dwell_trace.is_empty());
        assert_eq!(traced.rng.next_u64(), quiet.rng.next_u64());
        traced.force_set_target_pitch_log2(330.0_f32.log2());
        assert_eq!((traced.proposal_phase, traced.accumulated_time), (0.0, 0.0));
    }

    #[test]
    fn dwell_large_weights_stop_accelerate_and_keep_phase_finite() {
        for weight in [4.0, f32::MAX] {
            let pitch = PitchControl {
                dwell: weight,
                proposal_interval_sec: Some(f32::MAX),
                ..Default::default()
            };
            let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
            let mut ctl = controller(&pitch, &landscape.acoustic.space);
            for (score, stopped) in [(100.0, true), (-100.0, false)] {
                landscape.acoustic.consonance_field_score_eff.fill(score);
                for _ in 0..8 {
                    ctl.update_pitch_target(
                        220.0,
                        &NeuralRhythms::default(),
                        1.0,
                        &landscape,
                        &pitch,
                        &[],
                        &[],
                        0.0,
                    );
                    assert!(ctl.proposal_phase.is_finite());
                    assert!(ctl.proposal_phase >= 0.0);
                    if stopped {
                        assert_eq!(ctl.proposal_phase, 0.0);
                    }
                }
                if !stopped && weight == 4.0 {
                    assert_eq!(ctl.proposal_phase, 40.0);
                }
            }
        }
    }

    #[test]
    fn dwell_reads_pitch_ownership_then_erodes_and_converts_interpolated_score() {
        let pitch = PitchControl {
            dwell: 1.0,
            ..Default::default()
        };
        let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
        landscape.acoustic.consonance_field_score.fill(-10.0);
        landscape.acoustic.consonance_field_score_eff.fill(0.3);
        let mut ctl = controller(&pitch, &landscape.acoustic.space);
        let n = landscape.acoustic.space.n_bins();
        ctl.loo_input_mut().exact_score = Some(vec![1.2; n].into());
        for h in [0.0, 0.5, 1.0] {
            landscape.acoustic.perc_habituation_state_scan.fill(h);
            let (mixed, source) = ctl.core.dwell_level(&landscape, 220.0, false);
            assert!(!source);
            assert!((mixed - landscape.acoustic.consonance_representation.level(0.3)).abs() < 1e-6);
            let (own, source) = ctl.core.dwell_level(&landscape, 220.0, true);
            assert!(source);
            assert!(
                (own - landscape
                    .acoustic
                    .consonance_representation
                    .level(1.2 * (1.0 - h)))
                .abs()
                    < 1e-6
            );
        }
        ctl.loo_input_mut().exact_score = None;
        assert!(!ctl.core.dwell_level(&landscape, 220.0, true).1);
        landscape.acoustic.perc_habituation_state_scan.fill(0.0);
        landscape.acoustic.consonance_field_score_eff.fill(0.0);
        let i = n / 2;
        landscape.acoustic.consonance_field_score_eff[i] = 4.0;
        let log2 = (landscape.acoustic.space.centers_log2[i]
            + landscape.acoustic.space.centers_log2[i + 1])
            * 0.5;
        let raw =
            landscape.sample_linear_log2(&landscape.acoustic.consonance_field_score_eff, log2);
        assert!(
            (ctl.core.dwell_level(&landscape, log2.exp2(), false).0
                - landscape.acoustic.consonance_representation.level(raw))
            .abs()
                < 1e-6
        );
        landscape.acoustic.consonance_representation.theta = 0.7;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            landscape.acoustic.consonance_field_score_eff.fill(invalid);
            assert_eq!(ctl.core.dwell_level(&landscape, 220.0, false).0, 0.5);
            ctl.loo_input_mut().exact_score = Some(vec![invalid; n].into());
            assert_eq!(ctl.core.dwell_level(&landscape, 220.0, true).0, 0.5);
        }
    }
    #[test]
    fn autonomous_continuous_upper_bound_keeps_7980_hz_and_lock_keeps_10000() {
        let landscape = Landscape::new(crate::core::log2space::production_space(48_000).unwrap());
        for frequency in [7980.0_f32, 8000.0] {
            let pitch = PitchControl {
                freq: frequency,
                range_oct: 0.0,
                ..Default::default()
            };
            let bounds = PitchController::target_bounds(&pitch, frequency.log2(), &landscape);
            assert_eq!(bounds, (frequency.log2(), frequency.log2()));
        }
        let pitch = PitchControl {
            mode: PitchMode::Lock,
            freq: 10000.0,
            ..Default::default()
        };
        let mut ctl = controller(&pitch, &landscape.acoustic.space);
        ctl.update_pitch_target(
            10000.0,
            &NeuralRhythms::default(),
            1.0,
            &landscape,
            &pitch,
            &[],
            &[],
            0.0,
        );
        assert_eq!(ctl.target_pitch_log2, 10000.0_f32.log2());
    }
    #[test]
    fn hr2_free_target_from_10000_hz_uses_voice_upper_bound() {
        let landscape = Landscape::new(crate::core::log2space::production_space(48000).unwrap());
        let pitch = PitchControl {
            freq: 10000.0,
            range_oct: 0.0,
            ..Default::default()
        };
        let mut ctl = controller(&pitch, &landscape.acoustic.space);
        ctl.force_set_target_pitch_log2(10000.0_f32.log2());
        ctl.update_pitch_target(
            10000.0,
            &NeuralRhythms::default(),
            1.0,
            &landscape,
            &pitch,
            &[],
            &[],
            0.0,
        );
        assert!(ctl.target_pitch_log2 <= 8000.0_f32.log2());
    }
}

#[cfg(test)]
mod respond_tests {
    use super::super::pitch_core::LooInput;
    use super::*;
    use crate::life::adaptation::AdaptationConfig;
    use rand::SeedableRng;
    use std::sync::Arc;

    const DT: f32 = 0.05;
    const STEP: u64 = 2400;

    fn space() -> Log2Space {
        Log2Space::new(55.0, 8000.0, 96)
    }

    fn controller(pitch: &PitchControl, space: &Log2Space) -> PitchController {
        let mut ctl = PitchController::new(
            AnyPitchCore::from_control(pitch, 220.0_f32.log2()),
            AdaptationContext::from_config(&AdaptationConfig::default(), space.n_bins()),
            220.0_f32.log2(),
            2.0,
            SmallRng::seed_from_u64(21),
        );
        if pitch.respond_to_onsets > 0.0 {
            ctl.respond = Some(OnsetResponse::new(pitch.respond_to_onsets, space.n_bins()));
        }
        ctl.dwell_trace_enabled = true;
        ctl
    }

    fn observe(ctl: &mut PitchController, scan: Option<Arc<[f32]>>, end: u64, known: bool) {
        let input = ctl.loo_input_mut();
        input.onset_habituation_state_scan = scan;
        input.onset_habituation_end = end;
        input.onset_habituation_known = known;
    }

    fn input(scan: &Arc<[f32]>, end: u64, known: bool) -> LooInput {
        let mut input = LooInput::default();
        input.onset_habituation_state_scan = Some(scan.clone());
        input.onset_habituation_end = end;
        input.onset_habituation_known = known;
        input
    }

    fn tau() -> f64 {
        5.0 / std::f64::consts::LN_10
    }

    fn bins(space: &Log2Space, lo_hz: f32, hi_hz: f32) -> std::ops::Range<usize> {
        let c = &space.centers_log2;
        c.partition_point(|&x| x < lo_hz.log2())..c.partition_point(|&x| x <= hi_hz.log2())
    }

    /// A peak that moves every few seconds, so proposals both move and stay.
    fn terrain(landscape: &mut Landscape, step: usize) {
        let peak = [220.0_f32, 330.0, 330.0, 247.0][(step / 90) % 4].log2();
        let centers = landscape.acoustic.space.centers_log2.clone();
        for (score, x) in landscape
            .acoustic
            .consonance_field_score_eff
            .iter_mut()
            .zip(centers)
        {
            *score = 1.0 - 8.0 * (x - peak).abs();
        }
    }

    fn update(ctl: &mut PitchController, landscape: &Landscape, pitch: &PitchControl) -> bool {
        let current = ctl.target_pitch_log2().exp2();
        ctl.update_pitch_target(
            current,
            &NeuralRhythms::default(),
            DT,
            landscape,
            pitch,
            &[],
            &[],
            0.0,
        )
    }

    #[test]
    fn zero_weight_has_no_state_and_ignores_its_observation() {
        let space = space();
        let n = space.n_bins();
        let mut landscape = Landscape::new(space.clone());
        let zero = PitchControl::default();
        let positive = PitchControl {
            respond_to_onsets: 2.0,
            ..Default::default()
        };
        let mut observed = controller(&zero, &space);
        let mut plain = controller(&zero, &space);
        let mut flat = controller(&positive, &space);
        assert!(observed.respond.is_none() && plain.respond.is_none());
        let still: Arc<[f32]> = vec![0.1; n].into();
        let rhythms = NeuralRhythms::default();
        let mut moves = 0;
        for step in 0..1200_usize {
            terrain(&mut landscape, step);
            let at = step as u64 * STEP;
            let moving: Arc<[f32]> = (0..n)
                .map(|i| ((i * 7 + step * 13) % 17) as f32 / 17.0)
                .collect();
            observe(&mut observed, Some(moving), at, step % 5 != 2);
            observe(&mut flat, Some(still.clone()), at, true);
            for ctl in [&mut observed, &mut plain, &mut flat] {
                ctl.control_tick = Some(at);
            }
            let a = update(&mut observed, &landscape, &zero);
            let b = update(&mut plain, &landscape, &zero);
            let c = update(&mut flat, &landscape, &positive);
            assert_eq!(
                (
                    a,
                    observed.target_pitch_log2.to_bits(),
                    observed.accumulated_time.to_bits(),
                    observed.rng_state_for_test(),
                    observed.next_proposal_tick(at, &zero, &rhythms, 48_000.0, STEP),
                ),
                (
                    b,
                    plain.target_pitch_log2.to_bits(),
                    plain.accumulated_time.to_bits(),
                    plain.rng_state_for_test(),
                    plain.next_proposal_tick(at, &zero, &rhythms, 48_000.0, STEP),
                ),
                "step {step}"
            );
            // A positive weight that hears no rise proposes at the same samples.
            assert_eq!(
                (
                    c,
                    flat.target_pitch_log2.to_bits(),
                    flat.rng_state_for_test()
                ),
                (
                    b,
                    plain.target_pitch_log2.to_bits(),
                    plain.rng_state_for_test()
                ),
                "step {step}"
            );
            moves += usize::from(b);
        }
        assert!(moves > 2, "{moves}");
        assert!(observed.respond_trace.is_empty());
        let response = flat.respond.as_ref().unwrap();
        assert_eq!((response.extra_sec, response.pending), (0.0, 0.0));
        assert!(flat.respond_trace.iter().all(|r| r.event == "proposal"));
    }

    #[test]
    fn rises_inside_the_range_owe_w_tau_rho_and_other_changes_do_nothing() {
        let space = space();
        let n = space.n_bins();
        let range = bins(&space, 200.0, 400.0);
        let w = 3.0;
        let cases: [(&str, fn(f32) -> f32, f32); 3] = [
            ("inside", |r| r, 300.0),
            ("outside", |r| r, 1000.0),
            ("falling", |r| -r, 300.0),
        ];
        for (name, sign, hz) in cases {
            let mut response = OnsetResponse::new(w, n);
            let mut scan = vec![0.1_f32; n];
            let mut rises = 0.0_f64;
            for step in 0..80_u64 {
                let rise = if (1..=10).contains(&step) {
                    0.002 * step as f32
                } else {
                    0.0
                };
                for (x, &c) in scan.iter_mut().zip(&space.centers_hz) {
                    if (c - hz).abs() < 0.05 * hz {
                        *x += sign(rise);
                    }
                }
                let at = step * STEP;
                let shared: Arc<[f32]> = scan.clone().into();
                let spend = response.step(
                    &input(&shared, at, true),
                    Some(at),
                    &space,
                    range.clone(),
                    DT,
                    0.0,
                );
                assert!(spend <= f64::from(w) * f64::from(DT) + 1e-12);
                rises += f64::from(rise);
            }
            if name == "inside" {
                assert!((response.rise_sum - rises).abs() < 1e-6);
                assert_eq!(response.pending, 0.0);
                let expected = f64::from(w) * tau() * response.rise_sum;
                assert!((response.extra_sec - expected).abs() < 1e-12, "{expected}");
                assert_eq!(response.listening_bins as usize, range.len());
            } else {
                assert_eq!(
                    (response.extra_sec, response.rise_sum),
                    (0.0, 0.0),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn unknown_future_repeated_and_regridded_observations_only_reprime() {
        let space = space();
        let n = space.n_bins();
        let all = 0..n;
        let mut response = OnsetResponse::new(2.0, n);
        let mut level = 0.1_f32;
        let mut next = |end: u64, known: bool, now: u64, response: &mut OnsetResponse| {
            level += 0.003;
            let scan: Arc<[f32]> = vec![level; n].into();
            response.step(
                &input(&scan, end, known),
                Some(now),
                &space,
                all.clone(),
                DT,
                0.0,
            );
            response.rise_sum
        };
        let h = STEP;
        assert_eq!(next(0, true, 0, &mut response), 0.0);
        // A repeated end, even with changed values.
        assert_eq!(next(0, true, h, &mut response), 0.0);
        // Unknown, then the next known observation only re-primes.
        assert_eq!(next(2 * h, false, 2 * h, &mut response), 0.0);
        assert_eq!(next(3 * h, true, 3 * h, &mut response), 0.0);
        // Future, then a re-prime.
        assert_eq!(next(5 * h, true, 4 * h, &mut response), 0.0);
        assert_eq!(next(5 * h, true, 5 * h, &mut response), 0.0);
        assert!((next(6 * h, true, 6 * h, &mut response) - 0.003).abs() < 1e-6);
        // Missing own observation: never the shared history.
        let empty = LooInput::default();
        response.step(&empty, Some(7 * h), &space, all.clone(), DT, 0.0);
        assert!((next(8 * h, true, 8 * h, &mut response) - 0.003).abs() < 1e-6);
        // A grid change re-primes and keeps the debt.
        response.pending = 1.0;
        let coarse = Log2Space::new(55.0, 8000.0, 48);
        let scan: Arc<[f32]> = vec![0.9; coarse.n_bins()].into();
        let rise = response.rise_sum;
        response.step(
            &input(&scan, 9 * h, true),
            Some(9 * h),
            &coarse,
            0..coarse.n_bins(),
            0.0,
            0.0,
        );
        assert_eq!((response.rise_sum, response.pending), (rise, 1.0));
    }

    #[test]
    fn skipped_observations_keep_the_total_and_bound_each_step() {
        let space = space();
        let n = space.n_bins();
        let w = 2.0;
        let (mut every, mut sparse) = (OnsetResponse::new(w, n), OnsetResponse::new(w, n));
        let mut held = (Arc::<[f32]>::from(vec![0.0; n]), 0);
        let mut spread = false;
        for step in 0..200_u64 {
            let scan: Arc<[f32]> = vec![0.04 * step.min(20) as f32; n].into();
            let at = step * STEP;
            if step % 4 == 0 {
                held = (scan.clone(), at);
            }
            let a = every.step(&input(&scan, at, true), Some(at), &space, 0..n, DT, 0.0);
            let b = sparse.step(
                &input(&held.0, held.1, true),
                Some(at),
                &space,
                0..n,
                DT,
                0.0,
            );
            for spend in [a, b] {
                assert!(spend <= f64::from(w) * f64::from(DT) + 1e-12);
            }
            spread |= step % 4 != 0 && b > 0.0;
        }
        assert!(spread, "sparse debt must spill into later steps");
        assert!((every.rise_sum - sparse.rise_sum).abs() < 1e-5);
        assert!((every.extra_sec - sparse.extra_sec).abs() < 1e-4);
        assert_eq!((every.pending, sparse.pending), (0.0, 0.0));
    }

    #[test]
    fn a_moved_note_keeps_one_ordinary_cycle_and_protected_rises_are_dropped() {
        let space = space();
        let n = space.n_bins();
        let mut landscape = Landscape::new(space.clone());
        let w = 4.0;
        let pitch = PitchControl {
            respond_to_onsets: w,
            ..Default::default()
        };
        let mut ctl = controller(&pitch, &space);
        let mut quiet = controller(&pitch, &space);
        quiet.dwell_trace_enabled = false;
        let rise = 0.01_f32;
        let mut last_proposal: Option<(u64, bool)> = None;
        let (mut moved, mut stayed_early) = (0, 0);
        let mut checked_first_spend = 0;
        for step in 0..3000_usize {
            terrain(&mut landscape, step);
            let at = step as u64 * STEP;
            // Every step rises inside the range, except at the sawtooth wrap.
            let scan: Arc<[f32]> = vec![rise * (step % 50) as f32; n].into();
            for c in [&mut ctl, &mut quiet] {
                observe(c, Some(scan.clone()), at, true);
                c.control_tick = Some(at);
            }
            let before = ctl.respond_trace.len();
            let a = update(&mut ctl, &landscape, &pitch);
            let b = update(&mut quiet, &landscape, &pitch);
            // Reporting changes neither targets nor random draws.
            assert_eq!(
                (a, ctl.target_pitch_log2.to_bits(), ctl.rng_state_for_test()),
                (
                    b,
                    quiet.target_pitch_log2.to_bits(),
                    quiet.rng_state_for_test()
                )
            );
            // One ordinary cycle in f32 real time after the last pitch change.
            let threshold = ctl.integration_window();
            let mut sum = 0.0_f32;
            let mut cycle = 0_u64;
            while sum < threshold {
                sum += DT;
                cycle += 1;
            }
            for record in ctl.respond_trace[before..].iter().copied() {
                let record_at = record.at.unwrap();
                if let Some((last, last_moved)) = last_proposal
                    && last_moved
                {
                    // Nothing is spent inside the protected cycle.
                    assert!(record_at >= last + cycle * STEP, "{record:?}");
                    if record.event == "spend" && record_at == last + cycle * STEP {
                        // Rises heard during protection were dropped, not saved.
                        assert!(record.pending <= f64::from(w) * tau() * f64::from(rise) + 1e-9);
                        checked_first_spend += 1;
                    }
                }
                if record.event == "proposal" {
                    if let Some((last, last_moved)) = last_proposal {
                        if last_moved {
                            assert_eq!(record_at - last, cycle * STEP, "{record:?}");
                            assert!(!record.protected.unwrap());
                        } else if record_at - last < cycle * STEP {
                            stayed_early += 1;
                        }
                    }
                    moved += usize::from(record.moved.unwrap());
                    last_proposal = Some((record_at, record.moved.unwrap()));
                }
            }
        }
        assert!(moved > 2 && stayed_early > 2, "{moved} {stayed_early}");
        assert!(checked_first_spend > 0);
        assert!(quiet.respond_trace.is_empty());
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn observation_and_phase_updates_allocate_nothing() {
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        let space = space();
        let n = space.n_bins();
        let mut landscape = Landscape::new(space.clone());
        let pitch = PitchControl {
            respond_to_onsets: 4.0,
            ..Default::default()
        };
        let mut ctl = controller(&pitch, &space);
        ctl.dwell_trace_enabled = false;
        let scans: Vec<Arc<[f32]>> = (0..50).map(|k| vec![0.01 * k as f32; n].into()).collect();
        let (mut measured, mut spending) = (0, 0);
        for step in 0..3000_usize {
            terrain(&mut landscape, step);
            let at = step as u64 * STEP;
            observe(&mut ctl, Some(scans[step % 50].clone()), at, true);
            ctl.control_tick = Some(at);
            let before = ctl.respond.as_ref().unwrap().extra_sec;
            begin_allocations();
            update(&mut ctl, &landscape, &pitch);
            let counts = finish_allocations().unwrap();
            // Proposals belong to the pitch core; observation and clock steps must not allocate.
            if !ctl.harmonic_evaluated {
                assert_eq!((counts.count, counts.bytes), (0, 0), "step {step}");
                measured += 1;
                spending += usize::from(ctl.respond.as_ref().unwrap().extra_sec > before);
            }
        }
        assert!(measured > 0 && spending > 0, "{measured} {spending}");
        println!("RESPOND_CONTROL_ALLOC measured_steps={measured} spending_steps={spending}");
    }

    #[test]
    fn without_its_own_observation_the_voice_never_reads_the_shared_history() {
        let space = space();
        let n = space.n_bins();
        let mut landscape = Landscape::new(space.clone());
        let positive = PitchControl {
            respond_to_onsets: 8.0,
            ..Default::default()
        };
        let zero = PitchControl::default();
        let mut responding = controller(&positive, &space);
        let mut plain = controller(&zero, &space);
        for step in 0..600_usize {
            terrain(&mut landscape, step);
            let at = step as u64 * STEP;
            landscape.acoustic.onset_habituation_state_scan =
                vec![0.01 * (step % 40) as f32; n].into();
            landscape.acoustic.onset_habituation_end = at;
            landscape.acoustic.onset_habituation_known = true;
            responding.control_tick = Some(at);
            plain.control_tick = Some(at);
            assert_eq!(
                update(&mut responding, &landscape, &positive),
                update(&mut plain, &landscape, &zero)
            );
            assert_eq!(responding.target_pitch_log2, plain.target_pitch_log2);
        }
        assert_eq!(responding.respond.as_ref().unwrap().extra_sec, 0.0);
    }
}
