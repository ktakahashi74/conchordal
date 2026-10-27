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

#[derive(Debug)]
pub struct PitchController {
    #[cfg(test)]
    pub(crate) trace_owner: Option<(u64, u32)>,
    core: AnyPitchCore,
    pub(crate) own_tonality: Option<crate::life::own_tonality::OwnTonality>,
    pub(crate) own_sequence: Option<crate::life::own_sequence::OwnSequence>,
    pub(crate) sequence_decisions: Vec<crate::life::own_sequence::SequenceDecision>,
    adaptation: AdaptationContext,
    target_pitch_log2: f32,
    integration_window: f32,
    accumulated_time: f32,
    proposal_phase: f32,
    pub(crate) dwell_state: Option<DwellRecord>,
    pub(crate) dwell_trace: Vec<DwellRecord>,
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
            #[cfg(test)]
            trace_owner: None,
            core,
            own_sequence: None,
            own_tonality: None,
            sequence_decisions: Vec::new(),
            adaptation,
            target_pitch_log2,
            integration_window,
            accumulated_time: 0.0,
            proposal_phase: 0.0,
            dwell_state: None,
            dwell_trace: Vec::new(),
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
        let previous_target = self.target_pitch_log2;
        let dt_sec = dt_sec.max(0.0);
        let current_freq = current_freq_hz.max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let mut target_pitch_log2 = if self.target_pitch_log2 <= 0.0 {
            current_pitch_log2
        } else {
            self.target_pitch_log2
        };
        self.integration_window = Self::integration_window_for_freq(current_freq);
        self.accumulated_time += dt_sec;
        let dwelling = pitch.dwell > 0.0;
        if dwelling {
            let (level, excluded) =
                self.core
                    .dwell_level(landscape, current_freq, pitch.leave_self_out);
            // At extreme weights, retain a finite pending phase instead of overflowing.
            self.proposal_phase = (self.proposal_phase
                + dt_sec * (1.0 + pitch.dwell * (1.0 - 2.0 * level)).max(0.0))
            .min(f32::MAX);
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
        }
        let proposal_clock = if dwelling {
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
        let should_propose = if let Some(interval_sec) = proposal_interval_sec {
            proposal_clock >= interval_sec
        } else if has_rhythm {
            theta_cross && proposal_clock >= self.integration_window
        } else {
            proposal_clock >= self.integration_window
        };
        if should_propose {
            let elapsed = self.accumulated_time;
            if dwelling {
                self.proposal_phase = proposal_interval_sec
                    .map_or(0.0, |interval| (self.proposal_phase - interval).max(0.0));
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
            // Stage 1.5/2 of the occupancy redesign: adaptation's environment term is
            // fed by the predictive fundamental-occupancy of visible neighbors
            // (f0-based, zero-latency, self-excluded), built with the same shared
            // kernel and width as the crowding penalty so the two share one
            // occupancy definition. See docs/design-notes/voice-movement-redesign.md.
            self.build_neighbor_occupancy(
                &landscape.space,
                neighbor_pitch_log2,
                neighbor_salience,
                pitch.crowding_sigma_cents,
            );
            let features = FeaturesNow::from_occupancy_scan(&landscape.space, &self.occupancy_scan);
            self.adaptation.ensure_len(features.density.len());
            if let AnyPitchCore::PitchHillClimb(core) = &mut self.core {
                core.tonality_enabled = self.own_tonality.is_some();
                core.tonality_trace = self.own_tonality.as_ref().is_some_and(|s| s.trace_enabled);
                let center = pitch.freq.log2();
                let half = pitch.range_oct.clamp(0.0, 6.0) * 0.5;
                core.tonality = self.control_tick.and_then(|at| {
                    self.own_tonality
                        .as_ref()
                        .and_then(|t| t.snapshot(at, (center - half, center + half)))
                });
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
                && let Some(mut record) = core.tonality_decision.borrow_mut().take()
                && let Some(tonality) = &mut self.own_tonality
            {
                record.at = self.control_tick.unwrap_or(0);
                record.epoch = tonality.epoch;
                if record.reason.is_none() {
                    record.reason = tonality.reason;
                }
                if tonality.trace_enabled {
                    tonality.decisions.push(record);
                }
            }
            target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if self.adaptation_enabled
                && let Some(idx) = landscape.space.index_of_log2(target_pitch_log2)
            {
                self.adaptation.update(idx, &features, elapsed);
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
        let (fmin, fmax) = landscape.freq_bounds_log2();
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
        let min = fmin.max(min_range);
        let max = fmax.min(max_range);
        let clamped = if min <= max {
            target_pitch_log2.clamp(min, max)
        } else {
            target_pitch_log2.clamp(fmin, fmax)
        };
        target_pitch_log2 = clamped;
        self.target_pitch_log2 = target_pitch_log2;
        if should_propose
            && previous_target > 0.0
            && target_pitch_log2 != previous_target
            && let (Some(sequence), Some(at)) = (&mut self.own_sequence, self.control_tick)
        {
            sequence.bind_target(at, target_pitch_log2);
        }
        #[cfg(test)]
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
    fn dwell_clock_integrates_each_control_and_uses_first_legal_crossing() {
        for branch in 0..3 {
            for changing in [false, true] {
                let pitch = PitchControl {
                    dwell: 0.75,
                    proposal_interval_sec: (branch == 0).then_some(0.31),
                    ..Default::default()
                };
                let mut landscape = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
                let mut ctl = controller(&pitch, &landscape.space);
                let mut phase = 0.0_f64;
                let mut elapsed = 0.0_f32;
                let mut previous_phase = 0.0;
                let mut rhythms = NeuralRhythms::default();
                let mut proposals = 0;
                for step in 0..180 {
                    let raw = if changing && step >= 70 { -0.7 } else { 0.6 };
                    landscape.consonance_field_score_eff.fill(raw);
                    let level = landscape.consonance_representation.level(raw);
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
        landscape.consonance_field_score_eff.fill(0.7);
        let mut ctl = controller(&pitch, &landscape.space);
        let mut reference =
            AdaptationContext::from_config(&AdaptationConfig::default(), landscape.space.n_bins());
        let features = FeaturesNow::from_occupancy_scan(
            &landscape.space,
            &vec![0.0; landscape.space.n_bins()],
        );
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
                &[],
                &[],
                0.0,
            );
            if ctl.dwell_trace.len() > before {
                let idx = landscape
                    .space
                    .index_of_log2(ctl.target_pitch_log2())
                    .unwrap();
                reference.update(idx, &features, real);
                real = 0.0;
                for i in 0..landscape.space.n_bins() {
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
        let mut traced = controller(&pitch, &landscape.space);
        let mut quiet = controller(&pitch, &landscape.space);
        quiet.dwell_trace_enabled = false;
        for step in 0..80 {
            landscape
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
            let mut ctl = controller(&pitch, &landscape.space);
            for (score, stopped) in [(100.0, true), (-100.0, false)] {
                landscape.consonance_field_score_eff.fill(score);
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
        landscape.consonance_field_score.fill(-10.0);
        landscape.consonance_field_score_eff.fill(0.3);
        let mut ctl = controller(&pitch, &landscape.space);
        let n = landscape.space.n_bins();
        ctl.loo_input_mut().exact_score = Some(vec![1.2; n].into());
        for h in [0.0, 0.5, 1.0] {
            landscape.perc_habituation_state_scan.fill(h);
            let (mixed, source) = ctl.core.dwell_level(&landscape, 220.0, false);
            assert!(!source);
            assert!((mixed - landscape.consonance_representation.level(0.3)).abs() < 1e-6);
            let (own, source) = ctl.core.dwell_level(&landscape, 220.0, true);
            assert!(source);
            assert!(
                (own - landscape.consonance_representation.level(1.2 * (1.0 - h))).abs() < 1e-6
            );
        }
        ctl.loo_input_mut().exact_score = None;
        assert!(!ctl.core.dwell_level(&landscape, 220.0, true).1);
        landscape.perc_habituation_state_scan.fill(0.0);
        landscape.consonance_field_score_eff.fill(0.0);
        let i = n / 2;
        landscape.consonance_field_score_eff[i] = 4.0;
        let log2 = (landscape.space.centers_log2[i] + landscape.space.centers_log2[i + 1]) * 0.5;
        let raw = landscape.sample_linear_log2(&landscape.consonance_field_score_eff, log2);
        assert!(
            (ctl.core.dwell_level(&landscape, log2.exp2(), false).0
                - landscape.consonance_representation.level(raw))
            .abs()
                < 1e-6
        );
        landscape.consonance_representation.theta = 0.7;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            landscape.consonance_field_score_eff.fill(invalid);
            assert_eq!(ctl.core.dwell_level(&landscape, 220.0, false).0, 0.5);
            ctl.loo_input_mut().exact_score = Some(vec![invalid; n].into());
            assert_eq!(ctl.core.dwell_level(&landscape, 220.0, true).0, 0.5);
        }
    }
}
