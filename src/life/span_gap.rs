//! Private author-rule span and gap state. Acoustic evidence is read-only.

use crate::core::habituation::HabituationParams;
use crate::core::landscape::Landscape;
use crate::core::timebase::Tick;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Unstarted,
    Span,
    GapRequested,
    GapTail,
    GapActive,
    Ready,
    Finished,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Transition {
    pub request_at: Option<Tick>,
    pub execution_at: Tick,
    pub state: Phase,
    pub heard: f32,
    pub own: f64,
    pub observation_end: Tick,
    pub known: bool,
    pub f_now: f32,
    pub f_next: f32,
    pub exit_integral: f64,
    pub return_integral: f64,
    pub target: f64,
    pub weight: f32,
}

#[derive(Debug)]
pub(crate) struct SpanGap {
    pub acoustic_quiet: bool,
    pub phase: Phase,
    pub weight: f32,
    pub own: f64,
    pub exit_integral: f64,
    pub return_integral: f64,
    pub target: f64,
    pub pitch: f32,
    pub next_pitch: f32,
    pub last_onset_pitch: f32,
    pub trace: Vec<Transition>,
    pub trace_enabled: bool,
    pub last_adoption: Option<Tick>,
    pub last_control: Option<Tick>,
    pub pending_pitch: Option<f32>,
    pub glide_target: Option<f32>,
    params: HabituationParams,
    period: f64,
    fs: f64,
    rng: SmallRng,
    pub last_tick: Option<Tick>,
    last_on: Option<Tick>,
    released: Option<Tick>,
    requested: Option<Tick>,
    scan: Arc<[f32]>,
    first_log2: f32,
    step_log2: f32,
    observation_end: Tick,
    known: bool,
}

impl SpanGap {
    pub(crate) fn new(
        weight: f32,
        scene: u64,
        id: u64,
        generation: u32,
        fs: f32,
        pitch: f32,
    ) -> Self {
        Self {
            acoustic_quiet: false,
            phase: Phase::Unstarted,
            weight,
            own: 0.0,
            exit_integral: 0.0,
            return_integral: 0.0,
            target: 0.0,
            pitch,
            next_pitch: pitch,
            last_onset_pitch: pitch,
            trace: Vec::new(),
            trace_enabled: false,
            last_control: None,
            last_adoption: None,
            pending_pitch: None,
            glide_target: None,
            params: HabituationParams::default(),
            period: 0.5,
            fs: f64::from(fs),
            rng: SmallRng::seed_from_u64(
                scene
                    ^ id.rotate_left(21)
                    ^ u64::from(generation).rotate_left(43)
                    ^ 0x5350_414e_5f47_4150,
            ),
            last_tick: None,
            last_on: None,
            released: None,
            requested: None,
            scan: Arc::from([]),
            first_log2: 0.0,
            step_log2: 1.0,
            observation_end: 0,
            known: false,
        }
    }

    pub(crate) fn configure(&mut self, params: HabituationParams, period: f64) {
        self.params = params;
        self.period = period;
    }

    pub(crate) fn observe(&mut self, frame: &Landscape, now: Tick) {
        if frame.onset_habituation_end > now {
            if self.known
                && (self.scan.len() != frame.space.n_bins()
                    || self.first_log2 != frame.space.centers_log2[0]
                    || self.step_log2 != frame.space.step_log2)
            {
                self.known = false;
            }
            return;
        }
        if !frame.onset_habituation_known {
            self.known = false;
            self.observation_end = frame.onset_habituation_end;
        } else if frame.onset_habituation_end <= now {
            frame
                .space
                .assert_scan_len(&frame.onset_habituation_state_scan);
            self.scan = Arc::clone(&frame.onset_habituation_state_scan);
            self.first_log2 = frame.space.centers_log2[0];
            self.step_log2 = frame.space.step_log2;
            self.observation_end = frame.onset_habituation_end;
            self.known = true;
        }
    }

    pub(crate) fn heard(&self) -> f32 {
        if !self.known || self.scan.is_empty() {
            return 0.0;
        }
        let pos = (self.pitch.log2() - self.first_log2) / self.step_log2;
        crate::core::log2space::sample_scan_linear_at_pos(&self.scan, pos)
    }

    fn record(&mut self, at: Tick) {
        if self.trace_enabled {
            self.trace.push(Transition {
                request_at: self.requested,
                execution_at: at,
                state: self.phase,
                heard: self.heard(),
                own: self.own,
                observation_end: self.observation_end,
                known: self.known,
                f_now: self.pitch,
                f_next: self.next_pitch,
                exit_integral: self.exit_integral,
                return_integral: self.return_integral,
                target: self.target,
                weight: self.weight,
            });
        }
    }

    pub(crate) fn advance(&mut self, to: Tick, frozen: bool) {
        let from = self.last_tick.replace(to).unwrap_or(to);
        assert!(to >= from, "span time is monotonic");
        let expiry = self.last_on.map(|at| at as f64 + self.period * self.fs);
        let mut cursor = from as f64;
        let end = to as f64;
        while cursor < end {
            let driven = expiry.is_some_and(|expiry| cursor < expiry);
            let limit = if driven {
                end.min(expiry.unwrap())
            } else {
                end
            };
            let dt = (limit - cursor) / self.fs;
            let target = if driven { 1.0 } else { 0.0 };
            let tau = f64::from(if target > self.own {
                self.params.satiation_sec
            } else {
                self.params.recovery_sec
            }) / std::f64::consts::LN_10;
            let a = (-dt / tau).exp();
            let own_integral = target * dt + (self.own - target) * tau * (1.0 - a);
            if self.known && !frozen {
                let tau_e = f64::from(self.params.satiation_sec) / std::f64::consts::LN_10;
                match self.phase {
                    Phase::Span => {
                        self.exit_integral += f64::from(self.weight * self.heard()) * dt / tau_e
                    }
                    Phase::GapActive => {
                        self.return_integral += (dt - own_integral).max(0.0) / tau_e
                    }
                    _ => {}
                }
            }
            self.own = target + (self.own - target) * a;
            cursor = limit;
        }
    }

    pub(crate) fn opportunity(&mut self, at: Tick, frozen: bool) {
        if self.known && !frozen {
            if self.phase == Phase::Span && self.params.enabled && self.exit_integral >= self.target
            {
                self.phase = Phase::GapRequested;
                self.requested = Some(at);
                self.record(at);
            } else if self.phase == Phase::GapActive && self.return_integral >= self.target {
                self.phase = Phase::Ready;
                self.record(at);
            }
        }
    }

    pub(crate) fn allows_on(&self) -> bool {
        matches!(self.phase, Phase::Unstarted | Phase::Span | Phase::Ready)
    }

    pub(crate) fn on(&mut self, at: Tick, participation: bool) {
        self.last_on = Some(at);
        self.acoustic_quiet = false;
        self.pitch = self.next_pitch;
        self.last_onset_pitch = self.pitch;
        if participation {
            self.pending_pitch = None;
        }
        if matches!(self.phase, Phase::Unstarted | Phase::Ready) {
            self.phase = Phase::Span;
            self.exit_integral = 0.0;
            self.target = -self.rng.sample::<f64, _>(rand::distr::Open01).ln();
            self.requested = None;
            self.record(at);
        }
    }

    pub(crate) fn off(&mut self, at: Tick) {
        if self.phase == Phase::GapRequested {
            self.phase = Phase::GapTail;
            self.released = Some(at);
            self.record(at);
        }
    }

    pub(crate) fn quiet(&mut self, at: Tick, integrate_from: Option<Tick>) {
        if self.last_on.is_none_or(|on| on <= at) {
            self.acoustic_quiet = true;
        }
        if self.phase != Phase::GapTail || self.released.is_none_or(|off| at < off) {
            return;
        }
        self.phase = Phase::GapActive;
        self.return_integral = 0.0;
        self.target = -self.rng.sample::<f64, _>(rand::distr::Open01).ln();
        let mut own_at_quiet = self.own;
        // Notifications arrive after rendering. Reconstruct this attack-free interval once.
        if let Some(now) = self.last_tick
            && now > at
        {
            let expiry = self
                .last_on
                .map_or(0.0, |on| on as f64 + self.period * self.fs);
            let mut cursor = now as f64;
            let mut own = self.own;
            let mut eligible_seconds = 0.0;
            let mut own_integral = 0.0;
            while cursor > at as f64 {
                let drive = if cursor <= expiry { 1.0 } else { 0.0 };
                let start = if cursor > expiry {
                    (at as f64).max(expiry)
                } else {
                    at as f64
                };
                let dt = (cursor - start) / self.fs;
                let tau = f64::from(if drive > 0.0 {
                    self.params.satiation_sec
                } else {
                    self.params.recovery_sec
                }) / std::f64::consts::LN_10;
                let before = drive + (own - drive) * (dt / tau).exp();
                if let Some(resume) = integrate_from {
                    let eligible = ((cursor - start.max(resume as f64)) / self.fs).max(0.0);
                    eligible_seconds += eligible;
                    own_integral +=
                        drive * eligible + (own - drive) * tau * (eligible / tau).exp_m1();
                }
                own = before;
                cursor = start;
            }
            own_at_quiet = own;
            if self.known {
                self.return_integral = (eligible_seconds - own_integral).max(0.0)
                    * std::f64::consts::LN_10
                    / f64::from(self.params.satiation_sec);
            }
        }
        let current_own = self.own;
        let caught_up = self.return_integral;
        self.own = own_at_quiet;
        self.return_integral = 0.0;
        self.record(at);
        self.own = current_own;
        self.return_integral = caught_up;
    }

    pub(crate) fn finish(&mut self, at: Tick) {
        if self.phase != Phase::Finished {
            self.phase = Phase::Finished;
            self.record(at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;

    fn frame(value: f32, end: Tick) -> Landscape {
        let mut frame = Landscape::new(Log2Space::new(220.0, 880.0, 12));
        frame.onset_habituation_state_scan = vec![value; frame.space.n_bins()].into();
        frame.onset_habituation_end = end;
        frame.onset_habituation_known = true;
        frame
    }

    #[test]
    fn actual_attacks_drive_private_relaxation_without_mutating_audio_evidence() {
        let evidence = frame(0.3, 0);
        let mut state = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        state.configure(HabituationParams::default(), 5.0);
        state.observe(&evidence, 0);
        state.advance(0, false);
        state.on(0, true);
        state.advance(240000, true);
        assert!((state.own - 0.9).abs() < 1e-12);
        assert_eq!(state.exit_integral, 0.0);
        state.advance(624000, true);
        assert!((state.own - 0.09).abs() < 1e-12);
        assert!(
            evidence
                .onset_habituation_state_scan
                .iter()
                .all(|&v| v == 0.3)
        );
    }

    #[test]
    fn held_delivery_uses_voice_time_and_distinguishes_unknown_future_and_duplicates() {
        let mut state = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        state.observe(&frame(0.4, 0), 0);
        state.advance(0, false);
        state.on(0, true);
        state.advance(48000, false);
        let unit = 0.25 * f64::from(0.4_f32) * std::f64::consts::LN_10 / 5.0;
        assert!((state.exit_integral - unit).abs() < 1e-12);
        state.observe(&frame(0.4, 0), 48000);
        state.advance(48000, false);
        assert!((state.exit_integral - unit).abs() < 1e-12);
        state.observe(&frame(1.0, 96001), 48000);
        state.advance(96000, false);
        assert!((state.exit_integral - 2.0 * unit).abs() < 1e-12);
        let mut unknown = frame(1.0, 96000);
        unknown.onset_habituation_known = false;
        state.observe(&unknown, 96000);
        state.advance(144000, false);
        assert!((state.exit_integral - 2.0 * unit).abs() < 1e-12);
        assert!(!state.known);
        state.observe(&frame(0.2, 144000), 144000);
        state.advance(192000, false);
        assert!((state.exit_integral - 2.5 * unit).abs() < 1e-12);
    }

    #[test]
    fn return_ignores_shared_value_and_late_quiet_catches_up_once() {
        let mut timely = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        let mut delayed = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        for (state, heard) in [(&mut timely, 0.0), (&mut delayed, 1.0)] {
            state.observe(&frame(heard, 0), 0);
            state.advance(0, false);
            state.on(0, true);
            state.advance(12000, false);
            state.phase = Phase::GapRequested;
            state.off(12000);
            state.quiet(11999, Some(11999));
            assert_eq!(state.phase, Phase::GapTail);
        }
        timely.advance(16000, false);
        timely.quiet(16000, Some(16000));
        timely.advance(48000, false);
        delayed.advance(48000, false);
        delayed.quiet(16000, Some(16000));
        assert_eq!(timely.target, delayed.target);
        assert!((timely.return_integral - delayed.return_integral).abs() < 1e-12);
        assert!((timely.own - delayed.own).abs() < 1e-12);
        let integral = delayed.return_integral;
        delayed.quiet(16000, Some(16000));
        assert_eq!(delayed.return_integral, integral);
    }

    #[test]
    fn report_switch_and_suppressed_opportunities_do_not_draw_random_targets() {
        let mut quiet = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        let mut reported = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        reported.trace_enabled = true;
        for index in 0..20 {
            for state in [&mut quiet, &mut reported] {
                state.phase = Phase::Ready;
                state.opportunity(index, false);
                state.on(index, true);
            }
            assert_eq!(quiet.target, reported.target);
        }
        assert!(quiet.trace.is_empty());
        assert_eq!(reported.trace.len(), 20);
    }
}

#[cfg(test)]
mod late_rest_test {
    use super::*;
    #[test]
    fn tail_notification_integrates_only_after_fixed_rest_even_inside_the_same_hop() {
        let mut timely = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        let mut late = SpanGap::new(0.25, 21, 1, 0, 48000.0, 440.0);
        for state in [&mut timely, &mut late] {
            state.known = true;
            state.advance(0, false);
            state.on(0, true);
            state.advance(100, false);
            state.phase = Phase::GapRequested;
            state.off(100);
        }
        timely.advance(200, false);
        timely.quiet(200, Some(300));
        timely.advance(300, true);
        timely.advance(512, false);
        late.advance(512, true);
        late.quiet(200, Some(300));
        assert!((timely.return_integral - late.return_integral).abs() < 1e-12);
        assert!(late.return_integral > 0.0);
    }
}
