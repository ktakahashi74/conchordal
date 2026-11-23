//! Author rule for independent beat confirmation and half-cycle deferral.
use super::*;
use crate::core::felt_beat::FeltBeatState;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeltRhythmInput {
    pub snapshot: FeltBeatState,
    pub observed_end: Tick,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct FeltRhythmRecord {
    pub opportunity_tick: Tick,
    pub observed_end: Tick,
    pub sample_rate: f32,
    pub c: f32,
    pub m_plus: f32,
    pub p_beat: f32,
    pub r: f32,
    pub unknown: bool,
    pub felt_phase: f32,
    pub clock_phase: f32,
    pub phase_error: f32,
    pub frequency_hz: f32,
    pub meter_confidence: f32,
    pub flips: u32,
    pub beat_probability: f32,
    pub beat_draw: Option<f32>,
    pub half_draw: Option<f32>,
    pub result: &'static str,
    pub deferred: bool,
    pub scheduled_tick: Option<Tick>,
    pub executed_tick: Option<Tick>,
    pub execution_felt_phase: Option<f32>,
    pub execution_phase_error: Option<f32>,
    pub pitch_hz: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    gate: u64,
    kick: OnsetKick,
    record: FeltRhythmRecord,
}

#[derive(Debug)]
pub(crate) struct FeltRhythm {
    w: f32,
    rng: SmallRng,
    pub(super) input: Option<FeltRhythmInput>,
    pending: Option<Pending>,
    /// Integer and half crossings; separate from the normal veto opportunities.
    pub(super) points: Vec<CandidatePoint>,
    pub(super) delivery_grid: ThetaGrid,
    pub(super) delivering_half: bool,
    pub(crate) trace_enabled: bool,
    pub(crate) trace: Vec<FeltRhythmRecord>,
}

impl FeltRhythm {
    pub(crate) fn new(w: f32, seed: u64) -> Self {
        Self {
            w,
            rng: SmallRng::seed_from_u64(seed),
            input: None,
            pending: None,
            points: Vec::with_capacity(16),
            delivery_grid: ThetaGrid {
                boundaries: Vec::with_capacity(8),
            },
            delivering_half: false,
            trace_enabled: false,
            trace: Vec::new(),
        }
    }

    pub(crate) fn enable_trace(&mut self, enabled: bool) {
        if enabled && !self.trace_enabled {
            self.trace.reserve(16);
        }
        self.trace_enabled = enabled;
    }

    fn known(&self) -> Option<FeltRhythmInput> {
        self.input
            .filter(|i| !i.snapshot.unknown && i.snapshot.seeded)
    }

    pub(crate) fn set_input(&mut self, input: Option<FeltRhythmInput>) {
        self.input = input;
    }

    pub(crate) fn rhythms(&self, mut original: NeuralRhythms, now: Tick, fs: f32) -> NeuralRhythms {
        if let Some(i) = self.known() {
            original.delta.freq_hz = i.snapshot.meter_frequency_hz;
            original.delta.phase = TAU
                * (i.snapshot.phase / TAU
                    + now.saturating_sub(i.observed_end) as f32 / fs
                        * i.snapshot.meter_frequency_hz)
                    .rem_euclid(1.0);
            original.delta.alpha = i.snapshot.p_beat * i.snapshot.r;
        }
        original
    }

    fn record(&self, at: Tick, ctx: &CoreTickCtx, phase: f32) -> FeltRhythmRecord {
        let i = self.input;
        let s = i.map_or(FeltBeatState::default(), |i| i.snapshot);
        let elapsed = i.map_or(0.0, |i| at.saturating_sub(i.observed_end) as f32 / ctx.fs);
        let felt_phase = (s.phase / TAU + elapsed * s.meter_frequency_hz).rem_euclid(1.0);
        let err = felt_phase - phase;
        FeltRhythmRecord {
            opportunity_tick: at,
            observed_end: i.map_or(0, |i| i.observed_end),
            sample_rate: ctx.fs,
            c: s.c,
            m_plus: s.m_plus,
            p_beat: s.p_beat,
            r: s.r,
            unknown: self.known().is_none(),
            felt_phase,
            clock_phase: phase,
            phase_error: err - err.round(),
            frequency_hz: s.meter_frequency_hz,
            meter_confidence: s.meter_confidence,
            flips: s.flips,
            beat_probability: 1.0,
            beat_draw: None,
            half_draw: None,
            result: "beat",
            deferred: false,
            scheduled_tick: None,
            executed_tick: None,
            execution_felt_phase: None,
            execution_phase_error: None,
            pitch_hz: None,
        }
    }

    fn publish(&mut self, record: FeltRhythmRecord) {
        if self.trace_enabled {
            self.trace.push(record);
        }
    }

    pub(super) fn veto(&mut self, c: &CandidatePoint, ctx: &CoreTickCtx) {
        let mut record = self.record(c.tick, ctx, 0.0);
        record.result = "veto";
        self.publish(record);
    }

    pub(super) fn decide(
        &mut self,
        c: &CandidatePoint,
        kick: OnsetKick,
        ctx: &CoreTickCtx,
    ) -> bool {
        let mut r = self.record(c.tick, ctx, 0.0);
        if r.unknown {
            r.result = "unknown";
            r.scheduled_tick = Some(c.tick);
            self.publish(r);
            return true;
        }
        let mut half = false;
        if self.w > 0.0 {
            r.beat_probability = (1.0 - r.c).clamp(0.0, 1.0).powf(self.w);
            let draw = self.rng.random::<f32>();
            r.beat_draw = Some(draw);
            if draw >= r.beat_probability {
                let draw = self.rng.random::<f32>();
                r.half_draw = Some(draw);
                if draw < r.m_plus {
                    half = true;
                    r.result = "half";
                } else {
                    r.result = "omit";
                    self.publish(r);
                    return false;
                }
            }
        }
        if !half {
            r.scheduled_tick = Some(c.tick);
            self.publish(r);
            return true;
        }
        assert!(self.pending.is_none(), "at most one half-cycle deferral");
        self.pending = Some(Pending {
            gate: c.gate,
            kick,
            record: r,
        });
        self.publish(r);
        false
    }

    pub(super) fn take_due(&mut self, c: &CandidatePoint, half: bool) -> Option<OnsetKick> {
        if !half || self.pending.as_ref()?.gate != c.gate {
            return None;
        }
        let p = self.pending.take().unwrap();
        let mut r = p.record;
        r.deferred = true;
        r.scheduled_tick = Some(c.tick);
        self.publish(r);
        Some(p.kick)
    }

    pub(crate) fn executed(&mut self, at: Tick, hz: f32) {
        for r in self.trace.iter_mut().rev() {
            if r.scheduled_tick == Some(at) {
                r.executed_tick = Some(at);
                r.pitch_hz = Some(hz);
                if let Some(i) = self.input {
                    let phase = (i.snapshot.phase / TAU
                        + at.saturating_sub(i.observed_end) as f32 / r.sample_rate
                            * i.snapshot.meter_frequency_hz)
                        .rem_euclid(1.0);
                    r.execution_felt_phase = Some(phase);
                    r.execution_phase_error = Some(phase - phase.round());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx() -> CoreTickCtx {
        CoreTickCtx {
            now_tick: 0,
            frame_end: 512,
            fs: 48000.0,
            rhythms: NeuralRhythms::default(),
        }
    }
    fn input(c: f32, m: f32) -> FeltRhythmInput {
        FeltRhythmInput {
            snapshot: FeltBeatState {
                c,
                m_plus: m,
                p_beat: 1.0,
                r: 0.9,
                seeded: true,
                unknown: false,
                meter_frequency_hz: 2.0,
                phase: TAU * 0.25,
                ..Default::default()
            },
            observed_end: 0,
        }
    }
    #[test]
    fn unknown_bypasses_without_drawing_and_zero_has_no_state() {
        let mut f = FeltRhythm::new(1.0, 7);
        let mut before = f.rng.clone();
        assert!(f.decide(
            &CandidatePoint { tick: 0, gate: 0 },
            OnsetKick { strength: 1.0 },
            &ctx()
        ));
        assert_eq!(before.random::<u64>(), f.rng.random::<u64>());
        let e = PhonationEngine::from_spec(&PhonationSpec::default(), 7);
        assert!(e.felt.0.is_none());
    }
    #[test]
    fn confirmation_half_and_omission_follow_the_equations() {
        for (c, m, result) in [(0.0, 0.0, "beat"), (1.0, 1.0, "half"), (1.0, 0.0, "omit")] {
            let mut f = FeltRhythm::new(1.0, 9);
            f.trace_enabled = true;
            f.set_input(Some(input(c, m)));
            assert_eq!(
                f.decide(
                    &CandidatePoint { tick: 0, gate: 2 },
                    OnsetKick { strength: 0.7 },
                    &ctx()
                ),
                result == "beat"
            );
            assert_eq!(f.trace[0].result, result);
            assert_eq!(
                f.take_due(
                    &CandidatePoint {
                        tick: 12000,
                        gate: 2
                    },
                    true
                )
                .is_some(),
                result == "half"
            );
            assert!(
                f.take_due(
                    &CandidatePoint {
                        tick: 12000,
                        gate: 2
                    },
                    true
                )
                .is_none()
            );
        }
    }
    #[test]
    fn reading_changes_only_the_target_and_not_the_oscillator_phase() {
        let mut clock = CouplingClock::new(0.9, 2.0, 0.0, 0.0, 7);
        let mut f = FeltRhythm::new(1.0, 7);
        let mut i = input(0.0, 0.0);
        f.set_input(Some(i));
        let r = f.rhythms(NeuralRhythms::default(), 0, 48000.0);
        assert!((r.delta.phase - TAU * 0.25).abs() < 1e-6);
        assert!((r.delta.alpha - 0.9).abs() < 1e-6);
        let old = clock.phase;
        i.snapshot.phase += TAU * 0.5;
        f.set_input(Some(i));
        assert_eq!(clock.phase, old);
        let r = f.rhythms(r, 0, 48000.0);
        let mut out = Vec::new();
        clock.gather_rhythm(48000.0, 0, 512, &r, &mut out);
        assert!(clock.phase > old && clock.phase - old < 0.04);
    }
    #[cfg(feature = "profile-alloc")]
    #[test]
    fn steady_decisions_and_pending_delivery_allocate_nothing() {
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        let mut f = FeltRhythm::new(1.0, 7);
        f.set_input(Some(input(1.0, 1.0)));
        begin_allocations();
        for gate in 0..8000 {
            let c = CandidatePoint {
                tick: gate * 24000,
                gate,
            };
            f.decide(&c, OnsetKick { strength: 1.0 }, &ctx());
            std::hint::black_box(f.take_due(&c, true));
            std::hint::black_box(f.rhythms(NeuralRhythms::default(), c.tick, 48000.0));
            f.executed(c.tick, 220.0);
        }
        let count = finish_allocations().unwrap();
        assert_eq!((count.count, count.bytes), (0, 0));
    }
    #[cfg(feature = "profile-alloc")]
    #[test]
    fn engine_half_delivery_keeps_duration_and_allocates_nothing_after_birth() {
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        let spec = PhonationSpec {
            timing: crate::scenario::PhonationTiming::Coupled(
                crate::scenario::CoupledTimingSpec::default(),
            ),
            duration: crate::scenario::DurationSpec::Gates(1),
            sustain_participation: false,
        };
        let mut e = PhonationEngine::from_spec(&spec, 7);
        e.felt.0 = Some(Box::new(FeltRhythm::new(1.0, 7)));
        let mut cmds = Vec::with_capacity(64);
        let mut events = Vec::with_capacity(64);
        let mut onsets = Vec::with_capacity(64);
        let state = CoreState {
            is_alive: true,
            onset_allowed: true,
        };
        let mut total = 0;
        for j in 0..1600u64 {
            if j == 100 {
                begin_allocations();
            }
            let mut i = input(1.0, 1.0);
            i.observed_end = j * 512;
            i.snapshot.phase = TAU * (j as f32 * 512.0 / 24000.0).rem_euclid(1.0);
            let f = e.felt.0.as_mut().unwrap();
            f.set_input(Some(i));
            let mut ctx = ctx();
            ctx.now_tick = j * 512;
            ctx.frame_end = ctx.now_tick + 512;
            ctx.rhythms = f.rhythms(ctx.rhythms, ctx.now_tick, ctx.fs);
            cmds.clear();
            events.clear();
            onsets.clear();
            e.tick(
                &ctx,
                &state,
                None,
                0.0,
                1.0,
                None,
                &mut cmds,
                &mut events,
                &mut onsets,
            );
            total += onsets.len();
            for on in &onsets {
                assert!(
                    e.pending_off
                        .iter()
                        .any(|off| off.0.off_tick > on.onset_tick + 10000)
                );
            }
        }
        let n = finish_allocations().unwrap();
        assert!(total > 20, "half targets must actually sound");
        assert_eq!((n.count, n.bytes), (0, 0));
    }
}
