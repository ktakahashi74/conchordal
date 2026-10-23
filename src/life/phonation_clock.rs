use super::*;

#[derive(Debug, Default)]
pub struct ThetaGateClock {
    pub(super) last_gate_tick: Option<Tick>,
    pub(super) gate_index: u64,
    pub(super) has_gate: bool,
}

impl ThetaGateClock {
    pub(super) fn gather_candidates_core(
        &mut self,
        ctx: &CoreTickCtx,
        out: &mut Vec<CandidatePoint>,
        mut next_gate_tick_fn: impl FnMut(Tick, &CoreTickCtx) -> Option<Tick>,
    ) {
        let mut cursor = ctx.now_tick;
        while cursor < ctx.frame_end {
            let gate_tick = match next_gate_tick_fn(cursor, ctx) {
                Some(tick) => tick,
                None => return,
            };
            if gate_tick < cursor {
                cursor = cursor.saturating_add(1);
                continue;
            }
            if gate_tick < ctx.now_tick || gate_tick >= ctx.frame_end {
                return;
            }
            if self.last_gate_tick.is_some_and(|last| gate_tick <= last) {
                cursor = gate_tick.saturating_add(1);
                continue;
            }
            if self.has_gate {
                self.gate_index = self.gate_index.saturating_add(1);
            } else {
                self.gate_index = 0;
                self.has_gate = true;
            }
            self.last_gate_tick = Some(gate_tick);
            out.push(CandidatePoint {
                tick: gate_tick,
                gate: self.gate_index,
            });
            cursor = gate_tick.saturating_add(1);
        }
    }
}

impl ThetaGateClock {
    pub(super) fn gather_candidates_impl(
        &mut self,
        ctx: &CoreTickCtx,
        out: &mut Vec<CandidatePoint>,
    ) {
        self.gather_candidates_core(ctx, out, |cursor, ctx| {
            let mut theta = ctx.rhythms.theta;
            // The observation is stamped at frame start; advance it to the
            // search cursor before predicting another crossing in this hop.
            let elapsed_sec = cursor.saturating_sub(ctx.now_tick) as f64 / ctx.fs as f64;
            theta.phase = (theta.phase as f64 + TAU as f64 * theta.freq_hz as f64 * elapsed_sec)
                .rem_euclid(TAU as f64) as f32;
            next_gate_tick(cursor, ctx.fs, theta, 0.0).map(|next| {
                if cursor == ctx.now_tick && theta.phase == 0.0 {
                    cursor
                } else {
                    next
                }
            })
        });
    }
}

/// Upper bound on a clock's effective onset rate. Caps candidate density so a
/// pathological base/meter rate cannot collapse the period to ~1 tick and emit
/// a candidate at every sample.
pub(crate) const MAX_ONSET_RATE_HZ: f32 = 200.0;

/// Strength of the phase pull toward the shared meter beat. Kept at 1.0 so the
/// effective-frequency factor `1 + lock * PULL_K * err` stays in `[0.5, 1.5]`
/// (with `lock` in `[0,1]` and `err` in `[-0.5, 0.5]`) and the phase advances
/// monotonically. A backward phase jump would corrupt the in-hop integer-
/// crossing emission below.
pub(super) const COUPLING_PULL_K: f32 = 1.0;

/// Per-voice phase oscillator that entrains its onset phase to the shared
/// production meter beat with a coupling strength `coupling`. This single
/// mechanism spans the rhythm family continuum: `coupling -> 0` is a free
/// renewal process (flow texture), medium coupling entrains loosely as the
/// meter gains confidence (entrained beat), and high coupling locks into the
/// shared beat as a deep attractor (metric beat). Onsets are emitted at integer
/// crossings of the oscillator phase; a phase pull toward the meter beat phase
/// drags those crossings into a common alignment, so independently placed
/// voices synchronize without any externally imposed grid.
#[derive(Clone, Debug)]
pub struct CouplingClock {
    pub(crate) shape: Option<crate::life::shape_policy::ShapePolicy>,
    pub(super) coupling: f32,
    pub(super) base_rate_hz: f32,
    pub(super) flow_depth: f32,
    pub(super) microtiming: f32,
    /// Monotonic oscillator phase in cycles.
    pub(super) phase: f64,
    /// Intrinsic onset rate for the current interval, redrawn from the renewal
    /// distribution at each onset (constant when `flow_depth == 0`).
    pub(super) intrinsic_rate_hz: f32,
    /// Last effective rate, used to estimate a nominal period for duration grid
    /// extrapolation.
    pub(super) last_f_eff_hz: f32,
    pub(super) cluster_remaining: u8,
    pub(super) rng: SmallRng,
    pub(super) gate_index: u64,
    pub(super) has_gate: bool,
}

impl CouplingClock {
    pub(crate) fn new(
        coupling: f32,
        base_rate_hz: f32,
        flow_depth: f32,
        microtiming: f32,
        seed: u64,
    ) -> Self {
        let base_rate_hz = if base_rate_hz.is_finite() {
            base_rate_hz.clamp(0.01, MAX_ONSET_RATE_HZ)
        } else {
            1.0
        };
        let flow_depth = flow_depth.clamp(0.0, 1.0);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xC0D1_4E55_7A1C_0911);
        let mut cluster_remaining = 0u8;
        let first_ioi =
            OnsetRule::flow_next_ioi(base_rate_hz, flow_depth, &mut cluster_remaining, &mut rng);
        // Random initial phase so free-running voices placed together do not all
        // fire on the same tick before the meter entrains them.
        let phase: f64 = rng.random_range(0.0..1.0);
        Self {
            shape: None,
            coupling: coupling.clamp(0.0, 1.0),
            base_rate_hz,
            flow_depth,
            microtiming: if microtiming.is_finite() {
                microtiming
            } else {
                0.0
            },
            phase,
            intrinsic_rate_hz: (1.0 / first_ioi).clamp(0.01, MAX_ONSET_RATE_HZ),
            last_f_eff_hz: base_rate_hz,
            cluster_remaining,
            rng,
            gate_index: 0,
            has_gate: false,
        }
    }

    pub(crate) fn phase_cycles(&self) -> f64 {
        self.phase
    }

    pub(crate) fn nominal_period_ticks(&self, fs: f32) -> Option<f64> {
        if fs.is_finite() && fs > 0.0 {
            Some((fs as f64 / self.last_f_eff_hz.max(0.01) as f64).max(1.0))
        } else {
            None
        }
    }

    fn velocity_from_rhythms(&self, rhythms: &NeuralRhythms) -> (f32, f32, f32) {
        let conf = rhythms.delta.alpha.clamp(0.0, 1.0);
        let lock = (self.coupling * conf).clamp(0.0, 1.0);
        let f_band = {
            let f = rhythms.delta.freq_hz;
            if f.is_finite() && f > 0.0 { f } else { 2.0 }
        };
        // Meter beat phase in cycles, [0,1); microtiming offsets the lock target.
        let target_frac = ((rhythms.delta.phase / TAU) + self.microtiming).rem_euclid(1.0);

        let f_intrinsic = self.intrinsic_rate_hz.max(0.01);
        let mut f_eff = (1.0 - lock) * f_intrinsic + lock * f_band;
        if !f_eff.is_finite() {
            f_eff = f_intrinsic;
        }
        f_eff = f_eff.clamp(0.01, MAX_ONSET_RATE_HZ);

        let my_frac = self.phase.rem_euclid(1.0) as f32;
        let raw_err = target_frac - my_frac;
        let err = raw_err - raw_err.round(); // signed phase error in [-0.5, 0.5]
        let mut phase_dot = f_eff * (1.0 + lock * COUPLING_PULL_K * err);
        if !phase_dot.is_finite() || phase_dot <= 0.0 {
            phase_dot = f_eff.max(0.01);
        }

        (f_eff, phase_dot, my_frac)
    }

    pub(super) fn gather_candidates_impl(
        &mut self,
        ctx: &CoreTickCtx,
        out: &mut Vec<CandidatePoint>,
    ) {
        self.gather_rhythm(ctx.fs, ctx.now_tick, ctx.frame_end, &ctx.rhythms, out);
    }

    pub(crate) fn gather_rhythm(
        &mut self,
        fs: f32,
        now_tick: Tick,
        frame_end: Tick,
        rhythms: &NeuralRhythms,
        out: &mut Vec<CandidatePoint>,
    ) {
        if !fs.is_finite() || fs <= 0.0 || frame_end <= now_tick {
            return;
        }
        let end_tick = frame_end;
        let frame_end = frame_end as f64;
        let mut cursor = now_tick as f64;
        // Defensive bound on in-hop crossings (degenerate rates only).
        let mut guard = 0u32;
        loop {
            guard += 1;
            if guard > 100_000 {
                break;
            }
            let (f_eff, phase_dot, my_frac) = self.velocity_from_rhythms(rhythms);
            self.last_f_eff_hz = f_eff;

            // Cycles to the next integer crossing (a full cycle when on a beat).
            let frac_to_next = 1.0 - my_frac as f64;
            let dt_cross_sec = frac_to_next / phase_dot as f64;
            let cross_tick = cursor + dt_cross_sec * fs as f64;
            if !cross_tick.is_finite() || cross_tick >= frame_end {
                let remaining_sec = (frame_end - cursor) / fs as f64;
                if remaining_sec > 0.0 {
                    self.phase += phase_dot as f64 * remaining_sec;
                }
                break;
            }

            let adv_sec = (cross_tick - cursor) / fs as f64;
            self.phase += phase_dot as f64 * adv_sec;
            // Land exactly on the integer to stop floating drift accumulating.
            self.phase = self.phase.round();

            let tick = (cross_tick.round() as Tick).clamp(now_tick, end_tick - 1);
            if self.has_gate {
                self.gate_index = self.gate_index.saturating_add(1);
            } else {
                self.has_gate = true;
            }
            out.push(CandidatePoint {
                tick,
                gate: self.gate_index,
            });

            // Renewal: redraw the intrinsic rate for the next interval.
            let ioi = OnsetRule::flow_next_ioi(
                self.base_rate_hz,
                self.flow_depth,
                &mut self.cluster_remaining,
                &mut self.rng,
            );
            self.intrinsic_rate_hz = (1.0 / ioi).clamp(0.01, MAX_ONSET_RATE_HZ);
            cursor = cross_tick;
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ParticipationPrediction {
    pub(crate) issued_frame: u64,
    pub(crate) onset_frame: u64,
    pub(crate) forecast_observed_frame: u64,
    pub(crate) target_start_frame: u64,
    pub(crate) target_end_frame: u64,
    pub(crate) pred_continuation_habitat_band_energy: [f32; 3],
}

/// Public as the payload of PhonationClock; observations remain crate-internal.
pub struct ParticipationClock {
    pub(crate) shape: Option<crate::life::shape_policy::ShapePolicy>,
    pub(super) adoption_candidate: Option<Tick>,
    pub(super) initial_hold_on: Option<Tick>,
    pub(super) base_rate_hz: f32,
    pub(super) coupling: f32,
    pub(super) flow_depth: f32,
    pub(crate) hop_frames: u64,
    pub(super) steady: bool,
    pub(super) seed: u64,
    pub(crate) policy: Option<TemporalParticipation>,
    pub(super) forecast: Option<TemporalForecast>,
    pub(super) outcome_forecast: Option<TemporalForecast>,
    pub(super) pending_prediction: Option<ParticipationPrediction>,
    pub(super) issued_predictions: Vec<ParticipationPrediction>,
    pub(super) own_profile: [f32; 3],
    pub(super) hold_theta: f32,
    pub(super) adsr: Option<crate::life::sound::ToneAdsr>,
    pub(super) onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    pub(super) footprint: crate::life::temporal_participation::FootprintTracker,
    pub(super) decision_trace: bool,
    pub(super) gate_index: u64,
}

impl ParticipationClock {
    /// The representative hold and envelope of I11-1 §4.2 for either configured
    /// footprint source, so both sources can use the same record span.
    pub(super) fn footprint_hold(&self) -> Option<(f32, Option<crate::life::sound::ToneAdsr>)> {
        self.onset_comparison
            .map(|_| (self.hold_theta / self.base_rate_hz, self.adsr))
    }

    pub(super) fn candidate(
        &mut self,
        ctx: &CoreTickCtx,
        cursor: Tick,
        allowed: bool,
        arrival: Option<&ArrivalContext<'_>>,
    ) -> Option<CandidatePoint> {
        if !ctx.fs.is_finite() || ctx.fs < 1.0 {
            return None;
        }
        assert!(
            self.hop_frames > 0,
            "participation requires the configured hop length"
        );
        let policy = self.policy.get_or_insert_with(|| {
            let mut policy = TemporalParticipation::new(
                ctx.fs as u32,
                self.base_rate_hz,
                self.coupling,
                ctx.now_tick,
                self.seed,
                self.hop_frames,
            );
            policy.set_flow_depth(self.flow_depth);
            policy.set_steady(self.steady);
            policy
        });
        if let Some(at) = self.initial_hold_on.take() {
            policy.register_initial_hold(at);
        }
        if policy.needs_forecast(cursor, ctx.frame_end) {
            policy.update_reference(cursor, self.forecast.as_ref());
        }
        policy.set_overlap(0.8, self.own_profile);
        policy.set_sound_duration(self.hold_theta / self.base_rate_hz, self.adsr);
        policy.set_onset_comparison(self.onset_comparison);
        policy.set_body_footprint(self.footprint.selected());
        policy.set_current_identity(self.footprint.identity());
        policy.set_decision_trace(self.decision_trace);
        let extra = self.adoption_candidate.take();
        let tick = policy.candidate_with_shape(
            cursor,
            ctx.frame_end,
            allowed,
            self.forecast.as_ref(),
            arrival,
            extra,
            self.shape.as_mut().filter(|shape| shape.enabled),
        )?;
        self.pending_prediction = self.outcome_forecast.as_ref().and_then(|forecast| {
            let (start, end, energy) = forecast.energy_window_after(tick)?;
            Some(ParticipationPrediction {
                issued_frame: ctx.now_tick,
                onset_frame: tick,
                forecast_observed_frame: forecast.observed_frame(),
                target_start_frame: start,
                target_end_frame: end,
                pred_continuation_habitat_band_energy: energy,
            })
        });
        Some(CandidatePoint {
            tick,
            gate: self.gate_index,
        })
    }

    pub(super) fn resolve(&mut self, tick: Tick, sounded: bool) {
        self.policy
            .as_mut()
            .expect("selected participation candidate")
            .resolve(tick, sounded);
        // A scheduled onset is an intention; its acoustic outcome is observed later.
        if let Some(prediction) = self.pending_prediction.take()
            && sounded
        {
            self.issued_predictions.push(prediction);
        }
        if let Some(shape) = self.shape.as_mut() {
            shape.veto_plan = false;
        }
        self.gate_index = self.gate_index.saturating_add(1);
    }
}

// Keep coupling clocks inline to avoid an extra allocation per Voice.
#[allow(clippy::large_enum_variant)]
pub enum PhonationClock {
    ThetaGate(ThetaGateClock),
    Coupling(CouplingClock),
    Participation(Box<ParticipationClock>),
    #[cfg(test)]
    Custom(Box<dyn FnMut(&CoreTickCtx, &mut Vec<CandidatePoint>) + Send>),
}

impl PhonationClock {
    pub(crate) fn shape_slot(
        &mut self,
    ) -> Option<&mut Option<crate::life::shape_policy::ShapePolicy>> {
        match self {
            Self::Participation(c) => Some(&mut c.shape),
            Self::Coupling(c) => Some(&mut c.shape),
            _ => None,
        }
    }

    pub(crate) fn shape_mut(&mut self) -> Option<&mut crate::life::shape_policy::ShapePolicy> {
        self.shape_slot()?.as_mut()
    }

    pub(crate) fn intrinsic_period_sec(&self) -> Option<f64> {
        let rate = match self {
            Self::Coupling(c) => c.intrinsic_rate_hz,
            Self::Participation(c) => c.base_rate_hz,
            _ => return None,
        };
        (rate.is_finite() && rate > 0.).then(|| 1. / f64::from(rate))
    }
    #[cfg(test)]
    pub(crate) fn opportunity(
        &self,
        ctx: &CoreTickCtx,
    ) -> Option<crate::life::action_candidates::Opportunity> {
        use crate::life::action_candidates::{Opportunity, OpportunityBasis};
        if !ctx.fs.is_finite() || ctx.fs <= 0.0 || ctx.frame_end <= ctx.now_tick {
            return None;
        }
        let (at, basis) = match self {
            Self::Participation(clock) => return clock.policy.as_ref()?.opportunity(ctx.now_tick),
            Self::Coupling(clock) => {
                let (_, phase_dot, my_frac) = clock.velocity_from_rhythms(&ctx.rhythms);
                let crossing =
                    ctx.now_tick as f64 + (1.0 - my_frac as f64) / phase_dot as f64 * ctx.fs as f64;
                if !crossing.is_finite() || crossing.round() >= Tick::MAX as f64 {
                    return None;
                }
                let at = if crossing < ctx.frame_end as f64 {
                    (crossing.round() as Tick).clamp(ctx.now_tick, ctx.frame_end - 1)
                } else {
                    crossing.round() as Tick
                };
                (at, OpportunityBasis::CouplingProjection)
            }
            Self::ThetaGate(clock) => {
                let cursor = match clock.last_gate_tick {
                    Some(last) if last >= ctx.now_tick => last.checked_add(1)?,
                    _ => ctx.now_tick,
                };
                let mut theta = ctx.rhythms.theta;
                let elapsed = cursor.saturating_sub(ctx.now_tick) as f64 / ctx.fs as f64;
                theta.phase = (theta.phase as f64 + TAU as f64 * theta.freq_hz as f64 * elapsed)
                    .rem_euclid(TAU as f64) as f32;
                let next = next_gate_tick(cursor, ctx.fs, theta, 0.0)?;
                let at = if cursor == ctx.now_tick && theta.phase == 0.0 {
                    cursor
                } else {
                    next
                };
                (at, OpportunityBasis::ThetaProjection)
            }
            #[cfg(test)]
            Self::Custom(_) => return None,
        };
        Some(Opportunity {
            issued_at: ctx.now_tick,
            at,
            basis,
        })
    }

    pub fn gather_candidates(&mut self, ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>) {
        match self {
            PhonationClock::ThetaGate(clock) => {
                clock.gather_candidates_impl(ctx, out);
            }
            PhonationClock::Coupling(clock) => {
                clock.gather_candidates_impl(ctx, out);
            }
            PhonationClock::Participation(clock) => {
                clock.hop_frames = ctx.frame_end.saturating_sub(ctx.now_tick);
                let mut cursor = ctx.now_tick;
                while let Some(candidate) = clock.candidate(ctx, cursor, false, None) {
                    clock.resolve(candidate.tick, false);
                    cursor = candidate.tick.saturating_add(1);
                    out.push(candidate);
                }
            }
            #[cfg(test)]
            PhonationClock::Custom(f) => f(ctx, out),
        }
    }

    /// Whether onsets should bypass the adaptive `env_open`/social gating and use
    /// a flat timing field. The coupling clock owns its own timing (an internal
    /// entrained oscillator); gating it by theta would double-count the
    /// entrainment and destroy the renewal / lock continuum it produces.
    pub(super) fn uses_flat_field(&self) -> bool {
        matches!(
            self,
            PhonationClock::Coupling(_) | PhonationClock::Participation(_)
        )
    }

    /// Nominal tick spacing used to extend the duration grid without falling back
    /// to the adaptive theta. The last effective period for `Coupling`. `None`
    /// for theta-driven clocks, whose spacing is not constant.
    pub(super) fn fixed_period_ticks(&self, fs: f32) -> Option<f64> {
        match self {
            PhonationClock::Coupling(clock) => clock.nominal_period_ticks(fs),
            PhonationClock::Participation(clock) => Some(fs as f64 / clock.base_rate_hz as f64),
            _ => None,
        }
    }

    pub(crate) fn from_config(config: &PhonationClockConfig, seed: u64) -> Self {
        match config {
            PhonationClockConfig::ThetaGate => PhonationClock::ThetaGate(ThetaGateClock::default()),
            PhonationClockConfig::Coupling {
                coupling,
                base_rate_hz,
                flow_depth,
                microtiming,
            } => PhonationClock::Coupling(CouplingClock::new(
                *coupling,
                *base_rate_hz,
                *flow_depth,
                *microtiming,
                seed,
            )),
            PhonationClockConfig::Participation {
                coupling,
                base_rate_hz,
                flow_depth,
                steady,
            } => Self::Participation(Box::new(ParticipationClock {
                shape: None,
                hop_frames: 0,
                adoption_candidate: None,
                initial_hold_on: None,
                base_rate_hz: if base_rate_hz.is_finite() {
                    base_rate_hz.clamp(0.01, MAX_ONSET_RATE_HZ)
                } else {
                    1.0
                },
                coupling: sanitize01(*coupling),
                flow_depth: sanitize01(*flow_depth),
                steady: *steady && *flow_depth == 0.0,
                seed,
                policy: None,
                forecast: None,
                own_profile: [0.0; 3],
                outcome_forecast: None,
                pending_prediction: None,
                issued_predictions: Vec::new(),
                hold_theta: 1.0,
                adsr: None,
                onset_comparison: None,
                footprint: Default::default(),
                decision_trace: false,
                gate_index: 0,
            })),
        }
    }
}
