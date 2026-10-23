//! Coupled neural-oscillator meter core (Regime A + C of the rhythm design).
//!
//! This is the perception-side meter model: a self-sustaining limit-cycle beat
//! oscillator entrained by acoustic onset drive, plus integer-ratio subdivision
//! detection. Salience is reported as *entrainment confidence* (phase locking
//! and corroborated pattern recurrence), kept strictly distinct from the drive amplitude.
//!
//! Design rationale: `docs/design-notes/neural-rhythm-meter.md`.
//!
//! The tactus adapts to corroborated acoustic intervals. Integer-ratio layers
//! retain separate subdivision and accent evidence; they are not independent
//! noninteger tempi or a learned coupling network.

use std::f32::consts::TAU;

use crate::core::float::smoothstep;

use crate::core::onset::{Onset, OnsetDetector, REFRACTORY_SEC};
use crate::core::phase::wrap_pm_pi;

#[path = "meter/continuous_recurrence.rs"]
mod continuous;
pub(crate) use continuous::{Diagnostics as ContinuousDiagnostics, Packet as TemporalPacket};

#[derive(Debug)]
enum RecurrenceInput {
    Events(Recurrence),
    Continuous(Box<continuous::State>),
}
impl RecurrenceInput {
    fn events(&self) -> Option<&Recurrence> {
        match self {
            Self::Events(s) => Some(s),
            Self::Continuous(_) => None,
        }
    }
}

#[cfg(test)]
#[path = "meter_assay.rs"]
mod audio_assay;

#[cfg(test)]
#[path = "meter_hierarchy_tests.rs"]
mod hierarchy_tests;

// Existing tactus band; faster input is read through an integer subdivision.
const F_BEAT_MIN: f32 = 0.5;
const F_BEAT_MAX: f32 = 4.0;
const F_BEAT_INIT: f32 = 2.0;
// Existing integration floor, also bounds accumulated interval roundoff.
const MIN_STEP_SEC: f32 = 1e-4;

// `temporal_basin` restoring rate (per second): a weak pull of the beat
// frequency toward the basin center. Kept gentle so onset entrainment within
// the basin dominates -- the basin shapes the terrain, it does not schedule a
// beat.
const BASIN_PULL: f32 = 0.6;

// Forced Hopf normal form (Large neural resonance). ALPHA > 0 with BETA < 0
// gives a stable limit cycle of radius sqrt(-ALPHA/BETA) = 1.0, so the beat is
// self-sustaining and coasts through gaps (Regime C: persistence).
const ALPHA: f32 = 1.0;
const BETA: f32 = -1.0;
const FORCE_AMP: f32 = 1.0; // drive -> amplitude pumping (in phase)
// Large & Jones (1999) simulation gains, verified in the temporal design review.
const PHASE_GAIN: f32 = 0.6;
const PERIOD_GAIN: f32 = 0.1;

// Confidence accumulators decay over this timescale, so confidence persists
// through short gaps (a beat or two) but fades in sustained silence.
const PERSIST_TAU: f32 = 2.5;

// Candidate integer subdivisions of the beat.
const SUB_RATIOS: [u8; 3] = [2, 3, 4];

// Candidate measure groupings (beats per measure). The measure is an internally
// generated slow subharmonic of the beat, revealed by periodic accent (Regime B:
// the brain imposes meter endogenously by emphasis, not by a faster pulse).
const MEASURE_RATIOS: [u8; 3] = [2, 3, 4];
// Accent recurrence spans several beats, so the measure integrates over more
// history than the beat-level PLV.
const MEASURE_TAU: f32 = 6.0;
// Leaky tracking rate for the running onset-strength baseline (per onset). An
// accent is an onset louder than this baseline.
const STRENGTH_BL_RATE: f32 = 0.1;

#[inline]
fn ioi_window<T: rustfft::num_traits::Float>(period: T) -> T {
    (T::from(0.2).unwrap() * period).min(T::from(0.06).unwrap())
}

/// One metrical level's reported state.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MeterBand {
    /// Wrapped oscillator phase in [-pi, pi].
    pub phase: f32,
    /// Current frequency in Hz.
    pub freq_hz: f32,
    /// Limit-cycle amplitude (presence). Not salience.
    pub amplitude: f32,
    /// Entrainment confidence in [0, 1] from phase and pattern recurrence. This is the
    /// salience: a clean quiet beat reads high, a loud wandering beat reads low.
    pub confidence: f32,
}

/// Listener-side meter state.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MeterState {
    /// Tactus (delta-rate beat).
    pub beat: MeterBand,
    /// Tatum (theta-rate), mode-locked to the beat at the emergent ratio.
    pub subdivision: MeterBand,
    /// Measure: a slow subharmonic of the beat, induced by periodic accent.
    pub measure: MeterBand,
    /// Onset / flux drive salience (the force term), kept distinct from
    /// confidence so reports and UI never blur drive with entrainment.
    pub attention_level: f32,
    /// Emergent integer subdivision ratio (2, 3, or 4); 0 when none is detected.
    pub subdivision_ratio: u8,
    /// Emergent beats-per-measure (2, 3, or 4); 0 when no accent grouping holds.
    pub measure_ratio: u8,
}

/// Composer-set terrain shaping for the production meter. These are soft priors
/// on how a pulse forms, never a schedule: emergence (onset entrainment, accent
/// grouping) still does the work, the shaping only bends the terrain it forms on.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeterShaping {
    /// Attractor depth in [0, 1]. How strongly a pulse wants to form: scales the
    /// entrainment forcing and weights the preferred tempo. 0 = neutral baseline.
    pub stability: f32,
    /// Frequency-prior region in Hz `(min, max)`. Seeds the beat at its center
    /// and confines adaptation to the band. Stability weights the tempo
    /// preference and restoring pull. `None` = the default beat band.
    pub basin_hz: Option<(f32, f32)>,
}

pub(crate) const RECURRENCE_HISTORY_SECONDS: f64 = 2.0 / F_BEAT_MIN as f64 + 0.06;

#[derive(Debug)]
struct Echo {
    observation_deadline: f64,
    target: f64,
    window: f64,
    weight: Option<f64>,
    observation_done: bool,
    prediction_done: bool,
}

#[derive(Debug)]
struct Recurrence {
    history: Vec<f64>,
    echo: Vec<Echo>,
    time: f64,
    steps: u64,
    period: Option<f32>,
    multi: bool,
    uniform: bool,
    ratio_support: f32,
    observed: f64,
    predicted: f64,
    matched: f64,
}

impl Recurrence {
    fn new() -> Self {
        // One onset per refractory interval; two endpoint slots cover interpolation.
        let capacity = (RECURRENCE_HISTORY_SECONDS / REFRACTORY_SEC as f64).ceil() as usize + 2;
        Self {
            history: Vec::with_capacity(capacity),
            echo: Vec::with_capacity(capacity),
            time: 0.0,
            steps: 0,
            period: None,
            multi: false,
            uniform: false,
            ratio_support: 0.0,
            observed: 0.0,
            predicted: 0.0,
            matched: 0.0,
        }
    }

    fn observe(&mut self, dt: f32, fired: bool, frac: f32) -> Option<f32> {
        let onset_time = self.time + dt as f64 * frac as f64;
        self.time += dt as f64;
        self.steps += 1;
        let decay = (-dt as f64 / PERSIST_TAU as f64).exp();
        self.observed *= decay;
        self.predicted *= decay;
        self.matched *= decay;
        if fired {
            let mut best = None;
            let mut error = f64::INFINITY;
            for (index, echo) in self.echo.iter().enumerate() {
                let distance = (onset_time - echo.target).abs();
                if !echo.prediction_done
                    && echo.weight.is_none()
                    && distance <= echo.window
                    && distance < error
                {
                    best = Some(index);
                    error = distance;
                }
            }
            if let Some(index) = best {
                let echo = &mut self.echo[index];
                echo.weight = Some((1.0 - error / echo.window).max(0.0));
            }
        }
        for echo in &mut self.echo {
            if !echo.observation_done && self.time > echo.observation_deadline {
                self.observed += 1.0;
                echo.observation_done = true;
            }
            if !echo.prediction_done && self.time > echo.target + echo.window {
                self.predicted += 1.0;
                self.matched += echo.weight.unwrap_or(0.0);
                echo.prediction_done = true;
            }
        }
        self.echo
            .retain(|e| !(e.observation_done && e.prediction_done));
        let start = self
            .history
            .partition_point(|t| *t < onset_time - RECURRENCE_HISTORY_SECONDS);
        self.history.drain(..start);
        if !fired {
            return None;
        }
        assert!(self.history.len() < self.history.capacity());
        self.history.push(onset_time);
        let n = self.history.len();
        self.uniform = if n >= 3 {
            let p = self.history[n - 2] - self.history[n - 3];
            self.history
                .windows(2)
                .all(|x| ((x[1] - x[0]) - p).abs() <= ioi_window(p))
        } else {
            false
        };
        // Confirmation admits a new hypothesis; it does not delete an accepted one.
        let mut best_period = self.period.map(f64::from).unwrap_or(0.0);
        let mut best_coverage = self
            .period
            .map(|p| {
                let p = p as f64;
                self.coverage(p, ioi_window(p))
            })
            .unwrap_or(-1.0);
        for j in 1..n.saturating_sub(1) {
            for k in 0..j {
                let period = self.history[j] - self.history[k];
                if !(0.0625..=2.0).contains(&period) {
                    continue;
                }
                let window = ioi_window(period);
                if ((onset_time - self.history[j]) - period).abs() > window {
                    continue;
                }
                let coverage = self.coverage(period, window);
                // Bound timestamp summation and subsequent coverage arithmetic roundoff.
                let q = self.steps as f64 * f64::EPSILON;
                let gamma = q / (1.0 - q);
                let roundoff =
                    4.0 * gamma * self.time.abs() / window + (8 * n + 32) as f64 * f64::EPSILON;
                if coverage > best_coverage + roundoff
                    || ((coverage - best_coverage).abs() <= roundoff
                        && (best_period == 0.0 || period < best_period))
                {
                    best_period = period;
                    best_coverage = coverage;
                }
            }
        }
        (best_period > 0.0).then_some(best_period as f32)
    }

    fn coverage(&self, period: f64, window: f64) -> f64 {
        let start = self.history[0] + period;
        let end = *self.history.last().unwrap();
        let mut j = self.history.partition_point(|t| *t < start);
        let mut mass = 0.0;
        for t in &self.history {
            let target = *t + period;
            if target > end {
                break;
            }
            while j < self.history.len() && self.history[j] < target - window {
                j += 1;
            }
            if j < self.history.len() && (self.history[j] - target).abs() <= window {
                mass += (1.0 - (self.history[j] - target).abs() / window).max(0.0);
                j += 1;
            }
        }
        // All candidate lags share one observation denominator (autocorrelation).
        mass / self.history.len() as f64
    }

    fn accept(&mut self, period: f32, ratio_support: f32) {
        self.period = Some(period);
        self.ratio_support = ratio_support;
        self.multi = self
            .history
            .windows(2)
            .any(|x| x[1] - x[0] < period as f64 - ioi_window(period as f64));
    }

    fn reset_evidence(&mut self) {
        self.echo.clear();
        self.observed = 0.0;
        self.predicted = 0.0;
        self.matched = 0.0;
    }

    fn issue(&mut self, frac: f32, dt: f32) {
        if let Some(period) = self.period {
            let t = self.time - dt as f64 * (1.0 - frac as f64);
            assert!(self.echo.len() < self.echo.capacity());
            self.echo.push(Echo {
                observation_deadline: t + 0.06,
                target: t + period as f64,
                window: ioi_window(period as f64),
                weight: None,
                observation_done: false,
                prediction_done: false,
            });
        }
    }

    fn support(&self) -> f32 {
        let union = self.predicted + self.observed - self.matched;
        if union > 0.0 {
            (self.matched / union).clamp(0.0, 1.0) as f32 * self.ratio_support
        } else {
            0.0
        }
    }
}

/// Coupled limit-cycle meter network (single beat oscillator + subdivision
/// detection). One instance per listener.
#[derive(Debug)]
pub struct MeterNetwork {
    recurrence: RecurrenceInput,
    // Beat oscillator state.
    beat_phi: f32,
    beat_omega: f32,
    beat_r: f32,

    onset_detector: OnsetDetector,
    onset_age: Option<f32>,
    pending_period: Option<f32>,
    seeded: bool,
    previous_phase_error: Option<f32>,

    // Leaky resultant accumulators (numerator complex sum + count denominator),
    // all decayed at PERSIST_TAU so PLV is rate-independent and silence-fading.
    plv_count: f32,
    beat_re: f32,
    beat_im: f32,
    sub_re: [f32; 3],
    sub_im: [f32; 3],

    // Measure (accent subharmonic) state. beat_cycles is the unwrapped beat
    // count; accent-weighted resultants at candidate subharmonics select the
    // grouping, normalized by the accumulated accent mass.
    beat_cycles: f32,
    strength_baseline: f32,
    meas_re: [f32; 3],
    meas_im: [f32; 3],
    meas_norm: f32,

    attention_ema: f32,
    shaping: MeterShaping,
    last: MeterState,
}

impl Default for MeterNetwork {
    fn default() -> Self {
        Self::with_input(RecurrenceInput::Events(Recurrence::new()))
    }
}
impl MeterNetwork {
    fn with_input(recurrence: RecurrenceInput) -> Self {
        Self {
            recurrence,
            beat_phi: 0.0,
            beat_omega: TAU * F_BEAT_INIT,
            beat_r: 0.1,
            onset_detector: OnsetDetector::default(),
            onset_age: None,
            pending_period: None,
            seeded: false,
            previous_phase_error: None,
            plv_count: 0.0,
            beat_re: 0.0,
            beat_im: 0.0,
            sub_re: [0.0; 3],
            sub_im: [0.0; 3],
            beat_cycles: 0.0,
            strength_baseline: 0.0,
            meas_re: [0.0; 3],
            meas_im: [0.0; 3],
            meas_norm: 0.0,
            attention_ema: 0.0,
            shaping: MeterShaping::default(),
            last: MeterState::default(),
        }
    }
}

impl MeterNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install composer-set terrain shaping. A basin seeds the beat frequency at
    /// its center (cold-start prior) and confines later frequency learning to the
    /// band; stability takes effect on the next `process` call.
    pub fn set_shaping(&mut self, shaping: MeterShaping) {
        if let Some((min_hz, max_hz)) = shaping.basin_hz {
            let (lo, hi) = sane_basin(min_hz, max_hz);
            let center = 0.5 * (lo + hi);
            self.beat_omega = TAU * center;
        }
        self.shaping = shaping;
    }

    /// Advance the network by `dt` seconds under acoustic onset `drive` in
    /// [0, 1] (rectified spectral flux). Returns the updated meter state.
    pub fn process(&mut self, dt: f32, drive: f32) -> MeterState {
        self.process_evidence(dt, Some(drive))
    }
    fn continuous(&self) -> Option<&continuous::State> {
        match &self.recurrence {
            RecurrenceInput::Continuous(s) => Some(s),
            _ => None,
        }
    }
    fn continuous_mut(&mut self) -> Option<&mut continuous::State> {
        match &mut self.recurrence {
            RecurrenceInput::Continuous(s) => Some(s),
            _ => None,
        }
    }
    pub(crate) fn process_evidence(&mut self, dt: f32, drive: Option<f32>) -> MeterState {
        let preference_weight = self.shaping.stability.clamp(0.0, 1.0);
        let dt = dt.max(MIN_STEP_SEC);
        let drive_known = drive.is_some();
        let drive = drive.unwrap_or(0.0).clamp(0.0, 1.0);
        let onset = self.observe_onset(dt, drive_known, drive);
        let confirmed_period = self.confirm_period(dt, onset);
        let force_gain = 1.0 + self.shaping.stability.clamp(0.0, 1.0);
        let phi_before = self.beat_phi;
        let omega_before = self.beat_omega;
        self.integrate_hopf(dt, drive, force_gain);
        let (lo, hi, center) = self.apply_tempo_prior(dt, preference_weight);
        let (omega_lo, omega_hi) = (TAU * lo, TAU * hi);
        self.beat_omega = self.beat_omega.clamp(omega_lo, omega_hi);
        let observed_phi = wrap_pm_pi(phi_before + omega_before * dt * onset.frac);
        let mut seeded_now = false;
        if onset.fired {
            let proposal = self.propose_period(
                dt,
                onset.frac,
                confirmed_period,
                (lo, hi, center),
                preference_weight,
            );
            if !self.seeded
                && let Some(frequency) = proposal
            {
                self.seed_beat(frequency, dt, onset.frac);
                seeded_now = true;
            }
            if !seeded_now {
                self.correct_phase(
                    dt,
                    onset.frac,
                    force_gain,
                    observed_phi,
                    omega_before,
                    proposal,
                );
            }
            self.beat_omega = self.beat_omega.clamp(omega_lo, omega_hi);
            self.onset_age = Some(dt * (1.0 - onset.frac));
        } else if let Some(age) = self.onset_age.as_mut() {
            *age += dt;
        }
        self.accumulate_evidence(dt, onset.fired, seeded_now, observed_phi, drive);
        if let RecurrenceInput::Events(state) = &mut self.recurrence
            && onset.fired
            && self.seeded
            && !seeded_now
        {
            state.issue(onset.frac, dt);
        }
        let att_a = (-dt / 0.3).exp();
        self.attention_ema = att_a * self.attention_ema + (1.0 - att_a) * drive;
        self.last = self.build_state();
        self.last
    }

    fn observe_onset(&mut self, dt: f32, drive_known: bool, drive: f32) -> Onset {
        if drive_known {
            self.onset_detector.process(dt, drive)
        } else {
            self.onset_detector.advance_missing(dt);
            crate::core::onset::Onset::default()
        }
    }

    fn confirm_period(&mut self, dt: f32, onset: Onset) -> Option<f32> {
        let confirmed_period = match &mut self.recurrence {
            RecurrenceInput::Continuous(state) => {
                let onset_tick = state.fast_start as f64
                    + dt as f64 * onset.frac as f64 * state.sample_rate() as f64;
                state
                    .pending
                    .filter(|p| {
                        !state.suppressed
                            && onset.fired
                            && onset_tick > p.received as f64
                            && onset_tick > p.use_cut as f64
                    })
                    .map(|p| p.lag_sec as f32)
            }
            RecurrenceInput::Events(state) => state.observe(dt, onset.fired, onset.frac),
        };
        if self.recurrence.events().is_some_and(|s| s.uniform) {
            self.onset_age
                .zip(self.pending_period)
                .and_then(|(age, period)| {
                    ((age + dt * onset.frac - period).abs() <= ioi_window(period)).then_some(period)
                })
        } else {
            confirmed_period
        }
    }

    fn integrate_hopf(&mut self, dt: f32, drive: f32, force_gain: f32) {
        let phi_before = self.beat_phi;
        let omega_before = self.beat_omega;
        let dr = ALPHA * self.beat_r
            + BETA * self.beat_r.powi(3)
            + FORCE_AMP * force_gain * drive * phi_before.cos();
        self.beat_r = (self.beat_r + dr * dt).clamp(0.0, 2.0);
        self.beat_phi = wrap_pm_pi(phi_before + omega_before * dt);
        self.beat_cycles += omega_before * dt / TAU;
    }

    fn apply_tempo_prior(&mut self, dt: f32, preference_weight: f32) -> (f32, f32, f32) {
        match self.shaping.basin_hz {
            Some((min_hz, max_hz)) => {
                let (lo, hi) = sane_basin(min_hz, max_hz);
                let center = 0.5 * (lo + hi);
                // Existing restoring ceiling; auditory evidence suppresses the prior.
                self.beat_omega += preference_weight
                    * (1.0 - self.last.beat.confidence)
                    * BASIN_PULL
                    * (TAU * center - self.beat_omega)
                    * dt;
                (lo, hi, center)
            }
            None => (F_BEAT_MIN, F_BEAT_MAX, F_BEAT_INIT),
        }
    }

    fn propose_period(
        &mut self,
        dt: f32,
        onset_frac: f32,
        confirmed_period: Option<f32>,
        band: (f32, f32, f32),
        preference_weight: f32,
    ) -> Option<f32> {
        let (lo, hi, center) = band;
        let mut proposal = None;
        if let Some(age) = self.onset_age {
            let ioi = age + dt * onset_frac;
            if let Some(period) = confirmed_period {
                // Reuse temporal_participation's min(0.2*IOI, 60 ms) candidate window.
                {
                    let observed_frequency = 1.0 / period;
                    let mut best_score = -1.0;
                    // Positive f32 interval sums: gamma_n bounds roundoff at the band edge.
                    // At most 1/lo seconds at the existing minimum step, plus endpoint arithmetic.
                    let n_epsilon = ((1.0 / lo / MIN_STEP_SEC).ceil() + 3.0) * f32::EPSILON;
                    let relative_roundoff = n_epsilon / (1.0 - n_epsilon);
                    // Existing integer layers, evaluated at continuous observed frequencies.
                    for ratio in [1.0_f32, 0.5, 1.0 / 3.0, 0.25, 2.0, 3.0, 4.0] {
                        let raw_frequency = ratio * observed_frequency;
                        let frequency = raw_frequency.clamp(lo, hi);
                        let raw_period = 1.0 / raw_frequency;
                        if (raw_period - 1.0 / frequency).abs()
                            > (relative_roundoff * raw_period).min(ioi_window(raw_period))
                        {
                            continue;
                        }
                        // Free log-distance profile, not a fitted literature resonance curve.
                        let distance = if frequency <= center {
                            (frequency / center).ln() / (lo / center).ln()
                        } else {
                            (frequency / center).ln() / (hi / center).ln()
                        };
                        let preference = (1.0 - distance).clamp(0.0, 1.0);
                        // Ideal nested event trains: intersection / union coverage.
                        let support = ratio.min(1.0 / ratio);
                        let score =
                            support * ((1.0 - preference_weight) + preference_weight * preference);
                        if score > best_score {
                            best_score = score;
                            proposal = Some(frequency);
                        }
                    }
                    // Preserve an established tactus across whole missed beats.
                    if self.seeded {
                        let current_period = TAU / self.beat_omega;
                        for missing_ratio in [2.0_f32, 3.0, 4.0] {
                            let expected = missing_ratio * current_period;
                            let raw_frequency = missing_ratio / period;
                            let frequency = raw_frequency.clamp(lo, hi);
                            let raw_period = 1.0 / raw_frequency;
                            if (period - expected).abs() <= ioi_window(expected)
                                && (raw_period - 1.0 / frequency).abs()
                                    <= (relative_roundoff * raw_period).min(ioi_window(raw_period))
                            {
                                proposal = Some(frequency);
                                break;
                            }
                        }
                    }
                }
            }
            self.pending_period = if (1.0 / (F_BEAT_MAX * 4.0)..=1.0 / F_BEAT_MIN).contains(&ioi) {
                Some(ioi)
            } else {
                None
            };
        }
        if let (Some(period), Some(frequency)) = (confirmed_period, proposal) {
            let ratio = frequency * period;
            match &mut self.recurrence {
                RecurrenceInput::Continuous(state) => state.accept(ratio.min(1.0 / ratio)),
                RecurrenceInput::Events(state) => state.accept(period, ratio.min(1.0 / ratio)),
            }
        }
        proposal
    }

    fn seed_beat(&mut self, frequency: f32, dt: f32, onset_frac: f32) {
        self.beat_omega = TAU * frequency;
        self.beat_phi = wrap_pm_pi(self.beat_omega * dt * (1.0 - onset_frac));
        self.beat_cycles = self.beat_omega * dt * (1.0 - onset_frac) / TAU;
        // A new coordinate discards old evidence; this onset cannot certify itself.
        match &mut self.recurrence {
            RecurrenceInput::Continuous(state) => state.seed(),
            RecurrenceInput::Events(state) => state.reset_evidence(),
        }
        self.plv_count = 0.0;
        self.beat_re = 0.0;
        self.beat_im = 0.0;
        self.sub_re = [0.0; 3];
        self.sub_im = [0.0; 3];
        self.meas_re = [0.0; 3];
        self.meas_im = [0.0; 3];
        self.meas_norm = 0.0;
        self.previous_phase_error = Some(0.0);
        self.seeded = true;
    }

    fn correct_phase(
        &mut self,
        dt: f32,
        onset_frac: f32,
        force_gain: f32,
        observed_phi: f32,
        omega_before: f32,
        proposal: Option<f32>,
    ) {
        if let Some(previous_error) = self.previous_phase_error {
            let period = TAU / self.beat_omega;
            self.beat_omega =
                TAU / (period * (1.0 + PERIOD_GAIN * force_gain * previous_error.sin() / TAU));
        }
        let correction = -PHASE_GAIN * force_gain * observed_phi.sin();
        let tail = (self.beat_omega - omega_before) * dt * (1.0 - onset_frac);
        self.beat_phi = wrap_pm_pi(self.beat_phi + correction + tail);
        self.beat_cycles += (correction + tail) / TAU;
        self.previous_phase_error = Some(observed_phi);
        if let Some(frequency) = proposal {
            let omega = TAU * frequency;
            let tail = (omega - self.beat_omega) * dt * (1.0 - onset_frac);
            self.beat_phi = wrap_pm_pi(self.beat_phi + tail);
            self.beat_cycles += tail / TAU;
            self.beat_omega = omega;
        }
    }

    fn accumulate_evidence(
        &mut self,
        dt: f32,
        onset_fired: bool,
        seeded_now: bool,
        observed_phi: f32,
        drive: f32,
    ) {
        // --- Confidence accumulators: decay every tick (persistence), add an
        // impulse at each onset. ---
        let decay = (-dt / PERSIST_TAU).exp();
        self.plv_count *= decay;
        self.beat_re *= decay;
        self.beat_im *= decay;
        for k in 0..SUB_RATIOS.len() {
            self.sub_re[k] *= decay;
            self.sub_im[k] *= decay;
        }
        let m_decay = (-dt / MEASURE_TAU).exp();
        self.meas_norm *= m_decay;
        for k in 0..MEASURE_RATIOS.len() {
            self.meas_re[k] *= m_decay;
            self.meas_im[k] *= m_decay;
        }

        if onset_fired && !seeded_now {
            let onset_phi = observed_phi;
            // An uncorroborated coordinate is unknown, not weak beat evidence.
            if self.seeded {
                self.plv_count += 1.0;
                self.beat_re += onset_phi.cos();
                self.beat_im += onset_phi.sin();
                // ratio competition: which integer multiple of the beat phase do
                // onsets lock to.
                for (k, ratio) in SUB_RATIOS.iter().enumerate() {
                    let a = (*ratio as f32) * onset_phi;
                    self.sub_re[k] += a.cos();
                    self.sub_im[k] += a.sin();
                }
            }

            // Accent: how much louder this onset is than the running baseline.
            // Only positive emphasis counts; a uniform stream yields ~0 accent
            // mass, so no measure is induced.
            let strength = drive;
            self.strength_baseline += STRENGTH_BL_RATE * (strength - self.strength_baseline);
            let accent = (strength - self.strength_baseline).max(0.0);
            if self.seeded && accent > 0.0 {
                self.meas_norm += accent;
                let cycles = self.beat_cycles;
                for (k, m) in MEASURE_RATIOS.iter().enumerate() {
                    // Strong beats recurring every m beats align at this slow
                    // subharmonic phase; accent-weighting ignores even beats.
                    let a = TAU * cycles / (*m as f32);
                    self.meas_re[k] += accent * a.cos();
                    self.meas_im[k] += accent * a.sin();
                }
            }
        }
    }

    fn build_state(&self) -> MeterState {
        let count = self.plv_count.max(1e-6);
        // Presence gate: a resultant from too few onsets is statistically
        // unreliable (a couple of coincidentally aligned onsets read as a high
        // PLV), so confidence requires ~4 accumulated onsets of evidence before
        // it is trusted, matching beat-induction needing a few cycles. Decays in
        // silence as the leaky count fades. A deeper attractor (higher stability)
        // is a top-down prior that commits with less evidence, lowering this
        // threshold. It cannot fabricate a beat: scattered phases keep the PLV
        // resultant low regardless of how readily presence saturates.
        let stab = self.shaping.stability.clamp(0.0, 1.0);
        let presence = smoothstep(1.0 - 0.5 * stab, 4.0 - 2.0 * stab, self.plv_count);

        let beat_plv = (self.beat_re * self.beat_re + self.beat_im * self.beat_im).sqrt() / count;
        let beat_conf = match &self.recurrence {
            RecurrenceInput::Continuous(state) => state.confidence(beat_plv, presence),
            RecurrenceInput::Events(state) => {
                let support = if state.multi {
                    beat_plv.max(state.support())
                } else {
                    beat_plv
                };
                (support * presence).clamp(0.0, 1.0)
            }
        };
        let beat_freq = self.beat_omega / TAU;

        let beat = MeterBand {
            phase: wrap_pm_pi(self.beat_phi),
            freq_hz: beat_freq,
            amplitude: self.beat_r.clamp(0.0, 2.0),
            confidence: beat_conf,
        };

        // Pick the integer ratio whose harmonic of the beat phase the onsets
        // lock to best.
        let mut best_k = 0usize;
        let mut best_mag = 0.0f32;
        for k in 0..SUB_RATIOS.len() {
            let mag = (self.sub_re[k] * self.sub_re[k] + self.sub_im[k] * self.sub_im[k]).sqrt();
            if mag > best_mag {
                best_mag = mag;
                best_k = k;
            }
        }
        let sub_plv = best_mag / count;
        let offbeat_support = 0.5 * (1.0 - beat_plv.clamp(0.0, 1.0));
        // Subdivision is real only when onsets actually fall off the beat; a
        // beat-only signal yields ~0 here even though every harmonic is coherent.
        let sub_conf = (sub_plv * offbeat_support * presence).clamp(0.0, 1.0);
        let detected = sub_conf > 0.05;
        let ratio = if detected { SUB_RATIOS[best_k] } else { 0 };

        let subdivision = MeterBand {
            phase: wrap_pm_pi(SUB_RATIOS[best_k] as f32 * self.beat_phi),
            freq_hz: SUB_RATIOS[best_k] as f32 * beat_freq,
            amplitude: offbeat_support,
            confidence: sub_conf,
        };

        // Measure: the beats-per-bar whose accent recurrence is most coherent.
        let mut best_m = 0usize;
        let mut best_m_mag = 0.0f32;
        for k in 0..MEASURE_RATIOS.len() {
            let mag =
                (self.meas_re[k] * self.meas_re[k] + self.meas_im[k] * self.meas_im[k]).sqrt();
            if mag > best_m_mag {
                best_m_mag = mag;
                best_m = k;
            }
        }
        // Normalize by accent mass, and require enough accent evidence so a
        // uniform (accent-free) beat does not claim a measure.
        let meas_norm = self.meas_norm.max(1e-6);
        let meas_plv = (best_m_mag / meas_norm).clamp(0.0, 1.0);
        let accent_presence = smoothstep(0.6, 2.5, self.meas_norm);
        let meas_conf = (meas_plv * accent_presence * presence).clamp(0.0, 1.0);
        let meas_detected = meas_conf > 0.05;
        let measure_ratio = if meas_detected {
            MEASURE_RATIOS[best_m]
        } else {
            0
        };
        let measure = MeterBand {
            // Zero phase follows observed accents, not the arbitrary beat-count origin.
            phase: wrap_pm_pi(
                TAU * self.beat_cycles / MEASURE_RATIOS[best_m] as f32
                    - self.meas_im[best_m].atan2(self.meas_re[best_m]),
            ),
            freq_hz: beat_freq / MEASURE_RATIOS[best_m] as f32,
            amplitude: accent_presence,
            confidence: meas_conf,
        };

        MeterState {
            beat,
            subdivision,
            measure,
            attention_level: self.attention_ema.clamp(0.0, 1.0),
            subdivision_ratio: ratio,
            measure_ratio,
        }
    }

    #[cfg(test)]
    pub fn state(&self) -> MeterState {
        self.last
    }
}

/// Sanitize a basin `(min, max)` into an ordered, finite, in-band pair.
pub(crate) fn sane_basin(min_hz: f32, max_hz: f32) -> (f32, f32) {
    let a = if min_hz.is_finite() {
        min_hz
    } else {
        F_BEAT_MIN
    };
    let b = if max_hz.is_finite() {
        max_hz
    } else {
        F_BEAT_MAX
    };
    let lo = a.min(b).clamp(F_BEAT_MIN, F_BEAT_MAX);
    let hi = a.max(b).clamp(F_BEAT_MIN, F_BEAT_MAX);
    if hi - lo < 1e-3 {
        // Degenerate (point) basin: widen slightly so the clamp has room.
        let c = (0.5 * (lo + hi)).clamp(F_BEAT_MIN, F_BEAT_MAX);
        ((c - 0.05).max(F_BEAT_MIN), (c + 0.05).min(F_BEAT_MAX))
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.005; // 200 Hz control rate

    /// Build a drive pulse: 1.0 for `width` seconds after each event, else 0.
    fn pulse(t_in_period: f32, width: f32) -> f32 {
        if t_in_period < width { 1.0 } else { 0.0 }
    }

    /// Run a clean isochronous beat and return the final state.
    fn run_metric(beat_hz: f32, secs: f32) -> MeterState {
        let mut net = MeterNetwork::new();
        let period = 1.0 / beat_hz;
        let mut t = 0.0f32;
        let n = (secs / DT) as usize;
        for _ in 0..n {
            let drive = pulse(t % period, 0.02);
            net.process(DT, drive);
            t += DT;
        }
        net.state()
    }

    #[test]
    fn recurrence_capacity_covers_refractory_limited_onsets() {
        for dt in [0.0001_f32, 0.001, 0.005, 0.02, 0.07] {
            let mut net = MeterNetwork::new();
            let initial = net.recurrence.events().unwrap();
            let capacities = (initial.history.capacity(), initial.echo.capacity());
            assert!(capacities.0 < 100);
            let mut high = false;
            for _ in 0..(12.0 / dt) as usize {
                high = !high;
                net.process(dt, if high { 1.0 } else { 0.0 });
            }
            let recurrence = net.recurrence.events().unwrap();
            assert_eq!(
                capacities,
                (recurrence.history.capacity(), recurrence.echo.capacity())
            );
        }
    }

    #[test]
    fn basin_without_acoustic_drive_keeps_the_beat_unconfirmed() {
        for stability in [0.0, 0.85, 1.0] {
            let mut net = MeterNetwork::new();
            net.set_shaping(MeterShaping {
                stability,
                basin_hz: Some((2.8, 3.4)),
            });
            for _ in 0..(24.0 / DT) as usize {
                let state = net.process(DT, 0.0);
                assert!(!net.seeded);
                assert_eq!(state.beat.confidence, 0.0);
                assert_eq!(state.subdivision.confidence, 0.0);
                assert_eq!(state.measure.confidence, 0.0);
                assert_eq!(state.subdivision_ratio, 0);
                assert_eq!(state.measure_ratio, 0);
            }
        }
    }

    #[test]
    fn metric_beat_locks_with_high_confidence() {
        let s = run_metric(2.0, 20.0);
        assert!(
            (s.beat.freq_hz - 2.0).abs() < 0.3,
            "beat should lock near 2 Hz, got {}",
            s.beat.freq_hz
        );
        assert!(
            s.beat.confidence > 0.8,
            "clean isochronous beat should read high confidence, got {}",
            s.beat.confidence
        );
        // No off-beat content: subdivision must stay near zero.
        assert!(
            s.subdivision.confidence < 0.2,
            "beat-only signal must not claim subdivision, got {}",
            s.subdivision.confidence
        );
    }

    #[test]
    fn flow_rain_reads_low_confidence_despite_dense_drive() {
        // Deterministic pseudo-Poisson onset stream (renewal process).
        let mut net = MeterNetwork::new();
        let mut seed: u32 = 0x1234_5678;
        let mut next_rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let mean_ioi = 0.5; // same mean rate as a 2 Hz beat
        let mut t = 0.0f32;
        let mut next_onset = -mean_ioi * (1.0 - next_rng()).ln();
        let total = 25.0f32;
        let n = (total / DT) as usize;
        let mut drive_sum = 0.0f32;
        for _ in 0..n {
            let mut drive = 0.0;
            if t >= next_onset {
                drive = 1.0;
                next_onset = t - mean_ioi * (1.0 - next_rng()).ln();
            }
            drive_sum += drive;
            net.process(DT, drive);
            t += DT;
        }
        let s = net.state();
        // Drive is dense, but phases are spread: confidence must stay low.
        assert!(drive_sum > 10.0, "sanity: rain should have many onsets");
        assert!(
            s.beat.confidence < 0.5,
            "renewal rain must read low beat confidence, got {}",
            s.beat.confidence
        );
    }

    #[test]
    fn confidence_tracks_phase_lock_not_loudness() {
        // Quiet but perfectly periodic beat vs loud but random rain.
        let metric = run_metric(2.0, 20.0);

        let mut rain = MeterNetwork::new();
        let mut seed: u32 = 0x9E37_79B9;
        let mut next_rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut t = 0.0f32;
        let mut next_onset = 0.0f32;
        let n = (20.0f32 / DT) as usize;
        for _ in 0..n {
            let mut drive = 0.0;
            if t >= next_onset {
                drive = 1.0;
                next_onset = t - 0.4 * (1.0 - next_rng()).ln();
            }
            rain.process(DT, drive);
            t += DT;
        }
        assert!(
            metric.beat.confidence > rain.state().beat.confidence + 0.3,
            "periodic beat ({}) must out-confidence dense rain ({})",
            metric.beat.confidence,
            rain.state().beat.confidence
        );
    }

    #[test]
    fn entrainment_confidence_rises_as_timing_regularizes() {
        // Jittered beat whose jitter shrinks over time. Confidence should be
        // higher late than early.
        let mut net = MeterNetwork::new();
        let mut seed: u32 = 0x0BAD_F00D;
        let mut next_rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let period = 0.5f32;
        let total = 40.0f32;
        let n = (total / DT) as usize;
        let mut t = 0.0f32;
        let mut next_beat = 0.0f32;
        let mut early = 0.0f32;
        let mut late = 0.0f32;
        for i in 0..n {
            let mut drive = 0.0;
            if t >= next_beat {
                drive = 1.0;
                let frac = (t / total).clamp(0.0, 1.0);
                let jitter_amp = 0.18 * (1.0 - frac); // shrinks toward 0
                let jitter = jitter_amp * (next_rng() - 0.5);
                next_beat = t + period + jitter;
            }
            let s = net.process(DT, drive);
            let frac = i as f32 / n as f32;
            if (0.2..0.3).contains(&frac) {
                early = early.max(s.beat.confidence);
            }
            if frac > 0.9 {
                late = late.max(s.beat.confidence);
            }
            t += DT;
        }
        assert!(
            late > early + 0.05,
            "confidence should rise as timing regularizes: early {early} late {late}"
        );
    }

    #[test]
    fn detects_duple_subdivision_under_accent() {
        // Strong on-beat (2 Hz) plus a weaker half-beat onset. The accent keeps
        // the beat at 2 Hz while the off-beat onsets reveal a duple subdivision.
        let mut net = MeterNetwork::new();
        let period = 0.5f32;
        let mut t = 0.0f32;
        let n = (30.0f32 / DT) as usize;
        for _ in 0..n {
            let ph = t % period;
            let mut drive = pulse(ph, 0.02); // on-beat, strong
            let off = (ph - period * 0.5).rem_euclid(period);
            if off < 0.02 {
                drive = drive.max(0.55); // half-beat, weaker
            }
            net.process(DT, drive);
            t += DT;
        }
        let s = net.state();
        assert!(
            (s.beat.freq_hz - 2.0).abs() < 0.4,
            "accent should keep beat near 2 Hz, got {}",
            s.beat.freq_hz
        );
        assert_eq!(
            s.subdivision_ratio, 2,
            "expected duple subdivision, got ratio {}",
            s.subdivision_ratio
        );
        assert!(
            s.subdivision.confidence > 0.1,
            "duple subdivision should register, got {}",
            s.subdivision.confidence
        );
    }

    #[test]
    fn beat_persists_through_a_short_gap() {
        // Lock a beat, then go silent for ~1 s. The limit cycle must keep
        // oscillating (amplitude stays alive) rather than collapsing.
        let mut net = MeterNetwork::new();
        let period = 0.5f32;
        let mut t = 0.0f32;
        let lock_n = (12.0f32 / DT) as usize;
        for _ in 0..lock_n {
            let drive = pulse(t % period, 0.02);
            net.process(DT, drive);
            t += DT;
        }
        let locked = net.state();
        assert!(locked.beat.confidence > 0.7);

        // 1 s of silence.
        let gap_n = (1.0f32 / DT) as usize;
        for _ in 0..gap_n {
            net.process(DT, 0.0);
        }
        let after = net.state();
        assert!(
            after.beat.amplitude > 0.5,
            "limit-cycle beat should coast through a short gap, amp {}",
            after.beat.amplitude
        );
        assert!(
            after.beat.confidence > 0.4,
            "confidence should persist through a short gap, got {}",
            after.beat.confidence
        );
    }

    #[test]
    fn detects_duple_measure_from_accent() {
        // Isochronous 2 Hz beat with every other beat accented (louder). The
        // accent recurs at half the beat rate, so a 2-beat measure should emerge
        // while the beat itself stays at 2 Hz.
        let mut net = MeterNetwork::new();
        let period = 0.5f32;
        let mut t = 0.0f32;
        let n = (40.0f32 / DT) as usize;
        for _ in 0..n {
            let ph = t % period;
            let strong = (t / period).round() as i64 % 2 == 0;
            let drive = if ph < 0.02 {
                if strong { 1.0 } else { 0.55 }
            } else {
                0.0
            };
            net.process(DT, drive);
            t += DT;
        }
        let s = net.state();
        assert!(
            (s.beat.freq_hz - 2.0).abs() < 0.4,
            "beat should stay near 2 Hz, got {}",
            s.beat.freq_hz
        );
        assert_eq!(
            s.measure_ratio, 2,
            "alternating accent should induce a 2-beat measure, got {}",
            s.measure_ratio
        );
        assert!(
            s.measure.confidence > 0.1,
            "accent recurrence should register a measure, got {}",
            s.measure.confidence
        );
    }

    #[test]
    fn measure_phase_tracks_the_observed_accent_position() {
        for strong_parity in [0, 1] {
            let mut net = MeterNetwork::new();
            let mut strong_cos = Vec::new();
            let mut weak_cos = Vec::new();
            for step in 0..16000 {
                let beat = step / 100;
                let strong = beat % 2 == strong_parity;
                let drive = if step % 100 < 4 {
                    if strong { 1.0 } else { 0.55 }
                } else {
                    0.0
                };
                let state = net.process(DT, drive);
                if step >= 8000 && step % 100 == 4 {
                    assert_eq!(state.measure_ratio, 2);
                    if strong {
                        strong_cos.push(state.measure.phase.cos());
                    } else {
                        weak_cos.push(state.measure.phase.cos());
                    }
                }
            }
            let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
            assert!(
                mean(&strong_cos) > 0.7,
                "strong parity {strong_parity}: {}",
                mean(&strong_cos)
            );
            assert!(
                mean(&weak_cos) < -0.7,
                "weak parity {strong_parity}: {}",
                mean(&weak_cos)
            );
        }
    }

    #[test]
    fn uniform_beat_induces_no_measure() {
        // A perfectly uniform 2 Hz beat has no accent, so no measure grouping
        // should be claimed even though the beat itself locks strongly.
        let s = run_metric(2.0, 40.0);
        assert!(
            s.beat.confidence > 0.8,
            "uniform beat should still lock, got {}",
            s.beat.confidence
        );
        assert_eq!(
            s.measure_ratio, 0,
            "an accent-free beat must not claim a measure, got ratio {}",
            s.measure_ratio
        );
        assert!(
            s.measure.confidence < 0.1,
            "accent-free beat should read ~0 measure confidence, got {}",
            s.measure.confidence
        );
    }

    #[test]
    fn higher_stability_speeds_beat_lock() {
        // Compare after corroboration: resetting old coordinates invalidates the old 1.5 s cut.
        let mut neutral = MeterNetwork::new();
        let mut deep = MeterNetwork::new();
        deep.set_shaping(MeterShaping {
            stability: 1.0,
            basin_hz: None,
        });
        let mut t = 0.0_f32;
        let mut early_difference = 0.0_f32;
        let mut saturated_peaks = [0.0_f32; 2];
        // Reuse the clean metric fixture; compare saturated presence, not a lock deadline.
        for _ in 0..(20.0 / DT) as usize {
            let drive = pulse(t % 0.5, 0.02);
            let a = neutral.process(DT, drive);
            let b = deep.process(DT, drive);
            if neutral.seeded && deep.seeded {
                early_difference = early_difference.max(b.beat.confidence - a.beat.confidence);
            }
            if neutral.plv_count >= 4.0 && deep.plv_count >= 2.0 {
                saturated_peaks[0] = saturated_peaks[0].max(a.beat.confidence);
                saturated_peaks[1] = saturated_peaks[1].max(b.beat.confidence);
            }
            t += DT;
        }
        assert!(
            early_difference > 0.2,
            "high stability must precede neutral after corroboration"
        );
        assert!(
            saturated_peaks[0] > 0.8 && (saturated_peaks[1] - saturated_peaks[0]).abs() < 0.1,
            "stability must not raise the ceiling"
        );
    }

    #[test]
    fn high_stability_does_not_fabricate_beat_from_rain() {
        // A deep attractor must still read low confidence on renewal rain: it
        // amplifies forcing toward a real phase, it does not invent one.
        let mut net = MeterNetwork::new();
        net.set_shaping(MeterShaping {
            stability: 1.0,
            basin_hz: None,
        });
        let mut seed: u32 = 0x1234_5678;
        let mut next_rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let mean_ioi = 0.5;
        let mut t = 0.0f32;
        let mut next_onset = -mean_ioi * (1.0 - next_rng()).ln();
        let n = (25.0f32 / DT) as usize;
        for _ in 0..n {
            let mut drive = 0.0;
            if t >= next_onset {
                drive = 1.0;
                next_onset = t - mean_ioi * (1.0 - next_rng()).ln();
            }
            net.process(DT, drive);
            t += DT;
        }
        assert!(
            net.state().beat.confidence < 0.5,
            "deep attractor must not fabricate a beat from rain, got {}",
            net.state().beat.confidence
        );
    }

    #[test]
    fn temporal_basin_biases_free_running_beat_frequency() {
        // With only weak/sparse drive, the basin prior should pull the resting
        // beat frequency toward the basin center (terrain shaping), away from the
        // 2 Hz default seed.
        let mut net = MeterNetwork::new();
        net.set_shaping(MeterShaping {
            stability: 0.0,
            basin_hz: Some((2.8, 3.4)),
        });
        // Seeded at the basin center already; coast with no drive and confirm it
        // stays in-band rather than drifting back to the 2 Hz default.
        for _ in 0..(8.0f32 / DT) as usize {
            net.process(DT, 0.0);
        }
        let f = net.state().beat.freq_hz;
        assert!(
            (2.8..=3.4).contains(&f),
            "basin should hold the beat frequency in-band, got {f}"
        );
    }

    #[test]
    fn temporal_basin_alone_induces_no_measure() {
        // Setting a basin must shape only the beat frequency. A uniform beat in
        // the basin still has no accent, so no measure may be claimed: emergence,
        // not the prior, produces the grouping.
        let mut net = MeterNetwork::new();
        net.set_shaping(MeterShaping {
            stability: 0.5,
            basin_hz: Some((2.5, 3.5)),
        });
        let period = 1.0 / 3.0;
        let mut t = 0.0f32;
        for _ in 0..(40.0f32 / DT) as usize {
            let drive = pulse(t % period, 0.02);
            net.process(DT, drive);
            t += DT;
        }
        let s = net.state();
        assert!(
            s.beat.confidence > 0.7,
            "uniform in-basin beat should still lock, got {}",
            s.beat.confidence
        );
        assert_eq!(
            s.measure_ratio, 0,
            "a basin alone must not fabricate a measure, got ratio {}",
            s.measure_ratio
        );
    }
}
