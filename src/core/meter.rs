//! Coupled neural-oscillator meter core (Regime A + C of the rhythm design).
//!
//! This is the perception-side meter model: a self-sustaining limit-cycle beat
//! oscillator entrained by acoustic onset drive, plus integer-ratio subdivision
//! detection. Salience is reported as *entrainment confidence* (a windowed
//! phase-locking value, PLV), kept strictly distinct from the drive amplitude.
//!
//! Design rationale: `docs/design-notes/neural-rhythm-meter.md`.
//!
//! Phase boundary: the beat (tactus) limit cycle and PLV confidence are live and
//! observable from onset timing alone. The measure subharmonic and grouping
//! selection are deferred to the accent layer (Regime B), where they have real
//! support; shipping them here would be inert.

use std::f32::consts::TAU;

use crate::core::phase::wrap_pm_pi;

// Beat (tactus) range. Faster onset streams lock the beat at the top of this
// band; anything faster is subdivision territory, not a separate beat.
const F_BEAT_MIN: f32 = 0.5;
const F_BEAT_MAX: f32 = 4.0;
const F_BEAT_INIT: f32 = 2.0;

// Forced Hopf normal form (Large neural resonance). ALPHA > 0 with BETA < 0
// gives a stable limit cycle of radius sqrt(-ALPHA/BETA) = 1.0, so the beat is
// self-sustaining and coasts through gaps (Regime C: persistence).
const ALPHA: f32 = 1.0;
const BETA: f32 = -1.0;
const FORCE_AMP: f32 = 1.0; // drive -> amplitude pumping (in phase)
const FORCE_PHASE: f32 = 1.0; // drive -> phase entrainment
const ETA_OMEGA: f32 = 3.0; // Hebbian frequency-learning rate

// Onset detector (adaptive threshold + refractory).
const ONSET_TAU: f32 = 1.0;
const ONSET_K: f32 = 2.0;
const REFRACTORY: f32 = 0.06;

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

/// One metrical level's reported state.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeterBand {
    /// Wrapped oscillator phase in [-pi, pi].
    pub phase: f32,
    /// Current frequency in Hz.
    pub freq_hz: f32,
    /// Limit-cycle amplitude (presence). Not salience.
    pub amplitude: f32,
    /// Entrainment confidence in [0, 1] from phase prediction (PLV). This is the
    /// salience: a clean quiet beat reads high, a loud wandering beat reads low.
    pub confidence: f32,
}

/// Listener-side meter state.
#[derive(Clone, Copy, Debug, Default)]
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

/// Coupled limit-cycle meter network (single beat oscillator + subdivision
/// detection). One instance per listener.
#[derive(Clone, Debug)]
pub struct MeterNetwork {
    // Beat oscillator state.
    beat_phi: f32,
    beat_omega: f32,
    beat_r: f32,

    // Onset detector state.
    onset_baseline: f32,
    onset_var: f32,
    onset_timer: f32,
    prev_drive: f32,

    // Leaky resultant accumulators (numerator complex sum + count denominator),
    // all decayed at PERSIST_TAU so PLV is rate-independent and silence-fading.
    plv_count: f32,
    beat_re: f32,
    beat_im: f32,
    sub_re: [f32; 3],
    sub_im: [f32; 3],
    offbeat_num: f32,

    // Measure (accent subharmonic) state. beat_cycles is the unwrapped beat
    // count; accent-weighted resultants at candidate subharmonics select the
    // grouping, normalized by the accumulated accent mass.
    beat_cycles: f32,
    strength_baseline: f32,
    meas_re: [f32; 3],
    meas_im: [f32; 3],
    meas_norm: f32,

    attention_ema: f32,
    last: MeterState,
}

impl Default for MeterNetwork {
    fn default() -> Self {
        Self {
            beat_phi: 0.0,
            beat_omega: TAU * F_BEAT_INIT,
            beat_r: 0.1,
            onset_baseline: 0.0,
            onset_var: 0.0,
            onset_timer: 0.0,
            prev_drive: 0.0,
            plv_count: 0.0,
            beat_re: 0.0,
            beat_im: 0.0,
            sub_re: [0.0; 3],
            sub_im: [0.0; 3],
            offbeat_num: 0.0,
            beat_cycles: 0.0,
            strength_baseline: 0.0,
            meas_re: [0.0; 3],
            meas_im: [0.0; 3],
            meas_norm: 0.0,
            attention_ema: 0.0,
            last: MeterState::default(),
        }
    }
}

impl MeterNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the network by `dt` seconds under acoustic onset `drive` in
    /// [0, 1] (rectified spectral flux). Returns the updated meter state.
    pub fn process(&mut self, dt: f32, drive: f32) -> MeterState {
        let dt = dt.max(1e-4);
        let drive = drive.clamp(0.0, 1.0);

        let onset = self.detect_onset(dt, drive);

        // --- Beat oscillator: forced limit cycle, integrated in polar form so
        // the rotation is exact and only the slow amplitude/forcing terms use
        // Euler steps. ---
        let phi_before = self.beat_phi;
        let r = self.beat_r.max(1e-3);
        let dr =
            ALPHA * self.beat_r + BETA * self.beat_r.powi(3) + FORCE_AMP * drive * phi_before.cos();
        self.beat_r = (self.beat_r + dr * dt).clamp(0.0, 2.0);

        let dphi = self.beat_omega - FORCE_PHASE * (drive / r) * phi_before.sin();
        self.beat_phi = wrap_pm_pi(phi_before + dphi * dt);

        // Hebbian frequency learning: shift omega to reduce the phase error to
        // the stimulus. Random (renewal) input averages to ~0 net shift, so the
        // beat does not chase noise.
        self.beat_omega -= ETA_OMEGA * drive * phi_before.sin() * dt;
        self.beat_omega = self.beat_omega.clamp(TAU * F_BEAT_MIN, TAU * F_BEAT_MAX);

        // Unwrapped beat count, used to phase the slow measure subharmonic.
        self.beat_cycles += dphi * dt / TAU;

        // --- Confidence accumulators: decay every tick (persistence), add an
        // impulse at each onset. ---
        let decay = (-dt / PERSIST_TAU).exp();
        self.plv_count *= decay;
        self.beat_re *= decay;
        self.beat_im *= decay;
        self.offbeat_num *= decay;
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

        if onset.fired {
            let onset_phi = wrap_pm_pi(phi_before + dphi * dt * onset.frac);
            self.plv_count += 1.0;
            self.beat_re += onset_phi.cos();
            self.beat_im += onset_phi.sin();
            // off-beat support: 0 on the beat, 1 in anti-phase.
            self.offbeat_num += 0.5 * (1.0 - onset_phi.cos());
            // ratio competition: which integer multiple of the beat phase do
            // onsets lock to.
            for (k, ratio) in SUB_RATIOS.iter().enumerate() {
                let a = (*ratio as f32) * onset_phi;
                self.sub_re[k] += a.cos();
                self.sub_im[k] += a.sin();
            }

            // Accent: how much louder this onset is than the running baseline.
            // Only positive emphasis counts; a uniform stream yields ~0 accent
            // mass, so no measure is induced.
            let strength = drive;
            self.strength_baseline += STRENGTH_BL_RATE * (strength - self.strength_baseline);
            let accent = (strength - self.strength_baseline).max(0.0);
            if accent > 0.0 {
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

        let att_a = (-dt / 0.3).exp();
        self.attention_ema = att_a * self.attention_ema + (1.0 - att_a) * drive;

        self.last = self.build_state();
        self.last
    }

    fn detect_onset(&mut self, dt: f32, drive: f32) -> Onset {
        let a = (-dt / ONSET_TAU).exp();
        self.onset_baseline = a * self.onset_baseline + (1.0 - a) * drive;
        let dev = drive - self.onset_baseline;
        self.onset_var = a * self.onset_var + (1.0 - a) * dev * dev;
        let std = self.onset_var.max(1e-6).sqrt();
        let th = (self.onset_baseline + ONSET_K * std).clamp(0.01, 0.95);
        self.onset_timer = (self.onset_timer - dt).max(0.0);

        let mut out = Onset::default();
        if self.onset_timer <= 0.0 && self.prev_drive < th && drive >= th {
            let denom = (drive - self.prev_drive).max(1e-6);
            out.frac = ((th - self.prev_drive) / denom).clamp(0.0, 1.0);
            out.fired = true;
            self.onset_timer = REFRACTORY;
        }
        self.prev_drive = drive;
        out
    }

    fn build_state(&self) -> MeterState {
        let count = self.plv_count.max(1e-6);
        // Presence gate: a resultant from too few onsets is statistically
        // unreliable (a couple of coincidentally aligned onsets read as a high
        // PLV), so confidence requires ~4 accumulated onsets of evidence before
        // it is trusted, matching beat-induction needing a few cycles. Decays in
        // silence as the leaky count fades.
        let presence = smoothstep(1.0, 4.0, self.plv_count);

        let beat_plv = (self.beat_re * self.beat_re + self.beat_im * self.beat_im).sqrt() / count;
        let beat_conf = (beat_plv * presence).clamp(0.0, 1.0);
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
        let offbeat_support = (self.offbeat_num / count).clamp(0.0, 1.0);
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
            phase: wrap_pm_pi(TAU * self.beat_cycles / MEASURE_RATIOS[best_m] as f32),
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

    pub fn state(&self) -> MeterState {
        self.last
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Default)]
struct Onset {
    fired: bool,
    frac: f32,
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
}
