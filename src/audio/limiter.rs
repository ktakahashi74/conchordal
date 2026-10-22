use crate::core::float::finite_or;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct SoftClipParams {
    pub ceiling: f32,
    pub drive: f32,
}

impl Default for SoftClipParams {
    fn default() -> Self {
        Self {
            ceiling: 0.98,
            drive: 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PeakLimiterParams {
    pub ceiling: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub link_channels: bool,
}

impl Default for PeakLimiterParams {
    fn default() -> Self {
        Self {
            ceiling: 0.98,
            attack_ms: 0.5,
            release_ms: 50.0,
            link_channels: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LimiterMode {
    None,
    SoftClip(SoftClipParams),
    PeakLimiter(PeakLimiterParams),
}

impl Default for LimiterMode {
    fn default() -> Self {
        Self::PeakLimiter(PeakLimiterParams::default())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LimiterStats {
    pub max_abs_in: f32,
    pub max_abs_out: f32,
    pub max_reduction_db: f32,
    pub num_over: u64,
}

#[derive(Debug, Default)]
pub struct LimiterMeter {
    engaged_count: AtomicU64,
    over_count: AtomicU64,
    max_reduction_bits: AtomicU32,
    max_abs_in_bits: AtomicU32,
    max_abs_out_bits: AtomicU32,
}

impl LimiterMeter {
    pub fn record(&self, stats: &LimiterStats) {
        if stats.num_over == 0 && stats.max_reduction_db <= 0.0 {
            return;
        }
        self.engaged_count.fetch_add(1, Ordering::Relaxed);
        if stats.num_over > 0 {
            self.over_count.fetch_add(stats.num_over, Ordering::Relaxed);
        }
        self.max_reduction_bits
            .fetch_max(stats.max_reduction_db.to_bits(), Ordering::Relaxed);
        self.max_abs_in_bits
            .fetch_max(stats.max_abs_in.to_bits(), Ordering::Relaxed);
        self.max_abs_out_bits
            .fetch_max(stats.max_abs_out.to_bits(), Ordering::Relaxed);
    }

    pub fn take_snapshot(&self) -> Option<LimiterStats> {
        let engaged = self.engaged_count.swap(0, Ordering::Relaxed);
        if engaged == 0 {
            return None;
        }
        let over = self.over_count.swap(0, Ordering::Relaxed);
        let max_reduction_db = f32::from_bits(self.max_reduction_bits.swap(0, Ordering::Relaxed));
        let max_abs_in = f32::from_bits(self.max_abs_in_bits.swap(0, Ordering::Relaxed));
        let max_abs_out = f32::from_bits(self.max_abs_out_bits.swap(0, Ordering::Relaxed));
        Some(LimiterStats {
            max_abs_in,
            max_abs_out,
            max_reduction_db,
            num_over: over,
        })
    }
}

#[derive(Debug)]
struct PeakLimiterState {
    gain: f32,
    gains: Vec<f32>,
    attack_coeff: f32,
    release_coeff: f32,
}

#[derive(Debug)]
pub struct Limiter {
    mode: LimiterMode,
    input_gain: f32,
    channels: usize,
    limiter_state: Option<PeakLimiterState>,
    stats: LimiterStats,
    meter: Option<Arc<LimiterMeter>>,
}

impl Limiter {
    pub fn new(mode: LimiterMode, sample_rate: u32, channels: usize) -> Self {
        let sample_rate = (sample_rate as f32).max(1.0);
        let channels = channels.max(1);
        let limiter_state = match mode {
            LimiterMode::PeakLimiter(params) => {
                let attack_coeff = time_to_coeff(params.attack_ms, sample_rate);
                let release_coeff = time_to_coeff(params.release_ms, sample_rate);
                let gains = if params.link_channels {
                    Vec::new()
                } else {
                    vec![1.0; channels]
                };
                Some(PeakLimiterState {
                    gain: 1.0,
                    gains,
                    attack_coeff,
                    release_coeff,
                })
            }
            _ => None,
        };
        Self {
            mode,
            input_gain: 1.0,
            channels,
            limiter_state,
            stats: LimiterStats::default(),
            meter: None,
        }
    }

    pub fn with_meter(mut self, meter: Arc<LimiterMeter>) -> Self {
        self.meter = Some(meter);
        self
    }

    /// Apply fixed output gain before any guard, including the disabled guard.
    pub(crate) fn with_input_gain(mut self, gain: f32) -> Self {
        self.input_gain = gain;
        self
    }

    pub(crate) fn stats(&self) -> LimiterStats {
        self.stats
    }

    pub fn process_interleaved(&mut self, frames: &mut [f32], channels: usize) {
        if frames.is_empty() || channels == 0 {
            return;
        }
        debug_assert_eq!(channels, self.channels);
        self.stats = LimiterStats::default();
        if self.input_gain != 1.0 {
            for sample in frames.iter_mut() {
                *sample *= self.input_gain;
            }
        }
        match self.mode {
            LimiterMode::None => {
                for &sample in frames.iter() {
                    self.stats.max_abs_in = self.stats.max_abs_in.max(sample.abs());
                }
                self.stats.max_abs_out = self.stats.max_abs_in;
            }
            LimiterMode::SoftClip(params) => {
                let ceiling = finite_or(params.ceiling, SoftClipParams::default().ceiling)
                    .abs()
                    .max(1e-6);
                // Drive sets headroom: at drive=2 the identity knee is ceiling/2.
                let below_ceiling = f32::from_bits(ceiling.to_bits() - 1);
                let knee =
                    (ceiling - ceiling / finite_or(params.drive, 2.0).max(1.0)).min(below_ceiling);
                let headroom = ceiling - knee;
                for s in frames.iter_mut() {
                    let x = finite_or(*s, 0.0);
                    let abs_in = x.abs();
                    if abs_in > self.stats.max_abs_in {
                        self.stats.max_abs_in = abs_in;
                    }
                    if abs_in > ceiling {
                        self.stats.num_over += 1;
                    }
                    let y = if abs_in <= knee {
                        x
                    } else {
                        let magnitude = knee + headroom * ((abs_in - knee) / headroom).tanh();
                        // tanh rounds to one for large input; retain the strict ceiling.
                        magnitude.min(below_ceiling).copysign(x)
                    };
                    let abs_out = y.abs();
                    if abs_out > self.stats.max_abs_out {
                        self.stats.max_abs_out = abs_out;
                    }
                    update_reduction(&mut self.stats, abs_in, abs_out);
                    *s = y;
                }
            }
            LimiterMode::PeakLimiter(params) => {
                let ceiling = params.ceiling.abs().max(1e-6);
                let n_frames = frames.len() / channels;
                // Invariant: `mode` is fixed at construction and `limiter_state` is
                // `Some` for PeakLimiter, so this is unreachable. Bail out instead of
                // panicking because this runs on the cpal audio callback thread.
                let Some(state) = self.limiter_state.as_mut() else {
                    debug_assert!(false, "PeakLimiter mode without limiter state");
                    return;
                };
                if params.link_channels {
                    for frame in 0..n_frames {
                        let mut peak = 0.0f32;
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let x = frames[idx];
                            let abs_in = finite_or(x, 0.0).abs();
                            peak = peak.max(abs_in);
                        }
                        let target_gain = if peak > ceiling && peak > 0.0 {
                            ceiling / peak
                        } else {
                            1.0
                        };
                        let gain = smooth_gain(
                            state.gain,
                            target_gain,
                            state.attack_coeff,
                            state.release_coeff,
                        );
                        state.gain = gain;
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let x = finite_or(frames[idx], 0.0);
                            let abs_in = x.abs();
                            if abs_in > self.stats.max_abs_in {
                                self.stats.max_abs_in = abs_in;
                            }
                            if abs_in > ceiling {
                                self.stats.num_over += 1;
                            }
                            let mut y = x * gain;
                            if y.abs() > ceiling {
                                y = y.clamp(-ceiling, ceiling);
                            }
                            let abs_out = y.abs();
                            if abs_out > self.stats.max_abs_out {
                                self.stats.max_abs_out = abs_out;
                            }
                            update_reduction(&mut self.stats, abs_in, abs_out);
                            frames[idx] = y;
                        }
                    }
                } else {
                    if state.gains.len() != channels {
                        debug_assert!(
                            false,
                            "Peak limiter channel gains mismatch (state={}, input={})",
                            state.gains.len(),
                            channels
                        );
                        state.gains.resize(channels, 1.0);
                    }
                    for frame in 0..n_frames {
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let x = finite_or(frames[idx], 0.0);
                            let abs_in = x.abs();
                            let target_gain = if abs_in > ceiling && abs_in > 0.0 {
                                ceiling / abs_in
                            } else {
                                1.0
                            };
                            let gain = smooth_gain(
                                state.gains[ch],
                                target_gain,
                                state.attack_coeff,
                                state.release_coeff,
                            );
                            state.gains[ch] = gain;
                            if abs_in > self.stats.max_abs_in {
                                self.stats.max_abs_in = abs_in;
                            }
                            if abs_in > ceiling {
                                self.stats.num_over += 1;
                            }
                            let mut y = x * gain;
                            if y.abs() > ceiling {
                                y = y.clamp(-ceiling, ceiling);
                            }
                            let abs_out = y.abs();
                            if abs_out > self.stats.max_abs_out {
                                self.stats.max_abs_out = abs_out;
                            }
                            update_reduction(&mut self.stats, abs_in, abs_out);
                            frames[idx] = y;
                        }
                    }
                }
            }
        }
        if let Some(meter) = self.meter.as_ref() {
            meter.record(&self.stats);
        }
    }
}

fn time_to_coeff(time_ms: f32, sample_rate: f32) -> f32 {
    let time_s = (time_ms.max(0.0)) * 0.001;
    if time_s <= 0.0 {
        0.0
    } else {
        (-1.0 / (time_s * sample_rate)).exp()
    }
}

fn smooth_gain(current: f32, target: f32, attack_coeff: f32, release_coeff: f32) -> f32 {
    if target < current {
        attack_coeff * current + (1.0 - attack_coeff) * target
    } else {
        release_coeff * current + (1.0 - release_coeff) * target
    }
}

fn update_reduction(stats: &mut LimiterStats, abs_in: f32, abs_out: f32) {
    if abs_in <= 1e-12 || abs_out <= 0.0 {
        return;
    }
    let ratio = abs_out / abs_in;
    if ratio <= 0.0 || ratio >= 1.0 {
        return;
    }
    let db = 20.0 * ratio.log10();
    if db.is_finite() {
        let reduction = -db;
        if reduction > stats.max_reduction_db {
            stats.max_reduction_db = reduction;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_db_output_gain_preserves_guard_output_bits() {
        let mut config = crate::config::AudioConfig::default();
        config.output_gain_db = 0.0;
        for mode in [
            LimiterMode::None,
            LimiterMode::SoftClip(SoftClipParams::default()),
            LimiterMode::default(),
        ] {
            let mut original = Limiter::new(mode, 48_000, 2);
            let mut gained = Limiter::new(mode, 48_000, 2).with_input_gain(config.output_gain());
            for _ in 0..3 {
                let mut expected = [-0.0, 0.0, 0.1, -0.2, 2.0, -3.0];
                let mut actual = expected;
                original.process_interleaved(&mut expected, 2);
                gained.process_interleaved(&mut actual, 2);
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            }
        }
    }

    #[test]
    fn default_output_gain_scales_samples_with_every_guard_mode() {
        let config = crate::config::AudioConfig::default();
        let gain = 10.0f32.powf(config.output_gain_db / 20.0);
        for mode in [
            LimiterMode::None,
            LimiterMode::SoftClip(SoftClipParams::default()),
            LimiterMode::default(),
        ] {
            let mut guard = Limiter::new(mode, 48_000, 2).with_input_gain(config.output_gain());
            let mut samples = [0.001, -0.002, 0.0, -0.0];
            let expected = samples.map(|s| s * gain);
            guard.process_interleaved(&mut samples, 2);
            assert_eq!(samples.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }

    #[test]
    fn output_gain_precedes_limiter_and_its_input_meter() {
        let mut guard = Limiter::new(LimiterMode::default(), 48_000, 1).with_input_gain(2.0);
        let mut samples = [0.75, -0.75];
        guard.process_interleaved(&mut samples, 1);
        assert_eq!(guard.stats.max_abs_in, 1.5);
        assert_eq!(guard.stats.num_over, 2);
        assert!(guard.stats.max_reduction_db > 0.0);
        assert!(
            samples
                .iter()
                .all(|s| s.abs() <= PeakLimiterParams::default().ceiling)
        );

        let mut unguarded = Limiter::new(LimiterMode::None, 48_000, 1).with_input_gain(2.0);
        let mut samples = [0.75, -0.75];
        unguarded.process_interleaved(&mut samples, 1);
        assert_eq!(samples, [1.5, -1.5]);
    }

    #[test]
    fn softclip_preserves_small_signals_and_knee() {
        let params = SoftClipParams::default();
        let knee = params.ceiling - params.ceiling / params.drive;
        let mut guard = Limiter::new(LimiterMode::SoftClip(params), 48_000, 1);
        let mut buf = [-knee, -0.1, -1e-6, 0.0, 1e-6, 0.1, knee];
        let original = buf;
        guard.process_interleaved(&mut buf, 1);
        assert_eq!(buf.map(f32::to_bits), original.map(f32::to_bits));
    }

    #[test]
    fn softclip_is_monotone_odd_and_strictly_below_ceiling() {
        for drive in [0.0, 1.0, 2.0, 10.0, f32::MAX] {
            let params = SoftClipParams {
                drive,
                ..SoftClipParams::default()
            };
            let mut guard = Limiter::new(LimiterMode::SoftClip(params), 48_000, 1);
            let mut positive: Vec<_> = (0..=2000).map(|i| i as f32 * 0.01).collect();
            positive.push(f32::MAX);
            let mut negative: Vec<_> = positive.iter().map(|x| -x).collect();
            guard.process_interleaved(&mut positive, 1);
            guard.process_interleaved(&mut negative, 1);
            assert!(positive.windows(2).all(|w| w[0] <= w[1]));
            for (&a, &b) in positive.iter().zip(&negative) {
                assert!(a.abs() < params.ceiling);
                assert_eq!(a, -b);
            }
        }
    }

    #[test]
    fn meter_returns_interval_maxima_then_resets() {
        let meter = LimiterMeter::default();
        meter.record(&LimiterStats {
            max_abs_in: 4.0,
            max_abs_out: 0.9,
            max_reduction_db: 12.0,
            num_over: 2,
        });
        meter.record(&LimiterStats {
            max_abs_in: 2.0,
            max_abs_out: 0.98,
            max_reduction_db: 6.0,
            num_over: 3,
        });
        let snapshot = meter.take_snapshot().unwrap();
        assert_eq!(snapshot.max_abs_in, 4.0);
        assert_eq!(snapshot.max_abs_out, 0.98);
        assert_eq!(snapshot.max_reduction_db, 12.0);
        assert_eq!(snapshot.num_over, 5);
        assert!(meter.take_snapshot().is_none());
        meter.record(&LimiterStats {
            max_abs_in: 1.5,
            max_abs_out: 0.8,
            max_reduction_db: 3.0,
            num_over: 1,
        });
        let snapshot = meter.take_snapshot().unwrap();
        assert_eq!(snapshot.max_abs_in, 1.5);
        assert_eq!(snapshot.max_abs_out, 0.8);
        assert_eq!(snapshot.max_reduction_db, 3.0);
        assert_eq!(snapshot.num_over, 1);
    }

    #[test]
    fn safety_limiter() {
        let mut guard = Limiter::new(
            LimiterMode::PeakLimiter(PeakLimiterParams::default()),
            48_000,
            1,
        );
        let mut buf = [0.0f32, 2.0, -2.0, 0.25];
        guard.process_interleaved(&mut buf, 1);
        let ceiling = PeakLimiterParams::default().ceiling + 1e-6;
        for &v in &buf {
            assert!(v.abs() <= ceiling, "{v} exceeds ceiling");
        }
    }

    #[test]
    fn transparency_none() {
        let mut guard = Limiter::new(LimiterMode::None, 48_000, 1);
        let mut buf = [0.25f32, -0.5, 0.1, 0.0];
        let original = buf;
        guard.process_interleaved(&mut buf, 1);
        assert_eq!(buf, original);
    }

    #[test]
    fn transparency_limiter() {
        let mut guard = Limiter::new(
            LimiterMode::PeakLimiter(PeakLimiterParams::default()),
            48_000,
            1,
        );
        let mut buf = [0.25f32, -0.5, 0.1, 0.0];
        let original = buf;
        guard.process_interleaved(&mut buf, 1);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).abs() <= 1e-6);
        }
    }
}
