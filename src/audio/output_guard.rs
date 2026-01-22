use std::env;
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
pub enum OutputGuardMode {
    None,
    SoftClip(SoftClipParams),
    PeakLimiter(PeakLimiterParams),
}

impl Default for OutputGuardMode {
    fn default() -> Self {
        Self::PeakLimiter(PeakLimiterParams::default())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutputGuardStats {
    pub max_abs_in: f32,
    pub max_abs_out: f32,
    pub max_reduction_db: f32,
    pub num_over: u64,
}

#[derive(Debug, Default)]
pub struct OutputGuardMeter {
    engaged_count: AtomicU64,
    over_count: AtomicU64,
    max_reduction_bits: AtomicU32,
    max_abs_in_bits: AtomicU32,
    max_abs_out_bits: AtomicU32,
}

impl OutputGuardMeter {
    pub fn record(&self, stats: &OutputGuardStats) {
        if stats.num_over == 0 && stats.max_reduction_db <= 0.0 {
            return;
        }
        self.engaged_count.fetch_add(1, Ordering::Relaxed);
        if stats.num_over > 0 {
            self.over_count.fetch_add(stats.num_over, Ordering::Relaxed);
        }
        self.max_reduction_bits
            .store(stats.max_reduction_db.to_bits(), Ordering::Relaxed);
        self.max_abs_in_bits
            .store(stats.max_abs_in.to_bits(), Ordering::Relaxed);
        self.max_abs_out_bits
            .store(stats.max_abs_out.to_bits(), Ordering::Relaxed);
    }

    pub fn take_snapshot(&self) -> Option<OutputGuardStats> {
        let engaged = self.engaged_count.swap(0, Ordering::Relaxed);
        if engaged == 0 {
            return None;
        }
        let over = self.over_count.swap(0, Ordering::Relaxed);
        let max_reduction_db = f32::from_bits(self.max_reduction_bits.swap(0, Ordering::Relaxed));
        let max_abs_in = f32::from_bits(self.max_abs_in_bits.swap(0, Ordering::Relaxed));
        let max_abs_out = f32::from_bits(self.max_abs_out_bits.swap(0, Ordering::Relaxed));
        Some(OutputGuardStats {
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
pub struct OutputGuard {
    mode: OutputGuardMode,
    channels: usize,
    limiter_state: Option<PeakLimiterState>,
    stats: OutputGuardStats,
    meter: Option<Arc<OutputGuardMeter>>,
}

impl OutputGuard {
    pub fn new(mode: OutputGuardMode, sample_rate: u32, channels: usize) -> Self {
        let sample_rate = (sample_rate as f32).max(1.0);
        let channels = channels.max(1);
        let limiter_state = match mode {
            OutputGuardMode::PeakLimiter(params) => {
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
            channels,
            limiter_state,
            stats: OutputGuardStats::default(),
            meter: None,
        }
    }

    pub fn with_meter(mut self, meter: Arc<OutputGuardMeter>) -> Self {
        self.meter = Some(meter);
        self
    }

    pub fn process_interleaved(&mut self, frames: &mut [f32], channels: usize) {
        if frames.is_empty() || channels == 0 {
            return;
        }
        debug_assert_eq!(channels, self.channels);
        self.stats = OutputGuardStats::default();
        match self.mode {
            OutputGuardMode::None => {}
            OutputGuardMode::SoftClip(params) => {
                let ceiling = params.ceiling.abs().max(1e-6);
                let drive = params.drive.max(0.0);
                for s in frames.iter_mut() {
                    let x = if s.is_finite() { *s } else { 0.0 };
                    let abs_in = x.abs();
                    if abs_in > self.stats.max_abs_in {
                        self.stats.max_abs_in = abs_in;
                    }
                    if abs_in > ceiling {
                        self.stats.num_over += 1;
                    }
                    let y = (x * drive).tanh() * ceiling;
                    let abs_out = y.abs();
                    if abs_out > self.stats.max_abs_out {
                        self.stats.max_abs_out = abs_out;
                    }
                    update_reduction(&mut self.stats, abs_in, abs_out);
                    *s = y;
                }
            }
            OutputGuardMode::PeakLimiter(params) => {
                let ceiling = params.ceiling.abs().max(1e-6);
                let n_frames = frames.len() / channels;
                let state = self.limiter_state.as_mut().expect("limiter state");
                if params.link_channels {
                    for frame in 0..n_frames {
                        let mut peak = 0.0f32;
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let x = frames[idx];
                            let abs_in = if x.is_finite() { x.abs() } else { 0.0 };
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
                            let x = if frames[idx].is_finite() {
                                frames[idx]
                            } else {
                                0.0
                            };
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
                        return;
                    }
                    for frame in 0..n_frames {
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let x = if frames[idx].is_finite() {
                                frames[idx]
                            } else {
                                0.0
                            };
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

    pub fn stats(&self) -> OutputGuardStats {
        self.stats
    }

    pub fn from_env_or(config_mode: OutputGuardMode) -> OutputGuardMode {
        mode_from_env().unwrap_or(config_mode)
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

fn update_reduction(stats: &mut OutputGuardStats, abs_in: f32, abs_out: f32) {
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

fn mode_from_env() -> Option<OutputGuardMode> {
    let value = env::var("CONCHORDAL_OUTPUT_GUARD").ok()?;
    let norm = value.trim().to_ascii_lowercase();
    match norm.as_str() {
        "none" | "off" | "0" => Some(OutputGuardMode::None),
        "soft" | "softclip" => Some(OutputGuardMode::SoftClip(SoftClipParams::default())),
        "limiter" | "peak" | "peaklimiter" => {
            Some(OutputGuardMode::PeakLimiter(PeakLimiterParams::default()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_softclip() {
        let mut guard = OutputGuard::new(
            OutputGuardMode::SoftClip(SoftClipParams::default()),
            48_000,
            1,
        );
        let mut buf = [0.0f32, 1.5, -1.5, 0.5];
        guard.process_interleaved(&mut buf, 1);
        let ceiling = SoftClipParams::default().ceiling + 1e-6;
        for &v in &buf {
            assert!(v.abs() <= ceiling, "{v} exceeds ceiling");
        }
    }

    #[test]
    fn safety_limiter() {
        let mut guard = OutputGuard::new(
            OutputGuardMode::PeakLimiter(PeakLimiterParams::default()),
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
        let mut guard = OutputGuard::new(OutputGuardMode::None, 48_000, 1);
        let mut buf = [0.25f32, -0.5, 0.1, 0.0];
        let original = buf;
        guard.process_interleaved(&mut buf, 1);
        assert_eq!(buf, original);
    }

    #[test]
    fn transparency_limiter() {
        let mut guard = OutputGuard::new(
            OutputGuardMode::PeakLimiter(PeakLimiterParams::default()),
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
