use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::DEFAULT_MODE_COUNT;
use crate::life::articulation_core::PinkNoise;
use crate::life::individual::ArticulationSignal;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};
use crate::life::sound::control::VoiceControlBlock;
use crate::life::sound::mode_utils::{
    active_cluster_voices, cluster_detune_mul, cluster_gain, cluster_spread_cents_from_public,
};
use crate::life::sound::spectral::{
    add_log2_energy, harmonic_gain, harmonic_ratio, spectral_slope_from_brightness,
};
use crate::life::sound::{BodyKind, BodySnapshot};
use crate::synth::SynthError;
use std::f32::consts::TAU;
use std::sync::Arc;

#[derive(Debug, Clone)]
enum OscillatorProfile {
    Sine,
    Harmonic {
        partials: usize,
        cluster_spread_cents: f32,
        cluster_voices: usize,
        genotype: TimbreGenotype,
        ratios: Option<Arc<[f32]>>,
    },
}

#[derive(Debug, Clone)]
pub struct OscillatorBank {
    fs: f32,
    profile: OscillatorProfile,
    x: Vec<f32>,
    y: Vec<f32>,
    rot_c: Vec<f32>,
    rot_s: Vec<f32>,
    gain: Vec<f32>,
    last_modes_len: usize,
    last_built_pitch_hz: f32,
    last_nominal_pitch_hz: f32,
    pending_phase_seed: Option<u64>,
    lfo_phase_rad: f32,
    jitter_gen: PinkNoise,
    drive_env: f32,
    drive_decay: f32,
    #[cfg(test)]
    rebuild_count: usize,
}

impl OscillatorBank {
    pub fn from_snapshot(fs: f32, snapshot: &BodySnapshot) -> Result<Self, SynthError> {
        if !fs.is_finite() || fs <= 0.0 {
            return Err(SynthError::InvalidSampleRate);
        }

        let profile = match snapshot.kind {
            BodyKind::Sine => OscillatorProfile::Sine,
            BodyKind::Harmonic => {
                let partials = snapshot
                    .ratios
                    .as_deref()
                    .map_or(DEFAULT_MODE_COUNT, |ratios| ratios.len())
                    .max(1);
                OscillatorProfile::Harmonic {
                    partials,
                    cluster_spread_cents: cluster_spread_cents_from_public(snapshot.spread),
                    cluster_voices: snapshot.voices,
                    genotype: TimbreGenotype {
                        mode: HarmonicMode::Harmonic,
                        stiffness: snapshot.inharmonic.clamp(0.0, 1.0),
                        spectral_slope: spectral_slope_from_brightness(snapshot.brightness),
                        comb: 0.0,
                        damping: 0.5,
                        vibrato_rate: 5.0,
                        vibrato_depth: snapshot.motion.clamp(0.0, 1.0) * 0.02,
                        jitter: snapshot.motion.clamp(0.0, 1.0),
                    },
                    ratios: snapshot.ratios.clone(),
                }
            }
            BodyKind::Modal => {
                debug_assert!(
                    !matches!(snapshot.kind, BodyKind::Modal),
                    "modal snapshots must not be routed to OscillatorBank"
                );
                OscillatorProfile::Sine
            }
        };

        let capacity = match &profile {
            OscillatorProfile::Sine => 1,
            OscillatorProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                ..
            } => (*partials).max(1) * active_cluster_voices(*cluster_spread_cents, *cluster_voices),
        }
        .max(1);

        Ok(Self {
            fs,
            profile,
            x: vec![1.0; capacity],
            y: vec![0.0; capacity],
            rot_c: vec![1.0; capacity],
            rot_s: vec![0.0; capacity],
            gain: vec![0.0; capacity],
            last_modes_len: 0,
            last_built_pitch_hz: 0.0,
            last_nominal_pitch_hz: 0.0,
            pending_phase_seed: None,
            lfo_phase_rad: 0.0,
            jitter_gen: PinkNoise::new(0xA5A5_5A5A_DEAD_BEEF, 0.001),
            drive_env: 0.0,
            drive_decay: (-1.0 / (0.08 * fs)).exp(),
            #[cfg(test)]
            rebuild_count: 0,
        })
    }

    pub fn is_sine(&self) -> bool {
        matches!(self.profile, OscillatorProfile::Sine)
    }

    pub fn seed_phases(&mut self, seed: u64) {
        self.pending_phase_seed = Some(seed);
        self.jitter_gen = PinkNoise::new(seed ^ 0xA5A5_5A5A_DEAD_BEEF, 0.001);
    }

    pub fn last_modes_len(&self) -> usize {
        self.last_modes_len
    }

    #[cfg(test)]
    pub fn rebuild_count(&self) -> usize {
        self.rebuild_count
    }

    fn partial_ratio(ratios: Option<&Arc<[f32]>>, genotype: &TimbreGenotype, idx: usize) -> f32 {
        if let Some(ratios) = ratios
            && let Some(&ratio) = ratios.get(idx)
        {
            return ratio.max(0.1);
        }
        harmonic_ratio(genotype, idx.saturating_add(1))
    }

    fn rebuild_sine(&mut self, pitch_hz: f32) {
        let dphi = TAU * pitch_hz / self.fs;
        self.rot_c[0] = dphi.cos();
        self.rot_s[0] = dphi.sin();
        self.gain[0] = 1.0;
        self.last_modes_len = 1;
    }

    fn rebuild_harmonic(
        &mut self,
        pitch_hz: f32,
        partials: usize,
        cluster_spread_cents: f32,
        cluster_voices: usize,
        genotype: &TimbreGenotype,
        ratios: Option<&Arc<[f32]>>,
    ) {
        let max_freq_hz = (self.fs * 0.49).max(1.0);
        let cluster_voices = active_cluster_voices(cluster_spread_cents, cluster_voices);
        let mut out_idx = 0usize;
        let energy = 1.0;
        for partial_idx in 0..partials.max(1) {
            let ratio = Self::partial_ratio(ratios, genotype, partial_idx);
            let base_gain = cluster_gain(
                harmonic_gain(genotype, partial_idx.saturating_add(1), energy),
                cluster_spread_cents,
                cluster_voices,
            );
            let mut any_below_nyquist = false;
            for voice_idx in 0..cluster_voices {
                let detune = cluster_detune_mul(cluster_spread_cents, cluster_voices, voice_idx);
                let freq_hz = pitch_hz * ratio * detune;
                if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz > max_freq_hz {
                    continue;
                }
                any_below_nyquist = true;
                let dphi = TAU * freq_hz / self.fs;
                self.rot_c[out_idx] = dphi.cos();
                self.rot_s[out_idx] = dphi.sin();
                self.gain[out_idx] = base_gain;
                out_idx += 1;
            }
            if !any_below_nyquist && pitch_hz * ratio > max_freq_hz {
                break;
            }
        }
        self.last_modes_len = out_idx;
    }

    fn apply_phase_seed_if_needed(&mut self) {
        let Some(mut state) = self.pending_phase_seed.take() else {
            return;
        };
        for idx in 0..self.last_modes_len {
            let phase = splitmix64_unit_f32(&mut state) * TAU;
            self.x[idx] = phase.cos();
            self.y[idx] = phase.sin();
        }
    }

    fn rebuild_oscillators(&mut self, pitch_hz: f32) {
        if !pitch_hz.is_finite() || pitch_hz <= 0.0 {
            self.last_modes_len = 0;
            self.last_built_pitch_hz = pitch_hz;
            self.last_nominal_pitch_hz = pitch_hz;
            return;
        }

        #[cfg(test)]
        {
            self.rebuild_count = self.rebuild_count.saturating_add(1);
        }

        let profile = self.profile.clone();
        match profile {
            OscillatorProfile::Sine => self.rebuild_sine(pitch_hz),
            OscillatorProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                genotype,
                ratios,
            } => self.rebuild_harmonic(
                pitch_hz,
                partials,
                cluster_spread_cents,
                cluster_voices,
                &genotype,
                ratios.as_ref(),
            ),
        }
        self.apply_phase_seed_if_needed();
        self.last_built_pitch_hz = pitch_hz;
        self.last_nominal_pitch_hz = pitch_hz;
    }

    fn requires_per_sample_rebuild(&self, ctrl: VoiceControlBlock) -> bool {
        ctrl.pitch_hz.step != 0.0
    }

    fn motion_phase_offset_rad(&mut self, amp: f32) -> f32 {
        match &self.profile {
            OscillatorProfile::Sine => 0.0,
            OscillatorProfile::Harmonic { genotype, .. } => {
                if self.last_nominal_pitch_hz <= 0.0 {
                    return 0.0;
                }
                let dphi = TAU * genotype.vibrato_rate.max(0.0) / self.fs;
                self.lfo_phase_rad = (self.lfo_phase_rad + dphi).rem_euclid(TAU);
                let vibrato = genotype.vibrato_depth * self.lfo_phase_rad.sin();
                let jitter_scale = amp.max(0.0) + 0.1;
                let jitter = self.jitter_gen.sample() * genotype.jitter * jitter_scale;
                let freq_offset_hz = self.last_nominal_pitch_hz * (vibrato + jitter);
                TAU * freq_offset_hz / self.fs
            }
        }
    }

    fn excitation_gain(&mut self, drive: f32) -> f32 {
        match &self.profile {
            OscillatorProfile::Sine => 1.0,
            OscillatorProfile::Harmonic { .. } => {
                let drive = drive.max(0.0);
                self.drive_env = (self.drive_env * self.drive_decay)
                    .max(drive)
                    .clamp(0.0, 4.0);
                1.0 + 0.25 * self.drive_env
            }
        }
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }
        let rebuild_each_sample = self.requires_per_sample_rebuild(ctrl);
        for (idx, y) in out.iter_mut().enumerate() {
            let pitch_hz = ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32;
            let amp = ctrl.amp.start + ctrl.amp.step * idx as f32;
            if idx == 0
                || self.last_modes_len == 0
                || rebuild_each_sample
                || (pitch_hz - self.last_nominal_pitch_hz).abs() > 1.0e-6
            {
                self.rebuild_oscillators(pitch_hz.max(1.0));
            }
            let motion_angle = self.motion_phase_offset_rad(amp.max(0.0));
            let use_motion = motion_angle.abs() > 1.0e-9;
            let (motion_s, motion_c) = if use_motion {
                motion_angle.sin_cos()
            } else {
                (0.0, 1.0)
            };
            let mut sample = 0.0;
            for mode_idx in 0..self.last_modes_len {
                let x0 = self.x[mode_idx];
                let y0 = self.y[mode_idx];
                let c = self.rot_c[mode_idx];
                let s = self.rot_s[mode_idx];
                let base_x = c * x0 - s * y0;
                let base_y = s * x0 + c * y0;
                let (x1, y1) = if use_motion {
                    (
                        motion_c * base_x - motion_s * base_y,
                        motion_s * base_x + motion_c * base_y,
                    )
                } else {
                    (base_x, base_y)
                };
                self.x[mode_idx] = x1;
                self.y[mode_idx] = y1;
                sample += self.gain[mode_idx] * y1;
            }
            if amp.is_finite() {
                let drive_gain = self.excitation_gain(drive.get(idx).copied().unwrap_or(0.0));
                *y += amp.max(0.0) * drive_gain * sample;
            }
        }
    }

    pub fn project_spectral(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        debug_assert_eq!(amps.len(), space.n_bins());
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let pitch_hz = self.last_built_pitch_hz;
        if !pitch_hz.is_finite() || pitch_hz <= 0.0 {
            return;
        }
        let amp_scale = signal.amplitude;
        if amp_scale <= 0.0 {
            return;
        }
        match &self.profile {
            OscillatorProfile::Sine => add_log2_energy(amps, space, pitch_hz, amp_scale),
            OscillatorProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                genotype,
                ratios,
            } => {
                let max_freq_hz = (self.fs * 0.49).max(1.0);
                let cluster_voices = active_cluster_voices(*cluster_spread_cents, *cluster_voices);
                let energy = 1.0;
                for partial_idx in 0..(*partials).max(1) {
                    let ratio = Self::partial_ratio(ratios.as_ref(), genotype, partial_idx);
                    let base_gain = cluster_gain(
                        harmonic_gain(genotype, partial_idx.saturating_add(1), energy),
                        *cluster_spread_cents,
                        cluster_voices,
                    );
                    let mut any_below_nyquist = false;
                    for voice_idx in 0..cluster_voices {
                        let detune =
                            cluster_detune_mul(*cluster_spread_cents, cluster_voices, voice_idx);
                        let freq_hz = pitch_hz * ratio * detune;
                        if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz > max_freq_hz {
                            continue;
                        }
                        any_below_nyquist = true;
                        add_log2_energy(amps, space, freq_hz, amp_scale * base_gain);
                    }
                    if !any_below_nyquist && pitch_hz * ratio > max_freq_hz {
                        break;
                    }
                }
            }
        }
    }
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix64_unit_f32(state: &mut u64) -> f32 {
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    let bits = splitmix64_next(state) >> 11;
    (bits as f64 * SCALE) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control::ControlRamp;

    #[test]
    fn oscillator_bank_renders_finite_harmonic_signal() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.2,
            ratios: None,
        };
        let mut backend = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        backend.seed_phases(123);
        let mut out = [0.0f32; 16];
        backend.render_block(
            &[0.0; 16],
            VoiceControlBlock {
                pitch_hz: ControlRamp {
                    start: 220.0,
                    step: 0.0,
                },
                amp: ControlRamp {
                    start: 0.5,
                    step: 0.0,
                },
            },
            &mut out,
        );
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(out.iter().any(|s| s.abs() > 1e-7));
    }

    #[test]
    fn oscillator_bank_limits_modes_by_nominal_pitch() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.0,
            ratios: None,
        };
        let mut backend = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        let mut out = [0.0f32; 1];
        backend.render_block(
            &[0.0],
            VoiceControlBlock {
                pitch_hz: ControlRamp {
                    start: 4_000.0,
                    step: 0.0,
                },
                amp: ControlRamp {
                    start: 0.0,
                    step: 0.0,
                },
            },
            &mut out,
        );
        assert_eq!(backend.last_modes_len(), 5);
    }

    #[test]
    fn harmonic_drive_changes_rendered_level() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.0,
            ratios: None,
        };
        let mut dry = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        dry.seed_phases(123);
        let mut wet = dry.clone();
        let mut dry_out = [0.0f32; 16];
        let mut wet_out = [0.0f32; 16];
        let ctrl = VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: 220.0,
                step: 0.0,
            },
            amp: ControlRamp {
                start: 0.5,
                step: 0.0,
            },
        };
        dry.render_block(&[0.0; 16], ctrl, &mut dry_out);
        wet.render_block(&[1.0; 16], ctrl, &mut wet_out);
        let dry_peak = dry_out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let wet_peak = wet_out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            wet_peak > dry_peak,
            "drive should raise level: dry={dry_peak} wet={wet_peak}"
        );
    }

    #[test]
    fn static_harmonic_block_rebuilds_once() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.0,
            ratios: None,
        };
        let mut backend = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        let mut out = [0.0f32; 8];
        backend.render_block(
            &[0.0; 8],
            VoiceControlBlock {
                pitch_hz: ControlRamp {
                    start: 220.0,
                    step: 0.0,
                },
                amp: ControlRamp {
                    start: 0.5,
                    step: 0.0,
                },
            },
            &mut out,
        );
        assert_eq!(backend.rebuild_count(), 1);
    }

    #[test]
    fn motion_harmonic_block_rebuilds_once() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion: 0.4,
            ratios: None,
        };
        let mut backend = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        let mut out = [0.0f32; 8];
        backend.render_block(
            &[0.0; 8],
            VoiceControlBlock {
                pitch_hz: ControlRamp {
                    start: 220.0,
                    step: 0.0,
                },
                amp: ControlRamp {
                    start: 0.5,
                    step: 0.0,
                },
            },
            &mut out,
        );
        assert_eq!(backend.rebuild_count(), 1);
    }
}
