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
#[cfg(feature = "simd-wide")]
use wide::f32x8;

const PITCH_REFRESH_PERIOD_SAMPLES: usize = 64;
const MOTION_REFRESH_PERIOD_SAMPLES: usize = 8;

#[cfg(any(target_feature = "fma", target_feature = "neon"))]
#[inline(always)]
fn mul_add_fast(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(any(target_feature = "fma", target_feature = "neon")))]
#[inline(always)]
fn mul_add_fast(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[derive(Debug, Clone)]
enum OscillatorProfile {
    Sine,
    Harmonic { genotype: TimbreGenotype },
}

#[derive(Debug, Clone)]
pub struct OscillatorBank {
    fs: f32,
    profile: OscillatorProfile,
    freq_mul: Vec<f32>,
    base_gain: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    rot_c: Vec<f32>,
    rot_s: Vec<f32>,
    gain_mask: Vec<f32>,
    lane_len_real: usize,
    lane_len_simd: usize,
    active_lane_len: usize,
    last_modes_len: usize,
    last_built_pitch_hz: f32,
    pending_phase_seed: Option<u64>,
    lfo_phase_rad: f32,
    jitter_gen: PinkNoise,
    drive_env: f32,
    drive_decay: f32,
    pitch_counter: usize,
    motion_counter: usize,
    current_motion_s: f32,
    current_motion_c: f32,
    current_motion_enabled: bool,
    #[cfg(test)]
    rebuild_count: usize,
    #[cfg(test)]
    motion_refresh_count: usize,
}

impl OscillatorBank {
    pub fn from_snapshot(fs: f32, snapshot: &BodySnapshot) -> Result<Self, SynthError> {
        if !fs.is_finite() || fs <= 0.0 {
            return Err(SynthError::InvalidSampleRate);
        }

        let (profile, freq_mul, base_gain, lane_len_real) = match snapshot.kind {
            BodyKind::Sine => {
                let (freq_mul, base_gain, lane_len_real) = build_sine_lanes();
                (OscillatorProfile::Sine, freq_mul, base_gain, lane_len_real)
            }
            BodyKind::Harmonic => {
                let genotype = TimbreGenotype {
                    mode: HarmonicMode::Harmonic,
                    stiffness: snapshot.inharmonic.clamp(0.0, 1.0),
                    spectral_slope: spectral_slope_from_brightness(snapshot.brightness),
                    comb: 0.0,
                    damping: 0.5,
                    vibrato_rate: 5.0,
                    vibrato_depth: snapshot.motion.clamp(0.0, 1.0) * 0.02,
                    jitter: snapshot.motion.clamp(0.0, 1.0),
                };
                let partials = snapshot
                    .ratios
                    .as_deref()
                    .map_or(DEFAULT_MODE_COUNT, |ratios| ratios.len())
                    .max(1);
                let cluster_spread_cents = cluster_spread_cents_from_public(snapshot.spread);
                let (freq_mul, base_gain, lane_len_real) = build_harmonic_lanes(
                    partials,
                    cluster_spread_cents,
                    snapshot.voices,
                    &genotype,
                    snapshot.ratios.as_ref(),
                );
                (
                    OscillatorProfile::Harmonic { genotype },
                    freq_mul,
                    base_gain,
                    lane_len_real,
                )
            }
            BodyKind::Modal => {
                debug_assert!(
                    !matches!(snapshot.kind, BodyKind::Modal),
                    "modal snapshots must not be routed to OscillatorBank"
                );
                let (freq_mul, base_gain, lane_len_real) = build_sine_lanes();
                (OscillatorProfile::Sine, freq_mul, base_gain, lane_len_real)
            }
        };

        let lane_len_simd = round_up_to_8(lane_len_real.max(1));
        let mut freq_mul_padded = vec![0.0; lane_len_simd];
        let mut base_gain_padded = vec![0.0; lane_len_simd];
        freq_mul_padded[..lane_len_real].copy_from_slice(&freq_mul);
        base_gain_padded[..lane_len_real].copy_from_slice(&base_gain);

        Ok(Self {
            fs,
            profile,
            freq_mul: freq_mul_padded,
            base_gain: base_gain_padded,
            x: padded_unit_x(lane_len_simd),
            y: vec![0.0; lane_len_simd],
            rot_c: padded_unit_x(lane_len_simd),
            rot_s: vec![0.0; lane_len_simd],
            gain_mask: vec![0.0; lane_len_simd],
            lane_len_real,
            lane_len_simd,
            active_lane_len: 0,
            last_modes_len: 0,
            last_built_pitch_hz: 0.0,
            pending_phase_seed: None,
            lfo_phase_rad: 0.0,
            jitter_gen: PinkNoise::new(0xA5A5_5A5A_DEAD_BEEF, 0.001),
            drive_env: 0.0,
            drive_decay: (-1.0 / (0.08 * fs)).exp(),
            pitch_counter: 0,
            motion_counter: 0,
            current_motion_s: 0.0,
            current_motion_c: 1.0,
            current_motion_enabled: false,
            #[cfg(test)]
            rebuild_count: 0,
            #[cfg(test)]
            motion_refresh_count: 0,
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

    #[cfg(test)]
    pub fn motion_refresh_count(&self) -> usize {
        self.motion_refresh_count
    }

    fn apply_phase_seed_if_needed(&mut self) {
        let Some(mut state) = self.pending_phase_seed.take() else {
            return;
        };
        for idx in 0..self.lane_len_real {
            let phase = splitmix64_unit_f32(&mut state) * TAU;
            self.x[idx] = phase.cos();
            self.y[idx] = phase.sin();
        }
    }

    fn refresh_pitch_state(&mut self, nominal_pitch_hz: f32) {
        #[cfg(test)]
        {
            self.rebuild_count = self.rebuild_count.saturating_add(1);
        }

        if !nominal_pitch_hz.is_finite() || nominal_pitch_hz <= 0.0 {
            self.gain_mask.fill(0.0);
            self.active_lane_len = 0;
            self.last_modes_len = 0;
            self.last_built_pitch_hz = nominal_pitch_hz;
            return;
        }

        let max_freq_hz = (self.fs * 0.49).max(1.0);

        self.active_lane_len = 0;
        self.last_modes_len = 0;
        for idx in 0..self.lane_len_real {
            let freq_hz = nominal_pitch_hz * self.freq_mul[idx];
            let dphi = TAU * freq_hz / self.fs;
            self.rot_c[idx] = dphi.cos();
            self.rot_s[idx] = dphi.sin();

            let is_active = freq_hz.is_finite() && freq_hz > 0.0 && freq_hz <= max_freq_hz;
            self.gain_mask[idx] = if is_active { self.base_gain[idx] } else { 0.0 };
            if is_active {
                self.last_modes_len += 1;
                self.active_lane_len = idx + 1;
            }
        }
        for idx in self.lane_len_real..self.lane_len_simd {
            self.rot_c[idx] = 1.0;
            self.rot_s[idx] = 0.0;
            self.gain_mask[idx] = 0.0;
        }

        self.apply_phase_seed_if_needed();
        self.last_built_pitch_hz = nominal_pitch_hz;
    }

    fn refresh_motion_state(&mut self, amp: f32, nominal_pitch_hz: f32) {
        #[cfg(test)]
        {
            self.motion_refresh_count = self.motion_refresh_count.saturating_add(1);
        }

        let motion_angle = self.motion_phase_offset_rad(amp, nominal_pitch_hz);
        self.current_motion_enabled = motion_angle.abs() > 1.0e-9;
        if self.current_motion_enabled {
            let (motion_s, motion_c) = motion_angle.sin_cos();
            self.current_motion_s = motion_s;
            self.current_motion_c = motion_c;
        } else {
            self.current_motion_s = 0.0;
            self.current_motion_c = 1.0;
        }
    }

    fn needs_pitch_state_refresh(&self, nominal_pitch_hz: f32) -> bool {
        if self.last_modes_len == 0 {
            return true;
        }
        if !nominal_pitch_hz.is_finite() || !self.last_built_pitch_hz.is_finite() {
            return nominal_pitch_hz.to_bits() != self.last_built_pitch_hz.to_bits();
        }
        (nominal_pitch_hz - self.last_built_pitch_hz).abs() > 1.0e-6
    }

    fn motion_phase_offset_rad(&mut self, amp: f32, nominal_pitch_hz: f32) -> f32 {
        match &self.profile {
            OscillatorProfile::Sine => 0.0,
            OscillatorProfile::Harmonic { genotype } => {
                if nominal_pitch_hz <= 0.0 || !nominal_pitch_hz.is_finite() {
                    return 0.0;
                }
                let dphi =
                    TAU * genotype.vibrato_rate.max(0.0) * MOTION_REFRESH_PERIOD_SAMPLES as f32
                        / self.fs;
                self.lfo_phase_rad = (self.lfo_phase_rad + dphi).rem_euclid(TAU);
                let vibrato = genotype.vibrato_depth * self.lfo_phase_rad.sin();
                let jitter_scale = amp.max(0.0) + 0.1;
                let jitter = self.jitter_gen.sample() * genotype.jitter * jitter_scale;
                let freq_offset_hz = nominal_pitch_hz * (vibrato + jitter);
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

    #[allow(dead_code)]
    fn process_sample_basic(&mut self, use_motion: bool, motion_s: f32, motion_c: f32) -> f32 {
        let mut out = 0.0;
        for idx in 0..self.active_lane_len {
            let x0 = self.x[idx];
            let y0 = self.y[idx];
            let c = self.rot_c[idx];
            let s = self.rot_s[idx];

            let base_x = mul_add_fast(c, x0, -s * y0);
            let base_y = mul_add_fast(s, x0, c * y0);
            let (x1, y1) = if use_motion {
                (
                    mul_add_fast(motion_c, base_x, -motion_s * base_y),
                    mul_add_fast(motion_s, base_x, motion_c * base_y),
                )
            } else {
                (base_x, base_y)
            };

            self.x[idx] = x1;
            self.y[idx] = y1;
            out = mul_add_fast(self.gain_mask[idx], y1, out);
        }
        out
    }

    #[cfg(feature = "simd-wide")]
    fn process_sample_optimized(&mut self, use_motion: bool, motion_s: f32, motion_c: f32) -> f32 {
        if self.active_lane_len == 0 {
            return 0.0;
        }

        let x_ptr = self.x.as_mut_ptr();
        let y_ptr = self.y.as_mut_ptr();
        let rot_c_ptr = self.rot_c.as_ptr();
        let rot_s_ptr = self.rot_s.as_ptr();
        let gain_ptr = self.gain_mask.as_ptr();
        let motion_s_vec = f32x8::splat(motion_s);
        let motion_c_vec = f32x8::splat(motion_c);
        let mut out = 0.0;
        let simd_len = self.active_lane_len & !7;

        // Safety: all arrays have length >= lane_len_simd. The SIMD loop only accesses the
        // active prefix rounded down to a multiple of 8.
        unsafe {
            for idx in (0..simd_len).step_by(8) {
                let x0_arr = std::ptr::read_unaligned(x_ptr.add(idx) as *const [f32; 8]);
                let y0_arr = std::ptr::read_unaligned(y_ptr.add(idx) as *const [f32; 8]);
                let rot_c_arr = std::ptr::read_unaligned(rot_c_ptr.add(idx) as *const [f32; 8]);
                let rot_s_arr = std::ptr::read_unaligned(rot_s_ptr.add(idx) as *const [f32; 8]);
                let gain_arr = std::ptr::read_unaligned(gain_ptr.add(idx) as *const [f32; 8]);

                let x0 = f32x8::from(x0_arr);
                let y0 = f32x8::from(y0_arr);
                let c = f32x8::from(rot_c_arr);
                let s = f32x8::from(rot_s_arr);
                let base_x = c.mul_add(x0, -(s * y0));
                let base_y = s.mul_add(x0, c * y0);
                let (x1_vec, y1_vec) = if use_motion {
                    (
                        motion_c_vec.mul_add(base_x, -(motion_s_vec * base_y)),
                        motion_s_vec.mul_add(base_x, motion_c_vec * base_y),
                    )
                } else {
                    (base_x, base_y)
                };

                let x1_arr = x1_vec.to_array();
                let y1_arr = y1_vec.to_array();
                std::ptr::write_unaligned(x_ptr.add(idx) as *mut [f32; 8], x1_arr);
                std::ptr::write_unaligned(y_ptr.add(idx) as *mut [f32; 8], y1_arr);

                for lane in 0..8 {
                    out = mul_add_fast(gain_arr[lane], y1_arr[lane], out);
                }
            }
        }

        for idx in simd_len..self.active_lane_len {
            let x0 = self.x[idx];
            let y0 = self.y[idx];
            let c = self.rot_c[idx];
            let s = self.rot_s[idx];

            let base_x = mul_add_fast(c, x0, -s * y0);
            let base_y = mul_add_fast(s, x0, c * y0);
            let (x1, y1) = if use_motion {
                (
                    mul_add_fast(motion_c, base_x, -motion_s * base_y),
                    mul_add_fast(motion_s, base_x, motion_c * base_y),
                )
            } else {
                (base_x, base_y)
            };

            self.x[idx] = x1;
            self.y[idx] = y1;
            out = mul_add_fast(self.gain_mask[idx], y1, out);
        }

        out
    }

    #[cfg(all(
        feature = "simd-wide",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn process_sample(&mut self, use_motion: bool, motion_s: f32, motion_c: f32) -> f32 {
        self.process_sample_optimized(use_motion, motion_s, motion_c)
    }

    #[cfg(not(all(
        feature = "simd-wide",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    fn process_sample(&mut self, use_motion: bool, motion_s: f32, motion_c: f32) -> f32 {
        self.process_sample_basic(use_motion, motion_s, motion_c)
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }

        for (idx, y) in out.iter_mut().enumerate() {
            let pitch_hz = ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32;
            let nominal_pitch_hz = if pitch_hz.is_finite() {
                pitch_hz.max(1.0)
            } else {
                pitch_hz
            };
            let amp = ctrl.amp.start + ctrl.amp.step * idx as f32;
            if self.pitch_counter == 0 || self.needs_pitch_state_refresh(nominal_pitch_hz) {
                self.refresh_pitch_state(nominal_pitch_hz);
            }
            if self.motion_counter == 0 {
                self.refresh_motion_state(amp.max(0.0), nominal_pitch_hz);
            }

            let sample = self.process_sample(
                self.current_motion_enabled,
                self.current_motion_s,
                self.current_motion_c,
            );
            if amp.is_finite() {
                let drive_gain = self.excitation_gain(drive.get(idx).copied().unwrap_or(0.0));
                *y += amp.max(0.0) * drive_gain * sample;
            }

            self.pitch_counter = (self.pitch_counter + 1) % PITCH_REFRESH_PERIOD_SAMPLES;
            self.motion_counter = (self.motion_counter + 1) % MOTION_REFRESH_PERIOD_SAMPLES;
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

        let max_freq_hz = (self.fs * 0.49).max(1.0);
        for idx in 0..self.lane_len_real {
            let freq_hz = pitch_hz * self.freq_mul[idx];
            if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz > max_freq_hz {
                continue;
            }
            add_log2_energy(amps, space, freq_hz, amp_scale * self.base_gain[idx]);
        }
    }

    #[cfg(test)]
    fn render_block_for_test(
        &mut self,
        drive: &[f32],
        ctrl: crate::life::sound::control::VoiceControlBlock,
        out: &mut [f32],
        use_optimized: bool,
    ) {
        if out.is_empty() {
            return;
        }

        for (idx, y) in out.iter_mut().enumerate() {
            let pitch_hz = ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32;
            let nominal_pitch_hz = if pitch_hz.is_finite() {
                pitch_hz.max(1.0)
            } else {
                pitch_hz
            };
            let amp = ctrl.amp.start + ctrl.amp.step * idx as f32;
            if self.pitch_counter == 0 || self.needs_pitch_state_refresh(nominal_pitch_hz) {
                self.refresh_pitch_state(nominal_pitch_hz);
            }
            if self.motion_counter == 0 {
                self.refresh_motion_state(amp.max(0.0), nominal_pitch_hz);
            }

            let sample = if use_optimized {
                #[cfg(feature = "simd-wide")]
                {
                    self.process_sample_optimized(
                        self.current_motion_enabled,
                        self.current_motion_s,
                        self.current_motion_c,
                    )
                }
                #[cfg(not(feature = "simd-wide"))]
                {
                    self.process_sample_basic(
                        self.current_motion_enabled,
                        self.current_motion_s,
                        self.current_motion_c,
                    )
                }
            } else {
                self.process_sample_basic(
                    self.current_motion_enabled,
                    self.current_motion_s,
                    self.current_motion_c,
                )
            };

            if amp.is_finite() {
                let drive_gain = self.excitation_gain(drive.get(idx).copied().unwrap_or(0.0));
                *y += amp.max(0.0) * drive_gain * sample;
            }

            self.pitch_counter = (self.pitch_counter + 1) % PITCH_REFRESH_PERIOD_SAMPLES;
            self.motion_counter = (self.motion_counter + 1) % MOTION_REFRESH_PERIOD_SAMPLES;
        }
    }
}

fn build_sine_lanes() -> (Vec<f32>, Vec<f32>, usize) {
    (vec![1.0], vec![1.0], 1)
}

fn build_harmonic_lanes(
    partials: usize,
    cluster_spread_cents: f32,
    cluster_voices: usize,
    genotype: &TimbreGenotype,
    ratios: Option<&Arc<[f32]>>,
) -> (Vec<f32>, Vec<f32>, usize) {
    let cluster_voices = active_cluster_voices(cluster_spread_cents, cluster_voices);
    let lane_len_real = partials.max(1) * cluster_voices.max(1);
    let mut lanes = Vec::with_capacity(lane_len_real);
    let energy = 1.0;

    for partial_idx in 0..partials.max(1) {
        let ratio = partial_ratio(ratios, genotype, partial_idx);
        let partial_gain = cluster_gain(
            harmonic_gain(genotype, partial_idx.saturating_add(1), energy),
            cluster_spread_cents,
            cluster_voices,
        );
        for voice_idx in 0..cluster_voices {
            let detune = cluster_detune_mul(cluster_spread_cents, cluster_voices, voice_idx);
            lanes.push((ratio * detune, partial_gain));
        }
    }

    lanes.sort_by(|(lhs, _), (rhs, _)| lhs.total_cmp(rhs));
    let (freq_mul, base_gain): (Vec<_>, Vec<_>) = lanes.into_iter().unzip();
    (freq_mul, base_gain, lane_len_real)
}

fn partial_ratio(ratios: Option<&Arc<[f32]>>, genotype: &TimbreGenotype, idx: usize) -> f32 {
    if let Some(ratios) = ratios
        && let Some(&ratio) = ratios.get(idx)
    {
        return ratio.max(0.1);
    }
    harmonic_ratio(genotype, idx.saturating_add(1))
}

fn padded_unit_x(len: usize) -> Vec<f32> {
    vec![1.0; len]
}

fn round_up_to_8(len: usize) -> usize {
    (len + 7) & !7
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
    use crate::life::sound::control::{ControlRamp, VoiceControlBlock};

    fn harmonic_snapshot(motion: f32) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.0,
            voices: 1,
            motion,
            ratios: None,
        }
    }

    #[test]
    fn oscillator_bank_renders_finite_harmonic_signal() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.2)).expect("backend");
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
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
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
        let snapshot = harmonic_snapshot(0.0);
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
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
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
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.4)).expect("backend");
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
    fn pitch_refresh_runs_every_sixty_four_samples() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
        let mut out = [0.0f32; 128];
        backend.render_block(
            &[0.0; 128],
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
        assert_eq!(backend.rebuild_count(), 2);
    }

    #[test]
    fn motion_refresh_runs_every_eight_samples() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.4)).expect("backend");
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
        assert_eq!(backend.motion_refresh_count(), 2);
    }

    #[test]
    fn motion_lfo_phase_matches_refresh_cadence() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.4)).expect("backend");
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
        let expected = 2.0 * TAU * 5.0 * MOTION_REFRESH_PERIOD_SAMPLES as f32 / 48_000.0;
        assert!((backend.lfo_phase_rad - expected).abs() <= 1.0e-6);
    }

    #[test]
    fn pitch_start_change_refreshes_immediately() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
        let ctrl_a = VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: 220.0,
                step: 0.0,
            },
            amp: ControlRamp {
                start: 0.5,
                step: 0.0,
            },
        };
        let ctrl_b = VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: 330.0,
                step: 0.0,
            },
            amp: ControlRamp {
                start: 0.5,
                step: 0.0,
            },
        };
        let mut out = [0.0f32; 1];
        backend.render_block(&[0.0], ctrl_a, &mut out);
        backend.render_block(&[0.0], ctrl_b, &mut out);
        assert_eq!(backend.rebuild_count(), 2);
    }

    #[test]
    fn conservative_nyquist_mask_zeros_high_lanes() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
        backend.refresh_pitch_state(4_000.0);
        assert_eq!(
            backend.gain_mask[..backend.lane_len_real]
                .iter()
                .filter(|gain| **gain > 0.0)
                .count(),
            5
        );
    }

    #[test]
    fn upward_pitch_ramp_keeps_partials_until_actual_nyquist_crossing() {
        let mut backend =
            OscillatorBank::from_snapshot(48_000.0, &harmonic_snapshot(0.0)).expect("backend");
        backend.refresh_pitch_state(3_900.0);
        assert_eq!(backend.last_modes_len(), 6);
        assert_eq!(backend.active_lane_len, 6);
    }

    #[test]
    fn padding_lanes_stay_silent() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.5,
            voices: 3,
            motion: 0.0,
            ratios: Some(Arc::<[f32]>::from(vec![1.0, 2.0, 3.0])),
        };
        let backend = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("backend");
        assert_eq!(backend.lane_len_real, 9);
        assert_eq!(backend.lane_len_simd, 16);
        assert!(backend.base_gain[9..].iter().all(|gain| *gain == 0.0));
    }

    #[cfg(all(
        feature = "simd-wide",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn optimized_kernel_matches_basic_kernel() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            inharmonic: 0.0,
            spread: 0.5,
            voices: 3,
            motion: 0.4,
            ratios: Some(Arc::<[f32]>::from(vec![1.0, 2.3, 3.7])),
        };
        let ctrl = VoiceControlBlock {
            pitch_hz: ControlRamp {
                start: 220.0,
                step: 1.25,
            },
            amp: ControlRamp {
                start: 0.5,
                step: -0.001,
            },
        };
        let drive = vec![0.15; 96];
        let mut basic = OscillatorBank::from_snapshot(48_000.0, &snapshot).expect("basic");
        let mut optimized = basic.clone();
        basic.seed_phases(42);
        optimized.seed_phases(42);
        let mut out_basic = vec![0.0f32; drive.len()];
        let mut out_optimized = vec![0.0f32; drive.len()];

        basic.render_block_for_test(&drive, ctrl, &mut out_basic, false);
        optimized.render_block_for_test(&drive, ctrl, &mut out_optimized, true);

        for (lhs, rhs) in out_basic.iter().zip(out_optimized.iter()) {
            assert!(
                (lhs - rhs).abs() <= 1.0e-6,
                "output mismatch: {lhs} vs {rhs}"
            );
        }
        for idx in 0..basic.lane_len_simd {
            assert!(
                (basic.x[idx] - optimized.x[idx]).abs() <= 1.0e-6,
                "x mismatch at lane {idx}: {} vs {}",
                basic.x[idx],
                optimized.x[idx]
            );
            assert!(
                (basic.y[idx] - optimized.y[idx]).abs() <= 1.0e-6,
                "y mismatch at lane {idx}: {} vs {}",
                basic.y[idx],
                optimized.y[idx]
            );
        }
    }
}
