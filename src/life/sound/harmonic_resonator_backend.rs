use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::DEFAULT_MODE_COUNT;
use crate::life::individual::ArticulationSignal;
use crate::life::scenario::TimbreGenotype;
use crate::life::sound::BodySnapshot;
use crate::life::sound::control::VoiceControlBlock;
use crate::life::sound::modal_engine::ModalMode;
use crate::life::sound::mode_utils::{
    active_cluster_voices, cluster_detune_mul, cluster_gain, cluster_spread_cents_from_public,
    modal_modes_from_ratios, modal_tilt_from_brightness,
};
use crate::life::sound::spectral::{
    add_log2_energy, harmonic_gain, harmonic_ratio, spectral_slope_from_brightness,
};
use crate::synth::SynthError;
use crate::synth::modes::ModeParams;
use crate::synth::resonator::ResonatorBank;

const DEFAULT_UPDATE_PERIOD_SAMPLES: usize = 64;

#[derive(Debug, Clone)]
enum HarmonicProfile {
    Harmonic {
        partials: usize,
        cluster_spread_cents: f32,
        cluster_voices: usize,
        base_t60_s: f32,
        in_gain: f32,
        genotype: TimbreGenotype,
    },
    Ratios {
        modes: Vec<ModalMode>,
    },
}

#[derive(Debug, Clone)]
pub struct HarmonicResonatorBackend {
    bank: ResonatorBank,
    profile: HarmonicProfile,
    scratch: Vec<ModeParams>,
    last_built_pitch_hz: f32,
    update_period_samples: usize,
    counter: usize,
    last_modes_len: usize,
    pending_phase_seed: Option<u64>,
}

impl HarmonicResonatorBackend {
    pub fn from_snapshot(fs: f32, snapshot: &BodySnapshot) -> Result<Self, SynthError> {
        let brightness = snapshot.brightness.clamp(0.0, 1.0);
        let cluster_spread_cents = cluster_spread_cents_from_public(snapshot.spread);
        let profile = if let Some(ratios) = snapshot.ratios.as_deref() {
            HarmonicProfile::Ratios {
                modes: modal_modes_from_ratios(
                    ratios,
                    modal_tilt_from_brightness(brightness),
                    cluster_spread_cents,
                    snapshot.voices,
                ),
            }
        } else {
            HarmonicProfile::Harmonic {
                partials: DEFAULT_MODE_COUNT,
                cluster_spread_cents,
                cluster_voices: snapshot.voices,
                base_t60_s: 0.8,
                in_gain: 1.0,
                genotype: TimbreGenotype {
                    spectral_slope: spectral_slope_from_brightness(brightness),
                    jitter: snapshot.noise_mix.clamp(0.0, 1.0),
                    ..TimbreGenotype::default()
                },
            }
        };
        let max_modes = match &profile {
            HarmonicProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                ..
            } => (*partials).max(1) * active_cluster_voices(*cluster_spread_cents, *cluster_voices),
            HarmonicProfile::Ratios { modes } => modes.len().max(1),
        };
        let bank = ResonatorBank::new(fs, max_modes)?;
        let scratch = Vec::with_capacity(max_modes);
        Ok(Self {
            bank,
            profile,
            scratch,
            last_built_pitch_hz: 0.0,
            update_period_samples: DEFAULT_UPDATE_PERIOD_SAMPLES.max(1),
            counter: 0,
            last_modes_len: 0,
            pending_phase_seed: None,
        })
    }

    pub fn seed_phases(&mut self, seed: u64) {
        self.pending_phase_seed = Some(seed);
    }

    pub fn last_modes_len(&self) -> usize {
        self.last_modes_len
    }

    fn rebuild_modes(&mut self, pitch_hz: f32) {
        if !pitch_hz.is_finite() || pitch_hz <= 0.0 {
            self.last_modes_len = 0;
            let _ = self.bank.set_modes_preserve_state(&[]);
            self.last_built_pitch_hz = pitch_hz;
            return;
        }

        let limit = self.bank.capacity();
        let max_freq_hz = (self.bank.fs() * 0.49).max(1.0);
        self.scratch.clear();
        match &self.profile {
            HarmonicProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                base_t60_s,
                in_gain,
                genotype,
            } => {
                let energy = 1.0;
                let cluster_voices = active_cluster_voices(*cluster_spread_cents, *cluster_voices);
                for k in 1..=(*partials).max(1) {
                    let ratio = harmonic_ratio(genotype, k);
                    let gain = cluster_gain(
                        harmonic_gain(genotype, k, energy),
                        *cluster_spread_cents,
                        cluster_voices,
                    );
                    let t60_s = base_t60_s.max(1e-3) / (1.0 + 0.15 * k as f32);
                    let mut any_below_nyquist = false;
                    for voice_idx in 0..cluster_voices {
                        if self.scratch.len() >= limit {
                            break;
                        }
                        let detune =
                            cluster_detune_mul(*cluster_spread_cents, cluster_voices, voice_idx);
                        let freq_hz = pitch_hz * ratio * detune;
                        if !freq_hz.is_finite() || freq_hz <= 0.0 {
                            continue;
                        }
                        if freq_hz > max_freq_hz {
                            continue;
                        }
                        any_below_nyquist = true;
                        self.scratch.push(ModeParams {
                            freq_hz,
                            t60_s,
                            gain,
                            in_gain: *in_gain,
                        });
                    }
                    if !any_below_nyquist && pitch_hz * ratio > max_freq_hz {
                        break;
                    }
                }
            }
            HarmonicProfile::Ratios { modes } => {
                for mode in modes {
                    if self.scratch.len() >= limit {
                        break;
                    }
                    let freq_hz = pitch_hz * mode.ratio;
                    if !freq_hz.is_finite() || freq_hz <= 0.0 {
                        continue;
                    }
                    if freq_hz > max_freq_hz {
                        break;
                    }
                    self.scratch.push(ModeParams {
                        freq_hz,
                        t60_s: mode.t60_s.max(1e-3),
                        gain: mode.gain,
                        in_gain: mode.in_gain,
                    });
                }
            }
        }

        self.last_modes_len = self.scratch.len();
        let _ = self
            .bank
            .set_modes_preserve_state(&self.scratch[..self.last_modes_len]);
        if let Some(seed) = self.pending_phase_seed.take() {
            self.bank.randomize_input_phase_from_seed(seed);
        }
        self.last_built_pitch_hz = pitch_hz;
    }

    pub fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        debug_assert_eq!(drive.len(), out.len());
        if drive.is_empty() {
            return;
        }
        let mut counter = self.counter;
        let period = self.update_period_samples.max(1);
        for (idx, (u, y)) in drive.iter().copied().zip(out.iter_mut()).enumerate() {
            let pitch_hz = ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32;
            let amp = ctrl.amp.start + ctrl.amp.step * idx as f32;
            if counter == 0 {
                self.rebuild_modes(pitch_hz.max(1.0));
            }
            let sample = self.bank.process_sample(u);
            if amp.is_finite() {
                *y += amp.max(0.0) * sample;
            }
            counter += 1;
            if counter >= period {
                counter = 0;
            }
        }
        self.counter = counter;
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
        let max_freq_hz = (self.bank.fs() * 0.49).max(1.0);
        match &self.profile {
            HarmonicProfile::Harmonic {
                partials,
                cluster_spread_cents,
                cluster_voices,
                genotype,
                ..
            } => {
                let partials = (*partials).max(1);
                let energy = 1.0;
                let cluster_voices = active_cluster_voices(*cluster_spread_cents, *cluster_voices);
                for k in 1..=partials {
                    let ratio = harmonic_ratio(genotype, k);
                    let gain = cluster_gain(
                        harmonic_gain(genotype, k, energy),
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
                        add_log2_energy(amps, space, freq_hz, amp_scale * gain);
                    }
                    if !any_below_nyquist && pitch_hz * ratio > max_freq_hz {
                        break;
                    }
                }
            }
            HarmonicProfile::Ratios { modes } => {
                for mode in modes {
                    let freq_hz = pitch_hz * mode.ratio;
                    if freq_hz > max_freq_hz {
                        break;
                    }
                    add_log2_energy(amps, space, freq_hz, amp_scale * mode.gain);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::BodyKind;
    use crate::life::sound::control::ControlRamp;

    #[test]
    fn harmonic_backend_renders_finite_signal() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            spread: 0.0,
            voices: 1,
            noise_mix: 0.2,
            ratios: None,
        };
        let mut backend =
            HarmonicResonatorBackend::from_snapshot(48_000.0, &snapshot).expect("harmonic backend");
        backend.seed_phases(123);
        let mut out = [0.0f32; 16];
        backend.render_block(
            &[
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
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
    fn harmonic_backend_limits_modes_by_nominal_pitch() {
        let snapshot = BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: 0.6,
            spread: 0.0,
            voices: 1,
            noise_mix: 0.0,
            ratios: None,
        };
        let mut backend =
            HarmonicResonatorBackend::from_snapshot(48_000.0, &snapshot).expect("harmonic backend");
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
}
