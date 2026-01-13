use crate::core::log2space::Log2Space;
use crate::life::audio::backend::Backend;
use crate::life::audio::control::VoiceControlBlock;
use crate::life::individual::ArticulationSignal;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};
use crate::synth::SynthError;
use crate::synth::modes::ModeParams;
use crate::synth::resonator::ResonatorBank;

const DEFAULT_UPDATE_PERIOD_SAMPLES: usize = 64;

#[derive(Debug, Clone)]
pub enum ModeShape {
    Sine {
        t60_s: f32,
        out_gain: f32,
        in_gain: f32,
    },
    Harmonic {
        partials: usize,
        base_t60_s: f32,
        in_gain: f32,
        genotype: TimbreGenotype,
    },
    Modal {
        modes: Vec<ModalMode>,
    },
}

#[derive(Debug, Clone)]
pub struct ModalMode {
    pub ratio: f32,
    pub t60_s: f32,
    pub gain: f32,
    pub in_gain: f32,
}

#[derive(Debug, Clone)]
pub struct ModalEngine {
    bank: ResonatorBank,
    shape: ModeShape,
    scratch: Vec<ModeParams>,
    last_built_pitch_hz: f32,
    update_period_samples: usize,
    counter: usize,
    last_modes_len: usize,
}

impl ModalEngine {
    pub fn new(fs: f32, shape: ModeShape) -> Result<Self, SynthError> {
        let max_modes = shape.max_modes().max(1);
        let bank = ResonatorBank::new(fs, max_modes)?;
        let scratch = Vec::with_capacity(max_modes);
        Ok(Self {
            bank,
            shape,
            scratch,
            last_built_pitch_hz: 0.0,
            update_period_samples: DEFAULT_UPDATE_PERIOD_SAMPLES.max(1),
            counter: 0,
            last_modes_len: 0,
        })
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
        self.shape.build_modes(pitch_hz, limit, &mut self.scratch);
        self.last_modes_len = self.scratch.len();
        let _ = self
            .bank
            .set_modes_preserve_state(&self.scratch[..self.last_modes_len]);
        self.last_built_pitch_hz = pitch_hz;
    }
}

impl ModeShape {
    fn max_modes(&self) -> usize {
        match self {
            ModeShape::Sine { .. } => 1,
            ModeShape::Harmonic { partials, .. } => (*partials).max(1),
            ModeShape::Modal { modes } => modes.len().max(1),
        }
    }

    fn build_modes(&self, pitch_hz: f32, limit: usize, scratch: &mut Vec<ModeParams>) {
        scratch.clear();
        match self {
            ModeShape::Sine {
                t60_s,
                out_gain,
                in_gain,
            } => {
                scratch.push(ModeParams {
                    freq_hz: pitch_hz,
                    t60_s: t60_s.max(1e-3),
                    gain: *out_gain,
                    in_gain: *in_gain,
                });
            }
            ModeShape::Harmonic {
                partials,
                base_t60_s,
                in_gain,
                genotype,
            } => {
                let partials = (*partials).max(1);
                let energy = 1.0;
                for k in 1..=partials {
                    if scratch.len() >= limit {
                        break;
                    }
                    let ratio = harmonic_ratio(genotype, k);
                    let freq_hz = pitch_hz * ratio;
                    if !freq_hz.is_finite() || freq_hz <= 0.0 {
                        continue;
                    }
                    let gain = harmonic_gain(genotype, k, energy);
                    let t60_s = base_t60_s.max(1e-3) / (1.0 + 0.15 * k as f32);
                    scratch.push(ModeParams {
                        freq_hz,
                        t60_s,
                        gain,
                        in_gain: *in_gain,
                    });
                }
            }
            ModeShape::Modal { modes } => {
                for mode in modes {
                    if scratch.len() >= limit {
                        break;
                    }
                    let freq_hz = pitch_hz * mode.ratio;
                    if !freq_hz.is_finite() || freq_hz <= 0.0 {
                        continue;
                    }
                    scratch.push(ModeParams {
                        freq_hz,
                        t60_s: mode.t60_s.max(1e-3),
                        gain: mode.gain,
                        in_gain: mode.in_gain,
                    });
                }
            }
        }
    }
}

impl Backend for ModalEngine {
    fn render_block(&mut self, drive: &[f32], ctrl: VoiceControlBlock, out: &mut [f32]) {
        debug_assert_eq!(drive.len(), out.len());
        if drive.is_empty() {
            return;
        }
        let mut counter = self.counter;
        let period = self.update_period_samples.max(1);
        for (idx, (u, y)) in drive.iter().copied().zip(out.iter_mut()).enumerate() {
            let pitch_hz = ctrl.pitch_hz.start + ctrl.pitch_hz.step * idx as f32;
            let amp = ctrl.amp.start + ctrl.amp.step * idx as f32;
            if counter == 0 || self.last_modes_len == 0 {
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

    fn project_spectral(
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
        match &self.shape {
            ModeShape::Sine { out_gain, .. } => {
                add_log2_energy(amps, space, pitch_hz, amp_scale * out_gain.max(0.0));
            }
            ModeShape::Harmonic {
                partials, genotype, ..
            } => {
                let partials = (*partials).max(1);
                let energy = 1.0;
                for k in 1..=partials {
                    let ratio = harmonic_ratio(genotype, k);
                    let freq_hz = pitch_hz * ratio;
                    let gain = harmonic_gain(genotype, k, energy);
                    add_log2_energy(amps, space, freq_hz, amp_scale * gain);
                }
            }
            ModeShape::Modal { modes } => {
                for mode in modes {
                    let freq_hz = pitch_hz * mode.ratio;
                    add_log2_energy(amps, space, freq_hz, amp_scale * mode.gain);
                }
            }
        }
    }
}

fn harmonic_ratio(genotype: &TimbreGenotype, k: usize) -> f32 {
    let kf = k as f32;
    let base = match genotype.mode {
        HarmonicMode::Harmonic => kf,
        HarmonicMode::Metallic => kf.powf(1.4),
    };
    let stretch = 1.0 + genotype.stiffness * kf * kf;
    (base * stretch).max(0.1)
}

fn harmonic_gain(genotype: &TimbreGenotype, k: usize, energy: f32) -> f32 {
    let kf = k as f32;
    let slope = genotype.brightness.max(0.0);
    let mut amp = 1.0 / kf.powf(slope.max(1e-6));
    if k.is_multiple_of(2) {
        amp *= 1.0 - genotype.comb.clamp(0.0, 1.0);
    }
    let damping = genotype.damping.max(0.0);
    if damping > 0.0 {
        let energy = energy.clamp(0.0, 1.0);
        amp *= energy.powf(damping * kf);
    }
    amp
}

fn add_log2_energy(amps: &mut [f32], space: &Log2Space, freq_hz: f32, energy: f32) {
    if !freq_hz.is_finite() || energy == 0.0 {
        return;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return;
    }
    let log_f = freq_hz.log2();
    let base = space.centers_log2[0];
    let step = space.step();
    let pos = (log_f - base) / step;
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < amps.len() {
        amps[idx] += energy * (1.0 - frac);
        amps[idx + 1] += energy * frac;
    } else if idx < amps.len() {
        amps[idx] += energy;
    }
}
