use super::articulation_core::{ArticulationSignal, PinkNoise};
use crate::core::log2space::Log2Space;
use crate::life::scenario::{HarmonicMode, SoundBodyConfig, TimbreGenotype};
use rand::Rng;
use std::f32::consts::PI;

pub trait SoundBody {
    fn base_freq_hz(&self) -> f32;
    fn set_freq(&mut self, freq: f32);
    fn set_pitch_log2(&mut self, log_freq: f32);
    fn set_amp(&mut self, amp: f32);
    fn amp(&self) -> f32;
    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal);
    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    );
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

#[derive(Debug, Clone)]
pub struct SineBody {
    pub freq_hz: f32,
    pub amp: f32,
    pub audio_phase: f32,
}

impl SoundBody for SineBody {
    fn base_freq_hz(&self) -> f32 {
        self.freq_hz
    }

    fn set_freq(&mut self, freq: f32) {
        self.freq_hz = freq;
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        self.freq_hz = 2.0f32.powf(log_freq);
    }

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    fn amp(&self) -> f32 {
        self.amp
    }

    fn articulate_wave(
        &mut self,
        sample: &mut f32,
        _fs: f32,
        dt: f32,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.audio_phase = (self.audio_phase + 2.0 * PI * self.freq_hz * dt).rem_euclid(2.0 * PI);
        *sample += self.amp * signal.amplitude * self.audio_phase.sin();
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let energy = self.amp.max(0.0) * signal.amplitude;
        add_log2_energy(amps, space, self.freq_hz, energy);
    }
}

#[derive(Debug, Clone)]
pub struct HarmonicBody {
    pub base_freq_hz: f32,
    pub amp: f32,
    pub genotype: TimbreGenotype,
    pub lfo_phase: f32,
    pub phases: Vec<f32>,
    pub detune_phases: Vec<f32>,
    pub jitter_gen: PinkNoise,
}

impl HarmonicBody {
    fn partial_ratio(&self, idx: usize) -> f32 {
        let k = (idx + 1) as f32;
        let base = match self.genotype.mode {
            HarmonicMode::Harmonic => k,
            HarmonicMode::Metallic => k.powf(1.4),
        };
        let stretch = 1.0 + self.genotype.stiffness * k * k;
        (base * stretch).max(0.1)
    }

    fn compute_partial_amp(&self, idx: usize, current_energy: f32) -> f32 {
        let k = (idx + 1) as f32;
        let slope = self.genotype.brightness.max(0.0);
        let mut amp = 1.0 / k.powf(slope.max(1e-6));
        if (idx + 1).is_multiple_of(2) {
            amp *= 1.0 - self.genotype.comb.clamp(0.0, 1.0);
        }
        let damping = self.genotype.damping.max(0.0);
        if damping > 0.0 {
            let energy = current_energy.clamp(0.0, 1.0);
            amp *= energy.powf(damping * k);
        }
        amp
    }

    fn partial_count(&self) -> usize {
        self.phases.len().min(self.detune_phases.len())
    }
}

impl SoundBody for HarmonicBody {
    fn base_freq_hz(&self) -> f32 {
        self.base_freq_hz
    }

    fn set_freq(&mut self, freq: f32) {
        self.base_freq_hz = freq;
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        self.base_freq_hz = 2.0f32.powf(log_freq);
    }

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
    }

    fn amp(&self) -> f32 {
        self.amp
    }

    fn articulate_wave(
        &mut self,
        sample: &mut f32,
        _fs: f32,
        dt: f32,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }
        self.lfo_phase =
            (self.lfo_phase + 2.0 * PI * self.genotype.vibrato_rate * dt).rem_euclid(2.0 * PI);
        let vibrato =
            self.genotype.vibrato_depth * (1.0 + signal.relaxation * 0.5) * self.lfo_phase.sin();
        let jitter_scale = (1.0 + signal.tension * 0.5) * (signal.amplitude + 0.1);
        let jitter = self.jitter_gen.sample() * self.genotype.jitter * jitter_scale;
        let base_freq = (self.base_freq_hz * (1.0 + vibrato + jitter)).max(1.0);
        let unison = (self.genotype.unison * (1.0 + signal.relaxation * 0.5)).max(0.0);

        let mut acc = 0.0;
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq = base_freq * ratio;
            if !freq.is_finite() || freq <= 0.0 {
                continue;
            }
            let part_amp = self.compute_partial_amp(idx, signal.amplitude);
            if part_amp <= 0.0 {
                continue;
            }
            let phase = &mut self.phases[idx];
            *phase = (*phase + 2.0 * PI * freq * dt).rem_euclid(2.0 * PI);
            let mut part_sample = phase.sin();
            if unison > 0.0 {
                let detune_ratio = 1.0 + unison * 0.02;
                let detune_phase = &mut self.detune_phases[idx];
                *detune_phase =
                    (*detune_phase + 2.0 * PI * freq * detune_ratio * dt).rem_euclid(2.0 * PI);
                part_sample = 0.5 * (part_sample + detune_phase.sin());
            }
            acc += part_amp * part_sample;
        }
        *sample += self.amp * signal.amplitude * acc;
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }
        let amp_scale = self.amp.max(0.0) * signal.amplitude;
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq = self.base_freq_hz * ratio;
            let part_amp = self.compute_partial_amp(idx, signal.amplitude);
            add_log2_energy(amps, space, freq, amp_scale * part_amp);
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnySoundBody {
    Sine(SineBody),
    Harmonic(HarmonicBody),
}

impl AnySoundBody {
    pub fn from_config<R: Rng + ?Sized>(
        config: &SoundBodyConfig,
        freq_hz: f32,
        amp: f32,
        rng: &mut R,
    ) -> Self {
        match config {
            SoundBodyConfig::Sine { phase } => AnySoundBody::Sine(SineBody {
                freq_hz,
                amp,
                audio_phase: phase.unwrap_or_else(|| rng.random_range(0.0..std::f32::consts::TAU)),
            }),
            SoundBodyConfig::Harmonic { genotype, partials } => {
                let partials = partials.unwrap_or(16).max(1);
                let mut phases = Vec::with_capacity(partials);
                let mut detune_phases = Vec::with_capacity(partials);
                for _ in 0..partials {
                    phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                    detune_phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                }
                AnySoundBody::Harmonic(HarmonicBody {
                    base_freq_hz: freq_hz,
                    amp,
                    genotype: genotype.clone(),
                    lfo_phase: 0.0,
                    phases,
                    detune_phases,
                    jitter_gen: PinkNoise::new(rng.next_u64(), 0.001),
                })
            }
        }
    }
}

impl SoundBody for AnySoundBody {
    fn base_freq_hz(&self) -> f32 {
        match self {
            AnySoundBody::Sine(body) => body.base_freq_hz(),
            AnySoundBody::Harmonic(body) => body.base_freq_hz(),
        }
    }

    fn set_freq(&mut self, freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_freq(freq),
            AnySoundBody::Harmonic(body) => body.set_freq(freq),
        }
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_pitch_log2(log_freq),
            AnySoundBody::Harmonic(body) => body.set_pitch_log2(log_freq),
        }
    }

    fn set_amp(&mut self, amp: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_amp(amp),
            AnySoundBody::Harmonic(body) => body.set_amp(amp),
        }
    }

    fn amp(&self) -> f32 {
        match self {
            AnySoundBody::Sine(body) => body.amp(),
            AnySoundBody::Harmonic(body) => body.amp(),
        }
    }

    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal) {
        match self {
            AnySoundBody::Sine(body) => body.articulate_wave(sample, fs, dt, signal),
            AnySoundBody::Harmonic(body) => body.articulate_wave(sample, fs, dt, signal),
        }
    }

    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        space: &Log2Space,
        signal: &ArticulationSignal,
    ) {
        match self {
            AnySoundBody::Sine(body) => body.project_spectral_body(amps, space, signal),
            AnySoundBody::Harmonic(body) => body.project_spectral_body(amps, space, signal),
        }
    }
}
