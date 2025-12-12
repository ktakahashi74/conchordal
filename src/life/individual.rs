use crate::core::landscape::Landscape;
use crate::core::modulation::NeuralRhythms;
use crate::core::utils::pink_noise_tick;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::f32::consts::PI;

pub trait AudioAgent {
    fn id(&self) -> u64;
    fn metadata(&self) -> &AgentMetadata;
    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    );
    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    );
    fn is_alive(&self) -> bool;
}

/// Hybrid synthesis individuals render both time-domain audio and a spectral "body".
pub enum Individual {
    PureTone(PureTone),
    Harmonic(Harmonic),
}

impl Individual {
    pub fn id(&self) -> u64 {
        match self {
            Individual::PureTone(ind) => ind.id(),
            Individual::Harmonic(ind) => ind.id(),
        }
    }

    pub fn metadata(&self) -> &AgentMetadata {
        match self {
            Individual::PureTone(ind) => ind.metadata(),
            Individual::Harmonic(ind) => ind.metadata(),
        }
    }

    pub fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        match self {
            Individual::PureTone(ind) => {
                ind.render_wave(buffer, fs, current_frame, dt_sec, landscape)
            }
            Individual::Harmonic(ind) => {
                ind.render_wave(buffer, fs, current_frame, dt_sec, landscape)
            }
        }
    }

    pub fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    ) {
        match self {
            Individual::PureTone(ind) => ind.render_spectrum(amps, fs, nfft, current_frame, dt_sec),
            Individual::Harmonic(ind) => ind.render_spectrum(amps, fs, nfft, current_frame, dt_sec),
        }
    }

    pub fn is_alive(&self) -> bool {
        match self {
            Individual::PureTone(ind) => ind.is_alive(),
            Individual::Harmonic(ind) => ind.is_alive(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentMetadata {
    pub id: u64,
    pub tag: Option<String>,
    pub group_idx: usize,
    pub member_idx: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sensitivity {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioSample {
    pub sample: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArticulationState {
    Idle,
    Attack,
    Decay,
}

#[derive(Debug, Clone)]
pub struct PinkNoise {
    rng: SmallRng,
    b0: f32,
    b1: f32,
    b2: f32,
    gain: f32,
}

impl PinkNoise {
    pub fn new(seed: u64, gain: f32) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            gain,
        }
    }

    pub fn next(&mut self) -> f32 {
        let pink = pink_noise_tick(&mut self.rng, &mut self.b0, &mut self.b1, &mut self.b2);
        pink * self.gain
    }
}

/// Legacy pure tone voice (single oscillator).
#[derive(Debug, Clone)]
pub struct PureTone {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub freq_hz: f32,
    pub amp: f32,
    pub energy: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    // Rhythm domain
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    // Audio domain
    pub audio_phase: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
    pub retrigger: bool,
    pub omega: f32,
    pub noise_1f: PinkNoise,
    pub confidence: f32,
    pub gate_threshold: f32,
}

/// Additive harmonic voice.
#[derive(Debug, Clone)]
pub struct Harmonic {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub base_freq_hz: f32,
    pub amp: f32,
    pub genotype: TimbreGenotype,
    pub energy: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    pub lfo_phase: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
    pub retrigger: bool,
    pub confidence: f32,
    pub gate_threshold: f32,
    pub phases: Vec<f32>,
    pub detune_phases: Vec<f32>,
    pub jitter_gen: PinkNoise,
}

impl PureTone {
    pub fn update(&mut self, consonance: f32, rhythms: &NeuralRhythms, dt: f32) {
        let dt = dt.max(1e-4);

        // 1. Metabolism
        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return;
        }

        // 2. Rhythm sensing and phase update (Kuramoto on slow clock)
        let delta_phase = rhythms.delta.phase;
        let delta_mag = rhythms.delta.mag * self.sensitivity.delta;
        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.cos() * self.sensitivity.theta;
        let trigger_force = theta_signal * consonance;

        // 3. Rhythm phase dynamics (confidence-weighted coupling)
        let beta_err = rhythms.beta.mag * self.sensitivity.beta;
        self.confidence = 1.0 - beta_err;
        let coupling = (self.sensitivity.beta * (1.0 / (self.confidence + 1e-3))).min(10.0);
        let noise = self.noise_1f.next();
        let rhythm_omega = 2.0 * PI * self.rhythm_freq;
        let d_phi = rhythm_omega + noise + coupling * (delta_phase - self.rhythm_phase).sin();
        self.rhythm_phase += d_phi * dt;

        // 4. Trigger on rhythm wrap + gate
        if self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            let gate_bias = 0.2;
            let gate = delta_mag + trigger_force + gate_bias;
            if gate > self.gate_threshold && self.state == ArticulationState::Idle && self.retrigger
            {
                self.audio_phase = 0.0;
                self.env_level = 0.0;
                self.state = ArticulationState::Attack;
                self.energy -= self.action_cost;
                if consonance > 0.5 {
                    self.energy += self.recharge_rate * consonance;
                }
            }
        }
    }
}

impl PureTone {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn render_wave(
        &mut self,
        buffer: &mut [f32],
        _fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let rhythms = landscape.rhythm;
        let consonance = landscape.consonance_at(self.freq_hz);
        // Angular frequency in rad/s for audio oscillator
        let omega = 2.0 * PI * self.freq_hz;
        for s in buffer.iter_mut() {
            self.omega = omega;
            self.update(consonance, &rhythms, dt_per_sample);
            match self.state {
                ArticulationState::Idle => {
                    // silence
                }
                ArticulationState::Attack => {
                    self.env_level += self.attack_step;
                    if self.env_level >= 1.0 {
                        self.env_level = 1.0;
                        self.state = ArticulationState::Decay;
                    }
                    self.audio_phase =
                        (self.audio_phase + self.omega * dt_per_sample).rem_euclid(2.0 * PI);
                    *s += self.amp * self.env_level * self.audio_phase.sin();
                }
                ArticulationState::Decay => {
                    self.env_level *= self.decay_factor;
                    if self.env_level < 0.001 {
                        self.env_level = 0.0;
                        self.state = ArticulationState::Idle;
                    }
                    self.audio_phase =
                        (self.audio_phase + self.omega * dt_per_sample).rem_euclid(2.0 * PI);
                    *s += self.amp * self.env_level * self.audio_phase.sin();
                }
            }
        }
    }

    pub fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        _current_frame: u64,
        _dt_sec: f32,
    ) {
        // Use current envelope/state so spectral body matches audible body.
        if self.state == ArticulationState::Idle || self.env_level <= 0.0 {
            return;
        }
        let bin_f = self.freq_hz * nfft as f32 / fs;
        let k = bin_f.round() as isize;
        if k >= 0 && (k as usize) < amps.len() {
            amps[k as usize] += self.amp.max(0.0) * self.env_level;
        }
    }

    pub fn is_alive(&self) -> bool {
        if self.energy <= 0.0 {
            return false;
        }
        if !self.retrigger && self.state == ArticulationState::Idle {
            return false;
        }
        true
    }
}

impl AudioAgent for PureTone {
    fn id(&self) -> u64 {
        PureTone::id(self)
    }

    fn metadata(&self) -> &AgentMetadata {
        PureTone::metadata(self)
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        PureTone::render_wave(self, buffer, fs, current_frame, dt_sec, landscape);
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    ) {
        PureTone::render_spectrum(self, amps, fs, nfft, current_frame, dt_sec);
    }

    fn is_alive(&self) -> bool {
        PureTone::is_alive(self)
    }
}

impl Harmonic {
    fn update(&mut self, consonance: f32, rhythms: &NeuralRhythms, dt: f32) {
        let dt = dt.max(1e-4);

        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return;
        }

        let delta_phase = rhythms.delta.phase;
        let delta_mag = rhythms.delta.mag * self.sensitivity.delta;
        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.cos() * self.sensitivity.theta;
        let trigger_force = theta_signal * consonance;

        let beta_err = rhythms.beta.mag * self.sensitivity.beta;
        self.confidence = 1.0 - beta_err;
        let coupling = (self.sensitivity.beta * (1.0 / (self.confidence + 1e-3))).min(10.0);
        let noise = self.jitter_gen.next();
        let rhythm_omega = 2.0 * PI * self.rhythm_freq;
        let d_phi = rhythm_omega + noise + coupling * (delta_phase - self.rhythm_phase).sin();
        self.rhythm_phase += d_phi * dt;

        if self.rhythm_phase >= 2.0 * PI {
            self.rhythm_phase -= 2.0 * PI;
            let gate_bias = 0.2;
            let gate = delta_mag + trigger_force + gate_bias;
            if gate > self.gate_threshold && self.state == ArticulationState::Idle && self.retrigger
            {
                self.env_level = 0.0;
                self.state = ArticulationState::Attack;
                self.energy -= self.action_cost;
                if consonance > 0.5 {
                    self.energy += self.recharge_rate * consonance;
                }
            }
        }
    }

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

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn render_wave(
        &mut self,
        buffer: &mut [f32],
        _fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let rhythms = landscape.rhythm;
        let consonance = landscape.consonance_at(self.base_freq_hz);
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }

        for s in buffer.iter_mut() {
            self.update(consonance, &rhythms, dt_per_sample);
            match self.state {
                ArticulationState::Idle => {
                    continue;
                }
                ArticulationState::Attack => {
                    self.env_level += self.attack_step;
                    if self.env_level >= 1.0 {
                        self.env_level = 1.0;
                        self.state = ArticulationState::Decay;
                    }
                }
                ArticulationState::Decay => {
                    self.env_level *= self.decay_factor;
                    if self.env_level < 0.001 {
                        self.env_level = 0.0;
                        self.state = ArticulationState::Idle;
                        continue;
                    }
                }
            }

            if self.env_level <= 0.0 {
                continue;
            }

            self.lfo_phase = (self.lfo_phase
                + 2.0 * PI * self.genotype.vibrato_rate * dt_per_sample)
                .rem_euclid(2.0 * PI);
            let vibrato = self.genotype.vibrato_depth * self.lfo_phase.sin();
            let jitter_scale = (1.0 + rhythms.beta.mag * 0.5) * (self.env_level + 0.1);
            let jitter = self.jitter_gen.next() * self.genotype.jitter * jitter_scale;
            let base_freq = (self.base_freq_hz * (1.0 + vibrato + jitter)).max(1.0);
            let unison = (self.genotype.unison * (1.0 + rhythms.alpha.mag * 0.5)).max(0.0);

            let mut sample = 0.0;
            for idx in 0..partials {
                let ratio = self.partial_ratio(idx);
                let freq = base_freq * ratio;
                if !freq.is_finite() || freq <= 0.0 {
                    continue;
                }
                let part_amp = self.compute_partial_amp(idx, self.env_level);
                if part_amp <= 0.0 {
                    continue;
                }
                let phase = &mut self.phases[idx];
                *phase = (*phase + 2.0 * PI * freq * dt_per_sample).rem_euclid(2.0 * PI);
                let mut part_sample = phase.sin();
                if unison > 0.0 {
                    let detune_ratio = 1.0 + unison * 0.02;
                    let detune_phase = &mut self.detune_phases[idx];
                    *detune_phase = (*detune_phase
                        + 2.0 * PI * freq * detune_ratio * dt_per_sample)
                        .rem_euclid(2.0 * PI);
                    part_sample = 0.5 * (part_sample + detune_phase.sin());
                }
                sample += part_amp * part_sample;
            }
            *s += self.amp * self.env_level * sample;
        }
    }

    pub fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        _current_frame: u64,
        _dt_sec: f32,
    ) {
        if self.state == ArticulationState::Idle {
            return;
        }
        let partials = self.partial_count();
        if partials == 0 {
            return;
        }
        let amp_scale = self.amp.max(0.0);
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq = self.base_freq_hz * ratio;
            let bin_f = freq * nfft as f32 / fs;
            let k = bin_f.round() as isize;
            if k >= 0 && (k as usize) < amps.len() {
                let part_amp = self.compute_partial_amp(idx, 1.0);
                amps[k as usize] += amp_scale * part_amp;
            }
        }
    }

    pub fn is_alive(&self) -> bool {
        if self.energy <= 0.0 {
            return false;
        }
        if !self.retrigger && self.state == ArticulationState::Idle {
            return false;
        }
        true
    }
}

impl AudioAgent for Harmonic {
    fn id(&self) -> u64 {
        Harmonic::id(self)
    }

    fn metadata(&self) -> &AgentMetadata {
        Harmonic::metadata(self)
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
    ) {
        Harmonic::render_wave(self, buffer, fs, current_frame, dt_sec, landscape);
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    ) {
        Harmonic::render_spectrum(self, amps, fs, nfft, current_frame, dt_sec);
    }

    fn is_alive(&self) -> bool {
        Harmonic::is_alive(self)
    }
}
