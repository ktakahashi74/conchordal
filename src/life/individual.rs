use crate::core::landscape::Landscape;
use crate::core::modulation::NeuralRhythms;
use crate::core::utils::pink_noise_tick;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};
use rand::{Rng as _, SeedableRng, rngs::SmallRng};
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
        global_coupling: f32,
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

#[derive(Debug, Clone, Copy, Default)]
pub struct ArticulationSignal {
    pub amplitude: f32,
    pub is_active: bool,
    pub relaxation: f32,
    pub tension: f32,
}

pub trait NeuralCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal;
    fn is_alive(&self) -> bool;
}

pub trait SoundBody {
    fn base_freq_hz(&self) -> f32;
    fn set_freq(&mut self, freq: f32);
    fn set_amp(&mut self, amp: f32);
    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal);
    fn project_spectral_body(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        signal: &ArticulationSignal,
    );
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

#[derive(Debug, Clone)]
pub struct KuramotoCore {
    pub energy: f32,
    pub basal_cost: f32,
    pub action_cost: f32,
    pub recharge_rate: f32,
    pub sensitivity: Sensitivity,
    pub rhythm_phase: f32,
    pub rhythm_freq: f32,
    pub env_level: f32,
    pub state: ArticulationState,
    pub attack_step: f32,
    pub decay_factor: f32,
    pub retrigger: bool,
    pub noise_1f: PinkNoise,
    pub confidence: f32,
    pub gate_threshold: f32,
}

impl NeuralCore for KuramotoCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        let dt = dt.max(1e-4);

        self.energy -= self.basal_cost * dt;
        if self.energy <= 0.0 {
            self.state = ArticulationState::Idle;
            self.env_level = 0.0;
            return ArticulationSignal::default();
        }

        let delta_phase = rhythms.delta.phase;
        let delta_mag = rhythms.delta.mag * self.sensitivity.delta;
        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.cos() * self.sensitivity.theta;
        let trigger_force = theta_signal * consonance;

        let beta_err = rhythms.beta.mag * self.sensitivity.beta;
        self.confidence = 1.0 - beta_err;
        let coupling = (self.sensitivity.beta * (1.0 / (self.confidence + 1e-3))).min(10.0);
        let noise = self.noise_1f.next();
        let rhythm_omega = 2.0 * PI * self.rhythm_freq;
        let d_phi = rhythm_omega
            + noise
            + coupling * global_coupling * (delta_phase - self.rhythm_phase).sin();
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

        match self.state {
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
                }
            }
            ArticulationState::Idle => {}
        }

        let relaxation = rhythms.alpha.mag * self.sensitivity.alpha;
        let tension = rhythms.beta.mag * self.sensitivity.beta;
        let is_active = self.env_level > 0.0 && self.state != ArticulationState::Idle;
        ArticulationSignal {
            amplitude: self.env_level,
            is_active,
            relaxation,
            tension,
        }
    }

    fn is_alive(&self) -> bool {
        if self.energy <= 0.0 {
            return false;
        }
        if !self.retrigger && self.state == ArticulationState::Idle {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct SequencedCore {
    pub timer: f32,
    pub duration: f32,
    pub env_level: f32,
}

impl NeuralCore for SequencedCore {
    fn process(
        &mut self,
        _consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        _global_coupling: f32,
    ) -> ArticulationSignal {
        self.timer += dt.max(0.0);
        let active = self.timer < self.duration;
        self.env_level = if active { 1.0 } else { 0.0 };
        ArticulationSignal {
            amplitude: self.env_level,
            is_active: active,
            relaxation: rhythms.alpha.mag,
            tension: 0.0,
        }
    }

    fn is_alive(&self) -> bool {
        self.timer < self.duration
    }
}

#[derive(Debug, Clone)]
pub struct DroneCore {
    pub phase: f32,
    pub sway_rate: f32,
}

impl NeuralCore for DroneCore {
    fn process(
        &mut self,
        _consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        _global_coupling: f32,
    ) -> ArticulationSignal {
        let dt = dt.max(1e-4);
        let omega = 2.0 * PI * self.sway_rate.max(0.01);
        self.phase = (self.phase + omega * dt).rem_euclid(2.0 * PI);
        let lfo = 0.5 * (self.phase.sin() + 1.0);
        let relax_boost = 1.0 + rhythms.alpha.mag * 0.5;
        let amplitude = (0.3 + 0.7 * lfo) * relax_boost;
        ArticulationSignal {
            amplitude: amplitude.clamp(0.0, 1.0),
            is_active: true,
            relaxation: rhythms.alpha.mag,
            tension: rhythms.beta.mag * 0.25,
        }
    }

    fn is_alive(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub enum AnyCore {
    Entrain(KuramotoCore),
    Seq(SequencedCore),
    Drone(DroneCore),
}

impl NeuralCore for AnyCore {
    fn process(
        &mut self,
        consonance: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        global_coupling: f32,
    ) -> ArticulationSignal {
        match self {
            AnyCore::Entrain(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyCore::Seq(c) => c.process(consonance, rhythms, dt, global_coupling),
            AnyCore::Drone(c) => c.process(consonance, rhythms, dt, global_coupling),
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            AnyCore::Entrain(c) => c.is_alive(),
            AnyCore::Seq(c) => c.is_alive(),
            AnyCore::Drone(c) => c.is_alive(),
        }
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

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
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
        fs: f32,
        nfft: usize,
        signal: &ArticulationSignal,
    ) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let bin_f = self.freq_hz * nfft as f32 / fs;
        let k = bin_f.round() as isize;
        if k >= 0 && (k as usize) < amps.len() {
            amps[k as usize] += self.amp.max(0.0) * signal.amplitude;
        }
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

    fn set_amp(&mut self, amp: f32) {
        self.amp = amp;
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
        let jitter = self.jitter_gen.next() * self.genotype.jitter * jitter_scale;
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
        fs: f32,
        nfft: usize,
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
            let bin_f = freq * nfft as f32 / fs;
            let k = bin_f.round() as isize;
            if k >= 0 && (k as usize) < amps.len() {
                let part_amp = self.compute_partial_amp(idx, signal.amplitude);
                amps[k as usize] += amp_scale * part_amp;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Individual<N: NeuralCore, B: SoundBody> {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub core: N,
    pub body: B,
    pub last_signal: ArticulationSignal,
    pub target_freq: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
    pub breath_gain: f32,
    pub commitment: f32,
    pub habituation_sensitivity: f32,
    pub last_theta_sample: f32,
}

impl<N: NeuralCore, B: SoundBody> Individual<N, B> {
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn update_organic_movement(
        &mut self,
        rhythms: &NeuralRhythms,
        dt: f32,
        landscape: &Landscape,
    ) {
        let dt = dt.max(0.0);
        let current_freq = self.body.base_freq_hz().max(1.0);
        if self.target_freq <= 0.0 {
            self.target_freq = current_freq;
        }
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt;

        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.sin();
        let theta_cross = self.last_theta_sample <= 0.0 && theta_signal > 0.0;
        self.last_theta_sample = theta_signal;

        if theta_cross && self.accumulated_time >= self.integration_window {
            self.accumulated_time = 0.0;
            let neighbor_ratio = 2f32.powf(200.0 / 1200.0);
            let mut candidates = vec![
                self.target_freq,
                self.target_freq * neighbor_ratio,
                self.target_freq / neighbor_ratio,
                self.target_freq * 1.5,
                self.target_freq * 0.66,
            ];
            candidates.retain(|f| f.is_finite());

            let (fmin, fmax) = landscape.freq_bounds();
            let mut best_freq = self.target_freq.clamp(fmin, fmax);
            let mut best_score = f32::MIN;
            for f in candidates {
                let clamped = f.clamp(fmin, fmax);
                let score = landscape.evaluate_pitch(clamped);
                let distance_oct = (clamped.max(1.0).log2() - current_freq.log2()).abs();
                let penalty = distance_oct * self.integration_window * 0.5;
                // Spectral tilt pressure encourages 1/f balance (reduces upward masking and adapts to efficient auditory coding).
                let satiety = landscape.get_spectral_satiety(clamped);
                let overcrowding_weight = 2.0;
                let mut adjusted = score - penalty;
                if satiety > 1.0 {
                    adjusted -= (satiety - 1.0) * overcrowding_weight;
                }
                if adjusted > best_score {
                    best_score = adjusted;
                    best_freq = clamped;
                }
            }

            let current_score = landscape.evaluate_pitch(self.target_freq.clamp(fmin, fmax));
            let improvement = best_score - current_score;
            let satisfaction = ((current_score + 1.0) * 0.5).clamp(0.0, 1.0);
            let habituation_penalty = (1.0 - satisfaction) * self.habituation_sensitivity.max(0.0);
            let mut stay_prob =
                (self.commitment.clamp(0.0, 1.0) * satisfaction) - habituation_penalty;
            stay_prob = stay_prob.clamp(0.0, 1.0);

            if improvement > 0.1 {
                self.target_freq = best_freq;
            } else {
                let mut rng = rand::rng();
                if rng.random_range(0.0..1.0) > stay_prob {
                    self.target_freq = best_freq;
                }
            }
        }

        let (fmin, fmax) = landscape.freq_bounds();
        self.target_freq = self.target_freq.clamp(fmin, fmax);
        let distance_cents = 1200.0
            * (self.target_freq.max(1e-3) / current_freq.max(1e-3))
                .log2()
                .abs();
        let move_threshold = 10.0;
        if distance_cents > move_threshold {
            self.breath_gain = (self.breath_gain - dt * 1.5).max(0.0);
            if self.breath_gain < 0.1 {
                self.body.set_freq(self.target_freq);
            }
        } else {
            self.body.set_freq(self.target_freq);
            let attack_rate = 1.0 + rhythms.beta.mag;
            self.breath_gain = (self.breath_gain + dt * attack_rate).clamp(0.0, 1.0);
        }
    }
}

pub type PureTone = Individual<AnyCore, SineBody>;
pub type Harmonic = Individual<AnyCore, HarmonicBody>;

/// Hybrid synthesis individuals render both time-domain audio and a spectral "body".
pub enum IndividualWrapper {
    PureTone(PureTone),
    Harmonic(Harmonic),
}

impl<N: NeuralCore, B: SoundBody> AudioAgent for Individual<N, B> {
    fn id(&self) -> u64 {
        self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        self.metadata()
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        if buffer.is_empty() {
            return;
        }
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let rhythms = landscape.rhythm;
        for sample in buffer.iter_mut() {
            self.update_organic_movement(&rhythms, dt_per_sample, landscape);
            let consonance = landscape.evaluate_pitch(self.body.base_freq_hz());
            let mut signal =
                self.core
                    .process(consonance, &rhythms, dt_per_sample, global_coupling);
            signal.amplitude *= self.breath_gain;
            signal.is_active = signal.is_active && signal.amplitude > 0.0;
            self.last_signal = signal;
            if !signal.is_active {
                continue;
            }
            self.body
                .articulate_wave(sample, fs, dt_per_sample, &signal);
        }
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        _current_frame: u64,
        _dt_sec: f32,
    ) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, fs, nfft, &signal);
    }

    fn is_alive(&self) -> bool {
        self.core.is_alive()
    }
}

impl AudioAgent for IndividualWrapper {
    fn id(&self) -> u64 {
        match self {
            IndividualWrapper::PureTone(ind) => ind.id(),
            IndividualWrapper::Harmonic(ind) => ind.id(),
        }
    }

    fn metadata(&self) -> &AgentMetadata {
        match self {
            IndividualWrapper::PureTone(ind) => ind.metadata(),
            IndividualWrapper::Harmonic(ind) => ind.metadata(),
        }
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        match self {
            IndividualWrapper::PureTone(ind) => ind.render_wave(
                buffer,
                fs,
                current_frame,
                dt_sec,
                landscape,
                global_coupling,
            ),
            IndividualWrapper::Harmonic(ind) => ind.render_wave(
                buffer,
                fs,
                current_frame,
                dt_sec,
                landscape,
                global_coupling,
            ),
        }
    }

    fn render_spectrum(
        &mut self,
        amps: &mut [f32],
        fs: f32,
        nfft: usize,
        current_frame: u64,
        dt_sec: f32,
    ) {
        match self {
            IndividualWrapper::PureTone(ind) => {
                ind.render_spectrum(amps, fs, nfft, current_frame, dt_sec)
            }
            IndividualWrapper::Harmonic(ind) => {
                ind.render_spectrum(amps, fs, nfft, current_frame, dt_sec)
            }
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            IndividualWrapper::PureTone(ind) => ind.is_alive(),
            IndividualWrapper::Harmonic(ind) => ind.is_alive(),
        }
    }
}
