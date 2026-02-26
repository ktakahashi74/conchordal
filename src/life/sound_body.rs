use super::articulation_core::{ArticulationSignal, PinkNoise};
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::life::control::{AgentControl, BodyMethod};
use crate::life::scenario::{SoundBodyConfig, TimbreGenotype};
use crate::life::sound::spectral::{add_log2_energy, harmonic_gain, harmonic_ratio};
use crate::life::sound::{BodyKind, BodySnapshot};
use rand::{Rng, RngCore, rngs::SmallRng};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::warn;

pub trait SoundBody: Send {
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
    fn apply_timbre_controls(&mut self, brightness: f32, inharmonic: f32, width: f32, motion: f32);
    fn snapshot(&self) -> BodySnapshot;
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

    fn apply_timbre_controls(
        &mut self,
        _brightness: f32,
        _inharmonic: f32,
        _width: f32,
        _motion: f32,
    ) {
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Sine,
            amp_scale: 1.0,
            brightness: 0.0,
            width: 0.0,
            noise_mix: 0.0,
            ratios: None,
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
    pub custom_ratios: Option<Arc<[f32]>>,
}

impl HarmonicBody {
    fn partial_ratio(&self, idx: usize) -> f32 {
        if let Some(ratios) = &self.custom_ratios
            && let Some(&ratio) = ratios.get(idx)
        {
            return ratio;
        }
        harmonic_ratio(&self.genotype, idx.saturating_add(1))
    }

    fn compute_partial_amp(&self, idx: usize, current_energy: f32) -> f32 {
        harmonic_gain(&self.genotype, idx.saturating_add(1), current_energy)
    }

    fn partial_count(&self) -> usize {
        if let Some(ratios) = &self.custom_ratios {
            return ratios
                .len()
                .min(self.phases.len())
                .min(self.detune_phases.len());
        }
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

    fn apply_timbre_controls(&mut self, brightness: f32, inharmonic: f32, width: f32, motion: f32) {
        self.genotype.brightness = brightness.clamp(0.0, 1.0);
        self.genotype.stiffness = inharmonic.clamp(0.0, 1.0);
        self.genotype.unison = width.clamp(0.0, 1.0);
        self.genotype.jitter = motion.clamp(0.0, 1.0);
        self.genotype.vibrato_depth = self.genotype.jitter * 0.02;
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: self.genotype.brightness.clamp(0.0, 1.0),
            width: self.genotype.unison.clamp(0.0, 1.0),
            noise_mix: self.genotype.jitter.clamp(0.0, 1.0),
            ratios: self.custom_ratios.clone(),
        }
    }
}

pub enum AnySoundBody {
    Sine(SineBody),
    Harmonic(HarmonicBody),
    Dyn(Box<dyn SoundBody + Send>),
}

impl std::fmt::Debug for AnySoundBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnySoundBody::Sine(body) => f.debug_tuple("Sine").field(body).finish(),
            AnySoundBody::Harmonic(body) => f.debug_tuple("Harmonic").field(body).finish(),
            AnySoundBody::Dyn(_) => f.write_str("Dyn(..)"),
        }
    }
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
                    custom_ratios: None,
                })
            }
        }
    }

    pub fn from_dyn(body: Box<dyn SoundBody + Send>) -> Self {
        AnySoundBody::Dyn(body)
    }
}

impl SoundBody for AnySoundBody {
    fn base_freq_hz(&self) -> f32 {
        match self {
            AnySoundBody::Sine(body) => body.base_freq_hz(),
            AnySoundBody::Harmonic(body) => body.base_freq_hz(),
            AnySoundBody::Dyn(body) => body.base_freq_hz(),
        }
    }

    fn set_freq(&mut self, freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_freq(freq),
            AnySoundBody::Harmonic(body) => body.set_freq(freq),
            AnySoundBody::Dyn(body) => body.set_freq(freq),
        }
    }

    fn set_pitch_log2(&mut self, log_freq: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_pitch_log2(log_freq),
            AnySoundBody::Harmonic(body) => body.set_pitch_log2(log_freq),
            AnySoundBody::Dyn(body) => body.set_pitch_log2(log_freq),
        }
    }

    fn set_amp(&mut self, amp: f32) {
        match self {
            AnySoundBody::Sine(body) => body.set_amp(amp),
            AnySoundBody::Harmonic(body) => body.set_amp(amp),
            AnySoundBody::Dyn(body) => body.set_amp(amp),
        }
    }

    fn amp(&self) -> f32 {
        match self {
            AnySoundBody::Sine(body) => body.amp(),
            AnySoundBody::Harmonic(body) => body.amp(),
            AnySoundBody::Dyn(body) => body.amp(),
        }
    }

    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal) {
        match self {
            AnySoundBody::Sine(body) => body.articulate_wave(sample, fs, dt, signal),
            AnySoundBody::Harmonic(body) => body.articulate_wave(sample, fs, dt, signal),
            AnySoundBody::Dyn(body) => body.articulate_wave(sample, fs, dt, signal),
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
            AnySoundBody::Dyn(body) => body.project_spectral_body(amps, space, signal),
        }
    }

    fn apply_timbre_controls(&mut self, brightness: f32, inharmonic: f32, width: f32, motion: f32) {
        match self {
            AnySoundBody::Sine(body) => {
                body.apply_timbre_controls(brightness, inharmonic, width, motion)
            }
            AnySoundBody::Harmonic(body) => {
                body.apply_timbre_controls(brightness, inharmonic, width, motion)
            }
            AnySoundBody::Dyn(body) => {
                body.apply_timbre_controls(brightness, inharmonic, width, motion)
            }
        }
    }

    fn snapshot(&self) -> BodySnapshot {
        match self {
            AnySoundBody::Sine(body) => body.snapshot(),
            AnySoundBody::Harmonic(body) => body.snapshot(),
            AnySoundBody::Dyn(body) => body.snapshot(),
        }
    }
}

pub struct SoundBodyBuildInput<'a> {
    pub control: &'a AgentControl,
    pub base_freq_hz: f32,
    pub fs: f32,
    pub landscape: Option<&'a LandscapeFrame>,
}

pub trait SoundBodyFactory: Send + Sync {
    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody;
}

#[derive(Default)]
pub struct SoundBodyFactoryRegistry {
    factories: HashMap<String, Arc<dyn SoundBodyFactory>>,
}

impl SoundBodyFactoryRegistry {
    pub fn register(&mut self, id: &str, factory: Arc<dyn SoundBodyFactory>) {
        self.factories
            .insert(id.trim().to_ascii_lowercase(), factory);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn SoundBodyFactory>> {
        self.factories.get(&id.trim().to_ascii_lowercase()).cloned()
    }
}

pub fn body_method_id(method: BodyMethod) -> &'static str {
    match method {
        BodyMethod::Sine => "sine",
        BodyMethod::Harmonic => "harmonic",
        BodyMethod::Modal => "modal",
    }
}

fn global_factory_registry() -> &'static Mutex<SoundBodyFactoryRegistry> {
    static REGISTRY: OnceLock<Mutex<SoundBodyFactoryRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = SoundBodyFactoryRegistry::default();
        registry.register("sine", Arc::new(SineBodyFactory));
        registry.register("harmonic", Arc::new(HarmonicBodyFactory));
        Mutex::new(registry)
    })
}

pub fn register_sound_body_factory(id: &str, factory: Arc<dyn SoundBodyFactory>) {
    let mut registry = global_factory_registry()
        .lock()
        .expect("sound body factory registry");
    registry.register(id, factory);
}

pub fn build_sound_body_from_control(
    control: &AgentControl,
    base_freq_hz: f32,
    fs: f32,
    landscape: Option<&LandscapeFrame>,
    rng: &mut SmallRng,
) -> AnySoundBody {
    let method_id = body_method_id(control.body.method);
    let factory = {
        let registry = global_factory_registry()
            .lock()
            .expect("sound body factory registry");
        registry.get(method_id)
    };
    let input = SoundBodyBuildInput {
        control,
        base_freq_hz,
        fs,
        landscape,
    };

    if let Some(factory) = factory {
        return factory.build(&input, rng);
    }

    warn!(
        "Unknown sound body method '{}'; falling back to harmonic factory",
        method_id
    );
    HarmonicBodyFactory.build(&input, rng)
}

struct SineBodyFactory;

impl SoundBodyFactory for SineBodyFactory {
    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody {
        AnySoundBody::Sine(SineBody {
            freq_hz: input.base_freq_hz.max(1.0),
            amp: input.control.body.amp,
            audio_phase: rng.random_range(0.0..std::f32::consts::TAU),
        })
    }
}

struct HarmonicBodyFactory;

impl SoundBodyFactory for HarmonicBodyFactory {
    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody {
        let timbre = &input.control.body.timbre;
        let genotype = TimbreGenotype {
            mode: crate::life::scenario::HarmonicMode::Harmonic,
            stiffness: timbre.inharmonic,
            brightness: timbre.brightness,
            comb: 0.0,
            damping: 0.5,
            vibrato_rate: 5.0,
            vibrato_depth: timbre.motion * 0.02,
            jitter: timbre.motion,
            unison: timbre.width,
        };

        let fallback_space;
        let eval_space = if let Some(frame) = input.landscape {
            &frame.space
        } else {
            fallback_space = Log2Space::new(55.0, 8000.0, 96);
            &fallback_space
        };
        let custom_ratios = input
            .control
            .body
            .modes
            .as_ref()
            .map(|pattern| pattern.eval(input.base_freq_hz, eval_space, input.landscape, rng))
            .and_then(sanitize_mode_ratios);

        let partials = custom_ratios
            .as_ref()
            .map_or(16usize, |ratios| ratios.len());
        let partials = partials.max(1);
        let mut phases = Vec::with_capacity(partials);
        let mut detune_phases = Vec::with_capacity(partials);
        for _ in 0..partials {
            phases.push(rng.random_range(0.0..std::f32::consts::TAU));
            detune_phases.push(rng.random_range(0.0..std::f32::consts::TAU));
        }

        AnySoundBody::Harmonic(HarmonicBody {
            base_freq_hz: input.base_freq_hz.max(1.0),
            amp: input.control.body.amp,
            genotype,
            lfo_phase: 0.0,
            phases,
            detune_phases,
            jitter_gen: PinkNoise::new(rng.next_u64(), 0.001),
            custom_ratios,
        })
    }
}

fn sanitize_mode_ratios(mut ratios: Vec<f32>) -> Option<Arc<[f32]>> {
    ratios.retain(|r| r.is_finite() && *r > 0.0);
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
    if ratios.is_empty() {
        None
    } else {
        Some(Arc::<[f32]>::from(ratios))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_snapshot_preserves_custom_ratios_and_width() {
        let mut body = HarmonicBody {
            base_freq_hz: 220.0,
            amp: 0.2,
            genotype: TimbreGenotype::default(),
            lfo_phase: 0.0,
            phases: vec![0.0, 0.0, 0.0],
            detune_phases: vec![0.0, 0.0, 0.0],
            jitter_gen: PinkNoise::new(1, 0.001),
            custom_ratios: Some(Arc::<[f32]>::from(vec![1.0, 3.0, 5.0])),
        };
        body.apply_timbre_controls(0.4, 0.0, 0.55, 0.2);
        let snapshot = body.snapshot();
        assert_eq!(snapshot.kind, BodyKind::Harmonic);
        assert!((snapshot.width - 0.55).abs() <= 1.0e-6);
        assert!((snapshot.noise_mix - 0.2).abs() <= 1.0e-6);
        let ratios = snapshot.ratios.expect("ratios");
        assert_eq!(ratios.as_ref(), &[1.0, 3.0, 5.0]);
    }
}
