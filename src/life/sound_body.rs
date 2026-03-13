use super::articulation_core::{ArticulationSignal, PinkNoise};
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::DEFAULT_MODE_COUNT;
use crate::life::control::DEFAULT_TIMBRE_VOICES;
use crate::life::control::{AgentControl, BodyMethod};
use crate::life::scenario::{SoundBodyConfig, TimbreGenotype};
use crate::life::sound::mode_utils::{
    active_cluster_voices, cluster_detune_mul, cluster_gain, cluster_spread_cents_from_public,
    public_spread_from_cluster_spread_cents, sanitize_cluster_voices,
};
use crate::life::sound::spectral::{
    add_log2_energy, brightness_from_spectral_slope, harmonic_gain, harmonic_ratio,
    spectral_slope_from_brightness,
};
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
    fn apply_timbre_controls(
        &mut self,
        brightness: f32,
        inharmonic: f32,
        motion: f32,
        spread: f32,
        voices: usize,
    );
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
        _motion: f32,
        _spread: f32,
        _voices: usize,
    ) {
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Sine,
            amp_scale: 1.0,
            brightness: 0.0,
            inharmonic: 0.0,
            spread: 0.0,
            voices: DEFAULT_TIMBRE_VOICES,
            motion: 0.0,
            ratios: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HarmonicBody {
    pub base_freq_hz: f32,
    pub amp: f32,
    pub partials: usize,
    pub genotype: TimbreGenotype,
    pub cluster_spread_cents: f32,
    pub cluster_voices: usize,
    pub lfo_phase: f32,
    pub phases: Vec<f32>,
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
        self.partials.max(1)
    }

    fn active_cluster_voices(&self) -> usize {
        active_cluster_voices(self.cluster_spread_cents, self.cluster_voices)
    }

    fn sync_phase_storage(&mut self) {
        let phase_len = self
            .partials
            .max(1)
            .saturating_mul(self.active_cluster_voices());
        self.phases.resize(phase_len.max(1), 0.0);
    }

    fn partial_count_below(&self, max_freq_hz: f32) -> usize {
        let partials = self.partial_count();
        if partials == 0 || !max_freq_hz.is_finite() || max_freq_hz <= 0.0 {
            return partials;
        }
        let base_freq_hz = self.base_freq_hz.max(1.0);
        let mut active = 0usize;
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let freq_hz = base_freq_hz * ratio;
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                continue;
            }
            if freq_hz > max_freq_hz {
                break;
            }
            active += 1;
        }
        active
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

    fn articulate_wave(&mut self, sample: &mut f32, fs: f32, dt: f32, signal: &ArticulationSignal) {
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        let partials = self.partial_count_below((fs * 0.49).max(1.0));
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

        let mut acc = 0.0;
        let cluster_voices = self.active_cluster_voices();
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let part_amp = cluster_gain(
                self.compute_partial_amp(idx, signal.amplitude),
                self.cluster_spread_cents,
                cluster_voices,
            );
            if part_amp <= 0.0 {
                continue;
            }
            for voice_idx in 0..cluster_voices {
                let detune =
                    cluster_detune_mul(self.cluster_spread_cents, cluster_voices, voice_idx);
                let freq = base_freq * ratio * detune;
                if !freq.is_finite() || freq <= 0.0 || freq > (fs * 0.49).max(1.0) {
                    continue;
                }
                let phase = &mut self.phases[idx * cluster_voices + voice_idx];
                *phase = (*phase + 2.0 * PI * freq * dt).rem_euclid(2.0 * PI);
                acc += part_amp * phase.sin();
            }
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
        let partials = self.partial_count_below(space.fmax.max(1.0));
        if partials == 0 {
            return;
        }
        let amp_scale = self.amp.max(0.0) * signal.amplitude;
        let cluster_voices = self.active_cluster_voices();
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let part_amp = cluster_gain(
                self.compute_partial_amp(idx, signal.amplitude),
                self.cluster_spread_cents,
                cluster_voices,
            );
            for voice_idx in 0..cluster_voices {
                let detune =
                    cluster_detune_mul(self.cluster_spread_cents, cluster_voices, voice_idx);
                let freq = self.base_freq_hz * ratio * detune;
                add_log2_energy(amps, space, freq, amp_scale * part_amp);
            }
        }
    }

    fn apply_timbre_controls(
        &mut self,
        brightness: f32,
        inharmonic: f32,
        motion: f32,
        spread: f32,
        voices: usize,
    ) {
        self.genotype.spectral_slope = spectral_slope_from_brightness(brightness);
        self.genotype.stiffness = inharmonic.clamp(0.0, 1.0);
        self.genotype.jitter = motion.clamp(0.0, 1.0);
        self.genotype.vibrato_depth = self.genotype.jitter * 0.02;
        self.cluster_spread_cents = cluster_spread_cents_from_public(spread);
        self.cluster_voices = sanitize_cluster_voices(voices);
        self.sync_phase_storage();
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Harmonic,
            amp_scale: 1.0,
            brightness: brightness_from_spectral_slope(self.genotype.spectral_slope),
            inharmonic: self.genotype.stiffness.clamp(0.0, 1.0),
            spread: public_spread_from_cluster_spread_cents(self.cluster_spread_cents),
            voices: self.cluster_voices,
            motion: self.genotype.jitter.clamp(0.0, 1.0),
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
                let partials = partials.unwrap_or(DEFAULT_MODE_COUNT).max(1);
                let mut phases = Vec::with_capacity(partials * DEFAULT_TIMBRE_VOICES);
                for _ in 0..(partials * DEFAULT_TIMBRE_VOICES) {
                    phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                }
                AnySoundBody::Harmonic(HarmonicBody {
                    base_freq_hz: freq_hz,
                    amp,
                    partials,
                    genotype: genotype.clone(),
                    cluster_spread_cents: 0.0,
                    cluster_voices: DEFAULT_TIMBRE_VOICES,
                    lfo_phase: 0.0,
                    phases,
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

    fn apply_timbre_controls(
        &mut self,
        brightness: f32,
        inharmonic: f32,
        motion: f32,
        spread: f32,
        voices: usize,
    ) {
        match self {
            AnySoundBody::Sine(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, voices)
            }
            AnySoundBody::Harmonic(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, voices)
            }
            AnySoundBody::Dyn(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, voices)
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
            spectral_slope: spectral_slope_from_brightness(timbre.brightness),
            comb: 0.0,
            damping: 0.5,
            vibrato_rate: 5.0,
            vibrato_depth: timbre.motion * 0.02,
            jitter: timbre.motion,
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
            .map_or(DEFAULT_MODE_COUNT, |ratios| ratios.len());
        let partials = partials.max(1);
        let cluster_spread_cents = cluster_spread_cents_from_public(timbre.spread);
        let cluster_voices = sanitize_cluster_voices(timbre.voices);
        let active_cluster_voices = active_cluster_voices(cluster_spread_cents, cluster_voices);
        let mut phases = Vec::with_capacity(partials * active_cluster_voices);
        for _ in 0..(partials * active_cluster_voices) {
            phases.push(rng.random_range(0.0..std::f32::consts::TAU));
        }

        AnySoundBody::Harmonic(HarmonicBody {
            base_freq_hz: input.base_freq_hz.max(1.0),
            amp: input.control.body.amp,
            partials,
            genotype,
            cluster_spread_cents,
            cluster_voices,
            lfo_phase: 0.0,
            phases,
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
    fn harmonic_snapshot_preserves_custom_ratios() {
        let mut body = HarmonicBody {
            base_freq_hz: 220.0,
            amp: 0.2,
            partials: 3,
            genotype: TimbreGenotype::default(),
            cluster_spread_cents: 0.0,
            cluster_voices: DEFAULT_TIMBRE_VOICES,
            lfo_phase: 0.0,
            phases: vec![0.0, 0.0, 0.0],
            jitter_gen: PinkNoise::new(1, 0.001),
            custom_ratios: Some(Arc::<[f32]>::from(vec![1.0, 3.0, 5.0])),
        };
        body.apply_timbre_controls(0.4, 0.0, 0.2, 0.5, 3);
        let snapshot = body.snapshot();
        assert_eq!(snapshot.kind, BodyKind::Harmonic);
        assert!((snapshot.brightness - 0.4).abs() <= 1.0e-6);
        assert!((snapshot.inharmonic - 0.0).abs() <= 1.0e-6);
        assert!((snapshot.spread - 0.5).abs() <= 1.0e-6);
        assert_eq!(snapshot.voices, 3);
        assert!((snapshot.motion - 0.2).abs() <= 1.0e-6);
        let ratios = snapshot.ratios.expect("ratios");
        assert_eq!(ratios.as_ref(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn harmonic_body_limits_partials_by_nominal_pitch() {
        let body = HarmonicBody {
            base_freq_hz: 4_000.0,
            amp: 0.2,
            partials: DEFAULT_MODE_COUNT,
            genotype: TimbreGenotype::default(),
            cluster_spread_cents: 0.0,
            cluster_voices: DEFAULT_TIMBRE_VOICES,
            lfo_phase: 0.0,
            phases: vec![0.0; DEFAULT_MODE_COUNT],
            jitter_gen: PinkNoise::new(2, 0.001),
            custom_ratios: None,
        };
        assert_eq!(body.partial_count_below(48_000.0 * 0.49), 5);
    }
}
