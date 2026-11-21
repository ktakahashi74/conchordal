use super::articulation_core::ArticulationSignal;
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::DEFAULT_MODE_COUNT;
use crate::life::sound::mode_utils::{
    active_cluster_unison, cluster_detune_mul, cluster_spread_cents_from_public, cluster_weights,
    public_spread_from_cluster_spread_cents, sanitize_cluster_unison,
};
use crate::life::sound::pitch_motion::MotionClock;
use crate::life::sound::spectral::{
    add_log2_energy, add_vibrato_energy, brightness_from_spectral_slope, harmonic_gain,
    harmonic_ratio, spectral_slope_from_brightness,
};
use crate::life::sound::{BodyKind, BodySnapshot};
use crate::scenario::control::DEFAULT_TIMBRE_UNISON;
use crate::scenario::control::{BodyMethod, VoiceControl};
use crate::scenario::{SoundBodyConfig, TimbreGenotype};
use rand::{Rng, RngExt, rngs::SmallRng};
#[cfg(any(test, feature = "research"))]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::warn;

#[cfg(any(test, feature = "research"))]
impl BodySnapshot {
    pub(crate) fn hash_into(&self, hasher: &mut Sha256) {
        // Preserve the byte encoding of representative-onset recipe identities.
        hasher.update([match self.kind {
            BodyKind::Sine => 0u8,
            BodyKind::Harmonic => 1,
            BodyKind::Modal => 2,
        }]);
        for value in [
            self.amp_scale,
            self.brightness,
            self.inharmonic,
            self.spread,
            self.motion,
        ] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        if self.attack_noise > 0. {
            hasher.update(b"attack_noise");
            hasher.update(self.attack_noise.to_le_bytes());
        }
        hasher.update((self.unison as u64).to_le_bytes());
        match self.ratios.as_deref() {
            None => hasher.update([0u8]),
            Some(ratios) => {
                hasher.update([1u8]);
                hasher.update((ratios.len() as u64).to_le_bytes());
                for ratio in ratios {
                    hasher.update(ratio.to_bits().to_le_bytes());
                }
            }
        }
    }

    #[cfg(any(test, feature = "research"))]
    #[cfg(feature = "research")]
    pub(crate) fn supports_body_models(&self, renderer_phase3: bool) -> bool {
        self.kind.supports_body_models(renderer_phase3)
    }

    #[cfg(any(test, feature = "research"))]
    pub(crate) fn has_same_structure(&self, other: &Self) -> bool {
        self.kind == other.kind && self.unison == other.unison && self.ratios == other.ratios
    }
}

pub trait SoundBody: Send {
    /// Nominal active partial density from the body's spectral self-report.
    /// Build at birth/reconfiguration; it describes this body, not mixed PCM.
    /// Override with None when no spectral self-report is available. Empty is silence.
    fn project_support_density(
        &mut self,
        space: &Log2Space,
        ref_power: f32,
        loudness_exp: f32,
        signal: &ArticulationSignal,
    ) -> Option<Vec<(isize, f32)>> {
        let mut amplitudes = vec![0.; space.n_bins()];
        // Existing projections disagree on whether the body gain is included.
        // Project at unit gain, then apply the physical body gain exactly once.
        let amp = self.amp();
        self.set_amp(1.);
        self.project_spectral_body(&mut amplitudes, space, signal);
        self.set_amp(amp);
        for a in &mut amplitudes {
            *a *= amp.max(0.);
        }
        Some(crate::core::harmonic_support::body_density::project(
            &amplitudes,
            self.base_freq_hz(),
            space,
            ref_power,
            loudness_exp,
        ))
    }

    fn base_freq_hz(&self) -> f32;
    fn set_freq(&mut self, freq: f32);
    fn set_pitch_log2(&mut self, log_freq: f32);
    fn set_amp(&mut self, amp: f32);
    fn amp(&self) -> f32;
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
        unison: usize,
    );
    fn snapshot(&self) -> BodySnapshot;
}

#[derive(Debug, Clone)]
pub struct SineBody {
    pub freq_hz: f32,
    pub amp: f32,
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
        _unison: usize,
    ) {
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Sine,
            attack_noise: 0.,
            amp_scale: 1.0,
            brightness: 0.0,
            inharmonic: 0.0,
            spread: 0.0,
            unison: DEFAULT_TIMBRE_UNISON,
            motion: 0.0,
            ratios: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HarmonicBody {
    pub(crate) motion_clock: MotionClock,
    pub(crate) attack_noise: f32,
    pub base_freq_hz: f32,
    pub amp: f32,
    pub partials: usize,
    pub genotype: TimbreGenotype,
    pub cluster_spread_cents: f32,
    pub cluster_unison: usize,
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

    fn active_cluster_unison(&self) -> usize {
        active_cluster_unison(self.cluster_spread_cents, self.cluster_unison)
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
        let cluster_unison = self.active_cluster_unison();
        let weights = cluster_weights(self.cluster_spread_cents, cluster_unison);
        for idx in 0..partials {
            let ratio = self.partial_ratio(idx);
            let part_amp = self.compute_partial_amp(idx, signal.amplitude);
            for unison_idx in 0..cluster_unison {
                let detune =
                    cluster_detune_mul(self.cluster_spread_cents, cluster_unison, unison_idx);
                let freq = self.base_freq_hz * ratio * detune;
                let amplitude = amp_scale * part_amp * weights[unison_idx];
                add_vibrato_energy(amps, space, freq, amplitude, self.genotype.vibrato_depth);
            }
        }
    }

    fn apply_timbre_controls(
        &mut self,
        brightness: f32,
        inharmonic: f32,
        motion: f32,
        spread: f32,
        unison: usize,
    ) {
        self.genotype.spectral_slope = spectral_slope_from_brightness(brightness);
        self.genotype.stiffness = inharmonic.clamp(0.0, 1.0);
        self.genotype.jitter = motion.clamp(0.0, 1.0);
        self.genotype.vibrato_depth = self.genotype.jitter * 0.02;
        self.cluster_spread_cents = cluster_spread_cents_from_public(spread);
        self.cluster_unison = sanitize_cluster_unison(unison);
    }

    fn snapshot(&self) -> BodySnapshot {
        BodySnapshot {
            kind: BodyKind::Harmonic,
            attack_noise: self.attack_noise,
            amp_scale: 1.0,
            brightness: brightness_from_spectral_slope(self.genotype.spectral_slope),
            inharmonic: self.genotype.stiffness.clamp(0.0, 1.0),
            spread: public_spread_from_cluster_spread_cents(self.cluster_spread_cents),
            unison: self.cluster_unison,
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
            SoundBodyConfig::Sine { phase } => {
                // Keep the original phase draw in the caller's RNG stream.
                if phase.is_none() {
                    let _ = rng.random_range(0.0f32..std::f32::consts::TAU);
                }
                AnySoundBody::Sine(SineBody { freq_hz, amp })
            }
            SoundBodyConfig::Harmonic { genotype, partials } => {
                let partials = partials.unwrap_or(DEFAULT_MODE_COUNT).max(1);
                // Keep the original phase and jitter draws in the caller's RNG stream.
                for _ in 0..(partials * DEFAULT_TIMBRE_UNISON) {
                    let _ = rng.random_range(0.0f32..std::f32::consts::TAU);
                }
                let motion_clock = MotionClock {
                    seed: rng.next_u64(),
                    birth: 0,
                };
                AnySoundBody::Harmonic(HarmonicBody {
                    motion_clock,
                    attack_noise: 0.,
                    base_freq_hz: freq_hz,
                    amp,
                    partials,
                    genotype: genotype.clone(),
                    cluster_spread_cents: 0.0,
                    cluster_unison: DEFAULT_TIMBRE_UNISON,
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
    fn project_support_density(
        &mut self,
        space: &Log2Space,
        ref_power: f32,
        loudness_exp: f32,
        signal: &ArticulationSignal,
    ) -> Option<Vec<(isize, f32)>> {
        match self {
            Self::Sine(body) => {
                body.project_support_density(space, ref_power, loudness_exp, signal)
            }
            Self::Harmonic(body) => {
                body.project_support_density(space, ref_power, loudness_exp, signal)
            }
            Self::Dyn(body) => body.project_support_density(space, ref_power, loudness_exp, signal),
        }
    }

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
        unison: usize,
    ) {
        match self {
            AnySoundBody::Sine(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, unison)
            }
            AnySoundBody::Harmonic(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, unison)
            }
            AnySoundBody::Dyn(body) => {
                body.apply_timbre_controls(brightness, inharmonic, motion, spread, unison)
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
    pub control: &'a VoiceControl,
    pub base_freq_hz: f32,
    pub fs: f32,
    pub landscape: Option<&'a Landscape>,
}

pub trait SoundBodyFactory: Send + Sync {
    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody;

    /// Declare conformance of birth and PCM body models to the selected renderer.
    #[cfg(feature = "research")]
    fn model_kind(&self, _renderer_phase3: bool) -> Option<BodyKind> {
        None
    }
}

#[cfg(feature = "research")]
pub(crate) fn supports_body_model(
    control: &VoiceControl,
    snapshot: &BodySnapshot,
    renderer_phase3: bool,
) -> bool {
    model_kind(control, renderer_phase3) == Some(snapshot.kind)
}

#[cfg(feature = "research")]
pub(crate) fn model_kind(control: &VoiceControl, renderer_phase3: bool) -> Option<BodyKind> {
    let registry = global_factory_registry()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let method = body_method_id(control.body.method);
    registry.factories.get(method).map_or_else(
        || HarmonicBodyFactory.model_kind(renderer_phase3),
        |factory| factory.model_kind(renderer_phase3),
    )
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
    let mut registry = match global_factory_registry().lock() {
        Ok(registry) => registry,
        Err(poisoned) => {
            warn!("Sound body factory registry lock was poisoned; recovering");
            poisoned.into_inner()
        }
    };
    registry.register(id, factory);
}

pub fn build_sound_body_from_control(
    control: &VoiceControl,
    base_freq_hz: f32,
    fs: f32,
    landscape: Option<&Landscape>,
    rng: &mut SmallRng,
) -> AnySoundBody {
    let method_id = body_method_id(control.body.method);
    let factory = {
        // The registry is a plain map, so a poisoned lock leaves it usable. Recovering
        // beats panicking: this runs on the voice spawn path, which the audio thread
        // reaches during a performance.
        let registry = match global_factory_registry().lock() {
            Ok(registry) => registry,
            Err(poisoned) => {
                warn!("Sound body factory registry lock was poisoned; recovering");
                poisoned.into_inner()
            }
        };
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
    #[cfg(feature = "research")]
    fn model_kind(&self, renderer_phase3: bool) -> Option<BodyKind> {
        BodyKind::Sine
            .supports_body_models(renderer_phase3)
            .then_some(BodyKind::Sine)
    }

    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody {
        AnySoundBody::from_config(
            &SoundBodyConfig::Sine { phase: None },
            input.base_freq_hz.max(1.0),
            input.control.body.amp,
            rng,
        )
    }
}

struct HarmonicBodyFactory;

impl SoundBodyFactory for HarmonicBodyFactory {
    #[cfg(feature = "research")]
    fn model_kind(&self, renderer_phase3: bool) -> Option<BodyKind> {
        BodyKind::Harmonic
            .supports_body_models(renderer_phase3)
            .then_some(BodyKind::Harmonic)
    }

    fn build(&self, input: &SoundBodyBuildInput<'_>, rng: &mut SmallRng) -> AnySoundBody {
        let timbre = &input.control.body.timbre;
        let genotype = TimbreGenotype {
            mode: crate::scenario::HarmonicMode::Harmonic,
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
            fallback_space = crate::core::log2space::production_space(input.fs as u32)
                .expect("validated production sample rate");
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
        let cluster_unison = sanitize_cluster_unison(timbre.unison);
        let active_cluster_unison = active_cluster_unison(cluster_spread_cents, cluster_unison);
        // Keep the original phase and jitter draws in the caller's RNG stream.
        for _ in 0..(partials * active_cluster_unison) {
            let _ = rng.random_range(0.0f32..std::f32::consts::TAU);
        }
        let motion_clock = MotionClock {
            seed: rng.next_u64(),
            birth: 0,
        };

        AnySoundBody::Harmonic(HarmonicBody {
            motion_clock,
            attack_noise: timbre.attack_noise,
            base_freq_hz: input.base_freq_hz.max(1.0),
            amp: input.control.body.amp,
            partials,
            genotype,
            cluster_spread_cents,
            cluster_unison,
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
            motion_clock: Default::default(),
            attack_noise: 0.,
            base_freq_hz: 220.0,
            amp: 0.2,
            partials: 3,
            genotype: TimbreGenotype::default(),
            cluster_spread_cents: 0.0,
            cluster_unison: DEFAULT_TIMBRE_UNISON,
            custom_ratios: Some(Arc::<[f32]>::from(vec![1.0, 3.0, 5.0])),
        };
        body.apply_timbre_controls(0.4, 0.0, 0.2, 0.5, 3);
        let snapshot = body.snapshot();
        assert_eq!(snapshot.kind, BodyKind::Harmonic);
        assert!((snapshot.brightness - 0.4).abs() <= 1.0e-6);
        assert!((snapshot.inharmonic - 0.0).abs() <= 1.0e-6);
        assert!((snapshot.spread - 0.5).abs() <= 1.0e-6);
        assert_eq!(snapshot.unison, 3);
        assert!((snapshot.motion - 0.2).abs() <= 1.0e-6);
        let ratios = snapshot.ratios.expect("ratios");
        assert_eq!(ratios.as_ref(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn harmonic_body_limits_partials_by_nominal_pitch() {
        let body = HarmonicBody {
            motion_clock: Default::default(),
            attack_noise: 0.,
            base_freq_hz: 4_000.0,
            amp: 0.2,
            partials: DEFAULT_MODE_COUNT,
            genotype: TimbreGenotype::default(),
            cluster_spread_cents: 0.0,
            cluster_unison: DEFAULT_TIMBRE_UNISON,
            custom_ratios: None,
        };
        assert_eq!(body.partial_count_below(48_000.0 * 0.49), 5);
    }
}
