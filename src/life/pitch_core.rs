use crate::core::landscape::Landscape;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::scenario::PitchCoreConfig;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct TargetProposal {
    pub target_pitch_log2: f32,
    pub salience: f32,
}

#[allow(clippy::too_many_arguments)]
pub trait PitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal;
}

#[derive(Debug, Clone)]
pub struct PitchHillClimbPitchCore {
    neighbor_step_log2: f32,
    tessitura_center: f32,
    tessitura_gravity: f32,
    improvement_threshold: f32,
    exploration: f32,
    persistence: f32,
}

impl PitchHillClimbPitchCore {
    pub fn new(
        neighbor_step_cents: f32,
        tessitura_center: f32,
        tessitura_gravity: f32,
        improvement_threshold: f32,
        exploration: f32,
        persistence: f32,
    ) -> Self {
        Self {
            neighbor_step_log2: neighbor_step_cents / 1200.0,
            tessitura_center,
            tessitura_gravity,
            improvement_threshold,
            exploration: exploration.clamp(0.0, 1.0),
            persistence: persistence.clamp(0.0, 1.0),
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        self.exploration = value.clamp(0.0, 1.0);
    }

    pub fn set_persistence(&mut self, value: f32) {
        self.persistence = value.clamp(0.0, 1.0);
    }
}

impl PitchCore for PitchHillClimbPitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        _current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        _features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal {
        let (fmin, fmax) = landscape.freq_bounds_log2();
        let current_target_log2 = current_target_log2.clamp(fmin, fmax);
        let perfect_fifth = 1.5f32.log2();
        let imperfect_fifth = 0.66f32.log2();
        let mut candidates = vec![
            current_target_log2,
            current_target_log2 + self.neighbor_step_log2,
            current_target_log2 - self.neighbor_step_log2,
            current_target_log2 + perfect_fifth,
            current_target_log2 + imperfect_fifth,
        ];
        candidates.retain(|f| f.is_finite());

        let adjusted_score = |pitch_log2: f32| -> f32 {
            let clamped = pitch_log2.clamp(fmin, fmax);
            let score = landscape.evaluate_pitch01_log2(clamped);
            let distance_oct = (clamped - current_pitch_log2).abs();
            let penalty = distance_oct * integration_window * 0.5;
            let dist = clamped - self.tessitura_center;
            let gravity_penalty = dist * dist * self.tessitura_gravity;
            let base = score - penalty - gravity_penalty;
            let idx = landscape.space.index_of_log2(clamped).unwrap_or(0);
            base + perceptual.score_adjustment(idx)
        };

        let mut best_pitch = current_target_log2;
        let mut best_score = f32::MIN;
        for p in candidates {
            let clamped = p.clamp(fmin, fmax);
            let adjusted = adjusted_score(clamped);
            if adjusted > best_score {
                best_score = adjusted;
                best_pitch = clamped;
            }
        }

        let current_adjusted = adjusted_score(current_target_log2);
        let improvement = best_score - current_adjusted;
        let mut target_pitch_log2 = current_target_log2;

        if improvement > self.improvement_threshold {
            target_pitch_log2 = best_pitch;
        } else {
            let satisfaction = ((current_adjusted + 1.0) * 0.5).clamp(0.0, 1.0);
            let mut stay_prob = self.persistence.clamp(0.0, 1.0) * satisfaction;
            stay_prob = stay_prob.clamp(0.0, 1.0);
            let exploration = self.exploration.clamp(0.0, 1.0);
            stay_prob = (stay_prob * (1.0 - exploration)).clamp(0.0, 1.0);
            if rng.random_range(0.0..1.0) > stay_prob {
                target_pitch_log2 = best_pitch;
            }
        }

        TargetProposal {
            target_pitch_log2,
            salience: (improvement / 0.2).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnyPitchCore {
    PitchHillClimb(PitchHillClimbPitchCore),
}

impl PitchCore for AnyPitchCore {
    fn propose_target<R: Rng + ?Sized>(
        &mut self,
        current_pitch_log2: f32,
        current_target_log2: f32,
        current_freq_hz: f32,
        integration_window: f32,
        landscape: &Landscape,
        perceptual: &PerceptualContext,
        features: &FeaturesNow,
        rng: &mut R,
    ) -> TargetProposal {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.propose_target(
                current_pitch_log2,
                current_target_log2,
                current_freq_hz,
                integration_window,
                landscape,
                perceptual,
                features,
                rng,
            ),
        }
    }
}

impl AnyPitchCore {
    pub fn from_config<R: Rng + ?Sized>(
        config: &PitchCoreConfig,
        initial_pitch_log2: f32,
        _rng: &mut R,
    ) -> Self {
        match config {
            PitchCoreConfig::PitchHillClimb {
                neighbor_step_cents,
                tessitura_gravity,
                improvement_threshold,
                exploration,
                persistence,
            } => {
                let neighbor_step_cents = neighbor_step_cents.unwrap_or(200.0);
                let tessitura_gravity = tessitura_gravity.unwrap_or(0.1);
                let improvement_threshold = improvement_threshold.unwrap_or(0.1);
                let exploration = exploration.unwrap_or(0.0);
                let persistence = persistence.unwrap_or(0.5);
                AnyPitchCore::PitchHillClimb(PitchHillClimbPitchCore::new(
                    neighbor_step_cents,
                    initial_pitch_log2,
                    tessitura_gravity,
                    improvement_threshold,
                    exploration,
                    persistence,
                ))
            }
        }
    }

    pub fn set_exploration(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_exploration(value),
        }
    }

    pub fn set_persistence(&mut self, value: f32) {
        match self {
            AnyPitchCore::PitchHillClimb(core) => core.set_persistence(value),
        }
    }
}
