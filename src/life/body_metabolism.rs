//! Source identity, PCM history and representative bodies without asynchronous delivery.

use super::body_fitness::{BodyFitness, evaluate};
use super::onset_footprint::{Identity, Recipe};
use super::phonation_engine::OnsetKick;
use super::schedule_renderer::{SourcePcm, modal_phase_seed};
use super::sound::Tone;
use super::voice::Voice;
use crate::config::BodyMetabolismConfig;
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::modulation::NeuralRhythms;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::stream::synchronous::SynchronousAnalysis;
use crate::core::timebase::{Tick, Timebase};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unavailable {
    History,
    Tone,
    NoMass,
    InvalidInput,
}

struct Probe {
    identity: Identity,
    recipe: Recipe,
    template: Tone,
    tone: Tone,
}

pub(crate) struct Evaluator {
    pub id: u64,
    pub generation: u32,
    pub birth_sample: Tick,
    pub support_end_sample: Tick,
    pub evaluations: u64,
    environment: SynchronousAnalysis,
    representative: SynchronousAnalysis,
    probe: Option<Probe>,
    pcm: Vec<f32>,
    density_scan: Vec<f32>,
    time: Timebase,
    observation_frames: usize,
    current_sample: Tick,
    valid: bool,
    epoch: u64,
    selected_hop: Tick,
    last_evaluated: Option<Tick>,
    visit_number: u64,
    scored_epoch: u64,
    held: Option<BodyFitness>,
    density_pitch: Option<u32>,
    analysis_key: [u32; 4],
    pub density_builds: u64,
    pub density_pitch_changes: u64,
    pub density_recipe_changes: u64,
}

impl Evaluator {
    fn priority(&self) -> u64 {
        self.visit_number
    }

    fn prepare_body(&mut self, recipe: &Recipe) {
        // This identity selects allocated Tone storage, never a cached density.
        let mut fixed = recipe.clone();
        fixed.freq_hz = 1.0;
        let identity = Identity::new(self.id, self.generation, &fixed);
        if self
            .probe
            .as_ref()
            .is_some_and(|probe| probe.identity == identity)
        {
            return;
        }
        self.density_recipe_changes += u64::from(self.density_pitch.is_some());
        self.density_pitch = None;
        if let Some(probe) = self.probe.as_mut()
            && probe.recipe.body == recipe.body
            && probe.recipe.hold == recipe.hold
            && probe.recipe.adsr == recipe.adsr
        {
            probe
                .template
                .set_representative_modulator(recipe.modulator.clone());
            probe
                .template
                .set_smoothing_tau_sec(recipe.smoothing_tau_sec);
            probe.identity = identity;
            probe.recipe = recipe.clone();
            return;
        }
        let Some(mut template) = Tone::from_parts(
            self.time,
            0,
            recipe.hold,
            recipe.freq_hz,
            1.,
            Some(recipe.body.clone()),
            Some(recipe.modulator.clone()),
            recipe.adsr,
        ) else {
            self.probe = None;
            return;
        };
        template.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
        let kick = OnsetKick { strength: 1. };
        template.schedule_planned_kick(kick);
        template.arm_onset_trigger(kick.strength);
        let tone = Tone::from_parts(
            self.time,
            0,
            recipe.hold,
            recipe.freq_hz,
            1.,
            Some(recipe.body.clone()),
            Some(recipe.modulator.clone()),
            recipe.adsr,
        )
        .expect("the same representative body was just constructed");
        self.probe = Some(Probe {
            identity,
            recipe: recipe.clone(),
            tone,
            template,
        });
    }

    pub(crate) fn score(
        &mut self,
        pitch_log2: f32,
        params: &LandscapeParams,
        shared: &Landscape,
    ) -> Result<BodyFitness, Unavailable> {
        #[cfg(test)]
        let started = std::time::Instant::now();
        if self.scored_epoch != self.epoch {
            self.held = self.score_valid(pitch_log2, params, shared).ok();
            self.scored_epoch = self.epoch;
        }
        #[cfg(test)]
        assay::cost("score", started.elapsed().as_secs_f64());
        self.held.ok_or(Unavailable::History)
    }

    fn score_valid(
        &mut self,
        pitch_log2: f32,
        params: &LandscapeParams,
        shared: &Landscape,
    ) -> Result<BodyFitness, Unavailable> {
        if !self.valid || self.support_end_sample != self.current_sample {
            return Err(Unavailable::History);
        }
        let hz = pitch_log2.exp2();
        if !hz.is_finite() || hz <= 0. {
            return Err(Unavailable::InvalidInput);
        }
        let key = [
            params.fs.to_bits(),
            params.loudness_exp.to_bits(),
            params.ref_power.to_bits(),
            params.tau_ms.to_bits(),
        ];
        if self.analysis_key != key {
            self.density_pitch = None;
            self.analysis_key = key;
        }
        if self.density_pitch != Some(hz.to_bits()) {
            self.density_pitch_changes += u64::from(self.density_pitch.is_some());
            let probe = self.probe.as_mut().ok_or(Unavailable::Tone)?;
            probe.tone.reset_representative(&probe.template, hz);
            probe
                .tone
                .seed_modal_phases(modal_phase_seed(self.id, 0, 0));
            self.representative.reset_density();
            self.density_scan.fill(0.);
            let mut rhythms = NeuralRhythms::default();
            for frame in 0..self.observation_frames {
                let start = frame as u64 * self.time.hop as u64;
                probe.tone.kick_planned_if_due(start);
                probe.tone.render_block(
                    start,
                    self.time.fs,
                    1. / self.time.fs,
                    &mut rhythms,
                    &mut self.pcm,
                );
                let density = self.representative.density(&self.pcm, params);
                for (sum, &value) in self
                    .density_scan
                    .iter_mut()
                    .zip(density.subjective_intensity)
                {
                    *sum += value;
                }
            }
            let inv = 1. / self.observation_frames as f32;
            for value in &mut self.density_scan {
                *value *= inv;
            }
            self.density_pitch = Some(hz.to_bits());
            self.density_builds += 1;
        }
        let environment = &mut self.environment.landscape;
        environment.recompute_consonance(params);
        if params.habituation.enabled {
            environment.apply_habituation(
                &shared.perc_habituation_state_scan,
                params.consonance_representation.theta,
                &params.consonance_representation,
            );
        }
        let value = evaluate(
            &environment.space,
            &self.density_scan,
            &self.environment.du,
            &environment.consonance_field_score_eff,
            &params.consonance_representation,
        )
        .map_err(|error| match error {
            super::body_fitness::Unsupported::NoInBandMass => Unavailable::NoMass,
            super::body_fitness::Unsupported::InvalidInput => Unavailable::InvalidInput,
        })?;
        self.evaluations = self.evaluations.saturating_add(1);
        Ok(value)
    }
}

pub(crate) struct BodyMetabolism {
    pub pcm: Vec<SourcePcm>,
    sources: Vec<Evaluator>,
    shared: SynchronousAnalysis,
    subtraction: Vec<f32>,
    time: Timebase,
    settings: BodyMetabolismConfig,
    next_sample: Tick,
    latest_visit: u64,
    valid: bool,
    params: LandscapeParams,
}

impl BodyMetabolism {
    pub(crate) fn new(
        nsgt: RtNsgtKernelLog2,
        params: &LandscapeParams,
        settings: BodyMetabolismConfig,
    ) -> Self {
        let time = Timebase {
            fs: params.fs,
            hop: nsgt.hop(),
        };
        Self {
            pcm: Vec::new(),
            sources: Vec::new(),
            shared: SynchronousAnalysis::new(nsgt, params),
            subtraction: vec![0.; time.hop],
            time,
            settings,
            next_sample: 0,
            latest_visit: 0,
            valid: true,
            params: params.clone(),
        }
    }

    pub(crate) fn prepare(&mut self, voices: &[Voice], now: Tick, params: &LandscapeParams) {
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.params.consonance_kernel = params.consonance_kernel;
        self.params.consonance_representation = params.consonance_representation;
        self.params.consonance_density_roughness_gain = params.consonance_density_roughness_gain;
        self.params.roughness_aversion = params.roughness_aversion;
        self.params.habituation = params.habituation;
        self.params.loudness_exp = params.loudness_exp;
        self.params.ref_power = params.ref_power;
        self.params.tau_ms = params.tau_ms;
        if params.fs.to_bits() != self.time.fs.to_bits() {
            self.valid = false;
        }
        if now != self.next_sample {
            self.valid = false;
        }
        for index in (0..self.sources.len()).rev() {
            let source = &self.sources[index];
            if !voices.iter().any(|voice| {
                voice.id() == source.id && voice.metadata.generation == source.generation
            }) {
                self.sources.remove(index);
                self.pcm.remove(index);
            }
        }
        let hold = (f64::from(self.settings.representative_hold_sec) * f64::from(self.time.fs))
            .round() as Tick;
        for voice in voices {
            if !self.sources.iter().any(|source| {
                (source.id, source.generation) == (voice.id(), voice.metadata.generation)
            }) && !voice.supports_body_metabolism()
            {
                continue;
            }
            let index = self
                .sources
                .iter()
                .position(|source| {
                    source.id == voice.id() && source.generation == voice.metadata.generation
                })
                .unwrap_or_else(|| {
                    // Births join the back in Voice-list order, just like the former queue.
                    self.latest_visit = self
                        .latest_visit
                        .checked_add(1)
                        .expect("body visit number fits u64");
                    let bins = self.shared.landscape.space.n_bins();
                    self.sources.push(Evaluator {
                        id: voice.id(),
                        generation: voice.metadata.generation,
                        birth_sample: now,
                        support_end_sample: now,
                        current_sample: now,
                        valid: self.valid,
                        environment: self.shared.clone(),
                        representative: self.shared.clone(),
                        pcm: vec![0.; self.time.hop],
                        density_scan: vec![0.; bins],
                        probe: None,
                        time: self.time,
                        observation_frames: self.settings.observation_frames,
                        evaluations: 0,
                        epoch: 0,
                        selected_hop: Tick::MAX,
                        last_evaluated: None,
                        visit_number: self.latest_visit,
                        scored_epoch: 0,
                        held: None,
                        density_pitch: None,
                        analysis_key: [
                            params.fs.to_bits(),
                            params.loudness_exp.to_bits(),
                            params.ref_power.to_bits(),
                            params.tau_ms.to_bits(),
                        ],
                        density_builds: 0,
                        density_pitch_changes: 0,
                        density_recipe_changes: 0,
                    });
                    self.pcm.push(SourcePcm {
                        id: voice.id(),
                        generation: voice.metadata.generation,
                        habitat: vec![0.; self.time.hop],
                    });
                    self.sources.len() - 1
                });
            let source = &mut self.sources[index];
            source.current_sample = now;
            source.valid &= self.valid;
            if !source.valid {
                source.held = None;
            }
            source.prepare_body(&voice.representative_body_recipe(self.time.fs, hold));
        }
        #[cfg(test)]
        assay::cost("prepare", started.elapsed().as_secs_f64());
    }

    pub(crate) fn source(
        &mut self,
        id: u64,
        generation: u32,
    ) -> Option<(&mut Evaluator, &LandscapeParams)> {
        self.sources
            .iter_mut()
            .find(|source| source.id == id && source.generation == generation)
            .map(|source| (source, &self.params))
    }

    #[cfg(test)]
    pub(crate) fn evaluation_count(&self) -> u64 {
        self.sources.iter().map(|source| source.evaluations).sum()
    }

    fn swap_slots(&mut self, a: usize, b: usize) {
        self.sources.swap(a, b);
        self.pcm.swap(a, b);
    }

    fn sift_selected(&mut self, mut root: usize, len: usize) {
        loop {
            let left = 2 * root + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let worst =
                if right < len && self.sources[right].priority() > self.sources[left].priority() {
                    right
                } else {
                    left
                };
            if self.sources[root].priority() >= self.sources[worst].priority() {
                break;
            }
            self.swap_slots(root, worst);
            root = worst;
        }
    }

    fn select_next(&mut self, now: Tick, end: Tick) {
        let budget = self.settings.updates_per_hop.min(self.sources.len());
        if budget == 0 {
            return;
        }
        // The first budget slots are scratch for bounded selection, not another container.
        for root in (0..budget / 2).rev() {
            self.sift_selected(root, budget);
        }
        for index in budget..self.sources.len() {
            if self.sources[index].priority() < self.sources[0].priority() {
                self.swap_slots(0, index);
                self.sift_selected(0, budget);
            }
        }
        // Assign fresh numbers in the former queue's visit order, including within a hop.
        for len in (2..=budget).rev() {
            self.swap_slots(0, len - 1);
            self.sift_selected(0, len - 1);
        }
        for source in &mut self.sources[..budget] {
            source.epoch += 1;
            source.selected_hop = now;
            source.last_evaluated = Some(end);
            self.latest_visit = self
                .latest_visit
                .checked_add(1)
                .expect("body visit number fits u64");
            source.visit_number = self.latest_visit;
        }
    }

    pub(crate) fn observe(&mut self, now: Tick, mixed: &[f32], params: &LandscapeParams) {
        #[cfg(test)]
        let started = std::time::Instant::now();
        assert_eq!(mixed.len(), self.time.hop, "mixed PCM hop length");
        let Some(end) = now.checked_add(self.time.hop as Tick) else {
            self.valid = false;
            return;
        };
        if now != self.next_sample || mixed.iter().any(|value| !value.is_finite()) {
            self.valid = false;
        }
        self.select_next(now, end);
        for (source, pcm) in self.sources.iter_mut().zip(&self.pcm) {
            assert_eq!((source.id, source.generation), (pcm.id, pcm.generation));
            assert!(source.birth_sample <= now, "source before birth");
            for (out, (&mix, &own)) in self
                .subtraction
                .iter_mut()
                .zip(mixed.iter().zip(&pcm.habitat))
            {
                *out = mix - own;
            }
            source.valid &= self.valid && self.subtraction.iter().all(|value| value.is_finite());
            if !source.valid {
                source.held = None;
            }
            if source.valid {
                if source.selected_hop == now {
                    let gap =
                        i32::try_from((end - source.support_end_sample) / self.time.hop as Tick)
                            .expect("body environment hop clock fits i32");
                    source
                        .environment
                        .process_gap(&self.subtraction, gap, params);
                    source.support_end_sample = end;
                } else {
                    source.environment.skip(&self.subtraction);
                }
            }
        }
        if self.valid {
            self.shared.process(mixed, params);
        }
        self.next_sample = end;
        #[cfg(test)]
        assay::cost("observe", started.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
#[path = "body_metabolism_assay.rs"]
pub(crate) mod assay;
